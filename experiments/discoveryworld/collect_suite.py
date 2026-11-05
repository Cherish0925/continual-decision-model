"""Collect mixed-theme native trajectories with a local or hosted teacher."""
from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import redirect_stdout
from copy import deepcopy
import json
from pathlib import Path
import shutil
import subprocess
import sys
from time import perf_counter
import traceback

from .adapter import TrajectoryWriter
from .full_suite import THEMES, SKILLS, LIMITS, simulator_build, write_json
from .structured_observation import StructuredDiscoveryWorldEnv
from .native_worker import JsonConnection


def training_cases(seeds, difficulties=tuple(LIMITS), themes=THEMES, max_steps=None, small_skills=False):
    if any(seed < 100 for seed in seeds):
        raise ValueError("Collection seeds must be outside the official 0--4 evaluation set")
    if max_steps is not None and max_steps < 1:
        raise ValueError("The per-episode action cap must be positive")
    if small_skills:
        return [dict(id=f"skill{i:02d}-normal-{seed}", theme=theme, difficulty="Normal",
                     seed=seed, max_steps=min(max_steps or 100, 100))
                for seed in seeds for i, theme in enumerate(SKILLS, 1)]
    return [dict(id=f"{i:02d}-{difficulty.lower()}-{seed}", theme=theme, difficulty=difficulty,
                 seed=seed, max_steps=min(max_steps or LIMITS[difficulty], LIMITS[difficulty]))
            for seed in seeds for difficulty in difficulties for i, theme in enumerate(THEMES, 1) if theme in themes]


def environment_worker(pipe):
    env = writer = log = None
    case = path = None
    started = 0.
    restored_steps = 0

    def finish(status, error=None):
        nonlocal writer, log
        if writer:
            writer.__exit__(None, None, None)
            writer = None
        row = dict(case, status=status, steps=env.steps, task_metrics=env.metrics,
            progress=sum(r["progress"] for r in env.metrics)/len(env.metrics),
            success=all(r["completed_successfully"] for r in env.metrics),
            seconds=perf_counter()-started, resumed_prefix_steps=restored_steps)
        if error:
            row["error"] = error
        write_json(path / "summary.json", row)
        log.close()
        log = None
        return row

    try:
        while True:
            command = pipe.recv()
            if command[0] == "close":
                break
            if command[0] == "reset":
                case, root, resume, rng_source = command[1:5]
                request_settings = command[5] if len(command) > 5 else None
                path = Path(root) / case["id"]
                path.mkdir(parents=True, exist_ok=resume)
                recorded = path / 'trajectory.jsonl'
                prior = [json.loads(line) for line in recorded.read_text().splitlines()] if resume and recorded.exists() else []
                reuse = resume and (path / 'run.json').exists()
                prior_metadata = json.loads((path / 'run.json').read_text()) if reuse else None
                teacher_rows = {}
                if prior:
                    teacher_rows = {row['step']: row for line in (path / 'teacher.jsonl').read_text().splitlines()
                                    if 'action' in (row := json.loads(line)) and 'error' not in row}
                log = (path / "environment.log").open("a" if resume else "w")
                started = perf_counter()
                restored_steps = len(prior)
                with redirect_stdout(log):
                    if request_settings:
                        from .request_environment import RequestEnvironment
                        env = RequestEnvironment(path, case["max_steps"], **request_settings)
                    else:
                        env = StructuredDiscoveryWorldEnv(path, case["max_steps"])
                    observation = env.reset(case["theme"], case["difficulty"], case["seed"])
                    if reuse:
                        env.restore_initial_rng(path / "initial_rng.json.gz")
                        env.episode_id = prior_metadata['episode_id']
                    else:
                        if rng_source:
                            env.restore_initial_rng(Path(rng_source) / case["id"] / "initial_rng.json.gz")
                        env.save_initial_rng(path / "initial_rng.json.gz")
                    if request_settings:
                        env.record_native = False
                    for row in prior:
                        replayed = env.step(row['action'])
                        if replayed['action_result'] != row['action_result'] or replayed['next_observation']['ui'] != row['next_observation']['ui']:
                            raise ValueError(f"Native replay differs at {case['id']} step {row['index']}")
                    if request_settings:
                        env.record_native = True
                    observation = env.policy_input()
                    config = dict(env.configuration, policy="frozen_batched_language_teacher",
                        action_descriptions=env.actions, teleport_locations=env.teleports,
                        initial_task_metrics=env.metrics)
                    writer = TrajectoryWriter(path, config, resume=reuse)
                replay = [dict(transition={key: row[key] for key in ('observation','action','action_result','next_observation')},
                               generated=teacher_rows[row['index']]) for row in prior]
                response = dict(observation=observation, done=env.ended, replay=replay)
                if env.ended:
                    response['summary'] = finish('evaluated')
                pipe.send(response)
            elif command[0] == "step":
                generated = command[1]
                with (path / "teacher.jsonl").open("a") as stream:
                    stream.write(json.dumps(dict(step=getattr(env, 'decision_steps', env.steps), **generated), ensure_ascii=False)+"\n")
                if "error" in generated:
                    pipe.send(dict(done=True, summary=finish("policy_error", generated["error"])))
                    continue
                with redirect_stdout(log):
                    transition = env.step(generated["action"])
                    writer.append(transition)
                public = {k: deepcopy(transition[k]) for k in
                          ("index", "action", "action_result", "next_observation")}
                message = dict(observation=env.policy_input(), transition=public, done=env.ended)
                if env.ended:
                    message["summary"] = finish("evaluated")
                pipe.send(message)
    except (Exception, SystemExit):
        pipe.send(dict(error=traceback.format_exc(), case=case))
    finally:
        if writer:
            writer.__exit__(None, None, None)
        if log:
            log.close()
        pipe.close()


def collect(args):
    from .native_teacher import NativeTeacher
    cases = training_cases(args.seeds, args.difficulties, getattr(args, "themes", THEMES),
                           getattr(args, "max_steps", None), getattr(args, "small_skills", False))
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    pending = []
    for case in cases:
        path = args.output / case["id"] / "summary.json"
        if args.resume and path.exists() and (row := json.loads(path.read_text()))['status'] == 'evaluated':
            rows.append(row)
        else:
            pending.append(case)
    shared = dict(max_new_tokens=args.max_new_tokens, max_input_tokens=args.max_input_tokens,
                  temperature=args.temperature, reasoning_effort=args.reasoning_effort)
    hosted = getattr(args, "deepseek_flash", False) or getattr(args, "bailian_flash", False)
    if hosted:
        from .deepseek_teacher import DeepSeekFlashTeacher
        teacher_type = DeepSeekFlashTeacher
        if getattr(args, 'object_requests', False):
            from .request_teacher import RequestTeacher
            teacher_type = RequestTeacher
        teacher = teacher_type(**shared,
            provider="bailian" if getattr(args, "bailian_flash", False) else "deepseek",
            model=getattr(args, "teacher_model", None),
            action_corrections=getattr(args, "action_corrections", 0))
    else:
        teacher = NativeTeacher(args.model, device=args.device, dequantize=args.dequantize, attention=args.attention,
                                endpoint=args.endpoint, served_model=args.served_model,
                                model_source=args.model_source, api_style=args.api_style, **shared)
    collection = args.output / "collection.json"
    earlier = json.loads(collection.read_text()) if collection.exists() else {}
    rng_source = str(args.rng_from.resolve()) if getattr(args, "rng_from", None) else None
    if earlier and (hosted or earlier.get("teacher", {}).get("inference_backend") in
                    ("DeepSeek API", "Alibaba Cloud Bailian API")):
        if (not args.resume or (earlier.get("teacher") != teacher.configuration and not getattr(args,'teacher_continuation',False))
                or earlier.get("cases") != cases or earlier.get("rng_from") != rng_source):
            raise ValueError("Use a separate collection for a different teacher, case list or hosted configuration")
    interrupted = list(earlier.get('interrupted_episodes', []))
    if getattr(args, 'restart_interrupted', False):
        for case in pending:
            folder = args.output / case['id']
            if not (folder / 'run.json').exists():
                continue
            archive = args.output / 'interrupted' / case['id']
            archive.mkdir(parents=True, exist_ok=True)
            destination = archive / str(len(list(archive.iterdir())))
            trajectory = folder / 'trajectory.jsonl'
            steps = len(trajectory.read_text().splitlines()) if trajectory.exists() else 0
            shutil.move(str(folder), str(destination))
            interrupted.append(dict(id=case['id'], steps=steps,
                path=str(destination.relative_to(args.output)),
                continuation='fresh episode; original trajectory retained separately'))
    if not args.resume:
        earlier = {}
    attempts = earlier.get('previous_attempts', [])
    if earlier:
        attempts.append({key: value for key, value in earlier.items() if key != 'previous_attempts'})
    write_json(collection, dict(teacher=teacher.configuration, previous_attempts=attempts,
        cases=cases, status='running', simulator=simulator_build(), rng_from=rng_source,
        interrupted_episodes=interrupted,
        split="new native instances; law disjointness not asserted"))
    args.collection_started = True
    workers = []
    next_case = 0

    def start_worker():
        process = subprocess.Popen(
            [sys.executable, '-u', '-m', 'experiments.discoveryworld.native_worker', '--collection'],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
        return dict(pipe=JsonConnection(process.stdout, process.stdin), process=process)

    def close_worker(worker):
        process = worker['process']
        if process.poll() is None:
            try:
                worker['pipe'].send(('close',))
            except (BrokenPipeError, EOFError):
                pass
        try:
            process.communicate(timeout=30)
        except subprocess.TimeoutExpired:
            process.terminate()
            process.communicate()

    def receive(worker):
        try:
            return worker['pipe'].recv()
        except (EOFError, BrokenPipeError, ConnectionResetError) as error:
            # One crashed simulator does not discard other independent worlds.
            # Keep its trace as interrupted data, never as a policy failure.
            case = worker['case']
            folder = args.output / case['id']
            folder.mkdir(parents=True, exist_ok=True)
            trajectory = folder / 'trajectory.jsonl'
            records = trajectory.read_text().splitlines() if trajectory.exists() else []
            last = json.loads(records[-1]) if records else {}
            metrics = last.get('task_metrics', [])
            row = dict(case, status='environment_error', steps=len(records),
                task_metrics=metrics, progress=None, success=False,
                error=f'{type(error).__name__}: simulator subprocess disconnected',
                exit_code=worker['process'].poll())
            write_json(folder / 'summary.json', row)
            close_worker(worker)
            worker.update(start_worker())
            return dict(done=True, summary=row)

    def send(worker, message):
        try:
            worker['pipe'].send(message)
        except BrokenPipeError:
            # receive() records the closed simulator and retains its episode.
            pass

    def assign(worker):
        nonlocal next_case
        if next_case == len(pending):
            worker["active"] = False
            return
        case = pending[next_case]
        next_case += 1
        worker['case'] = case
        reset = ("reset", case, str(args.output), args.resume, rng_source)
        if getattr(args, 'object_requests', False):
            reset += (dict(max_request_actions=args.max_request_actions),)
        send(worker, reset)
        message = receive(worker)
        if "error" in message:
            raise RuntimeError(message["error"])
        if message['done']:
            rows.append(message['summary'])
            print(json.dumps(message['summary']), flush=True)
            write_json(args.output / 'summary.json', rows)
            assign(worker)
            return
        state = teacher.initial()
        for item in message.get('replay', []):
            state['objects'].observe(item['transition']['observation'])
            teacher.observe(state, item['transition'], item['generated'])
        worker.update(active=not message['done'], case=case, observation=message["observation"], state=state)

    try:
        for _ in range(min(args.workers, len(pending))):
            worker = start_worker()
            workers.append(worker)
            assign(worker)
        def consume(w, result):
            response = receive(w)
            if 'error' in response:
                raise RuntimeError(response['error'])
            if 'transition' in response:
                teacher.observe(w['state'], response['transition'], result)
            if response['done']:
                rows.append(response['summary'])
                print(json.dumps(response['summary']), flush=True)
                write_json(args.output / 'summary.json', rows)
                assign(w)
            else:
                w['observation'] = response['observation']

        if teacher.endpoint:
            # Each environment advances as soon as its own action is ready.
            # The inference server batches requests continuously; a long
            # generation does not hold up all the other scientific episodes.
            with ThreadPoolExecutor(args.workers) as pool:
                futures = {}
                def submit(w):
                    if w['active']:
                        future = pool.submit(teacher.batch, [w['observation']], [w['state']])
                        futures[future] = w
                for w in workers:
                    submit(w)
                decisions = 0
                while futures:
                    finished, _ = wait(futures, return_when=FIRST_COMPLETED)
                    for future in finished:
                        w = futures.pop(future)
                        try:
                            result = future.result()[0]
                        except Exception as error:
                            # Preserve a failed policy attempt and continue the
                            # other independent environments; no action is invented.
                            result = {'error': f'{type(error).__name__}: {error}',
                                      'prompt_version': teacher.configuration['prompt_version']}
                        send(w, ('step', result))
                        consume(w, result)
                        submit(w)
                        decisions += 1
                    if decisions // 100 != (decisions - len(finished)) // 100:
                        print(json.dumps(dict(decisions=decisions, completed=len(rows),
                                              total=len(cases), active=len(futures))), flush=True)
        else:
            batches = 0
            while active := [w for w in workers if w['active']]:
                generated = teacher.batch([w['observation'] for w in active], [w['state'] for w in active])
                for w, result in zip(active, generated):
                    send(w, ('step', result))
                for w, result in zip(active, generated):
                    consume(w, result)
                batches += 1
                if batches % 10 == 0:
                    print(json.dumps(dict(batches=batches, completed=len(rows), total=len(cases),
                        generation_seconds=generated[0]['batch_seconds'], active=len(active))), flush=True)
    finally:
        for w in workers:
            close_worker(w)
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="data/cache/gpt-oss-20b")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=[1000,1001,1002,1003])
    p.add_argument("--difficulties", nargs="+", choices=tuple(LIMITS), default=list(LIMITS))
    tasks = p.add_mutually_exclusive_group()
    tasks.add_argument("--themes", nargs="+", choices=THEMES, default=THEMES)
    tasks.add_argument("--small-skills", action="store_true", help="Collect the ten native basic-operation tasks, capped at 100 actions")
    p.add_argument("--max-steps", type=int, help="Optional collection cap below the native per-episode limit")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--object-requests", action="store_true", help="Use the shared executor with hosted planners")
    p.add_argument("--max-request-actions", type=int, default=12)
    p.add_argument("--max-new-tokens", type=int, default=1024)
    p.add_argument("--max-input-tokens", type=int, default=22000)
    p.add_argument("--dequantize", action="store_true")
    p.add_argument("--attention", default="flex_attention")
    backend = p.add_mutually_exclusive_group()
    backend.add_argument("--endpoint", help="Local completion-server URL, including /v1")
    backend.add_argument("--deepseek-flash", action="store_true", help="Use the official DeepSeek chat API")
    backend.add_argument("--bailian-flash", action="store_true", help="Use Flash through Alibaba Cloud Bailian")
    p.add_argument("--teacher-model", help="Hosted model ID; defaults to the provider's Flash model")
    p.add_argument("--action-corrections", type=int, default=0,
                   help="Return malformed hosted proposals to the teacher before ending an episode; all calls are recorded")
    p.add_argument("--served-model", default="teacher")
    p.add_argument("--api-style", choices=('chat', 'completion'), default='chat')
    p.add_argument("--reasoning-effort", choices=('none', 'low', 'medium', 'high', 'max'), default='low')
    p.add_argument("--model-source", default="unsloth/gpt-oss-20b")
    p.add_argument("--temperature", type=float, default=.7)
    p.add_argument("--resume", action="store_true")
    p.add_argument('--teacher-continuation',action='store_true',
                   help='With --resume, record a changed teacher configuration and continue recorded unfinished worlds')
    p.add_argument("--restart-interrupted", action="store_true",
                   help="With --resume, preserve unfinished trajectories separately and start fresh episodes")
    p.add_argument("--rng-from", type=Path, help="Use another teacher collection's initial simulator RNG snapshots")
    args=p.parse_args()
    if args.object_requests and not (args.deepseek_flash or args.bailian_flash):
        p.error('--object-requests requires a hosted planner')
    if args.restart_interrupted and not args.resume:
        p.error('--restart-interrupted requires --resume')
    if args.teacher_continuation and not args.resume:
        p.error('--teacher-continuation requires --resume')
    try:
        collect(args)
    except Exception as error:
        path = args.output / 'collection.json'
        if getattr(args, "collection_started", False) and path.exists():
            metadata = json.loads(path.read_text())
            write_json(path, {**metadata, 'status': 'error', 'error': f'{type(error).__name__}: {error}'})
        raise
    path = args.output / 'collection.json'
    metadata = json.loads(path.read_text())
    write_json(path, {**metadata, 'status': 'completed'})


if __name__ == "__main__":
    main()
