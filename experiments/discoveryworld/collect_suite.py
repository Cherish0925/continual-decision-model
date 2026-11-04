"""Collect mixed-theme native trajectories with one batched local teacher."""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
from copy import deepcopy
import json
import multiprocessing as mp
from pathlib import Path
from time import perf_counter
import traceback

from .adapter import TrajectoryWriter
from .full_suite import THEMES, LIMITS, simulator_build, write_json
from .structured_observation import StructuredDiscoveryWorldEnv


def training_cases(seeds, difficulties=tuple(LIMITS)):
    if any(seed < 100 for seed in seeds):
        raise ValueError("Collection seeds must be outside the official 0--4 evaluation set")
    return [dict(id=f"{i:02d}-{difficulty.lower()}-{seed}", theme=theme,
                 difficulty=difficulty, seed=seed, max_steps=LIMITS[difficulty])
            for seed in seeds for difficulty in difficulties for i, theme in enumerate(THEMES, 1)]


def environment_worker(pipe):
    import torch
    torch.set_num_threads(1)
    env = writer = log = None
    case = path = None
    started = 0.

    def finish(status, error=None):
        nonlocal writer, log
        if writer:
            writer.__exit__(None, None, None)
            writer = None
        row = dict(case, status=status, steps=env.steps, task_metrics=env.metrics,
            progress=sum(r["progress"] for r in env.metrics)/len(env.metrics),
            success=all(r["completed_successfully"] for r in env.metrics),
            seconds=perf_counter()-started)
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
                case, root = command[1:]
                path = Path(root) / case["id"]
                path.mkdir(parents=True, exist_ok=False)
                log = (path / "environment.log").open("w")
                started = perf_counter()
                with redirect_stdout(log):
                    env = StructuredDiscoveryWorldEnv(path, case["max_steps"])
                    observation = env.reset(case["theme"], case["difficulty"], case["seed"])
                    env.save_initial_rng(path / "initial_rng.json.gz")
                    config = dict(env.configuration, policy="frozen_batched_language_teacher",
                        action_descriptions=env.actions, teleport_locations=env.teleports,
                        initial_task_metrics=env.metrics)
                    writer = TrajectoryWriter(path, config)
                pipe.send(dict(observation=observation, done=False))
            elif command[0] == "step":
                generated = command[1]
                with (path / "teacher.jsonl").open("a") as stream:
                    stream.write(json.dumps(dict(step=env.steps, **generated), ensure_ascii=False)+"\n")
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
    cases = training_cases(args.seeds, args.difficulties)
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    pending = []
    for case in cases:
        path = args.output / case["id"] / "summary.json"
        if args.resume and path.exists():
            rows.append(json.loads(path.read_text()))
        else:
            pending.append(case)
    teacher = NativeTeacher(args.model, device=args.device, max_new_tokens=args.max_new_tokens,
                            dequantize=args.dequantize, attention=args.attention)
    write_json(args.output / "collection.json", dict(teacher=teacher.configuration,
        cases=cases, simulator=simulator_build(), split="new native instances; law disjointness not asserted"))
    ctx = mp.get_context("spawn")
    workers = []
    next_case = 0

    def assign(worker):
        nonlocal next_case
        if next_case == len(pending):
            worker["active"] = False
            return
        case = pending[next_case]
        next_case += 1
        worker["pipe"].send(("reset", case, str(args.output)))
        message = worker["pipe"].recv()
        if "error" in message:
            raise RuntimeError(message["error"])
        worker.update(active=True, case=case, observation=message["observation"], state=teacher.initial())

    try:
        for _ in range(min(args.workers, len(pending))):
            parent, child = ctx.Pipe()
            process = ctx.Process(target=environment_worker, args=(child,))
            process.start()
            child.close()
            worker = dict(pipe=parent, process=process)
            workers.append(worker)
            assign(worker)
        batches = 0
        while active := [w for w in workers if w["active"]]:
            generated = teacher.batch([w["observation"] for w in active], [w["state"] for w in active])
            for w, result in zip(active, generated):
                w["pipe"].send(("step", result))
            for w, result in zip(active, generated):
                response = w["pipe"].recv()
                if "error" in response:
                    raise RuntimeError(response["error"])
                if "transition" in response:
                    teacher.observe(w["state"], response["transition"], result)
                if response["done"]:
                    rows.append(response["summary"])
                    print(json.dumps(response["summary"]), flush=True)
                    write_json(args.output / "summary.json", rows)
                    assign(w)
                else:
                    w["observation"] = response["observation"]
            batches += 1
            if batches % 10 == 0:
                print(json.dumps(dict(batches=batches, completed=len(rows), total=len(cases),
                    generation_seconds=generated[0]["batch_seconds"], active=len(active))), flush=True)
    finally:
        for w in workers:
            if w["process"].is_alive():
                w["pipe"].send(("close",))
            w["process"].join(timeout=10)
            w["pipe"].close()
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="data/cache/gpt-oss-20b")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=[1000,1001,1002,1003])
    p.add_argument("--difficulties", nargs="+", choices=tuple(LIMITS), default=list(LIMITS))
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--max-new-tokens", type=int, default=1024)
    p.add_argument("--dequantize", action="store_true")
    p.add_argument("--attention", default="flex_attention")
    p.add_argument("--resume", action="store_true")
    args=p.parse_args()
    collect(args)


if __name__ == "__main__":
    main()
