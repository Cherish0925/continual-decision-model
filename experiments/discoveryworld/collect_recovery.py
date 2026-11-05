"""Collect teacher continuations from states reached by a frozen student.

The native environment executes every action. Student actions supply history
and outcome targets, never imitation labels. A teacher sees the same public
interface and the actual student history, then attempts the remaining task.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stdout
import json
import multiprocessing
from pathlib import Path
from time import perf_counter
import traceback

import numpy as np
import torch

from .adapter import TrajectoryWriter
from .collect_suite import training_cases
from .deepseek_teacher import DeepSeekFlashTeacher
from .evaluate import LearnedPolicy
from .full_suite import THEMES, seeded_object_initialization, write_json
from .structured_observation import StructuredDiscoveryWorldEnv


def run_case(job):
    case, config = job
    path = Path(config["output"]) / case["id"]
    path.mkdir()
    started = perf_counter()
    row = dict(case, student_steps=0, teacher_steps=0)
    torch.set_num_threads(config["threads"])
    try:
        student = LearnedPolicy(config["checkpoint"], config["feature_cache"], device="cpu",
                                temperature=config["student_temperature"], sampling_seed=case["seed"])
        teacher = DeepSeekFlashTeacher(provider=config["provider"], model=config["teacher_model"],
            max_input_tokens=config["max_input_tokens"], max_new_tokens=config["max_new_tokens"],
            reasoning_effort=config["reasoning_effort"], temperature=0., action_corrections=2)
        state = teacher.initial()
        np.random.seed((config["object_rng_seed"] + case["seed"]) % 2**32)
        with (path / "environment.log").open("w") as log, redirect_stdout(log), \
                seeded_object_initialization(config["object_rng_seed"]):
            env = StructuredDiscoveryWorldEnv(path, case["max_steps"])
            obs = env.reset(case["theme"], case["difficulty"], case["seed"])
            env.save_initial_rng(path / "initial_rng.json.gz")
            metadata = dict(env.configuration, action_descriptions=env.actions,
                teleport_locations=env.teleports, initial_task_metrics=env.metrics,
                policy="student_prefix_teacher_continuation", student_steps=config["student_steps"],
                student_epoch=student.checkpoint["epoch"],
                teacher={k: v for k, v in teacher.configuration.items() if k != "endpoint"})
            with TrajectoryWriter(path, metadata) as writer, (path / "teacher.jsonl").open("w") as replies:
                while not env.ended:
                    student_turn = env.steps < config["student_steps"]
                    if student_turn:
                        state["objects"].observe(obs)
                        state["measurements"].observe_initial(obs)
                        generated = dict(action=student(obs), memory=state["notebook"])
                    else:
                        generated = teacher.batch([obs], [state])[0]
                        replies.write(json.dumps(dict(step=env.steps, **generated), ensure_ascii=False) + "\n")
                        replies.flush()
                        if "error" in generated:
                            raise RuntimeError(generated["error"])
                    transition = env.step(generated["action"])
                    writer.append({**transition, "controller": "student" if student_turn else "teacher",
                                   "imitate_action": not student_turn})
                    teacher.observe(state, transition, generated)
                    if student_turn:
                        student.observe(transition)
                    row["student_steps" if student_turn else "teacher_steps"] += 1
                    obs = env.policy_input()
            row.update(status="evaluated", steps=env.steps, task_metrics=env.metrics,
                       success=all(task["completed_successfully"] for task in env.metrics),
                       progress=sum(task["progress"] for task in env.metrics) / len(env.metrics))
    except Exception:
        row.update(status="error", error=traceback.format_exc())
    row["seconds"] = perf_counter() - started
    write_json(path / "summary.json", row)
    return row


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--feature-cache", type=Path, required=True)
    p.add_argument("--seeds", type=int, nargs="+", required=True)
    p.add_argument("--difficulties", nargs="+", choices=("Easy", "Normal", "Challenge"), default=["Easy", "Normal"])
    p.add_argument("--themes", nargs="+", choices=THEMES, default=list(THEMES))
    p.add_argument("--student-steps", type=int, default=8)
    p.add_argument("--student-temperature", type=float, default=0.)
    p.add_argument("--max-steps", type=int)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--object-rng-seed", type=int, default=73105)
    p.add_argument("--provider", choices=("bailian", "deepseek"), default="bailian")
    p.add_argument("--teacher-model")
    p.add_argument("--max-input-tokens", type=int, default=65536)
    p.add_argument("--max-new-tokens", type=int, default=32768)
    p.add_argument("--reasoning-effort", choices=("none", "low", "medium", "high"), default="high")
    p.add_argument("--resume", action="store_true", help="Keep completed cases; interrupted cases require a separate output")
    args = p.parse_args(argv)
    if min(args.workers, args.threads) < 1 or args.student_steps < 0:
        p.error("Use positive workers/threads and nonnegative student steps")
    cases = training_cases(args.seeds, args.difficulties, args.themes, args.max_steps)
    # Source model development instances must not enter the training collection.
    source = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    validation = {tuple(item) for item in source["validation_instances"]}
    if any((case["theme"], case["difficulty"], case["seed"]) in validation for case in cases):
        raise ValueError("Recovery training overlaps the student's development instances")
    del source
    args.output.mkdir(parents=True, exist_ok=args.resume)
    config = {k: str(v.resolve()) if isinstance(v, Path) else v for k, v in vars(args).items()}
    rows, pending = [], []
    for case in cases:
        path = args.output / case["id"]
        if args.resume and (path / "summary.json").exists():
            prior = json.loads((path / "summary.json").read_text())
            if prior["status"] == "evaluated":
                rows.append(prior)
                continue
        if path.exists():
            raise ValueError(f"Preserve the interrupted case and use a new output: {case['id']}")
        pending.append(case)
    manifest = dict(configuration=config, cases=cases,
        protocol="student prefix followed by teacher recovery; training data, not independent student evaluation")
    write_json(args.output / "collection.json", dict(manifest, status="running", results=rows))
    with ProcessPoolExecutor(args.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = [pool.submit(run_case, (case, config)) for case in pending]
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            print(json.dumps({k: row[k] for k in ("id", "status", "student_steps", "teacher_steps")}), flush=True)
            write_json(args.output / "collection.json", dict(manifest, status="running", results=rows))
    status = "completed" if all(row["status"] == "evaluated" for row in rows) else "error"
    write_json(args.output / "collection.json", dict(manifest, status=status, results=rows))
    return int(status != "completed")


if __name__ == "__main__":
    raise SystemExit(main())
