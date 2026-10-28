"""Collect public native measurement histories without rendering images."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stdout
from dataclasses import asdict
import json
import multiprocessing
from pathlib import Path
from time import perf_counter

from .adapter import TrajectoryWriter
from .reference import ReactorLabReference
from .structured_observation import StructuredDiscoveryWorldEnv


def collect(job):
    seed, root, max_steps, split = job
    output = root / f"seed{seed}"
    output.mkdir(parents=True)
    reference = ReactorLabReference()
    started = perf_counter()
    with (output / "native.log").open("w") as log, redirect_stdout(log):
        env = StructuredDiscoveryWorldEnv(output, max_steps)
        observation = env.reset("Reactor Lab", "Normal", seed)
        initial = env.save_initial_rng(output / "initial_rng.json.gz")
        metadata = {**env.configuration, "policy": "scripted_public_reactor_reference_v1",
                    "is_learned_policy": False, "split": split,
                    "initial_rng_snapshot": {"path": "initial_rng.json.gz", **initial},
                    "action_descriptions": env.actions, "teleport_locations": env.teleports,
                    "collection": "public_UI_without_image_rendering"}
        error = None
        with TrajectoryWriter(output, metadata) as writer:
            try:
                while not env.ended:
                    action = reference(observation)
                    transition = env.step(action)
                    writer.append(transition)
                    observation = env.policy_input()
            except (RuntimeError, ValueError, StopIteration) as exc:
                error = f"{type(exc).__name__}: {exc}"
        if env.ended and error is None:
            try:
                reference(observation)
            except RuntimeError:
                pass
        result = {"seed": seed, "transitions": env.steps, "task_metrics": env.metrics,
                  "error": error, "elapsed_seconds": perf_counter() - started,
                  "ended": env.ended, "is_learned_policy": False,
                  "reference_evidence": reference.evidence,
                  "hypotheses": [asdict(h) for h in reference.hypotheses]}
        (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return {key: value for key, value in result.items() if key not in ("reference_evidence", "hypotheses")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--split", choices=("train", "validation"), default="train")
    args = parser.parse_args()
    if any(seed % 5 == 4 for seed in args.seeds):
        parser.error("Training collection excludes the reserved relation family")
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = [pool.submit(collect, (seed, args.output, args.max_steps, args.split)) for seed in args.seeds]
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            (args.output / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")
            print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
