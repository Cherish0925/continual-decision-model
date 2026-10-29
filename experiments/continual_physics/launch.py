"""Run two parent repetitions of the matched external-budget comparison."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import subprocess
import sys
import threading
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--environment-truth", type=Path, required=True)
    parser.add_argument("--execution-geometry", type=Path)
    parser.add_argument("--condition-failure-weights", action="store_true")
    parser.add_argument("--cache-observations", action="store_true")
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--updates", type=int, default=200)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--external-per-stream", type=int, default=32)
    parser.add_argument("--jobs-per-gpu", type=int, default=4)
    parser.add_argument("--seeds", type=int, nargs="+", choices=(231, 232), default=(231, 232))
    parser.add_argument("--modes", nargs="+", choices=("full", "discard_physical", "task_reset", "fixed_schedule", "recurrent_reference"),
                        default=("full", "discard_physical", "task_reset", "fixed_schedule"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    jobs = []
    for device, seed, campaign, parent in ((0, 231, "joint_20261029", 221),
                                           (1, 232, "joint_repeat_20261029", 222)):
        if seed not in args.seeds:
            continue
        seed += args.seed_offset
        parent_root = Path("runs/physical_belief") / campaign / f"carry_seed{parent}"
        for mode in args.modes:
            name = f"{mode}_seed{seed}"
            command = ["-m", "experiments.continual_physics.train",
                "--data", "runs/discoveryworld/active_experiments/data_large",
                "--checkpoint", str(parent_root / "policy.pt"),
                "--belief-checkpoint", str(parent_root / "belief.pt"),
                "--mixture-checkpoint", "runs/discoveryworld/calibration/mixture/shared_train_only_seed31/last.pt",
                "--environment-truth", str(args.environment_truth),
                "--output", str(args.output / name), "--mode", mode,
                "--device", f"cuda:{device}", "--seed", str(seed),
                "--updates", str(args.updates), "--batch", str(args.batch),
                "--external-per-stream", str(args.external_per_stream)]
            if args.execution_geometry is not None:
                command += ["--execution-geometry", str(args.execution_geometry)]
            if args.condition_failure_weights:
                command += ["--condition-failure-weights"]
            if args.cache_observations:
                command += ["--cache-observations"]
            jobs.append({"name": name, "gpu": device, "seed": seed, "parent_seed": parent,
                         "mode": mode, "arguments": command})
    plan = {"jobs": jobs,
            "external_requests_per_training": args.updates * args.batch * args.external_per_stream,
            "matched": "same parent within repetition, initialization seed, external-request budget and task plan",
            "recorded_separately": "internal activities, waiting ticks, native primitive actions and wall time",
            "execution": "native tool execution with 400 primitive-action task budget" if args.execution_geometry else "single-setting proxy",
            "condition_failure_weights": args.condition_failure_weights,
            "selection": "fixed final update"}
    (args.output / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    slots = {gpu: threading.Semaphore(args.jobs_per_gpu) for gpu in (0, 1)}

    def run(job):
        with slots[job["gpu"]]:
            started = time.monotonic()
            with (args.output / f"{job['name']}.log").open("w") as log:
                result = subprocess.run([sys.executable, *job["arguments"]], stdout=log, stderr=subprocess.STDOUT)
            return {"name": job["name"], "exit_code": result.returncode,
                    "seconds": time.monotonic() - started}

    completed = []
    with ThreadPoolExecutor(max_workers=len(jobs)) as executor:
        for future in as_completed([executor.submit(run, job) for job in jobs]):
            row = future.result(); completed.append(row)
            print(json.dumps(row), flush=True)
            (args.output / "completed.json").write_text(json.dumps(completed, indent=2) + "\n")
    if any(row["exit_code"] for row in completed):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
