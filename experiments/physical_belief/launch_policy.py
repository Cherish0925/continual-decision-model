"""Run the matched physical-belief/core comparison on two GPUs."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--updates", type=int, default=400)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--unroll", type=int, default=16)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    jobs = []
    for device, seed, parent, physical in ((0, 221, 183, 211), (1, 222, 184, 212)):
        for mode in ("ordinary", "reset", "carry"):
            name = f"{mode}_seed{seed}"
            command = [sys.executable, "-m", "experiments.physical_belief.train_policy",
                "--data", "runs/discoveryworld/active_experiments/data_large",
                "--checkpoint", f"runs/discoveryworld/active_experiments/rehearsal/counterfactual_seed{parent}/update1200.pt",
                "--belief-checkpoint", f"runs/physical_belief/pretrain{physical}/last.pt",
                "--mixture-checkpoint", "runs/discoveryworld/calibration/mixture/shared_train_only_seed31/last.pt",
                "--output", str(args.output / name), "--mode", mode,
                "--device", f"cuda:{device}", "--seed", str(seed),
                "--updates", str(args.updates), "--batch", str(args.batch), "--unroll", str(args.unroll)]
            jobs.append({"name": name, "device": device, "seed": seed, "parent_seed": parent,
                         "mode": mode, "command": command})
    plan = {"jobs": jobs, "comparisons": "matched recorded/generated task plan within each parent",
            "selection": "final fixed-budget checkpoint",
            "shared": "physical context tokens, policy architecture, native tools and outer update count",
            "difference": "fixed prior; trainable per-task prior; trainable prior and cross-task posterior memory"}
    (args.output / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")

    def run(job):
        started = time.monotonic()
        with (args.output / f"{job['name']}.log").open("w") as log:
            process = subprocess.Popen(job["command"], stdout=log, stderr=subprocess.STDOUT)
            row = {**job, "pid": process.pid}
            (args.output / f"{job['name']}.process.json").write_text(json.dumps(row, indent=2) + "\n")
            code = process.wait()
        return {"name": job["name"], "exit_code": code, "seconds": time.monotonic() - started}

    completed = []
    with ThreadPoolExecutor(max_workers=len(jobs)) as executor:
        futures = [executor.submit(run, job) for job in jobs]
        for future in as_completed(futures):
            row = future.result(); completed.append(row)
            print(json.dumps(row), flush=True)
            (args.output / "completed.json").write_text(json.dumps(completed, indent=2) + "\n")
    if any(row["exit_code"] for row in completed):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
