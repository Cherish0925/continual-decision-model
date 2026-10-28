"""Run a fixed two-seed comparison on a configurable GPU queue."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--mixture-checkpoint", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--gpus", nargs="+", default=["0", "1"], help="Physical GPU IDs, one training process per GPU")
    p.add_argument("--seeds", nargs="+", type=int, default=[201, 202])
    p.add_argument("--updates", type=int, default=1200)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    if len(set(args.gpus)) != len(args.gpus):
        p.error("GPU IDs must be unique")
    args.output.mkdir(parents=True, exist_ok=False)
    jobs = []
    # The fixed-depth curve permits comparisons at observed compute, not just equal maximum depth.
    arms = [("full", 1), ("no_adaptation", 1), ("task_reset", 1)] + [("fixed", n) for n in (0, 1, 2, 4)]
    for seed in args.seeds:
        for mode, refinements in arms:
            name = f"{mode}_depth{refinements}_seed{seed}"
            command = [sys.executable, "-m", "experiments.continual_experiment.train", "--data", args.data,
                       "--checkpoint", args.checkpoint, "--mixture-checkpoint", args.mixture_checkpoint,
                       "--output", str(args.output / name), "--mode", mode,
                       "--fixed-refinements", str(refinements), "--seed", str(seed),
                       "--updates", str(args.updates), "--batch", str(args.batch), "--device", "cuda:0"]
            jobs.append({"name": name, "command": command})
    (args.output / "jobs.json").write_text(json.dumps(jobs, indent=2) + "\n")
    if args.dry_run:
        print(json.dumps({"jobs": len(jobs), "gpus": args.gpus, "manifest": str(args.output / "jobs.json")}))
        return

    def worker(gpu, assigned):
        results = []
        for job in assigned:
            with (args.output / f"{job['name']}.log").open("w") as log:
                result = subprocess.run(job["command"], env={**os.environ, "CUDA_VISIBLE_DEVICES": gpu},
                                        stdout=log, stderr=subprocess.STDOUT, check=False)
            row = {"name": job["name"], "gpu": gpu, "returncode": result.returncode}
            print(json.dumps(row), flush=True)
            results.append(row)
        return results

    with ThreadPoolExecutor(max_workers=len(args.gpus)) as pool:
        futures = [pool.submit(worker, gpu, jobs[i::len(args.gpus)]) for i, gpu in enumerate(args.gpus)]
        results = [row for future in futures for row in future.result()]
    (args.output / "completion.json").write_text(json.dumps(results, indent=2) + "\n")
    if any(row["returncode"] for row in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
