"""Train four continual-decision conditions from one parent on a GPU queue."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
from queue import Empty, Queue
import subprocess
import sys
from time import monotonic


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--mixture-checkpoint", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--gpus", nargs="+", default=["0", "1"], help="Physical GPU IDs")
    p.add_argument("--jobs-per-gpu", type=int, default=1,
                   help="Concurrent training processes per GPU; use 2 when memory and throughput permit")
    p.add_argument("--seeds", nargs="+", type=int, default=[201, 202])
    p.add_argument("--updates", type=int, default=1200)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--unroll", type=int, default=32)
    p.add_argument("--memory-size", type=int, default=128)
    p.add_argument("--gamma", type=float, default=.99)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    if len(set(args.gpus)) != len(args.gpus):
        p.error("GPU IDs must be unique")
    if min(args.jobs_per_gpu, args.updates, args.batch, args.unroll, args.memory_size) <= 0:
        p.error("Worker count and training dimensions must be positive")
    if not 0 < args.gamma <= 1:
        p.error("gamma must be in (0, 1]")
    args.output.mkdir(parents=True, exist_ok=False)
    jobs = []
    arms = ("full", "no_adaptation", "task_reset", "fixed_schedule")
    for seed in args.seeds:
        for mode in arms:
            name = f"{mode}_seed{seed}"
            command = [sys.executable, "-m", "experiments.continual_experiment.train", "--data", args.data,
                       "--checkpoint", args.checkpoint, "--mixture-checkpoint", args.mixture_checkpoint,
                       "--output", str(args.output / name), "--mode", mode, "--seed", str(seed),
                       "--updates", str(args.updates), "--batch", str(args.batch),
                       "--unroll", str(args.unroll), "--memory-size", str(args.memory_size),
                       "--gamma", str(args.gamma), "--device", "cuda:0"]
            jobs.append({"name": name, "mode": mode, "seed": seed,
                         "planned_stream_ticks": args.updates * args.unroll * args.batch,
                         "command": command})
    (args.output / "jobs.json").write_text(json.dumps(jobs, indent=2) + "\n")
    if args.dry_run:
        print(json.dumps({"jobs": len(jobs), "gpus": args.gpus, "jobs_per_gpu": args.jobs_per_gpu,
                          "manifest": str(args.output / "jobs.json")}))
        return

    pending = Queue()
    for job in jobs:
        pending.put(job)

    def worker(gpu, slot):
        results = []
        while True:
            try:
                job = pending.get_nowait()
            except Empty:
                break
            started = monotonic()
            with (args.output / f"{job['name']}.log").open("w") as log:
                result = subprocess.run(job["command"], env={**os.environ, "CUDA_VISIBLE_DEVICES": gpu},
                                        stdout=log, stderr=subprocess.STDOUT, check=False)
            row = {"name": job["name"], "gpu": gpu, "slot": slot,
                   "returncode": result.returncode, "elapsed_seconds": monotonic() - started}
            print(json.dumps(row), flush=True)
            results.append(row)
        return results

    slots = [(gpu, slot) for slot in range(args.jobs_per_gpu) for gpu in args.gpus]
    with ThreadPoolExecutor(max_workers=len(slots)) as pool:
        futures = [pool.submit(worker, gpu, slot) for gpu, slot in slots]
        results = [row for future in futures for row in future.result()]
    (args.output / "completion.json").write_text(json.dumps(results, indent=2) + "\n")
    if any(row["returncode"] for row in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
