#!/usr/bin/env python3
"""Evaluate every trained model under a run root, spreading work over GPUs."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from launch import free_gb, in_progress  # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True, help="training root: <variant>/seed<k>/model.pt")
    p.add_argument("--out", type=Path, required=True, help="evaluation root")
    p.add_argument("--gpus", type=int, nargs="+", default=[0, 1])
    p.add_argument("--per-gpu", type=int, default=3)
    p.add_argument("--min-free-gb", type=float, default=0.0, help="start a job only if the GPU has this much free memory")
    p.add_argument("--stagger", type=float, default=0.0, help="seconds between two starts on the same GPU")
    p.add_argument("--extra", nargs=argparse.REMAINDER, default=[])
    args = p.parse_args(argv)
    jobs = []
    for model in sorted(args.root.glob("*/seed*/model.pt")):
        variant, seed = model.parent.parent.name, model.parent.name
        target = args.out / variant / f"{seed}.json"
        if not target.exists() and not in_progress(target, flag="--output"):
            jobs.append((target, [sys.executable, str(HERE / "evaluate.py"), "--model", str(model),
                                  "--output", str(target), *args.extra]))
    if not (args.out / "oracle.json").exists():
        jobs.insert(0, (args.out / "oracle.json", [sys.executable, str(HERE / "evaluate.py"), "--oracle",
                                                  "--output", str(args.out / "oracle.json"),
                                                  *[a for a in args.extra if a != "--latency"]]))
    slots = [g for g in args.gpus for _ in range(args.per_gpu)]
    running, last_start = {}, {}
    while jobs or running:
        for slot, gpu in enumerate(slots):
            if slot in running and running[slot][0].poll() is not None:
                print(f"done {running[slot][1]} rc={running[slot][0].returncode}", flush=True)
                running.pop(slot)
            if slot not in running and jobs and time.time() - last_start.get(gpu, -1e9) >= args.stagger \
                    and free_gb(gpu) >= args.min_free_gb:
                target, cmd = jobs.pop(0)
                last_start[gpu] = time.time()
                target.parent.mkdir(parents=True, exist_ok=True)
                log = target.with_suffix(".log").open("w")
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
                running[slot] = (subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT), target)
                print(f"start {target} gpu={gpu}", flush=True)
        time.sleep(5)


if __name__ == "__main__":
    main()
