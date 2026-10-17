#!/usr/bin/env python3
"""Train several variants with several seeds, spread over the GPUs.

A variant is a mode, or ``name=mode`` followed by its own training arguments,
for example ``"full_meta=full --meta 0.5"``; a trailing ``@0,1`` restricts it to
those seeds. Runs start in order of seed, then of variant. Runs go to
``<output-root>/<name>/seed<k>``. A run whose ``config.json`` records a finished
training time is skipped, and so is one whose log changed in the last
``--busy-minutes``. A run starts on a GPU when fewer than ``--per-gpu`` runs
occupy it, at least ``--min-free-gb`` of its memory is free, and ``--stagger``
seconds have passed since the last start there. Arguments after ``--`` are
passed to every run.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def free_gb(gpu: int) -> float:
    out = subprocess.run(["nvidia-smi", f"--id={gpu}", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, check=True).stdout
    return float(out.strip()) / 1024


def finished(run: Path) -> bool:
    config = run / "config.json"
    return config.exists() and "train_seconds" in json.loads(config.read_text())


def busy(run: Path, minutes: float) -> bool:
    log = run / "train_log.jsonl"
    return log.exists() and time.time() - log.stat().st_mtime < 60 * minutes


def variant(text: str) -> tuple[str, str, list[str], set[int] | None]:
    text, _, seeds = text.partition(" @")
    head, *extra = shlex.split(text)
    name, _, mode = head.partition("=")
    return name, mode or name, extra, ({int(k) for k in seeds.split(",")} if seeds else None)


def main():
    argv = sys.argv[1:]
    extra = argv[argv.index("--") + 1:] if "--" in argv else []
    argv = argv[:argv.index("--")] if "--" in argv else argv
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--variants", nargs="+", default=["full", "fixed_learn", "frozen", "world_model", "recurrent"])
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--gpus", type=int, nargs="+", default=[0, 1])
    p.add_argument("--per-gpu", type=int, default=5, help="concurrent runs per GPU")
    p.add_argument("--min-free-gb", type=float, default=20.0)
    p.add_argument("--stagger", type=float, default=90.0, help="seconds between starts on one GPU")
    p.add_argument("--busy-minutes", type=float, default=15.0)
    p.add_argument("--output-root", type=Path, required=True)
    args = p.parse_args(argv)
    queue = [(seed, name, mode, own) for seed, v in itertools.product(args.seeds, args.variants)
             for name, mode, own, only in [variant(v)] if only is None or seed in only]
    queue = [job for job in queue if not finished(run := args.output_root / job[1] / f"seed{job[0]}")
             and not busy(run, args.busy_minutes)]
    running: dict[int, list] = {g: [] for g in args.gpus}
    last = dict.fromkeys(args.gpus, 0.0)
    while queue or any(running.values()):
        for g in args.gpus:
            running[g] = [job for job in running[g] if job.poll() is None]
            if (queue and len(running[g]) < args.per_gpu and time.time() - last[g] >= args.stagger
                    and free_gb(g) >= args.min_free_gb):
                last[g] = time.time()
                seed, name, mode, own = queue.pop(0)
                run = args.output_root / name / f"seed{seed}"
                run.mkdir(parents=True, exist_ok=True)
                cmd = [sys.executable, str(HERE / "train.py"), "--mode", mode, "--seed", str(seed),
                       "--output-dir", str(run), *extra, *own]
                env = {**os.environ, "CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_VISIBLE_DEVICES": str(g),
                       "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
                running[g].append(subprocess.Popen(cmd, env=env, stdout=(run / "stdout.log").open("a"),
                                                   stderr=subprocess.STDOUT))
                print(f"gpu {g}: {name} seed {seed}", flush=True)
        time.sleep(20)


if __name__ == "__main__":
    main()
