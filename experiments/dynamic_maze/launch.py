#!/usr/bin/env python3
"""Run a matrix of training or evaluation jobs across the visible GPUs."""
from __future__ import annotations

import argparse
import itertools
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
VARIANTS = {
    "restart": ["--mode", "restart"],
    "carry": ["--mode", "carry"],
    "goal_restart": ["--mode", "goal_restart"],
    "carry_read": ["--mode", "carry", "--channels", "86", "--goal-read"],
    "carry_gate": ["--mode", "carry", "--channels", "86", "--goal-read", "--change-gate"],
    "snapshot": ["--mode", "snapshot"],
    "nearest": ["--mode", "nearest"],
    "memory_local": ["--mode", "memory_local"],
    "memory_global": ["--mode", "memory_global"],
    "carry_static": ["--mode", "carry", "--regimes", "static"],
    "factorized": ["--mode", "factorized", "--channels", "60"],
    "factorized_local": ["--mode", "factorized", "--channels", "60", "--no-goal-broadcast"],
    "factorized_carry": ["--mode", "factorized", "--channels", "60", "--no-reseed"],
    "factorized_memory": ["--mode", "factorized", "--channels", "60", "--map-memory", "8"],
    # trained on what a moving agent sees (train_partial.py)
    "partial_read": ["--mode", "carry", "--channels", "86", "--goal-read"],
    "partial_read_age": ["--mode", "carry", "--channels", "86", "--goal-read", "--age-input"],
    "partial_factorized_age": ["--mode", "factorized", "--channels", "60", "--age-input"],
}
PARTIAL = {"partial_read", "partial_read_age", "partial_factorized_age"}


def in_progress(out: Path, flag: str = "--output-dir") -> bool:
    """True if a live process already writes to ``out`` (given after ``flag`` on its command line)."""
    needle = f"{flag}\0{out}\0".encode()
    for cmdline in Path("/proc").glob("[0-9]*/cmdline"):
        try:
            if needle in cmdline.read_bytes() + b"\0":
                return True
        except OSError:
            continue
    return False


def free_gb(gpu: int) -> float:
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits", "-i", str(gpu)],
                         capture_output=True, text=True, check=True).stdout
    return float(out.strip()) / 1024


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--variants", nargs="+", default=list(VARIANTS))
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--gpus", type=int, nargs="+", default=[0, 1])
    p.add_argument("--per-gpu", type=int, default=4)
    p.add_argument("--min-free-gb", type=float, default=0.0, help="start a job only if the GPU has this much free memory")
    p.add_argument("--stagger", type=float, default=0.0, help="seconds between two starts on the same GPU")
    p.add_argument("--extra", nargs=argparse.REMAINDER, default=[])
    args = p.parse_args(argv)
    jobs = []
    for variant, seed in itertools.product(args.variants, args.seeds):
        out = args.root / variant / f"seed{seed}"
        if (out / "model.pt").exists() or in_progress(out):
            continue
        script = "train_partial.py" if variant in PARTIAL else "train.py"
        cmd = [sys.executable, str(HERE / script), *VARIANTS[variant], "--seed", str(seed),
               "--output-dir", str(out), *args.extra]
        jobs.append((out, cmd))
    slots = [g for g in args.gpus for _ in range(args.per_gpu)]
    running: dict[int, tuple[subprocess.Popen, Path]] = {}
    last_start: dict[int, float] = {}
    while jobs or running:
        for slot, gpu in enumerate(slots):
            proc = running.get(slot)
            if proc and proc[0].poll() is not None:
                print(f"done {proc[1]} rc={proc[0].returncode}", flush=True)
                running.pop(slot)
            if slot not in running and jobs and time.time() - last_start.get(gpu, -1e9) >= args.stagger \
                    and free_gb(gpu) >= args.min_free_gb:
                out, cmd = jobs.pop(0)
                last_start[gpu] = time.time()
                out.mkdir(parents=True, exist_ok=True)
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
                log = (out / "stdout.log").open("w")
                running[slot] = (subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT), out)
                print(f"start {out} gpu={gpu}", flush=True)
        time.sleep(5)


if __name__ == "__main__":
    main()
