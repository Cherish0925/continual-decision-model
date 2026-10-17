#!/usr/bin/env python3
"""Evaluate every finished model under one or more run roots, spread over the GPUs.

Each model is evaluated once per named family into ``<output>/<family>/<variant>_<seed>.json``; an existing
result is skipped. A family is a name, the process family, and changes to its specification:

    training=instrument  thermal=thermal  slow_narrow=instrument:time_constant=8,16:width=0.15,0.25

Families named in ``--main`` (by default ``training``) use ``--devices`` processes over ``--ticks``; every other
family a quarter of each.
At most ``--per-gpu`` evaluations run on a GPU, and one starts only when ``--min-free-gb`` are free.
With ``--watch`` the run roots are scanned again every that many seconds for models that have finished since,
and the script keeps running. Arguments after ``--`` are passed to every evaluation, for example
``-- --adapter-lr 0.015``.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def free_gb(gpu: int) -> float:
    out = subprocess.run(["nvidia-smi", f"--id={gpu}", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, check=True).stdout
    return float(out.strip()) / 1024


def finished_models(roots):
    for root in roots:
        for config in sorted(Path(root).glob("*/seed*/config.json")):
            if "train_seconds" in json.loads(config.read_text()):
                yield config.parent / "model.pt"


def main():
    argv = sys.argv[1:]
    extra = argv[argv.index("--") + 1:] if "--" in argv else []
    argv = argv[:argv.index("--")] if "--" in argv else argv
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", type=Path, nargs="+", required=True)
    p.add_argument("--families", nargs="+", default=["training=instrument"])
    p.add_argument("--main", nargs="*", default=["training"], help="families evaluated at full size")
    p.add_argument("--also", nargs="*", default=["frozen", "discard"])
    p.add_argument("--devices", type=int, default=512)
    p.add_argument("--ticks", type=int, default=16384)
    p.add_argument("--gpus", type=int, nargs="+", default=[0, 1])
    p.add_argument("--per-gpu", type=int, default=3)
    p.add_argument("--min-free-gb", type=float, default=8.0)
    p.add_argument("--watch", type=float, default=0.0, help="seconds between scans for newly finished models")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args(argv)
    taken = set()                                          # outputs queued or started by this run

    def scan():
        found = []
        for model in finished_models(args.runs):
            for family in args.families:
                name, _, rest = family.partition("=")
                kind, *changes = rest.split(":")
                out = args.output / name / f"{model.parent.parent.name}_{model.parent.name}.json"
                if out.exists() or out in taken:
                    continue
                devices, ticks = (args.devices, args.ticks) if name in args.main else (args.devices // 4, args.ticks // 4)
                cmd = [sys.executable, str(HERE / "evaluate.py"), "--model", str(model), "--family", kind,
                       "--devices", str(devices), "--ticks", str(ticks), "--output", str(out)]
                cmd += ["--also", *args.also] if args.also else []
                cmd += ["--instrument", *changes] if changes else []
                found.append((cmd + extra, out))
                taken.add(out)
        if found:
            print(f"{len(found)} evaluations", flush=True)
        return found

    jobs, scanned = scan(), time.time()
    running = {g: [] for g in args.gpus}
    while jobs or any(running.values()) or args.watch:
        if args.watch and time.time() - scanned >= args.watch:
            jobs, scanned = jobs + scan(), time.time()
        for g in args.gpus:
            running[g] = [job for job in running[g] if job.poll() is None]
            if jobs and len(running[g]) < args.per_gpu and free_gb(g) >= args.min_free_gb:
                cmd, out = jobs.pop(0)
                out.parent.mkdir(parents=True, exist_ok=True)
                env = {**os.environ, "CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_VISIBLE_DEVICES": str(g),
                       "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
                running[g].append(subprocess.Popen(cmd, env=env, stdout=out.with_suffix(".log").open("w"),
                                                   stderr=subprocess.STDOUT))
                print(f"gpu {g}: {out.parent.name}/{out.name}", flush=True)
                time.sleep(10)
        time.sleep(10)


if __name__ == "__main__":
    main()
