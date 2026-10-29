"""Evaluate fixed final checkpoints on the reserved native task streams."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def make_jobs(output, training_root, manifest, workers, *, delay_ticks=0):
    common = ["--manifest", str(manifest), "--initial-rng-root", str(output / "initial_rng"),
              "--max-actions", "400", "--max-decisions", "160", "--workers", str(workers),
              "--threads", "1", "--device", "cpu"]
    jobs = []
    parents = {231: (221, Path("runs/physical_belief/joint_20261029/carry_seed221")),
               232: (222, Path("runs/physical_belief/joint_repeat_20261029/carry_seed222"))}
    for seed, (parent_seed, parent) in parents.items():
        name = f"parent_seed{parent_seed}"
        jobs.append({"name": name, "ready": None,
                     "command": [sys.executable, "-m", "experiments.physical_belief.evaluate_native",
                                 "--output", str(output/name), "--checkpoint", str(parent/"policy.pt"),
                                 "--belief-checkpoint", str(parent/"belief.pt"), "--belief-mode", "carry",
                                 "--state-mode", "reset", "--mixture-checkpoint",
                                 "runs/discoveryworld/calibration/mixture/shared_train_only_seed31/last.pt",
                                 *common]})
        for mode in ("full", "discard_physical", "task_reset", "fixed_schedule"):
            directory = training_root/f"{mode}_seed{seed}"
            variants = [(f"{mode}_seed{seed}", mode == "discard_physical")]
            if mode == "full":
                variants.append((f"full_seed{seed}_discard_physical", True))
            for name, discard in variants:
                command = [sys.executable, "-m", "experiments.continual_physics.evaluate_native",
                           "--output", str(output/name), "--checkpoint", str(directory/"last.pt"),
                           "--state-mode", "reset" if mode == "task_reset" else "carry",
                           "--feedback-delay-ticks", str(delay_ticks), *common]
                if discard:
                    command.append("--discard-physical-updates")
                jobs.append({"name": name, "ready": str(directory/"finished.json"), "command": command})
    return jobs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--training-root", type=Path, default=Path("runs/continual_physics/joint_20261029"))
    parser.add_argument("--manifest", type=Path, default=Path("experiments/continual_physics/native_confirmation.json"))
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--parallel", type=int, default=8)
    parser.add_argument("--feedback-delay-ticks", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--only", nargs="+", help="Execute a named subset of the fixed comparison")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    jobs = make_jobs(args.output, args.training_root, args.manifest, args.workers,
                     delay_ticks=args.feedback_delay_ticks)
    cohort = json.loads(args.manifest.read_text())["streams"]
    instances = sum(len(stream["seeds"]) for stream in cohort)
    plan = {"jobs": jobs, "manifest": str(args.manifest), "instances_per_condition": instances,
            "total_native_executions": instances*len(jobs), "maximum_workers": args.parallel*args.workers,
            "selection": "Every final fixed-budget condition; no best-checkpoint or native-score selection",
            "same_weight_intervention": "Discard physical fast/slow corrections; preserve cost adapters and Bayes memory",
            "training_repetitions": [231, 232], "bootstrap_unit": "complete six-instance ordered stream",
            "delivery_delay_ticks": args.feedback_delay_ticks,
            "timing_scope": "Delivery delay does not advance the native world independently"}
    (args.output/"evaluation_plan.json").write_text(json.dumps(plan, indent=2)+"\n")
    if args.dry_run:
        print(json.dumps({"conditions": len(jobs), "instances": instances, "total": plan["total_native_executions"]}))
        return
    completion = args.output/"evaluation_completion.json"
    completed = json.loads(completion.read_text()) if completion.exists() else []
    finished = {row["name"] for row in completed if row["exit_code"] == 0}
    pending = [job for job in jobs if job["name"] not in finished
               and (args.only is None or job["name"] in args.only)]
    running = []
    while pending or running:
        for job in list(pending):
            if len(running) >= args.parallel:
                break
            if job["ready"] and not Path(job["ready"]).exists():
                continue
            if (args.output/job["name"]).exists():
                raise FileExistsError(f"Existing evaluation output: {job['name']}")
            log = (args.output/f"{job['name']}.log").open("w")
            process = subprocess.Popen(job["command"], stdout=log, stderr=subprocess.STDOUT,
                                       env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1",
                                            "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"})
            running.append((job, process, log, time.monotonic()))
            pending.remove(job)
            print(json.dumps({"started": job["name"], "pid": process.pid}), flush=True)
        for entry in list(running):
            job, process, log, started = entry
            code = process.poll()
            if code is None:
                continue
            log.close()
            row = {"name": job["name"], "exit_code": code, "seconds": time.monotonic()-started}
            completed.append(row)
            running.remove(entry)
            (args.output/"evaluation_completion.json").write_text(json.dumps(completed, indent=2)+"\n")
            print(json.dumps({"finished": job["name"], **row}), flush=True)
        if pending or running:
            time.sleep(2)
    if any(row["exit_code"] for row in completed):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
