"""Queue fixed native comparisons after their training processes have finished."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from time import monotonic, sleep


def completed_training(directory):
    result = {}
    for filename in ("finished.json", "completion.json"):
        path = directory / filename
        if not path.exists():
            continue
        try:
            rows = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue  # A launcher may be publishing its completion record.
        result.update({row["name"]: row["returncode"] for row in rows})
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--training-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--manifest", default="experiments/continual_experiment/native_comparison.json")
    p.add_argument("--mixture-checkpoint", default="runs/discoveryworld/calibration/mixture/shared_train_only_seed31/last.pt")
    p.add_argument("--initial-rng-root", type=Path)
    p.add_argument("--wait-for", type=Path, help="Optional completion file for an earlier baseline batch")
    p.add_argument("--seeds", type=int, nargs="+", default=[201, 202])
    p.add_argument("--parallel", type=int, default=3)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rng = args.initial_rng_root or args.output / "initial_rng"
    jobs = []
    for seed in args.seeds:
        conditions = [(mode, mode, "reset" if mode == "task_reset" else "carry", False)
                      for mode in ("full", "no_adaptation", "task_reset", "fixed_schedule")]
        conditions += [("full_reset", "full", "reset", False), ("full_discard_updates", "full", "carry", True)]
        for name, training_mode, state_mode, discard in conditions:
            name, source = f"{name}_seed{seed}", f"{training_mode}_seed{seed}"
            command = [sys.executable, "-m", "experiments.discoveryworld.evaluate_active_stream", "--continual",
                       "--checkpoint", str(args.training_root / source / "last.pt"),
                       "--manifest", args.manifest, "--mixture-checkpoint", args.mixture_checkpoint,
                       "--initial-rng-root", str(rng), "--state-mode", state_mode, "--feedback-cost", "native",
                       "--max-actions", "400", "--max-decisions", "160", "--device", "cpu",
                       "--workers", str(args.workers), "--threads", str(args.threads),
                       "--output", str(args.output / name)]
            if discard:
                command.append("--discard-adaptation")
            jobs.append({"name": name, "source": source, "command": command})
    (args.output / "evaluation_plan.json").write_text(json.dumps(jobs, indent=2) + "\n")
    if args.dry_run:
        print(json.dumps({"conditions": len(jobs), "parallel": args.parallel}))
        return
    completion_path = args.output / "evaluation_completion.json"
    completed = json.loads(completion_path.read_text()) if completion_path.exists() else {}
    pending = [job for job in jobs if job["name"] not in completed]
    running = {}
    while pending or running:
        for name, (process, log, started) in list(running.items()):
            code = process.poll()
            if code is None:
                continue
            log.close()
            row = {"returncode": code, "elapsed_seconds": monotonic() - started}
            summary = args.output / name / "summary.json"
            if summary.exists():
                streams = json.loads(summary.read_text())
                row.update(instances=sum(s["instances"] for s in streams),
                           completed=sum(s["completed"] for s in streams),
                           native_actions=sum(s["native_actions"] for s in streams),
                           execution_errors=sum(bool(e.get("error")) for s in streams for e in s["episodes"]))
            completed[name] = row
            completion_path.write_text(json.dumps(completed, indent=2) + "\n")
            print(json.dumps({"finished": name, **row}), flush=True)
            del running[name]
        training = completed_training(args.training_root)
        if args.wait_for is None or args.wait_for.exists():
            for job in list(pending):
                if len(running) >= args.parallel:
                    break
                if job["source"] not in training:
                    continue
                if training[job["source"]] != 0:
                    raise RuntimeError(f"Training failed: {job['source']}")
                if (args.output / job["name"]).exists():
                    raise FileExistsError(f"Unfinished output already exists: {job['name']}")
                log = (args.output / f"{job['name']}.log").open("w")
                process = subprocess.Popen(job["command"], stdout=log, stderr=subprocess.STDOUT,
                                           env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1",
                                                "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"})
                running[job["name"]] = (process, log, monotonic())
                pending.remove(job)
                print(json.dumps({"started": job["name"], "training_completed": job["source"]}), flush=True)
        if pending or running:
            sleep(5)
    if any(row["returncode"] or row.get("execution_errors", 0) for row in completed.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
