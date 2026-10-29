"""Run the separately declared fixed-computation recurrent references."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from experiments.physical_belief.summarize_native import load_rows
from .summarize_native import comparison


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-root", type=Path, default=Path("runs/continual_physics/recurrent_reference_20261029"))
    parser.add_argument("--output", type=Path, default=Path("runs/continual_physics/recurrent_reference_native"))
    parser.add_argument("--main-native", type=Path, default=Path("runs/continual_physics/native_confirmation"))
    parser.add_argument("--manifest", default="experiments/continual_physics/native_confirmation.json")
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    jobs = []
    for seed in (231, 232):
        name = f"recurrent_reference_seed{seed}"
        directory = args.training_root/name
        command = [sys.executable, "-m", "experiments.continual_physics.evaluate_native", "--checkpoint", str(directory/"last.pt"),
                   "--output", str(args.output/name), "--manifest", args.manifest, "--initial-rng-root", str(args.main_native/"initial_rng"),
                   "--feedback-delay-ticks", "2", "--workers", str(args.workers), "--threads", "1"]
        jobs.append({"name": name, "seed": seed, "ready": str(directory/"finished.json"), "command": command})
    (args.output/"evaluation_plan.json").write_text(json.dumps({"jobs": jobs,
        "scope": "Separate recurrent reference; does not change the original twelve-condition comparison",
        "policy": "One GRU/workspace assimilation per observation, shared physical tools, fixed neural parameters during deployment"}, indent=2)+"\n")
    pending, running, complete = list(jobs), [], []
    while pending or running:
        status = args.training_root/"completed.json"
        try:
            training = {r["name"]: r for r in json.loads(status.read_text())} if status.exists() else {}
        except json.JSONDecodeError:
            training = {}
        for job in list(pending):
            if job["name"] in training and training[job["name"]]["exit_code"]:
                complete.append({"name": job["name"], "exit_code": training[job["name"]]["exit_code"], "reason": "training_failed"})
                pending.remove(job)
                continue
            if not Path(job["ready"]).exists():
                continue
            log = (args.output/f"{job['name']}.log").open("w")
            process = subprocess.Popen(job["command"], stdout=log, stderr=subprocess.STDOUT)
            running.append((job, process, log)); pending.remove(job)
            print(json.dumps({"started": job["name"]}), flush=True)
        for entry in list(running):
            job, process, log = entry
            code = process.poll()
            if code is None:
                continue
            log.close(); running.remove(entry)
            complete.append({"name": job["name"], "exit_code": code})
            print(json.dumps({"finished": job["name"], "exit_code": code}), flush=True)
        (args.output/"evaluation_completion.json").write_text(json.dumps(complete, indent=2)+"\n")
        if pending or running:
            time.sleep(2)
    if any(row["exit_code"] for row in complete):
        raise SystemExit(1)
    result = {"conditions": {}, "comparisons": {}}
    for job in jobs:
        rows = load_rows(args.output/job["name"])
        result["conditions"][job["name"]] = {"instances": len(rows), "completed": sum(r["completed_successfully"] for r in rows),
                                            "mean_actions": sum(r["native_actions"] for r in rows)/len(rows)}
        for control in (f"full_seed{job['seed']}", f"parent_seed{job['seed']-10}"):
            result["comparisons"][f"{control}:{job['name']}"] = comparison(load_rows(args.main_native/control), rows)
    (args.output/"summary.json").write_text(json.dumps(result, indent=2)+"\n")


if __name__ == "__main__":
    main()
