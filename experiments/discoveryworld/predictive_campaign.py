"""Run the fixed prediction-training comparison and paired native executions."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys

NEW_INSTANCES = [s for s in range(2400, 2420) if s % 5 != 4]
NEW_RULE = list(range(2404, 2440, 5))


def run(command, log):
    log.parent.mkdir(parents=True, exist_ok=True)
    stream = log.open("w")
    child = subprocess.Popen([sys.executable, "-m", *command], stdout=stream, stderr=subprocess.STDOUT)
    stream.close()
    return child


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("runs/discoveryworld/predictive_actions"))
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/discoveryworld/belief_actions/full_seed61/last.pt"))
    parser.add_argument("--skip-training", action="store_true")
    args = parser.parse_args()
    root = args.root
    if not args.skip_training:
        # The campaign consumes one completed collection, not a moving subset.
        for split, expected in (("train", 141), ("validation", 24)):
            count = len(list((root / "prepared" / split).glob("*.pt")))
            if count != expected:
                raise ValueError(f"Prepared {split}: {count} episodes; expected {expected}")
        jobs = []
        for index, arm in enumerate(("feedback", "no_feedback")):
            command = ["experiments.discoveryworld.train_predictive_actions", "--checkpoint", str(args.checkpoint),
                       "--prepared", str(root / "prepared"), "--output", str(root / arm),
                       "--device", f"cuda:{index}", "--threads", "4"]
            if arm == "no_feedback":
                command.append("--no-feedback")
            jobs.append((arm, run(command, root / f"{arm}.log")))
        for arm, process in jobs:
            if process.wait() != 0:
                raise RuntimeError(f"Training failed: {arm}; see {root / (arm + '.log')}")
            print(json.dumps({"stage": "training_finished", "arm": arm}), flush=True)
    evaluations = [
        ("reference", args.checkpoint, "belief", []),
        ("feedback", root / "feedback/last.pt", "predictive", []),
        ("no_feedback", root / "no_feedback/last.pt", "predictive", []),
        ("zero_extra_cycles", root / "feedback/last.pt", "predictive", ["--refinements", "0"]),
        ("forecasts_removed", root / "feedback/last.pt", "predictive", ["--no-use-forecasts"]),
    ]
    rows = []
    for arm, checkpoint, kind, extra in evaluations:
        output = root / "native" / arm
        if not (output / "summary.json").exists():
            command = ["experiments.discoveryworld.evaluate", "--policy-kind", kind, "--checkpoint", str(checkpoint),
                       "--feature-cache", "runs/discoveryworld/features", "--encoder-path", "data/cache/siglip2",
                       "--output", str(output), "--seeds", *map(str, NEW_INSTANCES + NEW_RULE),
                       "--max-steps", "400", "--devices", "cuda:0", "cuda:1", "--workers", "8", "--threads", "2",
                       "--local-probes", *extra]
            if arm != "reference":
                command += ["--initial-rng-root", str(root / "native/reference")]
            if run(command, root / f"native_{arm}.log").wait() != 0:
                raise RuntimeError(f"Native execution failed: {arm}; see {root / ('native_' + arm + '.log')}")
        for seed in NEW_INSTANCES + NEW_RULE:
            result = json.loads((output / f"seed{seed}/summary.json").read_text())
            decisions = [json.loads(line) for line in (output / f"seed{seed}/decisions.jsonl").read_text().splitlines()]
            changed = [d["selected_by_cycle"][0] != d["selected_by_cycle"][-1]
                       for d in decisions if "selected_by_cycle" in d]
            rows.append({"arm": arm, "seed": seed, "cohort": "new_instance" if seed in NEW_INSTANCES else "heldout_rule",
                         "completed": int(result["completed_successfully"]), "progress": result["progress"],
                         "actions": result["steps"], "decision_seconds": result["decision_latency_mean_seconds"],
                         "proposal_changed_by_computation": sum(changed) / len(changed) if changed else 0.})
        for cohort in ("new_instance", "heldout_rule"):
            subset = [row for row in rows if row["arm"] == arm and row["cohort"] == cohort]
            print(json.dumps({"arm": arm, "cohort": cohort, "completed": sum(r["completed"] for r in subset),
                              "instances": len(subset), "progress": sum(r["progress"] for r in subset) / len(subset),
                              "actions": sum(r["actions"] for r in subset) / len(subset)}), flush=True)
        with (root / "comparison.csv").open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
