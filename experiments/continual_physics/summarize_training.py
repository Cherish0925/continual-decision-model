"""Export training budgets and activity counts from completed or running jobs.

These are on-policy training measurements, not held-out evaluation results.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


def records(path):
    text = path.read_text() if path.exists() else ""
    # A running writer may still be appending its final line.
    if text and not text.endswith("\n"):
        text = text.rpartition("\n")[0]
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def summarize(directory):
    config = json.loads((directory / "config.json").read_text())
    rows = records(directory / "training.jsonl")
    tasks = records(directory / "tasks.jsonl")
    finish = directory / "finished.json"
    completed = json.loads(finish.read_text()) if finish.exists() else {}
    counts = [sum(row["activity_counts"][i] for row in rows) for i in range(6)]
    requests = sum(row["external_requests"] for row in rows)
    result = dict(condition=directory.name, seed=config["seed"], mode=config["mode"],
        status="complete" if completed else "incomplete", updates=len(rows),
        planned_updates=config["updates"], batch=config["batch"],
        external_requests_per_window=config["batch"] * config["external_per_stream"],
        external_requests=requests,
        planned_external_requests=config["updates"] * config["batch"] * config["external_per_stream"],
        returned_requests=completed.get("returned_requests"),
        seconds=completed.get("seconds", rows[-1]["seconds"] if rows else 0.),
        gradients_finite=all(math.isfinite(row["gradient_norm"]) for row in rows) if rows else None,
        peak_allocated_gib=max((row.get("peak_allocated_gib", 0.) for row in rows), default=0.),
        physical_learn_computations=sum(row["physical_learn_computations"] for row in rows),
        physical_consolidation_computations=sum(row["physical_consolidation_computations"] for row in rows),
        completed_training_tasks=len(tasks),
        solved_training_tasks=sum(row["successes"] == row["queries"] for row in tasks))
    for name, count in zip(("act", "refine", "retrieve", "learn", "consolidate", "wait"), counts):
        result[name] = count
    result["internal_activities_per_external_request"] = sum(counts[1:5]) / requests if requests else None
    for phase, label in enumerate(("initial_a", "interference_b", "return_a")):
        result[f"{label}_training_tasks"] = sum(row["phase"] == phase for row in tasks)
    for field in ("physical_nll", "streams_crossing_task_fraction", "streams_crossing_lifetime_fraction"):
        result[f"mean_{field}"] = sum(row[field] for row in rows) / len(rows) if rows else None
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaigns", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = [summarize(path.parent) for campaign in args.campaigns
            for path in sorted(campaign.glob("*/config.json"))]
    if not rows:
        raise SystemExit("No training configurations found")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "training_summary.json").write_text(json.dumps({
        "measurement": "on-policy training; task counts include repeated source worlds",
        "gradient_window": "state persists across tasks; gradient is truncated at each external-request window",
        "rows": rows}, indent=2) + "\n")
    with (args.output / "training_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(dict(trainings=len(rows), complete=sum(row["status"] == "complete" for row in rows),
                         external_requests=sum(row["external_requests"] for row in rows))))


if __name__ == "__main__":
    main()
