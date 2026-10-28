"""Separate native setup, measurement and experimental execution costs.

Use one invocation per training repetition. The same reference run can appear
in both repetitions; those repeated rows are not independent measurements.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .summarize_active_stream import GROUPS, read_run, selected, write_csv


COSTS = ("bootstrap_native_actions", "measurement_native_actions", "test_native_actions",
         "native_actions", "measurement_requests", "test_requests")


def decompose(arm, directory):
    rows, _ = read_run(arm, directory)
    for row in rows:
        path = (directory / row["stream"] /
                f"episode{row['episode_index']:03d}_seed{row['scenario_seed']}" / "summary.json")
        episode = json.loads(path.read_text())
        measurements = [decision for decision in episode["decisions"] if decision["action_index"] < 20]
        tests = [decision for decision in episode["decisions"] if decision["action_index"] >= 20]
        row["bootstrap_native_actions"] = episode["bootstrap_actions"]
        row["measurement_native_actions"] = sum(decision["native_actions"] for decision in measurements)
        row["test_native_actions"] = sum(decision["native_actions"] for decision in tests)
        if sum(row[key] for key in COSTS[:3]) != row["native_actions"]:
            raise ValueError(f"Native action components do not sum to the recorded total: {path}")
        if len(measurements) != row["measurement_requests"] or len(tests) != row["test_requests"]:
            raise ValueError(f"Request counts do not match the executed decisions: {path}")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True, metavar="ARM=DIRECTORY")
    parser.add_argument("--repetition", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    instances, summaries = [], []
    for item in args.run:
        arm, directory = item.split("=", 1)
        rows = decompose(arm, Path(directory))
        instances.extend({"repetition": args.repetition, **row} for row in rows)
        for group in GROUPS:
            members = [row for row in rows if selected(row, group)]
            n = len(members)
            totals = {key: sum(row[key] for row in members) for key in COSTS}
            summaries.append({"repetition": args.repetition, "arm": arm, "group": group,
                              "streams": len({row["stream"] for row in members}), "instances": n,
                              "completed": sum(row["completed"] for row in members),
                              **{f"total_{key}": value for key, value in totals.items()},
                              **{f"mean_{key}": value / n if n else None for key, value in totals.items()}})
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "native_cost_instances.csv", instances)
    write_csv(args.output / "native_cost_breakdown.csv", summaries)
    result = {
        "repetition": args.repetition,
        "unit": "Native simulator actions, including failures and actions preceding termination",
        "components": {
            "bootstrap": "Public station discovery and reading initial working-device settings",
            "measurement": "Actions executed for measurement requests, including navigation and isolation",
            "test": "Actions executed for candidate tests, including navigation, device adjustments and local attempts",
        },
        "identity": "bootstrap + measurement + test = recorded native actions, checked for each instance",
        "groups": "All instances and initial/switch/return segments, including first and subsequent instances",
        "shared_references": "Runs reused across training repetitions remain the same observations; do not pool them",
        "rows": summaries,
    }
    (args.output / "native_cost_breakdown.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"repetition": args.repetition, "instance_rows": len(instances),
                      "summary_rows": len(summaries), "all_action_totals_reconciled": True}))


if __name__ == "__main__":
    main()
