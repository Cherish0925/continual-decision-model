"""Paired task, runtime and physical-learning results for the fixed comparison."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from cdm.continual_experiment import Activity
from experiments.physical_belief.summarize_native import load_rows, paired


def subset(rows, phase):
    return [row for row in rows if row.get("segment_role") == phase
            and row.get("position_in_segment") == 0]


def comparison(left, right):
    return {"all": paired(left, right), **{f"first_{phase}": paired(subset(left, phase), subset(right, phase))
                                           for phase in ("initial", "switch", "return")}}


def trajectory(path, row):
    return json.loads((path/row["stream"]/f"episode{row['episode_index']:03d}_seed{row['seed']}"/"summary.json").read_text())["decisions"]


def signatures(decisions):
    actions, settings = [], []
    for decision in decisions:
        actions.append((decision["action_index"], decision["query_index"]))
        result = decision["result"]
        attempts = result.get("attempts", [])
        setting = result.get("setting")
        if setting is not None:
            attempts = [*attempts, setting]
        settings.append(tuple((row.get("frequency_hz"), row.get("activated")) for row in attempts))
    return actions, settings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=Path("experiments/continual_physics/native_confirmation.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())["streams"]
    expected = {(s["name"], i, seed) for s in manifest for i, seed in enumerate(s["seeds"])}
    plan = json.loads((args.root/"evaluation_plan.json").read_text())
    conditions = {}
    rows = []
    result = {"manifest": str(args.manifest), "bootstrap_unit": "whole_six_instance_stream",
              "conditions": {}, "comparisons": {}, "same_weight_trajectories": {}}
    for job in plan["jobs"]:
        name = job["name"]
        values = load_rows(args.root/name)
        actual = {(r["stream"], r["episode_index"], r["seed"]) for r in values}
        if actual != expected or len(values) != len(expected):
            raise ValueError(f"Incomplete declared cohort for {name}")
        conditions[name] = values
        updates = [r["physical_learning"] for r in values if "physical_learning" in r]
        tests = sum(r["setting_decisions"] for r in updates)
        activities = [r["internal_activity_counts"] for r in values if "internal_activity_counts" in r]
        result["conditions"][name] = {
            "instances": len(values), "completed": sum(r["completed_successfully"] for r in values),
            "total_actions": sum(r["native_actions"] for r in values),
            "mean_actions": float(np.mean([r["native_actions"] for r in values])),
            "mean_episode_seconds": float(np.mean([r["seconds"] for r in values])),
            "activity_counts": ({kind.name: int(count) for kind, count in zip(Activity, np.sum(activities, axis=0))}
                                if activities else None),
            "mean_absolute_physical_correction_hz": (sum(r["sum_absolute_mean_correction_hz"] for r in updates)/tests if tests else None),
            "max_absolute_physical_correction_hz": max((r["max_absolute_mean_correction_hz"] for r in updates), default=None),
            "phase_seconds": ({key: sum(r["timing"][key] or 0. for r in values)
                               for key in values[0]["timing"]} if "timing" in values[0] else None)}
        for row in values:
            record = {"condition": name, **{key: row.get(key) for key in
                      ("stream", "episode_index", "seed", "segment_role", "position_in_segment",
                       "completed_successfully", "native_actions", "measurement_requests", "test_requests", "seconds")}}
            record.update(row.get("timing", {}))
            record.update({"activities_"+kind.name: count for kind, count in zip(Activity, row.get("internal_activity_counts", []))})
            record.update(row.get("physical_learning", {}))
            rows.append(record)
    for seed, parent in ((231, 221), (232, 222)):
        full = f"full_seed{seed}"
        controls = [f"parent_seed{parent}", f"discard_physical_seed{seed}", f"task_reset_seed{seed}",
                    f"fixed_schedule_seed{seed}", f"full_seed{seed}_discard_physical"]
        for control in controls:
            result["comparisons"][f"{full}:{control}"] = comparison(conditions[full], conditions[control])
        discarded = controls[-1]
        index = {(r["stream"], r["episode_index"]): r for r in conditions[discarded]}
        differences = []
        for row in conditions[full]:
            other = index[row["stream"], row["episode_index"]]
            actions, settings = signatures(trajectory(args.root/full, row))
            other_actions, other_settings = signatures(trajectory(args.root/discarded, other))
            differences.append({"stream": row["stream"], "seed": row["seed"],
                                "action_sequence_changed": actions != other_actions,
                                "actual_setting_sequence_changed": settings != other_settings,
                                "completion_difference": int(row["completed_successfully"])-int(other["completed_successfully"]),
                                "native_action_difference": row["native_actions"]-other["native_actions"]})
        result["same_weight_trajectories"][str(seed)] = differences
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/"summary.json").write_text(json.dumps(result, indent=2)+"\n")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (args.output/"instances.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(result["conditions"], indent=2))


if __name__ == "__main__":
    main()
