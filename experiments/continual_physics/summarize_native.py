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
                                           for phase in ("initial", "switch", "return")},
            "second_return": paired([r for r in left if r.get("segment_role") == "return" and r.get("position_in_segment") == 1],
                                    [r for r in right if r.get("segment_role") == "return" and r.get("position_in_segment") == 1])}


def paired_seconds(left, right, key):
    def value(row):
        return row.get("seconds") if key == "episode_seconds" else row.get("timing", {}).get(key)
    if any(value(row) is None for row in [*left, *right]):
        return None
    other = {(r["stream"], r["episode_index"]): r for r in right}
    groups = {}
    for row in left:
        groups.setdefault(row["stream"], []).append(value(row)-value(other[row["stream"], row["episode_index"]]))
    totals = np.array([sum(groups[name]) for name in sorted(groups)])
    counts = np.array([len(groups[name]) for name in sorted(groups)])
    sample = np.random.default_rng(231).integers(len(groups), size=(10000, len(groups)))
    estimates = totals[sample].sum(1)/counts[sample].sum(1)
    return {"mean_difference_seconds": float(totals.sum()/counts.sum()),
            "difference_95ci_seconds": np.quantile(estimates, [.025, .975]).tolist(),
            "bootstrap_unit": "whole_six_instance_stream"}


def trajectory(path, row):
    return json.loads((path/row["stream"]/f"episode{row['episode_index']:03d}_seed{row['seed']}"/"summary.json").read_text())["decisions"]


def signatures(decisions):
    actions, settings = [], []
    for decision in decisions:
        actions.append((decision["action_index"], decision["query_index"]))
        result = decision["result"]
        if decision["action_index"] < 20:
            continue
        attempts = result.get("attempts", [])
        setting = result.get("setting")
        request = result.get("request", {})
        if (setting is not None and setting.get("object_handle") == request.get("target_handle")
                and setting.get("reactor_handle") == request.get("reactor_handle")):
            attempts = [*attempts, setting]
        settings.extend((decision["query_index"], row.get("frequency_hz"), row.get("activated"))
                        for row in attempts)
    return actions, settings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=Path("experiments/continual_physics/native_confirmation.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--completed-only", action="store_true", help="Summarize only whole completed conditions; mark the campaign incomplete")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())["streams"]
    expected = {(s["name"], i, seed) for s in manifest for i, seed in enumerate(s["seeds"])}
    plan = json.loads((args.root/"evaluation_plan.json").read_text())
    jobs = plan["jobs"]
    if args.completed_only:
        completed = json.loads((args.root/"evaluation_completion.json").read_text())
        names = {row["name"] for row in completed if row["exit_code"] == 0}
        jobs = [job for job in jobs if job["name"] in names]
    conditions = {}
    rows = []
    result = {"manifest": str(args.manifest), "bootstrap_unit": "whole_six_instance_stream",
              "conditions": {}, "comparisons": {}, "same_weight_trajectories": {},
              "planned_conditions": [job["name"] for job in plan["jobs"]],
              "completed_conditions": [job["name"] for job in jobs],
              "campaign_complete": len(jobs) == len(plan["jobs"])}
    for job in jobs:
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
            if full not in conditions or control not in conditions:
                continue
            compared = comparison(conditions[full], conditions[control])
            compared["timing"] = {key: paired_seconds(conditions[full], conditions[control], key)
                                  for key in ("episode_seconds", "inference_seconds", "execution_seconds", "feedback_seconds")}
            result["comparisons"][f"{full}:{control}"] = compared
        discarded = controls[-1]
        if full not in conditions or discarded not in conditions:
            continue
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
        result["same_weight_trajectories"][str(seed)] = {
            "changed_action_sequences": sum(row["action_sequence_changed"] for row in differences),
            "changed_actual_setting_sequences": sum(row["actual_setting_sequence_changed"] for row in differences),
            "instances": differences}
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
