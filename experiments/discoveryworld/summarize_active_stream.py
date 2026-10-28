"""Summarize one training repetition of paired native A–B–A task streams.

Bootstrap samples are whole streams, including every instance in each sampled
stream. Groups and jointly successful subsets retain those stream identities.
Run this separately for each training repetition; repeated evaluation on the
same scenarios does not create additional independent test instances.

Example:
    python -m experiments.discoveryworld.summarize_active_stream \\
        --repetition 191 --candidate carry \\
        --run carry=runs/native/carry --run taskreset=runs/native/taskreset \\
        --run resettrained=runs/native/resettrained --run parent=runs/native/parent \\
        --output runs/native/comparison191
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


ROLES = ("initial", "switch", "return")
GROUPS = ("all", *(f"{role}_{position}" for role in ROLES
                    for position in ("all", "first", "subsequent")))
COSTS = ("native_actions", "measurement_requests", "test_requests")


def segment_lengths(stream, rows):
    if "segment_lengths" in stream:
        lengths = stream["segment_lengths"]
    else:
        # Earlier development runs recorded relation labels but no segment map.
        relations, lengths = [], []
        for row in rows:
            if not relations or row["relation"] != relations[-1]:
                relations.append(row["relation"])
                lengths.append(0)
            lengths[-1] += 1
        if len(relations) != 3 or relations[0] != relations[2]:
            raise ValueError(f"No A–B–A segment definition for {stream['name']}")
    if len(lengths) != 3 or min(lengths) < 1 or sum(lengths) != len(rows):
        raise ValueError(f"Invalid segment lengths for {stream['name']}: {lengths}")
    return lengths


def read_run(arm, directory):
    config = json.loads((directory / "configuration.json").read_text())
    raw = json.loads((directory / "summary.json").read_text())
    results = {entry["stream"]: entry["episodes"] for entry in raw}
    planned = {stream["name"]: stream for stream in config["streams"]}
    if len(results) != len(raw) or results.keys() != planned.keys():
        raise ValueError(f"{arm}: evaluation does not contain every scheduled stream exactly once")
    rows = []
    for name, stream in sorted(planned.items()):
        episodes = sorted(results[name], key=lambda row: row["episode_index"])
        expected = list(enumerate(stream["seeds"]))
        actual = [(row["episode_index"], row["seed"]) for row in episodes]
        if actual != expected:
            raise ValueError(f"{arm}/{name}: incomplete or different instance sequence: {actual}; expected {expected}")
        lengths = segment_lengths(stream, episodes)
        offset = 0
        for segment, length in enumerate(lengths):
            for position, row in enumerate(episodes[offset:offset + length]):
                role = ROLES[segment]
                if row.get("segment_role", role) != role:
                    raise ValueError(f"{arm}/{name}: recorded segment does not match the manifest")
                rows.append({
                    "arm": arm, "stream": name, "episode_index": row["episode_index"],
                    "scenario_seed": row["seed"], "relation": row["relation"],
                    "segment_role": role, "position_in_segment": position,
                    "first_in_segment": int(position == 0),
                    "completed": int(row["completed_successfully"]),
                    "progress": row["progress"],
                    **{metric: row[metric] for metric in COSTS},
                    "seconds": row["seconds"], "ending": row["ending"],
                    "execution_error": row.get("error") or "",
                })
            offset += length
    return rows, {"state_mode": config["state_mode"],
                  "native_action_budget": config["max_actions"],
                  "policy_decision_budget": config["max_decisions"],
                  "instances": len(rows), "streams": len(planned)}


def selected(row, group):
    if group == "all":
        return True
    role, position = group.split("_", 1)
    return row["segment_role"] == role and (
        position == "all" or bool(row["first_in_segment"]) == (position == "first"))


def aggregate(rows):
    n = len(rows)
    return {
        "streams": len({row["stream"] for row in rows}), "instances": n,
        "completed": sum(row["completed"] for row in rows),
        "completion_rate": sum(row["completed"] for row in rows) / n if n else None,
        "mean_progress": float(np.mean([row["progress"] for row in rows])) if n else None,
        **{f"total_{key}": sum(row[key] for row in rows) for key in COSTS},
        **{f"mean_{key}": float(np.mean([row[key] for row in rows])) if n else None
           for key in (*COSTS, "seconds")},
        "execution_errors": sum(bool(row["execution_error"]) for row in rows),
    }


def identity(row):
    return row["stream"], row["episode_index"], row["scenario_seed"]


def location(row):
    return {key: row[key] for key in ("stream", "episode_index", "scenario_seed")}


def mean_difference(reference, candidate, metric, streams, bootstrap):
    counts = np.zeros(len(streams), dtype=np.int64)
    differences = np.zeros(len(streams))
    positions = {name: index for index, name in enumerate(streams)}
    for old, new in zip(reference, candidate):
        index = positions[new["stream"]]
        counts[index] += 1
        differences[index] += new[metric] - old[metric]
    point = float(differences.sum() / counts.sum()) if counts.sum() else None
    interval = [None, None]
    valid = 0
    if np.count_nonzero(counts) >= 2:
        sampled_counts = counts[bootstrap].sum(axis=1)
        keep = sampled_counts > 0
        estimates = differences[bootstrap].sum(axis=1)[keep] / sampled_counts[keep]
        interval = np.quantile(estimates, [.025, .975]).tolist()
        valid = int(keep.sum())
    return {"candidate_minus_reference": point,
            "paired_stream_bootstrap_95_interval": interval,
            "contributing_streams": int(np.count_nonzero(counts)),
            "valid_bootstrap_samples": valid}


def compare(reference, candidate, streams, bootstrap):
    old = {identity(row): row for row in reference}
    new = {identity(row): row for row in candidate}
    if old.keys() != new.keys():
        raise ValueError("Paired arms must contain the same stream/position/scenario identities")
    keys = sorted(old)
    before, after = [old[key] for key in keys], [new[key] for key in keys]
    common = [key for key in keys if old[key]["completed"] and new[key]["completed"]]
    common_before, common_after = [old[key] for key in common], [new[key] for key in common]
    return {
        "reference": aggregate(before), "candidate": aggregate(after),
        "gained_completions": [location(new[key]) for key in keys
                               if new[key]["completed"] and not old[key]["completed"]],
        "lost_completions": [location(old[key]) for key in keys
                             if old[key]["completed"] and not new[key]["completed"]],
        "completion_rate_difference": mean_difference(before, after, "completed", streams, bootstrap),
        "all_instance_costs": {metric: mean_difference(before, after, metric, streams, bootstrap)
                               for metric in COSTS},
        "joint_success": {
            "reference": aggregate(common_before), "candidate": aggregate(common_after),
            "costs": {metric: mean_difference(common_before, common_after, metric, streams, bootstrap)
                      for metric in COSTS},
        },
    }


def paired_csv_row(repetition, candidate, reference, group, pair):
    row = {
        "repetition": repetition, "candidate": candidate, "reference": reference, "group": group,
        "streams": pair["candidate"]["streams"], "instances": pair["candidate"]["instances"],
        "candidate_completed": pair["candidate"]["completed"],
        "reference_completed": pair["reference"]["completed"],
        "gained_completions": len(pair["gained_completions"]),
        "lost_completions": len(pair["lost_completions"]),
        "gained_scenario_seeds": ";".join(str(x["scenario_seed"]) for x in pair["gained_completions"]),
        "lost_scenario_seeds": ";".join(str(x["scenario_seed"]) for x in pair["lost_completions"]),
        "joint_success_instances": pair["joint_success"]["candidate"]["instances"],
        "joint_success_streams": pair["joint_success"]["candidate"]["streams"],
    }
    completion = pair["completion_rate_difference"]
    row["completion_rate_difference"] = completion["candidate_minus_reference"]
    row["completion_difference_ci_low"], row["completion_difference_ci_high"] = completion["paired_stream_bootstrap_95_interval"]
    for prefix, scope, costs in (("all", pair, pair["all_instance_costs"]),
                                 ("joint_success", pair["joint_success"], pair["joint_success"]["costs"])):
        for metric in COSTS:
            row[f"{prefix}_candidate_mean_{metric}"] = scope["candidate"][f"mean_{metric}"]
            row[f"{prefix}_reference_mean_{metric}"] = scope["reference"][f"mean_{metric}"]
            row[f"{prefix}_{metric}_difference"] = costs[metric]["candidate_minus_reference"]
            low, high = costs[metric]["paired_stream_bootstrap_95_interval"]
            row[f"{prefix}_{metric}_ci_low"] = low
            row[f"{prefix}_{metric}_ci_high"] = high
    return row


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True, metavar="ARM=DIRECTORY")
    parser.add_argument("--candidate", required=True, help="Arm compared against every other run")
    parser.add_argument("--repetition", required=True, help="One training repetition; never pool repetitions")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resamples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=191)
    args = parser.parse_args()
    runs, configs = {}, {}
    for item in args.run:
        arm, directory = item.split("=", 1)
        if arm in runs:
            parser.error(f"Duplicate arm: {arm}")
        runs[arm], configs[arm] = read_run(arm, Path(directory))
    if args.candidate not in runs or len(runs) < 2:
        parser.error("Supply the candidate arm and at least one paired reference")
    candidate = runs[args.candidate]
    expected = [identity(row) for row in candidate]
    for arm, rows in runs.items():
        if [identity(row) for row in rows] != expected:
            parser.error(f"{arm} has a different stream schedule")
        if [(row["segment_role"], row["position_in_segment"]) for row in rows] != [
                (row["segment_role"], row["position_in_segment"]) for row in candidate]:
            parser.error(f"{arm} has different segment assignments")
    streams = sorted({row["stream"] for row in candidate})
    bootstrap = np.random.default_rng(args.bootstrap_seed).integers(
        len(streams), size=(args.resamples, len(streams)))
    methods = {arm: {group: aggregate([row for row in rows if selected(row, group)])
                     for group in GROUPS} for arm, rows in runs.items()}
    pairs = {arm: {group: compare([row for row in rows if selected(row, group)],
                                 [row for row in candidate if selected(row, group)], streams, bootstrap)
                   for group in GROUPS} for arm, rows in runs.items() if arm != args.candidate}
    summary = {
        "repetition": args.repetition, "candidate": args.candidate,
        "setting": "Native Reactor Lab Normal A–B–A streams; fresh ledger and numerical posterior per instance",
        "completion": "Official completed_successfully; all evaluated failures retained in costs",
        "uncertainty": {
            "unit": "complete stream", "streams": len(streams),
            "resamples": args.resamples, "seed": args.bootstrap_seed,
            "estimator": "Instance-weighted mean difference; resample streams jointly across arms, preserving all their instances",
            "joint_success": "Use the same stream draws, retaining zero-count streams; condition costs on instances completed by both arms",
            "scope": "Within one fixed training repetition; repeated evaluations on the same scenarios are not pooled",
            "single_stream": "Intervals omitted when fewer than two streams contribute",
        },
        "runs": configs, "methods": methods, "comparisons": pairs,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "instances.csv", [{"repetition": args.repetition, **row}
              for rows in runs.values() for row in rows])
    write_csv(args.output / "summary.csv", [{"repetition": args.repetition, "arm": arm,
              "group": group, **values} for arm, groups in methods.items() for group, values in groups.items()])
    write_csv(args.output / "paired.csv", [paired_csv_row(args.repetition, args.candidate, arm, group, pair)
              for arm, groups in pairs.items() for group, pair in groups.items()])
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"repetition": args.repetition, "streams": len(streams),
                      "instances_per_arm": len(candidate),
                      "methods": {arm: groups["all"] for arm, groups in methods.items()}}, indent=2))


if __name__ == "__main__":
    main()
