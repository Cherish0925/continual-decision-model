"""Summarize matched native streams, including failed instances and task returns."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def load_rows(path):
    streams = json.loads((Path(path) / "summary.json").read_text())
    rows = [row for stream in streams for row in stream["episodes"]]
    if any(row.get("error") for row in rows):
        raise ValueError(f"Unresolved execution error in {path}")
    return rows


def paired(left, right, *, seed=211, samples=10000):
    key = lambda row: (row["stream"], row["episode_index"], row["seed"])
    right = {key(row): row for row in right}
    if len(left) != len(right) or {key(row) for row in left} != set(right):
        raise ValueError("Comparisons require identical complete stream instances")
    grouped = {}
    for row in left:
        other = right[key(row)]
        difference = [row["native_actions"] - other["native_actions"],
                      int(row["completed_successfully"]) - int(other["completed_successfully"])]
        grouped.setdefault(row["stream"], []).append(difference)
    streams = sorted(grouped)
    totals = np.array([np.sum(grouped[name], axis=0) for name in streams])
    counts = np.array([len(grouped[name]) for name in streams])
    rng = np.random.default_rng(seed)
    selected = rng.integers(len(streams), size=(samples, len(streams)))
    estimates = totals[selected].sum(1) / counts[selected].sum(1)[:, None]
    interval = np.quantile(estimates, [.025, .975], axis=0)
    difference = totals.sum(0) / counts.sum()
    return {"instances": len(left), "streams": len(streams),
            "mean_action_difference": float(difference[0]),
            "mean_action_difference_95ci": interval[:, 0].tolist(),
            "completion_fraction_difference": float(difference[1]),
            "completion_fraction_difference_95ci": interval[:, 1].tolist(),
            "bootstrap_unit": "whole_ordered_stream", "bootstrap_samples": samples}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", required=True, help="condition=directory")
    parser.add_argument("--reference", required=True)
    parser.add_argument("--pair", action="append", default=[], help="Additional matched comparison left:right")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    expected = json.loads(args.manifest.read_text())
    keys = {(stream["name"], index, seed) for stream in expected["streams"]
            for index, seed in enumerate(stream["seeds"])}
    conditions = {}
    for specification in args.input:
        name, path = specification.split("=", 1)
        rows = load_rows(path)
        if {(r["stream"], r["episode_index"], r["seed"]) for r in rows} != keys:
            raise ValueError(f"{name} has not completed its declared cohort")
        conditions[name] = rows
    reference = conditions[args.reference]
    results = {"cohort_purpose": expected.get("purpose"), "reference": args.reference,
               "inputs": dict(specification.split("=", 1) for specification in args.input),
               "specified_pair_requests": args.pair, "conditions": {}, "comparisons": {}}
    flattened = []
    for name, rows in conditions.items():
        results["conditions"][name] = {
            "instances": len(rows),
            "completed": sum(r["completed_successfully"] for r in rows),
            "mean_actions_including_failures": float(np.mean([r["native_actions"] for r in rows])),
            "total_actions": sum(r["native_actions"] for r in rows),
            "mean_measurement_requests": float(np.mean([r["measurement_requests"] for r in rows])),
            "mean_setting_requests": float(np.mean([r["test_requests"] for r in rows]))}
        comparisons = {"all": paired(rows, reference)}
        for segment in ("initial", "switch", "return"):
            choose = lambda xs: [r for r in xs if r.get("segment_role") == segment
                                and r.get("position_in_segment") == 0]
            subset, subset_reference = choose(rows), choose(reference)
            if subset:
                comparisons[f"first_{segment}"] = paired(subset, subset_reference)
        results["comparisons"][name] = comparisons
        for row in rows:
            flattened.append({"condition": name, **{key: row.get(key) for key in
                              ("stream", "seed", "episode_index", "segment_role", "position_in_segment",
                               "completed_successfully", "native_actions", "measurement_requests",
                               "test_requests", "ending")}})
    results["specified_pairs"] = {}
    for specification in args.pair:
        left_name, right_name = specification.split(":", 1)
        left, right = conditions[left_name], conditions[right_name]
        comparison = {"left": left_name, "right": right_name, "difference": "left minus right",
                      "all": paired(left, right)}
        for segment in ("initial", "switch", "return"):
            choose = lambda xs: [r for r in xs if r.get("segment_role") == segment
                                and r.get("position_in_segment") == 0]
            subset, counterpart = choose(left), choose(right)
            if subset:
                comparison[f"first_{segment}"] = paired(subset, counterpart)
        results["specified_pairs"][specification] = comparison
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(json.dumps(results, indent=2) + "\n")
    with (args.output / "instances.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flattened[0]))
        writer.writeheader()
        writer.writerows(flattened)
    print(json.dumps(results["conditions"], indent=2))


if __name__ == "__main__":
    main()
