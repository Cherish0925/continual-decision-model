"""Summarize paired native active-experiment runs, including their real action costs."""
from __future__ import annotations

import argparse
import csv
import json
from math import comb
from pathlib import Path

import numpy as np


def read_run(label, directory):
    records = json.loads((directory / "summary.json").read_text())
    rows = []
    for record in records:
        metrics = record["task_metrics"]
        rows.append({
            "method": label,
            "seed": record["seed"],
            "split": record["rule_split"],
            "completed": int(all(metric["completed_successfully"] for metric in metrics)),
            "progress": float(np.mean([metric["progress"] for metric in metrics])),
            "actions": record["native_actions"],
            "measurements": record["measurement_requests"],
            "tests": record["test_requests"],
            "seconds": record["seconds"],
            "ending": record["ending"],
            "error": record.get("error"),
        })
    return sorted(rows, key=lambda row: row["seed"])


def aggregate(rows):
    return {
        "instances": len(rows),
        "completed": sum(row["completed"] for row in rows),
        "completion_rate": float(np.mean([row["completed"] for row in rows])),
        **{f"mean_{key}": float(np.mean([row[key] for row in rows]))
           for key in ("progress", "actions", "measurements", "tests", "seconds")},
        "execution_errors": sum(row["error"] is not None for row in rows),
    }


def paired_difference(reference, candidate):
    old = {row["seed"]: row for row in reference}
    new = {row["seed"]: row for row in candidate}
    if old.keys() != new.keys():
        raise ValueError("Paired runs must contain the same completed evaluation instances")
    seeds = sorted(old)
    gains = [seed for seed in seeds if new[seed]["completed"] > old[seed]["completed"]]
    losses = [seed for seed in seeds if new[seed]["completed"] < old[seed]["completed"]]
    discordant = len(gains) + len(losses)
    exact_p = min(1., 2 * sum(comb(discordant, k) for k in range(min(len(gains), len(losses)) + 1))
                  / 2 ** discordant) if discordant else 1.
    rng = np.random.default_rng(181)
    samples = rng.integers(len(seeds), size=(10000, len(seeds)))
    differences = {}
    for key in ("completed", "actions", "measurements", "tests"):
        delta = np.array([new[seed][key] - old[seed][key] for seed in seeds], dtype=float)
        interval = np.quantile(delta[samples].mean(-1), [.025, .975]).tolist()
        differences[key] = {"mean_difference": float(delta.mean()), "paired_bootstrap_95_interval": interval}
    return {"instances": len(seeds), "new_completions": gains, "lost_completions": losses,
            "completion_exact_two_sided_p": exact_p, "differences_candidate_minus_reference": differences}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True, metavar="NAME=DIRECTORY")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    runs = {}
    for item in args.run:
        label, directory = item.split("=", 1)
        if label in runs:
            parser.error(f"Duplicate method: {label}")
        runs[label] = read_run(label, Path(directory))
    first = next(iter(runs))
    summary = {
        "setting": "Reactor Lab Normal, native from-reset execution, paired initial RNG",
        "completion": "Official completed_successfully; optional intermediate progress need not equal one",
        "cost": "Actual native actions, including discovery, navigation, measurement and scalar adjustment",
        "uncertainty": "Paired instance resampling; does not estimate variation across training seeds",
        "methods": {label: {"all": aggregate(rows),
                            **{split: aggregate([row for row in rows if row["split"] == split])
                               for split in sorted({row["split"] for row in rows})}}
                    for label, rows in runs.items()},
        "comparisons": {f"{label}_vs_{first}": paired_difference(runs[first], rows)
                        for label, rows in runs.items() if label != first},
    }
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "instances.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(runs[first][0]))
        writer.writeheader()
        writer.writerows(row for rows in runs.values() for row in rows)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
