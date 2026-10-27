"""Rebuild the experimental-plan result tables from released run records."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean


HEADS = ("state", "numeric", "state_seed172", "numeric_seed172")


def read(path):
    return json.loads(path.read_text())


def write(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def native(root, arms, seeds):
    rows = []
    for arm in arms:
        for seed in seeds:
            result = read(root / arm / f"seed{seed}" / "summary.json")
            rows.append({"arm": arm, "seed": seed,
                         "cohort": "reserved_family" if seed % 5 == 4 else "new_instance",
                         "completed": int(result["completed_successfully"]),
                         "progress": result["progress"], "actions": result["steps"]})
    return rows


def summarize(root, output, include_transfer):
    output.mkdir(parents=True, exist_ok=True)
    trials = []
    for split in ("train", "validation"):
        for row in read(root / split / "comparison.json"):
            trials.append({"split": split, "seed": int(Path(row["source"]).name[4:]),
                           "prefix_actions": row["prefix"], "hypothesis": row["hypothesis"],
                           "finished": row["finished"], "activated": row["success"],
                           "actions": row["actions"], "final_progress": row["progress"],
                           "prefix_matched": row["prefix_replay_matched"], "record": row["output"]})
    trials.sort(key=lambda r: (r["split"], r["seed"], r["hypothesis"]))
    write(output / "experimental_plan_trials.csv", trials)
    training = []
    for arm in HEADS:
        data = read(root / arm / "training.json")
        for point in data["history"]:
            for split in ("train", "validation"):
                row = point[split]
                training.append({"arm": arm, "training_seed": data["seed"],
                                 "epoch": point["epoch"], "split": split, "loss": row["loss"],
                                 "selected_success": row["learned"]["success"],
                                 "selected_actions": row["learned"]["actions"],
                                 "prior_success": row["prior"]["success"],
                                 "prior_actions": row["prior"]["actions"],
                                 "observed_best_success": row["observed_oracle"]["success"],
                                 "observed_best_actions": row["observed_oracle"]["actions"],
                                 "success_brier": row["success_brier"]})
    write(output / "experimental_plan_training.csv", training)
    cohorts = {"native": native(root / "native", ("reference", "parent", "posterior", *HEADS),
                                read(root / "design.json")["native_evaluation_instances"])}
    if include_transfer:
        cohorts["transfer"] = native(root / "transfer",
                                     ("reference", "posterior", "numeric", "numeric_seed172"),
                                     read(root / "transfer/design.json")["instances"])
    for cohort, rows in cohorts.items():
        write(output / f"experimental_plan_{cohort}.csv", rows)
        baseline = {r["seed"]: r for r in rows if r["arm"] == "posterior"}
        for arm in dict.fromkeys(r["arm"] for r in rows):
            group = [r for r in rows if r["arm"] == arm]
            print(json.dumps({"comparison": cohort, "arm": arm, "instances": len(group),
                              "completed": sum(r["completed"] for r in group),
                              "mean_progress": mean(r["progress"] for r in group),
                              "mean_actions": mean(r["actions"] for r in group),
                              "gained_vs_posterior": [r["seed"] for r in group if
                                                      r["completed"] > baseline[r["seed"]]["completed"]],
                              "lost_vs_posterior": [r["seed"] for r in group if
                                                     r["completed"] < baseline[r["seed"]]["completed"]]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("runs/discoveryworld/experimental_plans"))
    parser.add_argument("--output", type=Path, default=Path("experiments/discoveryworld/results"))
    parser.add_argument("--include-transfer", action="store_true")
    args = parser.parse_args()
    summarize(args.root, args.output, args.include_transfer)
