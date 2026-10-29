"""Build prediction tables and relation-error decompositions from saved records."""
from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path

import numpy as np


def write_csv(path, rows):
    with path.open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def decomposition(rows):
    absolute = np.abs([r["map_mean_hz"] - r["target_hz"] for r in rows])
    correct = np.asarray([r["map_quantity"] == r["environment_quantity"] for r in rows])
    conditional = np.abs([r["means_hz"][r["environment_quantity"]] - r["target_hz"] for r in rows])
    truth_weight = np.asarray([r["weights"][r["environment_quantity"]] for r in rows])
    return {"queries": len(rows), "map_mae_hz": float(absolute.mean()),
            "wrong_relation_rate": float(1 - correct.mean()),
            "wrong_relation_error_fraction": float(absolute[~correct].sum() / max(absolute.sum(), 1e-30)),
            "conditional_true_relation_mae_hz": float(conditional.mean()),
            "mean_true_relation_posterior": float(truth_weight.mean()),
            "mean_nll_hz": float(np.mean([r["nll"] for r in rows]))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=Path("runs/physical_belief"))
    parser.add_argument("--output", type=Path, default=Path("experiments/physical_belief/results"))
    args = parser.parse_args()
    comparisons, phases, training, mechanisms = [], [], [], []
    for prefix, training_mode in (("pretrain", "through_memory"), ("episodic", "episodic")):
        for seed in (211, 212):
            run = args.runs / f"{prefix}{seed}"
            config = json.loads((run / "config.json").read_text())
            completed = json.loads((run / "completed.json").read_text())
            training.append({"training": training_mode, "seed": seed, "parameters": config["parameters"],
                             "updates": config["updates"], "batch": config["batch"],
                             "episodes_per_stream": 3 * config["segment_length"],
                             "episode_presentations": config["updates"] * config["batch"] * 3 * config["segment_length"],
                             "elapsed_seconds": completed["elapsed_seconds"], "final_training_nll_hz": completed["nll_hz"]})
            folder = run / "prediction_unique"
            summary = json.loads((folder / "summary.json").read_text())
            comparisons.extend({"training": training_mode, "seed": seed, **r} for r in summary["groups"])
            phases.extend({"training": training_mode, "seed": seed, **r} for r in summary["phase_groups"])
            with gzip.open(folder / "predictions.jsonl.gz", "rt") as stream:
                rows = [json.loads(line) for line in stream]
            for kind in sorted({r["kind"] for r in rows}):
                for method in sorted({r["method"] for r in rows}):
                    for target_number in (0, 1):
                        selected = [r for r in rows if (r["kind"], r["method"], r["target_number"]) == (kind, method, target_number)]
                        mechanisms.append({"training": training_mode, "seed": seed, "kind": kind,
                                           "method": method, "target_number": target_number,
                                           **decomposition(selected)})
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "pretraining.csv", training)
    write_csv(args.output / "prediction.csv", comparisons)
    write_csv(args.output / "prediction_phases.csv", phases)
    write_csv(args.output / "prediction_mechanisms.csv", mechanisms)
    print(json.dumps({"trainings": len(training), "prediction_groups": len(comparisons),
                      "mechanism_groups": len(mechanisms)}))


if __name__ == "__main__":
    main()
