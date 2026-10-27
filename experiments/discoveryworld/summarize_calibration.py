"""Build compact tables from completed numerical prediction and native runs."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize_native(paths):
    rows = []
    for path in paths:
        config = json.loads((path / "config.json").read_text())
        for result in json.loads((path / "results.json").read_text()):
            rows.append({"run": path.name, "cohort": result["cohort"],
                         "instance": result["seed"], "calibrations": result["support_count"],
                         "updates_retained": result["retain_updates"],
                         "local_probes": config.get("local_probes", False),
                         "provided_progress": result["takeover_start_progress"],
                         "actual_takeover_actions": result["takeover_steps"],
                         "new_crystals": result["takeover_new_activations"],
                         "completed": result["completed_successfully"],
                         "final_progress": result["progress"],
                         "error": result["error"], "raw_directory": result["output"]})
    return rows


def summarize_transfer(path):
    rows = []
    for name, model in json.loads(path.read_text())["models"].items():
        for cohort, comparisons in model["cohorts"].items():
            for condition in ("retained", "computed_discarded"):
                metrics = comparisons[condition]
                for count, result in metrics["by_support_count"].items():
                    rows.append({"model": name, "cohort": cohort,
                                 "updates_retained": condition == "retained",
                                 "calibrations": int(count), "queries": result["queries"],
                                 "predictive_nll": result["nll"],
                                 "map_mae_hz": result["map_expert"]["mae_hz"],
                                 "map_within_2hz": result["map_expert"]["within_2hz"],
                                 "mixture_mean_mae_hz": result["mixture_mean"]["mae_hz"],
                                 "known_support_mae_hz": result["known_support_mae_hz"]})
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native", type=Path, nargs="+", required=True)
    parser.add_argument("--transfer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    native, transfer = summarize_native(args.native), summarize_transfer(args.transfer)
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "calibration_native_comparison.csv", native)
    write_csv(args.output / "calibration_transfer.csv", transfer)
    print(json.dumps({"native_trials": len(native), "transfer_groups": len(transfer)}))


if __name__ == "__main__":
    main()
