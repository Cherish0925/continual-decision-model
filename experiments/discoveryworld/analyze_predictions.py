"""Compare refinement forecasts against what the executed action returned."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from .train import outcome_target
from .train_predictive_actions import future_targets
from .predictive_actions import HORIZONS


def analyze(root):
    rows = []
    for directory in sorted(root.glob("*/seed*")):
        if not (directory / "summary.json").exists():
            continue
        decisions = [json.loads(line) for line in (directory / "decisions.jsonl").read_text().splitlines()]
        if not decisions or "executed_forecast_by_cycle" not in decisions[0]:
            continue
        transitions = [json.loads(line) for line in (directory / "trajectory.jsonl").read_text().splitlines()]
        if len(decisions) != len(transitions):
            raise ValueError("Prediction and actual action histories differ")
        summary = json.loads((directory / "summary.json").read_text())
        training = json.loads((directory / "run.json").read_text())["policy_details"]["predictive_training"]
        target = np.asarray([outcome_target(t) for t in transitions])
        returns, masks = map(np.asarray, future_targets(transitions, training["gamma"], training["action_cost"]))
        prediction = np.asarray([d["executed_forecast_by_cycle"] for d in decisions])
        numeric_mask = target[:, 9:14]
        for depth in range(prediction.shape[1]):
            p = prediction[:, depth]
            probabilities = torch.as_tensor(p[:, [0, 3, 9, 10, 11, 12, 13]]).sigmoid().numpy()
            item = {"arm": directory.parent.name, "seed": summary["seed"],
                    "cohort": "heldout_rule" if summary["seed"] % 5 == 4 else "new_instance",
                    "cycle": depth, "actions": len(transitions), "completed": int(summary["completed_successfully"]),
                    "operation_success_brier": float(np.mean((probabilities[:, 0] - target[:, 0]) ** 2)),
                    "numeric_presence_brier": float(np.mean((probabilities[:, 1:] - target[:, [3, 9, 10, 11, 12, 13]]) ** 2)),
                    "received_number_count": int(numeric_mask.sum()),
                    "received_number_mae_logscale": float((np.abs(p[:, 4:9] - target[:, 4:9]) * numeric_mask).sum() / numeric_mask.sum()) if numeric_mask.sum() else None,
                    "numeric_tool_fraction": sum(d.get("action_source") == "explicit_native_scalar_control" for d in decisions) / len(decisions)}
            for index, horizon in enumerate(HORIZONS):
                mask = masks[:, index]
                item[f"return_{horizon}_count"] = int(mask.sum())
                item[f"return_{horizon}_mae"] = float((np.abs(p[:, 14 + index] - returns[:, index]) * mask).sum() / mask.sum()) if mask.sum() else None
            rows.append(item)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = analyze(args.root)
    if not rows:
        raise ValueError("No completed native prediction records")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for arm, cohort, cycle in sorted({(r["arm"], r["cohort"], r["cycle"]) for r in rows}):
        group = [r for r in rows if (r["arm"], r["cohort"], r["cycle"]) == (arm, cohort, cycle)]
        print(json.dumps({"arm": arm, "cohort": cohort, "cycle": cycle, "instances": len(group),
                          **{k: float(np.mean([r[k] for r in group if r[k] is not None])) if any(r[k] is not None for r in group) else None for k in group[0]
                             if k not in ("arm", "cohort", "cycle", "seed")}}), flush=True)


if __name__ == "__main__":
    main()
