"""Reinterpret actually observed failed settings without reexecuting a task.

Each comparison starts from the prediction logged before a real setting test.
Only the settings actually observed to fail update that distribution. A later
publicly observed successful setting supplies an evaluation interval, never an
input to the update. This is a conditional-forecast comparison, not a rollout
of the alternative policy.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from cdm.censored_observation import condition_gaussian_mixture, interval_log_probability
from experiments.discoveryworld.calibration import FEATURE_QUANTITIES


class PublicFailureMask:
    """Reconstruct the native controller's mask from the received ledger.

    A failed relation remains unavailable until a measurement changes its
    support or current-target input. Finishing the first target clears the
    mask. This is the existing action rule, not Bayesian conditioning.
    """

    def __init__(self, bootstrap):
        self.slots = {row["public_handle"]: row["slot"] for row in bootstrap["public_object_order"]}
        self.values = torch.zeros(4, len(FEATURE_QUANTITIES), dtype=torch.float64)
        self.counts = torch.zeros_like(self.values, dtype=torch.long)
        self.known = torch.zeros(4, dtype=torch.bool)
        self.known[bootstrap["known_support_slots"]] = True
        self.failed = torch.zeros(len(FEATURE_QUANTITIES), dtype=torch.bool)

    def receive(self, decision):
        action, query = decision["action_index"], decision["query_index"]
        previous_values, previous_counts = self.values.clone(), self.counts.clone()
        for row in decision["result"].get("measurements", []):
            if (row["kind"] != "instrument" or row["target_handle"] not in self.slots
                    or row["quantity"] not in FEATURE_QUANTITIES):
                continue
            obj, quantity = self.slots[row["target_handle"]], FEATURE_QUANTITIES.index(row["quantity"])
            count, value = int(self.counts[obj, quantity]), float(row["value"])
            if quantity == 2 and count:
                self.values[obj, quantity] = min(float(self.values[obj, quantity]), value)
            else:
                self.values[obj, quantity] = (self.values[obj, quantity] * count + value) / (count + 1)
            self.counts[obj, quantity] += 1
        if action < 20:
            changed = (self.counts > 0) & ((previous_counts == 0) | (self.values != previous_values))
            relevant = self.known | (torch.arange(4) == query)
            self.failed &= ~(changed & relevant[:, None]).any(0)
        elif decision["success"]:
            self.known[query] = True
            if query == 2:
                self.failed.zero_()
        else:
            self.failed[action - 20] = True


def mask_relations(prediction, mask):
    weights = prediction["weights"].masked_fill(mask[None], 0)
    if not bool(weights.sum() > 0):
        raise ValueError("Mask-only control has no remaining predictive probability")
    return {**prediction, "weights": weights / weights.sum(-1, keepdim=True)}


def successful_setting(decision):
    if decision["action_index"] < 20 or not decision["success"]:
        return None
    result = decision["result"]
    activated = [row for row in result.get("attempts", []) if row.get("activated") is True]
    if activated:
        return float(activated[0]["frequency_hz"])
    setting = result.get("setting") or {}
    if setting.get("activated") is True:
        request = result["request"]
        if (setting.get("object_handle"), setting.get("reactor_handle")) == (request["target_handle"], request["reactor_handle"]):
            return float(setting["frequency_hz"])
    return None


def intervals(values, tolerance):
    return torch.tensor([[[value - tolerance, value + tolerance] for value in sorted(set(values))]],
                        dtype=torch.float64).reshape(1, -1, 2)


def summarize(rows):
    by_method = {}
    for method in sorted({row["method"] for row in rows}):
        selected = [row for row in rows if row["method"] == method]
        scored = [row for row in selected if row["future_success_interval_log_probability"] is not None]
        logs = np.asarray([row["future_success_interval_log_probability"] for row in scored])
        by_method[method] = {
            "failed_boundaries": len(selected), "scored_boundaries": len(scored),
            "mean_future_success_interval_nll": float(-logs.mean()) if len(logs) else None,
            "zero_probability_future_intervals": int(np.isneginf(logs).sum()),
            "mean_future_working_setting_absolute_error_hz": float(np.mean([row["map_error_hz"] for row in scored])) if scored else None,
            "changed_map_relations": sum(row["map_relation_changed"] for row in selected),
            "mean_weight_l1_change": float(np.mean([row["weight_l1_change"] for row in selected])),
        }
    return by_method


@torch.no_grad()
def replay(root, tolerance):
    rows, exclusions, consumed = [], {}, []
    skipped = {"not_complete_setting_failure": 0, "missing_prediction": 0,
               "missing_explicit_false_attempt": 0, "missing_later_public_success": 0}
    for path in sorted(root.glob("*/episode*/summary.json")):
        episode = json.loads(path.read_text())
        # A new physical instance receives its own false-setting history.
        exclusions.clear()
        native_mask = PublicFailureMask(episode["bootstrap"])
        for position, decision in enumerate(episode["decisions"]):
            native_mask.receive(decision)
            if decision["action_index"] < 20 or decision["success"]:
                continue
            result = decision["result"]
            request = result.get("request") or {}
            # Navigation errors, budget exhaustion and incomplete dialogs do
            # not establish that a frequency failed for the intended object.
            if (result.get("ending") != "local_experiment_exhausted"
                    or request.get("kind") != "test"
                    or request.get("target_handle") is None
                    or request.get("reactor_handle") is None):
                skipped["not_complete_setting_failure"] += 1
                continue
            if "physical_prediction" not in decision:
                skipped["missing_prediction"] += 1
                continue
            attempts = [entry for entry in result.get("attempts", []) if entry.get("activated") is False]
            if not attempts:
                skipped["missing_explicit_false_attempt"] += 1
                continue
            key = (request["target_handle"], request["reactor_handle"])
            current_values = [float(entry["frequency_hz"]) for entry in attempts]
            exclusions.setdefault(key, []).extend(current_values)
            current_intervals = intervals(current_values, tolerance)
            all_intervals = intervals(exclusions[key], tolerance)
            source = decision["physical_prediction"]
            prediction = {key: torch.tensor([source[key]], dtype=torch.float64) for key in ("means", "std", "weights")}
            chosen = decision["action_index"] - 20
            current_mask = torch.arange(prediction["weights"].shape[-1]) == chosen
            masked = mask_relations(prediction, current_mask)
            active_masked = mask_relations(prediction, native_mask.failed)
            current_conditioned = condition_gaussian_mixture(prediction, current_intervals)
            all_conditioned = condition_gaussian_mixture(prediction, all_intervals)
            methods = (("unchanged", prediction, None), ("mask_chosen_relation", masked, None),
                       ("mask_active_failed_relations", active_masked, None),
                       ("exclude_current_test", current_conditioned, current_intervals),
                       ("exclude_observed_history", all_conditioned, all_intervals))
            future = None
            for later in episode["decisions"][position + 1:]:
                later_request = later["result"].get("request") or {}
                if (later_request.get("target_handle"), later_request.get("reactor_handle")) == key:
                    value = successful_setting(later)
                    if value is not None:
                        future = {"decision": later["decision"], "frequency_hz": value}
                        break
            if future is None:
                skipped["missing_later_public_success"] += 1
            original_choice = int(prediction["weights"].argmax(-1))
            metadata = {"source": str(path.relative_to(root)), "stream": episode["stream"],
                        "episode_index": episode["episode_index"], "source_seed": episode["seed"],
                        "decision": decision["decision"], "query_index": decision["query_index"],
                        "chosen_relation": chosen, "failed_attempts": len(attempts),
                        "distinct_failed_settings": len(set(current_values)),
                        "cumulative_distinct_failed_settings": len(set(exclusions[key])),
                        "active_failed_relations": native_mask.failed.nonzero().flatten().tolist(),
                        "future_public_success": future}
            consumed.append({**metadata, "original_prediction": source,
                             "current_failed_settings_hz": current_values,
                             "cumulative_failed_settings_hz": sorted(set(exclusions[key]))})
            for name, updated, excluded in methods:
                selected = int(updated["weights"].argmax(-1))
                mean = float(updated["means"][0, selected])
                probability, error = None, None
                if future is not None:
                    successful_interval = torch.tensor([[future["frequency_hz"] - tolerance,
                                                        future["frequency_hz"] + tolerance]], dtype=torch.float64)
                    # Interval conditioning uses the original Gaussian mixture,
                    # not its moment-matched approximation after truncation.
                    scoring_prior = prediction if excluded is not None else updated
                    probability = float(interval_log_probability(scoring_prior, successful_interval, excluded))
                    error = abs(mean - future["frequency_hz"])
                rows.append({**metadata, "method": name,
                             "future_success_interval_log_probability": probability,
                             "map_relation": selected, "map_mean_hz": mean, "map_error_hz": error,
                             "map_relation_changed": selected != original_choice,
                             "weight_l1_change": float((updated["weights"] - prediction["weights"]).abs().sum()),
                             "posterior_weights": updated["weights"][0].tolist(),
                             "posterior_means_hz": updated["means"][0].tolist(),
                             "posterior_std_hz": updated["std"][0].tolist()})
    return {"protocol": "offline counterfactual forecast updates at completed observed failure boundaries",
            "native_policy_rerun": False, "tolerance_hz": tolerance,
            "evidence": "operator attempts logged after matching public object and reactor identities",
            "scoring": "later observed successful-setting interval; no hidden exact frequency read",
            "conditioned_distribution": "exact excluded-interval union likelihood; Gaussian moments only for point prediction",
            "source_condition": root.name, "distinct_source_instances": len({r["source"] for r in consumed}),
            "failed_boundaries": len(consumed), "attempt_entries": sum(r["failed_attempts"] for r in consumed),
            "skipped": skipped, "methods": summarize(rows), "records": rows,
            "public_inputs": consumed}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tolerance-hz", type=float, default=2.)
    args = parser.parse_args()
    torch.set_num_threads(2)
    result = replay(args.native_root, args.tolerance_hz)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    compact = [{"method": method, **values} for method, values in result["methods"].items()]
    with (args.output / "methods.csv").open("w") as out:
        writer = csv.DictWriter(out, fieldnames=list(compact[0]))
        writer.writeheader(); writer.writerows(compact)
    print(json.dumps({key: value for key, value in result.items() if key not in ("records", "public_inputs")}, indent=2))


if __name__ == "__main__":
    main()
