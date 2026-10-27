"""Summarize observed behaviour in completed model-policy native evaluations.

Uses executed actions, public observations, returned measurements and official
scores. No teacher phase, expert recommendation, hypothesis or stopping count
is consulted. Ending a sampling run does not establish scientific sufficiency.
"""
from __future__ import annotations

import argparse
import csv
from hashlib import sha256
import json
from pathlib import Path
import re
from statistics import mean

from .measurements import MeasurementLog
from .reference import INSTRUMENTS, observed_objects


EVALUATION_POLICIES = {
    "learned_dynamic_action_policy", "frozen_semantic_native_generation",
    "learned_relation_mixture_with_native_scalar_control",
    "cdm_belief_action_policy_with_explicit_numeric_primitive",
    "joint_cdm_action_and_compute_policy_with_explicit_numeric_primitive",
    "predictive_cdm_with_explicit_numeric_primitive",
}


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _progress(metrics):
    return mean(float(item["progress"]) for item in metrics) if metrics else None


def action_runs(rows: list[dict]) -> list[dict]:
    runs = []
    for index, row in enumerate(rows):
        action = row["action"]
        if runs and _canonical(action) == _canonical(runs[-1]["action"]):
            runs[-1]["length"] += 1
            runs[-1]["end_index"] = index
            runs[-1]["successes"] += int(bool(row["action_result"].get("success")))
        else:
            runs.append({"start_index": index, "end_index": index, "length": 1,
                         "action": action, "successes": int(bool(row["action_result"].get("success")))})
    return runs


def _active_pairs(observation):
    objects = observed_objects(observation)
    pairs = set()
    for reactor, obj in objects.items():
        if obj.get("name") != "crystal reactor (activated)":
            continue
        for crystal, item in objects.items():
            if item.get("name", "").startswith("quantum crystal") and \
                    f"[uuid: {reactor}]" in item.get("description", ""):
                pairs.add((reactor, crystal))
    return pairs


def analyze_episode(directory: Path) -> dict:
    directory = Path(directory).resolve()
    meta = json.loads((directory / "run.json").read_text())
    summary = json.loads((directory / "summary.json").read_text())
    if meta.get("policy") not in EVALUATION_POLICIES:
        raise ValueError(f"Not an independent model-policy evaluation: {directory}")
    raw = (directory / "trajectory.jsonl").read_bytes()
    rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    if any(row["index"] != index for index, row in enumerate(rows)):
        raise ValueError(f"Trajectory is not contiguous from index zero: {directory}")
    if summary["steps"] != len(rows):
        raise ValueError(f"Summary/trajectory step mismatch: {directory}")

    progress = [_progress(meta.get("initial_task_metrics", []))]
    progress += [_progress(row.get("task_metrics", [])) for row in rows]
    final_progress = summary.get("progress", _progress(summary.get("task_metrics", [])))
    if rows and progress[-1] is not None and final_progress is not None and abs(progress[-1] - final_progress) > 1e-8:
        raise ValueError(f"Summary/trajectory progress mismatch: {directory}")

    log = MeasurementLog()
    tools, tool_events, targets, measured_pairs = {}, [], {}, set()
    measurement_keys, measurement_events, new_evidence = {}, [], set()
    sampling_runs, current_sampling = [], None
    frequencies_seen, calibration_events, tuning_events, activation_events = set(), [], [], []
    activated_crystals = set()
    first = rows[0]["observation"] if rows else {"ui": {}}
    for _, crystal in _active_pairs(first):
        activated_crystals.add(crystal)
    for item in first["ui"].get("inventoryObjects", []):
        if item.get("name") in INSTRUMENTS:
            tools[item["uuid"]] = item["name"]

    for index, row in enumerate(rows):
        before, after = row["observation"], row["next_observation"]
        action, success = row["action"], bool(row["action_result"].get("success"))
        for item in after["ui"].get("inventoryObjects", []):
            if item.get("name") in INSTRUMENTS and item["uuid"] not in tools:
                tools[item["uuid"]] = item["name"]
                tool_events.append({"index": index, "handle": item["uuid"], "name": item["name"]})

        records = log.append_transition(row)
        readings = [record for record in records if record.kind == "instrument"]
        if readings:
            record = readings[0]
            pair = (record.instrument_handle, record.target_handle)
            if pair not in measured_pairs:
                new_evidence.add(index)
                measured_pairs.add(pair)
            target = targets.setdefault(record.target_handle, {"handle": record.target_handle,
                                                               "name": record.target_name, "instruments": set(), "actions": 0})
            target["instruments"].add(record.instrument_name)
            target["actions"] += 1
            condition = sha256(_canonical(record.conditions).encode()).hexdigest()
            key = (*pair, condition)
            measurement_keys[index] = key
            measurement_events.append({"index": index, "instrument": record.instrument_name,
                                       "instrument_handle": record.instrument_handle,
                                       "target": record.target_name, "target_handle": record.target_handle,
                                       "values": {item.quantity: item.value for item in readings}})
            if current_sampling and key == current_sampling["key"]:
                current_sampling["length"] += 1
                current_sampling["end_index"] = index
            else:
                if current_sampling:
                    sampling_runs.append(current_sampling)
                current_sampling = {"key": key, "start_index": index, "end_index": index, "length": 1,
                                    "instrument": record.instrument_name, "target": record.target_name,
                                    "instrument_handle": record.instrument_handle, "target_handle": record.target_handle}
        elif current_sampling:
            sampling_runs.append(current_sampling)
            current_sampling = None

        for record in (record for record in records if record.kind == "setting" and record.bound):
            if record.target_handle not in frequencies_seen:
                new_evidence.add(index)
                frequencies_seen.add(record.target_handle)
            if action.get("action") in ("TALK", "USE") and record.instrument_name == "crystal reactor (activated)" \
                    and record.contained_objects:
                calibration_events.append({"index": index, "reactor": record.target_handle,
                                           "crystals": list(record.contained_objects), "frequency_hz": record.value})
        if "chosen_dialog_option_int" in action:
            options = before["ui"].get("dialog_box", {}).get("dialogOptions", {})
            key = action["chosen_dialog_option_int"]
            label = options.get(str(key), options.get(key, ""))
            if re.fullmatch(r"(?:Increase|Decrease) frequency by [\d.]+ Hz", label):
                tuning_events.append({"index": index, "choice": label, "success": success,
                                      "returned_settings_hz": [record.value for record in records if record.kind == "setting"]})

        before_objects, after_objects = observed_objects(before), observed_objects(after)
        # An activated device first discovered after navigation is not evidence
        # of an activation caused by the preceding action.
        for _, crystal in _active_pairs(before):
            activated_crystals.add(crystal)
        after_pairs = _active_pairs(after)
        for reactor, obj in after_objects.items():
            if obj.get("name") != "crystal reactor (activated)" or reactor not in before_objects:
                continue
            if before_objects[reactor].get("name") == "crystal reactor (activated)":
                continue
            crystals = [crystal for r, crystal in after_pairs if r == reactor]
            novel = [crystal for crystal in crystals if crystal not in activated_crystals]
            activation_events.append({"index": index, "reactor": reactor, "crystals": crystals,
                                      "newly_activated_crystals": novel})
            if novel:
                new_evidence.add(index)
        activated_crystals.update(crystal for _, crystal in after_pairs)
    if current_sampling:
        sampling_runs.append(current_sampling)

    repeated_sampling = []
    for run in sampling_runs:
        if run["length"] < 2:
            continue
        key = run.pop("key")
        next_index = run["end_index"] + 1
        exit_success = (next_index < len(rows)
                        and bool(rows[next_index]["action_result"].get("success"))
                        and measurement_keys.get(next_index, ())[:2] != key[:2])
        # A successful change of action is an exit, but statistical adequacy is
        # unknown. Separately look for new evidence before returning to the pair.
        next_evidence = None
        for index in range(next_index, len(rows)):
            if measurement_keys.get(index, ())[:2] == key[:2]:
                break
            if index in new_evidence:
                next_evidence = index
                break
        repeated_sampling.append({**run, "condition_key": key[2], "exited": exit_success,
                                  "next_action": rows[next_index]["action"] if next_index < len(rows) else None,
                                  "next_new_evidence_index": next_evidence if exit_success else None})

    runs = action_runs(rows)
    longest = max(runs, key=lambda run: run["length"]) if runs else None
    measured = [{**target, "instruments": sorted(target["instruments"])} for target in targets.values()]
    return {"seed": meta["seed"], "scenario": meta["scenario"], "difficulty": meta["difficulty"],
            "policy": meta["policy"],
            "steps": len(rows), "progress": final_progress, "success": bool(summary["completed_successfully"]),
            "completed": bool(summary["completed"]), "truncated": bool(summary["truncated"]),
            "progress_by_action": progress, "official_summary": str(directory / "summary.json"),
            "trajectory": str(directory / "trajectory.jsonl"), "trajectory_sha256": sha256(raw).hexdigest(),
            "observation_adapter": meta.get("observation_adapter"),
            "native_successes": sum(bool(row["action_result"].get("success")) for row in rows),
            "tools_acquired": sorted(set(tools.values())), "tool_acquisition_events": tool_events,
            "measured_targets": measured, "measurement_actions": len(measurement_events),
            "measurement_values": len([record for record in log.records if record.kind == "instrument"]),
            "radiation_samples": sum(event["instrument"] == "radiation meter" for event in measurement_events),
            "max_same_action_run": longest["length"] if longest else 0, "longest_same_action_run": longest,
            "terminal_same_action_run": runs[-1] if runs else None,
            "exit_after_sampling": sum(run["exited"] for run in repeated_sampling),
            "sampling_exits_with_new_evidence": sum(run["next_new_evidence_index"] is not None for run in repeated_sampling),
            "repeated_sampling_runs": repeated_sampling,
            "calibration_dialogues": len(calibration_events), "calibration_events": calibration_events,
            "tuning_actions": sum(event["success"] for event in tuning_events), "tuning_events": tuning_events,
            "activation_transitions": len(activation_events), "activation_events": activation_events,
            "new_activations": sum(len(event["newly_activated_crystals"]) for event in activation_events)}


def analyze_models(models: dict[str, Path]) -> dict:
    report = {"kind": "actual_closed_loop_trace_analysis_v2", "models": {}, "max_actions": 0,
              "definitions": {
                  "progress": "Official episode summary; curves use initial_task_metrics then each native task_metrics return.",
                  "exit_after_sampling": "A successful action immediately following >=2 consecutive received measurements of one tool/target/observable condition, other than another reading of the same pair.",
                  "sampling_exits_with_new_evidence": "Exit followed by a first tool/target reading, first reactor frequency, or new activation before measuring that pair again.",
                  "sampling_sufficiency": "Not inferred. No teacher phase or fixed sample count is consulted.",
                  "calibration_dialogues": "Successful dialogue opening exposing frequency and a visible crystal in an activated reactor.",
                  "new_activations": "Crystals not previously observed activated in a directly observed inactive-to-active reactor transition; known restorations and first sightings excluded."}}
    for name, root in models.items():
        paths = sorted(Path(root).rglob("trajectory.jsonl"))
        if not paths:
            raise ValueError(f"No trajectories under {root}")
        episodes = sorted((analyze_episode(path.parent) for path in paths), key=lambda episode: episode["seed"])
        if len({(e["scenario"], e["difficulty"], e["seed"]) for e in episodes}) != len(episodes):
            raise ValueError(f"Duplicate evaluation instances in {root}")
        report["models"][name] = {"evaluation_root": str(Path(root).resolve()), "episodes": episodes,
                                  "policy_types": sorted({episode["policy"] for episode in episodes}),
                                  "seeds": [episode["seed"] for episode in episodes],
                                  "endpoint_progress": [episode["progress"] for episode in episodes],
                                  "progress_by_action": [episode["progress_by_action"] for episode in episodes],
                                  "mean_progress": mean(episode["progress"] for episode in episodes),
                                  "successful_episodes": sum(episode["success"] for episode in episodes),
                                  "episode_count": len(episodes)}
        report["max_actions"] = max(report["max_actions"], max(episode["steps"] for episode in episodes))
    return report


def markdown_table(report: dict) -> str:
    lines = ["| Evaluation | Policy type | Episodes | Mean progress | Completed | Tools / measured targets (mean) | Longest repeat | Radiation samples (mean) | Sampling exits / new evidence | Calibration / tuning | New activations |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for name, model in report["models"].items():
        episodes = model["episodes"]
        lines.append(f"| {name.replace('|', '/')} | {', '.join(model['policy_types'])} | "
                     f"{len(episodes)} | {model['mean_progress']:.3f} | "
                     f"{model['successful_episodes']} / {len(episodes)} | "
                     f"{mean(len(e['tools_acquired']) for e in episodes):.1f} / {mean(len(e['measured_targets']) for e in episodes):.1f} | "
                     f"{max(e['max_same_action_run'] for e in episodes)} | {mean(e['radiation_samples'] for e in episodes):.1f} | "
                     f"{sum(e['exit_after_sampling'] for e in episodes)} / {sum(e['sampling_exits_with_new_evidence'] for e in episodes)} | "
                     f"{sum(e['calibration_dialogues'] for e in episodes)} / {sum(e['tuning_actions'] for e in episodes)} | "
                     f"{sum(e['new_activations'] for e in episodes)} |")
    lines.append("\nRepeat is the maximum observed run across episodes. Exit, calibration, tuning and activation columns are totals. Sampling exits do not establish sufficient measurement. Completion is the official native evaluation result for the recorded policy type.\n")
    return "\n".join(lines)


def write_summary_csv(report, path):
    """Compact per-instance results, including failed and truncated runs."""
    keys = ["seed", "scenario", "difficulty", "steps", "progress", "success",
            "new_activations", "measurement_actions", "radiation_samples",
            "tuning_actions", "max_same_action_run"]
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["model", *keys])
        writer.writeheader()
        for name, model in report["models"].items():
            for episode in model["episodes"]:
                writer.writerow({"model": name, **{key: episode[key] for key in keys}})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", action="append", required=True, metavar="LABEL=EVALUATION_ROOT")
    parser.add_argument("--output", type=Path, required=True, help="New output prefix for .json and .md")
    args = parser.parse_args(argv)
    models = {}
    for spec in args.model:
        label, separator, directory = spec.partition("=")
        if not separator or not label or label in models:
            parser.error("Each model needs a distinct LABEL=EVALUATION_ROOT")
        models[label] = Path(directory)
    if any(args.output.with_suffix(suffix).exists() for suffix in (".json", ".md", ".csv")):
        parser.error("Choose a new diagnostic output prefix")
    report = analyze_models(models)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    table = markdown_table(report)
    args.output.with_suffix(".md").write_text(table)
    write_summary_csv(report, args.output.with_suffix(".csv"))
    print(table)


if __name__ == "__main__":
    main()
