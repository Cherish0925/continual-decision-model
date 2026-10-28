"""Summarize joint training and paired native task streams without loading models.

Run from the repository root. Only complete native streams from a complete
scheduled evaluation enter the comparison; training progress may still be partial.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import shlex

import numpy as np

from experiments.discoveryworld.summarize_active_stream import (
    GROUPS, ROLES, aggregate, compare, identity, mean_difference, read_run, selected,
)
from .evaluate import completed_training


ACTIVITIES = ("act", "refine", "retrieve", "learn", "consolidate", "wait")
BASELINES = ("baseline_parent", "baseline_rehearsal", "baseline_explicit")
MODES = ("full", "no_adaptation", "task_reset", "fixed_schedule", "full_reset", "full_discard_updates")


def read_json(path):
    return json.loads(path.read_text())


def json_lines(path):
    """The final unterminated line of an active writer is not a complete record."""
    if path.exists():
        with path.open() as stream:
            for line in stream:
                if not line.endswith("\n"):
                    break
                if line.strip():
                    yield json.loads(line)


def relative(path, project):
    return path.absolute().relative_to(project.absolute()).as_posix()


def prediction_errors(decisions):
    """Score pre-action predictions against the subsequent public outcome."""
    result = {}
    for kind in ("measurement", "test"):
        rows = [row for row in decisions if (row["action_index"] < 20) == (kind == "measurement")]
        predicted = [row for row in rows if "model_prediction" in row]
        result[f"{kind}_decisions"] = len(rows)
        result[f"{kind}_predictions"] = len(predicted)
        result[f"{kind}_nonzero_adaptation"] = sum(
            max(abs(value) for value in row["model_prediction"]["adaptation"]) > 1e-6 for row in predicted)
        for component in ("shared", "total"):
            for index, target in enumerate(("test_success", "log_native_cost")):
                errors = []
                for row in predicted:
                    truth = float(row["success"]) if index == 0 else math.log1p(row["native_actions"])
                    errors.append((row["model_prediction"][component][index] - truth) ** 2)
                result[f"{kind}_{component}_{target}_squared_error_sum"] = sum(errors) if predicted else None
    return result


def native_rows(name, directory):
    rows, metadata = read_run(name, directory)
    config = read_json(directory / "configuration.json")
    metadata.update({key: config.get(key) for key in
                     ("continual", "feedback_cost", "discard_adaptation", "device", "workers", "threads")})
    for row in rows:
        record = read_json(directory / row["stream"] /
                           f"episode{row['episode_index']:03d}_seed{row['scenario_seed']}" / "summary.json")
        row["execution_error"] = bool(row["execution_error"])
        internal = record.get("internal_activity")
        row["activity_recorded"] = internal is not None
        for i, activity in enumerate(ACTIVITIES):
            row[f"activity_{activity}"] = internal["counts"][i] if internal else None
        for key in ("fast_adapter_norm", "slow_adapter_norm", "memory_entries"):
            row[key] = internal[key] if internal else None
        row.update(prediction_errors(record["decisions"]))
    return rows, metadata


def summarize_native(rows):
    result = aggregate(rows)
    # perf_counter elapsed time of CPU execution; it is not process CPU time.
    result["mean_cpu_elapsed_seconds"] = result.pop("mean_seconds")
    result["total_cpu_elapsed_seconds"] = sum(row["seconds"] for row in rows)
    observed = [row for row in rows if row["activity_recorded"]]
    result["activity_recorded_instances"] = len(observed)
    for activity in ACTIVITIES:
        total = sum(row[f"activity_{activity}"] for row in observed) if observed else None
        result[f"total_{activity}"] = total
        result[f"mean_{activity}"] = total / len(observed) if observed else None
    for adapter, activity in (("fast", "learn"), ("slow", "consolidate")):
        norms = [row[f"{adapter}_adapter_norm"] for row in observed]
        result[f"{activity}_requested_instances"] = sum(row[f"activity_{activity}"] > 0 for row in observed) if observed else None
        result[f"{adapter}_nonzero_instances"] = sum(value > 1e-6 for value in norms) if observed else None
        result[f"mean_final_{adapter}_norm"] = float(np.mean(norms)) if norms else None
        result[f"max_final_{adapter}_norm"] = max(norms) if norms else None
    result["mean_memory_entries"] = float(np.mean([row["memory_entries"] for row in observed])) if observed else None
    for kind in ("measurement", "test"):
        for suffix in ("decisions", "predictions", "nonzero_adaptation"):
            result[f"{kind}_{suffix}"] = sum(row[f"{kind}_{suffix}"] for row in rows)
        n = result[f"{kind}_predictions"]
        for component in ("shared", "total"):
            for target in ("test_success", "log_native_cost"):
                key = f"{kind}_{component}_{target}"
                result[f"{key}_mse"] = sum(row[f"{key}_squared_error_sum"] or 0 for row in rows) / n if n else None
    return result


def training_tasks(path):
    groups = {name: {"tasks": 0, "completed": 0, "queries": 0, "successful_queries": 0,
                     "measurements": 0, "tests": 0} for name in GROUPS}
    segments = set()
    for row in json_lines(path):
        key = row["stream"], row["lifetime"], row["phase"]
        position = "subsequent" if key in segments else "first"
        segments.add(key)
        role = ROLES[row["phase"]]
        for group in ("all", f"{role}_all", f"{role}_{position}"):
            data = groups[group]
            data["tasks"] += 1
            data["completed"] += int(row["successes"] == row["queries"])
            data["queries"] += row["queries"]
            data["successful_queries"] += row["successes"]
            data["measurements"] += row["measurements"]
            data["tests"] += row["tests"]
    return groups


def training_results(root, project):
    planned = read_json(root / "jobs.json")
    completion = completed_training(root)
    results = {}
    for job in planned:
        name = job["name"]
        directory = root / name
        row = {"source": relative(directory, project), "mode": job["mode"], "seed": job["seed"],
               "planned_stream_ticks": job["planned_stream_ticks"], "status": "not_started"}
        if not (directory / "config.json").exists():
            results[name] = row
            continue
        config = read_json(directory / "config.json")
        log = list(json_lines(directory / "training.jsonl"))
        last = log[-1] if log else {}
        if [entry["update"] for entry in log] != list(range(1, len(log) + 1)):
            raise ValueError(f"{name}: training log has missing or duplicate optimizer updates")
        row.update({key: config[key] for key in
                    ("updates", "batch", "unroll", "memory_size", "gamma", "compute_cost")})
        row.update(status=("finished" if completion[name] == 0 else "failed") if name in completion else "in_progress",
                   completed_updates=len(log), returncode=completion.get(name),
                   elapsed_seconds=last.get("seconds"),
                   world_interactions=sum(entry["external_actions_this_update"] for entry in log),
                   stream_ticks=sum(sum(entry["activity_counts_this_update"]) for entry in log),
                   final_prediction_loss=last.get("prediction_loss"),
                   final_memory_records_mean=last.get("memory_records_mean"),
                   tasks=training_tasks(directory / "tasks.jsonl"))
        for i, activity in enumerate(ACTIVITIES):
            row[f"total_{activity}"] = sum(entry["activity_counts_this_update"][i] for entry in log)
        for adapter in ("fast", "slow"):
            row[f"final_{adapter}_norm"] = last.get(f"{adapter}_norm")
            row[f"max_{adapter}_norm"] = max((entry[f"{adapter}_norm"] for entry in log), default=None)
        results[name] = row
    return results


def collect_native(root, project):
    plan = read_json(root / "evaluation_plan.json")
    sources = {entry["name"]: entry["source"] for entry in plan}
    names = list(BASELINES) + list(sources)
    runs, metadata, pending = {}, {}, {}
    for name in names:
        directory = root / name
        if not (directory / "summary.json").exists():
            pending[name] = {"completed_streams": 0, "instances": 0}
            continue
        config, raw = read_json(directory / "configuration.json"), read_json(directory / "summary.json")
        n = sum(len(stream["episodes"]) for stream in raw)
        expected = sum(len(stream["seeds"]) for stream in config["streams"])
        if len(raw) < len(config["streams"]) or n < expected:
            pending[name] = {"completed_streams": len(raw), "instances": n,
                             "expected_streams": len(config["streams"]), "expected_instances": expected}
            continue
        rows, meta = native_rows(name, directory)
        meta.update(source=relative(directory, project), training_source=sources.get(name))
        runs[name], metadata[name] = rows, meta
    if runs:
        schedule = lambda rows: [(identity(row), row["segment_role"], row["position_in_segment"]) for row in rows]
        expected = schedule(next(iter(runs.values())))
        for name, rows in runs.items():
            if schedule(rows) != expected:
                raise ValueError(f"{name}: native comparisons require the same ordered task streams")
    return runs, metadata, pending


def paired_results(runs, repetitions, resamples, seed):
    results = {}
    if not runs:
        return results
    streams = sorted({row["stream"] for row in next(iter(runs.values()))})
    bootstrap = np.random.default_rng(seed).integers(0, len(streams), (resamples, len(streams)))
    for repetition in repetitions:
        candidate = f"full_seed{repetition}"
        if candidate not in runs:
            continue
        references = [*BASELINES, *(f"{mode}_seed{repetition}" for mode in MODES[1:])]
        results[str(repetition)] = {}
        for reference in references:
            if reference not in runs:
                continue
            groups = {}
            for group in GROUPS:
                old = [row for row in runs[reference] if selected(row, group)]
                new = [row for row in runs[candidate] if selected(row, group)]
                pair = compare(old, new, streams, bootstrap)
                pair["all_instance_costs"]["cpu_elapsed_seconds"] = mean_difference(old, new, "seconds", streams, bootstrap)
                common = [(a, b) for a, b in zip(old, new) if a["completed"] and b["completed"]]
                pair["joint_success"]["costs"]["cpu_elapsed_seconds"] = mean_difference(
                    [a for a, _ in common], [b for _, b in common], "seconds", streams, bootstrap)
                if all(row["activity_recorded"] for row in old + new):
                    pair["internal_activities"] = {activity: mean_difference(old, new, f"activity_{activity}", streams, bootstrap)
                                                   for activity in ACTIVITIES}
                groups[group] = pair
            results[str(repetition)][reference] = {"candidate": candidate, "groups": groups}
    return results


def number(value, digits=2):
    return "—" if value is None else f"{value:.{digits}f}"


def interval(row, scale=1):
    point = row["candidate_minus_reference"]
    low, high = row["paired_stream_bootstrap_95_interval"]
    if point is None:
        return "—"
    bounds = "—" if low is None else f"{low * scale:+.2f}, {high * scale:+.2f}"
    return f"{point * scale:+.2f} [{bounds}]"


def markdown(summary):
    lines = ["# Continual experimental decisions: results", "",
             "Each native condition follows the same ordered task instances. Training repetitions are reported separately.",
             "[Method and stage-level results](methods.csv) · [Data, paired intervals and training records](summary.json)", "",
             "| Condition | Completed | Actions / instance | Measurements / instance | CPU-evaluated elapsed s / instance |",
             "| --- | ---: | ---: | ---: | ---: |"]
    for name, method in summary["native"].items():
        row = method["groups"]["all"]
        lines.append(f"| `{name}` | {row['completed']}/{row['instances']} | {number(row['mean_native_actions'])} | "
                     f"{number(row['mean_measurement_requests'])} | {number(row['mean_cpu_elapsed_seconds'])} |")
    if not summary["native"]:
        lines.append("| No complete native evaluations | — | — | — | — |")
    lines += ["", "Time includes environment construction, model execution and tool operations on CPU. It is elapsed time under concurrent evaluation, not process CPU time or isolated model latency."]
    if summary["pending_native"]:
        lines += ["", f"{len(summary['pending_native'])} native conditions are not yet complete and are excluded from these results."]
    continual = [(name, method["groups"]["all"]) for name, method in summary["native"].items()
                 if method["groups"]["all"]["activity_recorded_instances"]]
    if continual:
        lines += ["", "## Internal activities and received-feedback prediction", "",
                  "Counts below are totals across the same native instances. Predictions were recorded before each selected action; their targets are its subsequently returned test success and log(1 + native actions).", "",
                  "| Condition | REFINE | RETRIEVE | LEARN | CONSOLIDATE | Fast / slow nonzero instances | Test success MSE, shared → adapted | Measurement cost MSE, shared → adapted | Test cost MSE, shared → adapted |",
                  "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
        for name, row in continual:
            mse = lambda kind, target: " → ".join(number(row[f"{kind}_{component}_{target}_mse"], 4) for component in ("shared", "total"))
            lines.append(f"| `{name}` | {row['total_refine']} | {row['total_retrieve']} | {row['total_learn']} | "
                         f"{row['total_consolidate']} | {row['fast_nonzero_instances']} / {row['slow_nonzero_instances']} | "
                         f"{mse('test', 'test_success')} | {mse('measurement', 'log_native_cost')} | {mse('test', 'log_native_cost')} |")
        lines += ["", "LEARN/CONSOLIDATE counts show requested updates; nonzero norms show retained adapter state and may include earlier tasks. The discard-updates intervention can request updates while retaining zero adapters. Success predictions are raw regression outputs. Measurement actions have a zero test-success target and are scored separately. Shared/adapted errors use the same executed actions; policy benefit requires the paired intervention results."]
    lines += ["", "## Training exposure", "",
              "| Training | Status | Updates | Stream-ticks | External interactions | Finished proxy tasks | Elapsed s |",
              "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for name, row in summary["training"].items():
        lines.append(f"| `{name}` | {row['status']} | {row.get('completed_updates', 0)} | {row.get('stream_ticks', 0)} | "
                     f"{row.get('world_interactions', 0)} | {row.get('tasks', {}).get('all', {}).get('tasks', 0)} | "
                     f"{number(row.get('elapsed_seconds'), 1)} |")
    lines += ["", "Equal stream-tick budgets do not imply equal external training data, FLOPs or elapsed time. Training tasks use recorded/synthetic calibration outcomes and proxy costs; the table above separates them from native execution. Intervention conditions reuse their corresponding full-model training, not additional runs.", "",
              "## Paired interpretation", "",
              "Intervals resample complete task streams, retaining every instance within each sampled stream. Initial, switched and returning segments include separate first/subsequent-instance results. Cost differences are reported both for all instances and for the subset completed by both conditions. Training repetitions share test instances and are not pooled as independent test sets."]
    for repetition, comparisons in summary["paired"].items():
        lines += ["", f"Full model, training repetition {repetition}. Differences are full minus reference, with 95% stream-bootstrap intervals. Completion differences are percentage points; cost differences are native actions per instance.", "",
                  "| Reference | Gained / lost completions | Completion difference | Action difference, all | Action difference, joint success (n) |",
                  "| --- | ---: | ---: | ---: | ---: |"]
        for reference, comparison in comparisons.items():
            pair = comparison["groups"]["all"]
            common = pair["joint_success"]
            lines.append(f"| `{reference}` | {len(pair['gained_completions'])} / {len(pair['lost_completions'])} | "
                         f"{interval(pair['completion_rate_difference'], 100)} | "
                         f"{interval(pair['all_instance_costs']['native_actions'])} | "
                         f"{interval(common['costs']['native_actions'])} ({common['candidate']['instances']}) |")
    lines += ["", "## Regenerate", "", "```bash", summary["command"], "```", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path.cwd())
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--native-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resamples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=201)
    args = parser.parse_args()
    project = args.project.absolute()
    training_root = project / args.training_root
    native_root = project / args.native_root
    output = project / args.output
    training = training_results(training_root, project)
    runs, metadata, pending = collect_native(native_root, project)
    native = {name: {"metadata": metadata[name], "groups": {
        group: summarize_native([row for row in rows if selected(row, group)]) for group in GROUPS},
        "instances": rows} for name, rows in runs.items()}
    repetitions = sorted({row["seed"] for row in training.values()})
    command = ("python -m experiments.continual_experiment.summarize \\\n"
               f"  --training-root {shlex.quote(relative(training_root, project))} \\\n"
               f"  --native-root {shlex.quote(relative(native_root, project))} \\\n"
               f"  --output {shlex.quote(relative(output, project))} \\\n"
               f"  --resamples {args.resamples} --bootstrap-seed {args.bootstrap_seed}")
    summary = {"command": command,
               "sources": {"training": relative(training_root, project), "native": relative(native_root, project)},
               "bootstrap": {"unit": "complete ordered task stream", "resamples": args.resamples,
                             "seed": args.bootstrap_seed, "training_repetitions_pooled": False},
               "training": training, "native": native, "pending_native": pending,
               "paired": paired_results(runs, repetitions, args.resamples, args.bootstrap_seed)}
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    methods = [{"condition": name, "training_source": method["metadata"]["training_source"], "group": group, **row}
               for name, method in native.items() for group, row in method["groups"].items()]
    with (output / "methods.csv").open("w", newline="") as stream:
        fields = list(methods[0]) if methods else ["condition", "training_source", "group"]
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(methods)
    (output / "README.md").write_text(markdown(summary))
    print(json.dumps({"complete_native_conditions": len(native), "pending_native_conditions": len(pending),
                      "training_conditions": len(training), "output": relative(output, project)}))


if __name__ == "__main__":
    main()
