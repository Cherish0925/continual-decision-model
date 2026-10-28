"""Compare structured and visual native execution, then exercise partial evidence."""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import json
from pathlib import Path
from time import perf_counter

from .active_measurement_policy import NativeExperimentOperator
from .adapter import DiscoveryWorldEnv
from .reference import observed_objects
from .structured_observation import StructuredDiscoveryWorldEnv


def replay(source, output, env_class):
    rows = [json.loads(line) for line in (source / "trajectory.jsonl").read_text().splitlines()]
    metadata = json.loads((source / "run.json").read_text())
    env = env_class(output, len(rows) + 1)
    env.reset(metadata["scenario"], metadata["difficulty"], metadata["seed"])
    env.restore_initial_rng(source / metadata["initial_rng_snapshot"]["path"])
    results = []
    start = perf_counter()
    for row in rows:
        transition = env.step(row["action"])
        results.append({"before": transition["observation"]["ui"],
                        "return": transition["action_result"],
                        "after": transition["next_observation"]["ui"],
                        "metrics": transition["task_metrics"]})
    return results, perf_counter() - start


def partial_probe(source, output, *, check_radiation=False):
    metadata = json.loads((source / "run.json").read_text())
    env = StructuredDiscoveryWorldEnv(output, 180)
    observation = env.reset(metadata["scenario"], metadata["difficulty"], metadata["seed"])
    env.restore_initial_rng(source / metadata["initial_rng_snapshot"]["path"])
    operator = NativeExperimentOperator()

    def step(action):
        nonlocal observation
        transition = env.step(action)
        operator.observe(transition)
        observation = env.policy_input()

    for fragment in ("instruments", "crystal bench", "reactor lab"):
        location = next(name for name in observation["teleport_locations"] if fragment in name)
        step({"action": "TELEPORT_TO_LOCATION", "arg1": location})
    operator.next_action(observation)
    objects = operator.objects
    working = [handle for handle, obj in objects.items() if obj["name"] == "crystal reactor (activated)"]
    empty = [handle for handle, obj in objects.items() if "no crystal present" in obj["name"]]
    loose = [handle for handle, obj in objects.items()
             if obj["name"].startswith("quantum crystal") and "(on table" in obj["description"]]

    def execute(request):
        operator.request(request)
        while (action := operator.next_action(observation)) is not None:
            step(action)
        return operator.completed[-1]

    supports = []
    for reactor in working[:2]:
        setting = execute({"kind": "read_setting", "reactor_handle": reactor})["setting"]
        target = setting["object_handle"]
        measurement = execute({"kind": "measure", "group_index": 0, "target_handle": target})
        density = next(r["value"] for r in measurement["measurements"] if r["quantity"] == "density_g_cm3")
        supports.append((density, setting["frequency_hz"]))
    query = execute({"kind": "measure", "group_index": 0, "target_handle": loose[0]})
    density = next(r["value"] for r in query["measurements"] if r["quantity"] == "density_g_cm3")
    slope = (supports[1][1] - supports[0][1]) / (supports[1][0] - supports[0][0])
    mean = supports[0][1] + slope * (density - supports[0][0])
    trial = execute({"kind": "test", "target_handle": loose[0], "reactor_handle": empty[0],
                     "mean_hz": mean, "std_hz": 2.})
    presence = operator.calibration.query(loose[0]).presence
    result = {"native_actions": env.steps, "measured_query_quantities": [q for q, yes in presence.items() if yes],
              "trial_success": trial["success"]}
    if check_radiation:
        repeated = [execute({"kind": "measure", "group_index": 2,
                             "target_handle": loose[0]}) for _ in range(3)]
        rows = [row for request in repeated for row in request["measurements"]
                if row["kind"] == "instrument" and row["quantity"] == "radiation_usv_h"]
        assert len(rows) == 3
        assert [request["actions"] for request in repeated[1:]] == [1, 1]
        execute({"kind": "measure", "group_index": 0, "target_handle": loose[0]})
        restored = execute({"kind": "read_setting", "reactor_handle": empty[0]})
        assert restored["setting"]["activated"] is True
        result["radiation"] = {"returned_samples": len(rows),
                               "request_native_actions": [request["actions"] for request in repeated],
                               "values": [row["value"] for row in rows],
                               "restored_activation": restored["setting"]["activated"]}
    result["requests"] = operator.completed
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True)
    with (args.output / "native.log").open("w") as log, redirect_stdout(log):
        png, png_time = replay(args.source, args.output / "png", DiscoveryWorldEnv)
        structured, structured_time = replay(args.source, args.output / "structured", StructuredDiscoveryWorldEnv)
        mismatches = [i for i, (a, b) in enumerate(zip(png, structured)) if a != b]
        if mismatches:
            raise AssertionError(f"Public execution differs at {mismatches}")
        partial = partial_probe(args.source, args.output / "partial", check_radiation=True)
    result = {"actions_compared": len(png), "public_ui_and_returns_identical": not mismatches,
              "png_execution_seconds": png_time, "structured_execution_seconds": structured_time,
              "partial_evidence": partial}
    (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "partial_evidence"}))
    print(json.dumps({key: value for key, value in partial.items() if key != "requests"}))


if __name__ == "__main__":
    main()
