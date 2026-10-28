"""Collect native chemical experiments through public observations and actions.

The reference tries bounded integer recipes over the visible dispensers. It
receives no scenario seed, simulator object, hidden recipe or task score.
Experiment records preserve the observed key state before and after treatment:
partial rust removal can persist, so trials are not independent fresh-key tests.
This is executable reference data for training, not a learned-policy result.

Example:
    python -m experiments.discoveryworld.collect_chemistry \\
        --output runs/discoveryworld/task_transfer/chemistry/train \\
        --seeds 7100 7101 7102 7103 --workers 4
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stdout
from copy import deepcopy
import json
import multiprocessing
from pathlib import Path
import re
from time import perf_counter

from .adapter import TrajectoryWriter
from .reference import observed_objects
from .structured_observation import StructuredDiscoveryWorldEnv


def integer_recipes(component_count: int, max_parts: int):
    """Enumerate all nonempty recipes by total amount, without a hidden recipe prior."""
    def partition(total, count):
        if count == 1:
            yield (total,)
        else:
            for amount in range(total, -1, -1):
                for tail in partition(total - amount, count - 1):
                    yield (amount, *tail)

    for total in range(1, max_parts + 1):
        yield from partition(total, component_count)


def observed_rust(name: str) -> int:
    for label, value in (("no rust", 0), ("lightly rusted", 1),
                         ("moderately rusted", 2), ("heavily rusted", 3)):
        if label in name:
            return value
    raise ValueError(f"No public rust description in {name!r}")


class ChemistryExperiments:
    """Execute a proposed recipe and return public feedback plus native action cost.

    A future policy can call ``state()``, select component counts, and call
    ``experiment(counts)``. The same methods are used by this reference collector.
    ``state()`` contains public task text and experiment history, without scores.
    """

    def __init__(self, env, writer):
        self.env = env
        self.writer = writer
        self.observation = env.policy_input()
        self.seen = observed_objects(self.observation)
        self.history = []
        self.components = []
        self.dispensers = []
        self.jar = self.key = self.cleaner = self.door = None

    def _action(self, action, arg1=None, arg2=None):
        payload = {"action": action}
        if arg1 is not None:
            payload["arg1"] = arg1
        if arg2 is not None:
            payload["arg2"] = arg2
        transition = self.env.step(payload)
        self.writer.append(transition)
        self.observation = self.env.policy_input()
        self.seen.update(observed_objects(self.observation))
        if not transition["action_result"].get("success", False):
            raise RuntimeError(self.observation["ui"]["lastActionMessage"])
        return transition

    def _approach(self, uuid):
        if uuid not in self.seen:
            raise ValueError("The target has not been observed")
        ui = self.observation["ui"]
        reachable = {o["uuid"] for field in ("accessibleEnvironmentObjects", "inventoryObjects")
                     for o in ui.get(field, [])}
        if uuid not in reachable:
            self._action("TELEPORT_TO_OBJECT", uuid)

    def _wait_for_reaction(self):
        # The public task instructions allow up to two ticks for mixing/treatment.
        # Rotation is a documented action; all ticks remain in the native trace.
        for _ in range(2):
            facing = self.observation["ui"]["agentLocation"]["faceDirection"]
            self._action("ROTATE_DIRECTION", facing)

    def _find(self, predicate):
        matches = [o for o in self.seen.values() if predicate(o["name"])]
        if len(matches) != 1:
            raise ValueError(f"Expected one observed object, found {len(matches)}")
        return matches[0]["uuid"]

    def prepare(self):
        """Acquire the public apparatus and discover its visible component names."""
        self.jar = self._find(lambda n: n == "jar")
        self.key = self._find(lambda n: "key" in n and "rust" in n)
        self.door = self._find(lambda n: n == "door")
        for uuid in (self.jar, self.key):
            self._approach(uuid)
            self._action("PICKUP", uuid)
        visited = set()
        while not any(o["name"] == "bottle cleaner" for o in self.seen.values()):
            apparatus = sorted((o["name"], o["uuid"]) for o in self.seen.values()
                               if o["name"].startswith("Dispenser (") and o["uuid"] not in visited)
            if not apparatus:
                raise ValueError("No unvisited public apparatus leads to the bottle cleaner")
            _, uuid = apparatus[-1]
            visited.add(uuid)
            self._approach(uuid)
        self.cleaner = self._find(lambda n: n == "bottle cleaner")
        # Walking to the apparatus may reveal more dispensers in larger rooms.
        self._approach(self.cleaner)
        pairs = []
        for obj in self.seen.values():
            match = re.fullmatch(r"Dispenser \((.+)\)", obj["name"])
            if match:
                pairs.append((match.group(1), obj["uuid"]))
        pairs.sort()
        if not pairs:
            raise ValueError("No public chemical dispensers were observed")
        self.components = [name for name, _ in pairs]
        self.dispensers = [uuid for _, uuid in pairs]
        return self.state()

    def state(self):
        key = self.seen[self.key]
        return {
            "task_description": self.observation["ui"]["taskProgress"][0]["description"],
            "components": list(self.components),
            "key_description": key["description"],
            "rust_level": observed_rust(key["name"]),
            "history": [{k: deepcopy(v) for k, v in trial.items()
                         if k not in ("state", "next_state")}
                        for trial in self.history],
        }

    def experiment(self, counts):
        counts = tuple(counts)
        if len(counts) != len(self.dispensers) or not any(counts):
            raise ValueError("Supply one nonnegative count per visible component")
        if any(not isinstance(v, int) or v < 0 for v in counts):
            raise ValueError("Recipe quantities must be nonnegative integers")
        before = self.state()
        start = self.env.steps
        if self.history:
            self._approach(self.cleaner)
            self._action("USE", self.cleaner, self.jar)
        # The key remains outside the jar while ingredients are added and mixed.
        for dispenser, amount in zip(self.dispensers, counts):
            if amount:
                self._approach(dispenser)
                for _ in range(amount):
                    self._action("USE", dispenser, self.jar)
        self._wait_for_reaction()
        mixture = [o["description"] for o in self.observation["ui"]["inventoryObjects"]
                   if "substance" in o["name"].lower() or "mixture" in o["name"].lower()]
        self._action("PUT", self.key, self.jar)
        self._wait_for_reaction()
        after = self.state()
        self._action("PICKUP", self.key)
        record = {
            "index": len(self.history), "state": before,
            "component_counts": list(counts), "observed_mixture": mixture,
            "rust_before": before["rust_level"], "rust_after": after["rust_level"],
            "key_after": after["key_description"],
            "rust_removed": after["rust_level"] == 0,
            "native_actions": self.env.steps - start,
            "transition_start": start, "transition_stop": self.env.steps,
        }
        self.history.append(record)
        record["next_state"] = self.state()
        return record

    def leave(self):
        """Use the cleaned key to open the observed door and walk through it."""
        self._approach(self.door)
        self._action("OPEN", self.door)
        facing = self.observation["ui"]["agentLocation"]["faceDirection"]
        for _ in range(3):
            if self.env.ended:
                break
            self._action("MOVE_DIRECTION", facing)


def collect(job):
    seed, root, difficulty, split, max_steps, max_parts = job
    output = root / f"seed{seed}"
    output.mkdir(parents=True)
    started = perf_counter()
    error = None
    with (output / "native.log").open("w") as log, redirect_stdout(log):
        env = StructuredDiscoveryWorldEnv(output, max_steps)
        env.reset("Combinatorial Chemistry", difficulty, seed)
        metadata = {**env.configuration, "policy": "public_integer_recipe_reference",
                    "is_learned_policy": False, "split": split, "max_recipe_parts": max_parts,
                    "action_descriptions": env.actions, "teleport_locations": env.teleports,
                    "experiment_state": "public component names, pre-treatment rust and past trials"}
        with TrajectoryWriter(output, metadata) as writer, \
                (output / "experiments.jsonl").open("x") as experiments:
            executor = ChemistryExperiments(env, writer)
            try:
                executor.prepare()
                for counts in integer_recipes(len(executor.components), max_parts):
                    record = executor.experiment(counts)
                    experiments.write(json.dumps(record) + "\n")
                    experiments.flush()
                    if record["rust_removed"]:
                        executor.leave()
                        break
                    if env.ended:
                        break
            except (RuntimeError, ValueError) as exc:
                error = f"{type(exc).__name__}: {exc}"
        summary = {
            "seed": seed, "scenario": "Combinatorial Chemistry", "difficulty": difficulty,
            "split": split, "is_learned_policy": False, "native_actions": env.steps,
            "experiments": len(executor.history),
            "unsuccessful_recipes": sum(not r["rust_removed"] for r in executor.history),
            "successful_recipes": sum(r["rust_removed"] for r in executor.history),
            "task_metrics": env.metrics, "error": error,
            "seconds": perf_counter() - started,
        }
        (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--difficulty", choices=("Normal", "Challenge"), default="Normal")
    parser.add_argument("--split", choices=("train", "validation"), default="train")
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--max-parts", type=int, default=6)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    jobs = [(seed, args.output, args.difficulty, args.split, args.max_steps, args.max_parts)
            for seed in args.seeds]
    summaries = []
    with ProcessPoolExecutor(max_workers=args.workers,
                             mp_context=multiprocessing.get_context("spawn")) as pool:
        for future in as_completed([pool.submit(collect, job) for job in jobs]):
            summary = future.result()
            summaries.append(summary)
            (args.output / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
            print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
