"""A native task-stream reference that remembers the last successful hypothesis.

At each new target, try the most recently successful candidate first. Acquire
that candidate's two initial support readings and current target reading, then
use the existing numerical posterior and scalar-setting operator. On failure
or an unavailable prediction, select the highest current posterior among the
remaining candidates. Commit to each selected candidate until its measurements
and test finish. Spectrometry shares all five returned channels; radiation uses
32 separately paid samples per object and the existing running minimum.

Only a candidate index crosses task boundaries. All readings and numerical
coefficients are rebuilt in the current task. The reset condition clears that
index at each task boundary; its within-task policy is identical. A successful
candidate can be an approximate or accidental fit, not the simulator's hidden
relation. No scenario seed, schedule label or task score enters the selector.
This is a fixed reference policy, not a trained CDM or a new learning method.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stdout
import fcntl
import gc
import json
import multiprocessing
from pathlib import Path
from time import perf_counter

import torch

from .active_native import NativeActiveEpisode
from .adapter import TrajectoryWriter
from .calibration import FEATURE_QUANTITIES
from .calibration_mixture import load_mixture_checkpoint
from .evaluate_active_native import _public_test_mask
from .evaluate_active_stream import NATIVE_RELATIONS, public_object_order, schedule_position
from .structured_observation import StructuredDiscoveryWorldEnv
from .train_active_core import GROUP_QUANTITIES, QUANTITY_GROUP, visible_beliefs


class SuccessfulRelationMemory:
    """An explicit candidate index, with a fresh within-task search state."""

    def __init__(self, radiation_samples=32):
        self.radiation_samples = radiation_samples
        self.last_success = None
        self.start_task(carry=True)

    def start_task(self, *, carry):
        if not carry:
            self.last_success = None
        self.query = None
        self.pending = None
        self.unavailable = set()

    def choose(self, episode, mixture):
        query = int(episode.query[0])
        if query != self.query:
            self.query = query
            self.pending = None
            self.unavailable.clear()
        tensors = episode.tensors(mixture.device)
        beliefs = visible_beliefs(tensors["values"], tensors["seen"], tensors["known_hz"],
                                 tensors["known"], tensors["query"], mixture)
        while True:
            if self.pending is None:
                choices = [q for q in range(9) if not episode.failed[0, q]
                           and q not in self.unavailable]
                if not choices:
                    return None, beliefs
                if self.last_success in choices:
                    self.pending = self.last_success
                else:
                    self.pending = max(choices, key=lambda q: (float(beliefs["weights"][0, q]), -q))
            quantity = self.pending
            group = QUANTITY_GROUP[quantity]
            required = self.radiation_samples if group == 2 else 1
            for obj in (0, 1, query):
                count = int(episode.counts[0, obj, list(GROUP_QUANTITIES[group])].min())
                if count < required:
                    return obj * 5 + group, beliefs
            if bool(_public_test_mask(episode, beliefs)[quantity]):
                return 20 + quantity, beliefs
            self.unavailable.add(quantity)
            self.pending = None

    def observe(self, decision):
        if decision["action_index"] >= 20:
            if decision["success"]:
                self.last_success = decision["action_index"] - 20
            self.pending = None


def candidate_name(index):
    return None if index is None else FEATURE_QUANTITIES[index]


def run_stream(job):
    options, stream = job
    torch.set_num_threads(options["threads"])
    mixture, _ = load_mixture_checkpoint(options["mixture_checkpoint"], "cpu")
    mixture.eval()
    policy = SuccessfulRelationMemory(options["radiation_samples"])
    root = Path(options["output"]) / stream["name"]
    root.mkdir(parents=True)
    started = perf_counter()
    rows = []
    for index, seed in enumerate(stream["seeds"]):
        policy.start_task(carry=options["state_mode"] == "carry")
        entry = policy.last_success
        output = root / f"episode{index:03d}_seed{seed}"
        output.mkdir()
        episode_start = perf_counter()
        error, ending, bootstrap = None, "decision_budget", None
        with (output / "native.log").open("w") as log, redirect_stdout(log), torch.inference_mode():
            env = StructuredDiscoveryWorldEnv(output, options["max_actions"])
            env.reset("Reactor Lab", "Normal", seed)
            rng_root = Path(options["initial_rng_root"])
            rng_root.mkdir(parents=True, exist_ok=True)
            snapshot = rng_root / f"seed{seed}.json.gz"
            with (rng_root / f"seed{seed}.lock").open("a") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                rng = env.restore_initial_rng(snapshot) if snapshot.exists() else env.save_initial_rng(snapshot)
            (output / "initial_rng.json.gz").write_bytes(snapshot.read_bytes())
            metadata = {
                **env.configuration, "policy": "last_successful_relation_reference",
                "is_learned_policy": False, "state_mode": options["state_mode"],
                "carried_state": "last_actual_successful_candidate_index" if options["state_mode"] == "carry" else None,
                "ledger": "fresh_for_each_instance",
                "numerical_posterior": "current_instance_public_measurements_only",
                "initial_rng_snapshot": {"path": "initial_rng.json.gz", **rng},
                "action_descriptions": env.actions, "teleport_locations": env.teleports,
                "mixture_checkpoint": options["mixture_checkpoint"],
                "max_decisions": options["max_decisions"],
                "radiation_samples_per_object": options["radiation_samples"],
                "failed_test_retention": "until_its_own_observed_support_or_query_changes",
            }
            with TrajectoryWriter(output, metadata) as writer:
                episode = NativeActiveEpisode(env, writer)
                try:
                    episode.bootstrap()
                    bootstrap = {
                        "measurement_samples": int(episode.counts.sum()),
                        "known_support_slots": torch.where(episode.known[0])[0].tolist(),
                        "known_support_hz": episode.known_hz[0, :2].tolist(),
                        "failed_hypotheses": int(episode.failed.sum()),
                        "public_object_order": public_object_order(episode),
                    }
                    for _ in range(options["max_decisions"]):
                        if env.ended:
                            ending = "native_termination" if all(m["completed"] for m in env.metrics) else "native_action_budget"
                            break
                        action, beliefs = policy.choose(episode, mixture)
                        if action is None:
                            ending = "available_hypotheses_exhausted"
                            break
                        selected = policy.pending
                        mean = float(beliefs["means"][0, action - 20]) if action >= 20 else None
                        spread = float(beliefs["std"][0, action - 20]) if action >= 20 else None
                        decision = episode.apply(action, mean_hz=mean, std_hz=spread)
                        decision["selected_candidate"] = selected
                        decision["selected_quantity"] = candidate_name(selected)
                        policy.observe(decision)
                    else:
                        if env.ended:
                            ending = "native_termination" if all(m["completed"] for m in env.metrics) else "native_action_budget"
                except (ValueError, RuntimeError) as exc:
                    error = f"{type(exc).__name__}: {exc}"
                    ending = "execution_error"
            row = {
                "stream": stream["name"], "episode_index": index, "seed": seed,
                "relation": NATIVE_RELATIONS[seed % 5], "state_mode": options["state_mode"],
                **schedule_position(stream, index),
                "ending": ending, "error": error, "seconds": perf_counter() - episode_start,
                "bootstrap": bootstrap,
                "relation_memory": {"entry_candidate": entry, "entry_quantity": candidate_name(entry),
                                    "exit_candidate": policy.last_success,
                                    "exit_quantity": candidate_name(policy.last_success)},
                **episode.summary(),
            }
            (output / "summary.json").write_text(json.dumps(row, indent=2) + "\n")
        rows.append({key: value for key, value in row.items() if key != "decisions"})
        (root / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")
        print(json.dumps({key: row[key] for key in ("stream", "episode_index", "seed", "state_mode",
                                                   "completed_successfully", "native_actions", "ending")}), flush=True)
        del episode, env
        gc.collect()
        if error is not None:
            break
    return {"stream": stream["name"], "state_mode": options["state_mode"],
            "instances": len(rows), "completed": sum(row["completed_successfully"] for row in rows),
            "native_actions": sum(row["native_actions"] for row in rows),
            "seconds": perf_counter() - started, "episodes": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--manifest", type=Path)
    source.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--state-mode", choices=("carry", "reset"), required=True)
    parser.add_argument("--mixture-checkpoint", required=True)
    parser.add_argument("--initial-rng-root", required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--max-actions", type=int, default=400)
    parser.add_argument("--max-decisions", type=int, default=160)
    parser.add_argument("--radiation-samples", type=int, default=32)
    args = parser.parse_args()
    if args.manifest:
        streams = json.loads(args.manifest.read_text())["streams"]
        args.manifest = str(args.manifest)
    else:
        streams = [{"name": "stream", "seeds": args.seeds}]
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    options = vars(args)
    (root / "configuration.json").write_text(json.dumps({**options, "streams": streams}, indent=2) + "\n")
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        for future in as_completed([pool.submit(run_stream, (options, stream)) for stream in streams]):
            rows.append(future.result())
            (root / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
