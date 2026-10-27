"""Native policy takeover after an explicitly supplied, actually replayed prefix.

This is a conditional evaluation, never an autonomous end-to-end result. Only
the requested source prefix is read. Its actions are replayed in a fresh native
world; the policy receives actual public feedback, not recorded answers.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
from hashlib import sha256
from importlib import import_module
import json
from pathlib import Path
import shutil
from time import perf_counter
import warnings

from .adapter import DiscoveryWorldEnv, TrajectoryWriter
from .analyze_rollouts import _active_pairs
from .reference import observed_objects


FACTORIES = {
    "experiments.discoveryworld.evaluate:create": "learned_dynamic_action_policy",
    "experiments.discoveryworld.semantic_policy:create": "frozen_semantic_native_generation",
    "experiments.discoveryworld.quantitative_policy:create": "learned_calibration_with_native_scalar_control",
    "experiments.discoveryworld.mixture_policy:create": "learned_relation_mixture_with_native_scalar_control",
    "experiments.discoveryworld.affine_reference_policy:create": "affine_reference_with_native_scalar_control",
    "experiments.discoveryworld.belief_actions:create": "cdm_belief_action_policy_with_explicit_numeric_primitive",
    "experiments.discoveryworld.adaptive_compute:create": "joint_cdm_action_and_compute_policy_with_explicit_numeric_primitive",
    "experiments.discoveryworld.predictive_actions:create": "predictive_cdm_with_explicit_numeric_primitive",
    "experiments.discoveryworld.experimental_plans:create": "cdm_with_persistent_experimental_plans",
}
FORWARD_PRIMED_POLICIES = {"learned_dynamic_action_policy", "learned_calibration_with_native_scalar_control",
                         "learned_relation_mixture_with_native_scalar_control",
                         "cdm_belief_action_policy_with_explicit_numeric_primitive",
                         "joint_cdm_action_and_compute_policy_with_explicit_numeric_primitive",
                         "predictive_cdm_with_explicit_numeric_primitive",
                         "cdm_with_persistent_experimental_plans",
                         "affine_reference_with_native_scalar_control"}


def load_prefix(source: Path, steps: int) -> dict:
    """Read exactly the first N transitions, never the remaining trace/summary."""
    if steps < 0:
        raise ValueError("prefix_steps must be nonnegative")
    source = Path(source).resolve()
    trajectory = source / "trajectory.jsonl" if source.is_dir() else source
    metadata = json.loads((trajectory.parent / "run.json").read_text())
    identity = {key: metadata[key] for key in ("scenario", "difficulty", "seed")}
    snapshot = metadata.get("initial_rng_snapshot")
    if snapshot is None and (trajectory.parent / "initial_rng.json.gz").exists():
        # Earlier native and on-policy collectors wrote this same snapshot
        # under a shorter metadata key or as a companion file only.
        snapshot = {"path": "initial_rng.json.gz", **metadata.get("initial_rng", {})}
    digest, rows = sha256(), []
    with trajectory.open("rb") as stream:
        for index in range(steps):
            line = stream.readline()
            if not line:
                raise ValueError(f"Source has fewer than {steps} prefix actions")
            digest.update(line)
            row = json.loads(line)
            if row["index"] != index:
                raise ValueError(f"Source prefix is not contiguous at index {index}")
            # Exclude annotations, teacher phase, hypothesis and task score.
            rows.append({"index": index, "action": row["action"], "action_result": row["action_result"],
                         "before_ui": row["observation"]["ui"], "after_ui": row["next_observation"]["ui"]})
    return {"identity": identity, "trajectory": str(trajectory), "prefix_sha256": digest.hexdigest(),
            "source_policy": metadata.get("policy"), "source_observation_adapter": metadata.get("observation_adapter"),
            "initial_rng_snapshot": snapshot,
            "rows": rows}


def _differences(expected: dict, actual: dict, field: str) -> list[dict]:
    """Report exact public-field differences; timestamps/images are not compared."""
    # JSON trajectories stringify native integer dialogue-option keys. Compare
    # both values in the same public serialization, retaining every reading.
    expected = json.loads(json.dumps(expected))
    actual = json.loads(json.dumps(actual))
    return [{"field": f"{field}.{key}", "expected": expected.get(key), "actual": actual.get(key)}
            for key in sorted(expected.keys() | actual.keys()) if expected.get(key) != actual.get(key)]


def _progress(metrics):
    return sum(row["progress"] for row in metrics) / len(metrics)


def _public_transition(transition):
    return {key: deepcopy(transition[key]) for key in
            ("index", "observation", "action", "action_result", "next_observation")}


class _Activations:
    """Track directly observed transitions, excluding known restorations."""
    def __init__(self, observation):
        self.seen = {crystal for _, crystal in _active_pairs(observation)}
        self.events = []

    def observe(self, transition, phase):
        before, after = transition["observation"], transition["next_observation"]
        self.seen.update(crystal for _, crystal in _active_pairs(before))
        objects_before, objects_after = observed_objects(before), observed_objects(after)
        pairs = _active_pairs(after)
        for handle, obj in objects_after.items():
            if (obj.get("name") == "crystal reactor (activated)" and handle in objects_before
                    and objects_before[handle].get("name") != "crystal reactor (activated)"):
                crystals = sorted(crystal for reactor, crystal in pairs if reactor == handle)
                self.events.append({"index": transition["index"], "control": phase, "reactor": handle,
                                    "crystals": crystals, "newly_activated_crystals": [
                                        crystal for crystal in crystals if crystal not in self.seen]})
        self.seen.update(crystal for _, crystal in pairs)


def evaluate_prefix(agent, source: Path, output: Path, *, prefix_steps: int,
                    takeover_steps: int, policy_type: str, policy_metadata: dict | None = None,
                    first_action: dict | None = None, first_refinements: int | None = None) -> dict:
    """Replay a verified prefix, prime the existing policy, then let it act.

    DigitalPolicy is advanced once per prefix observation through LearnedPolicy
    __call__; its proposed actions are discarded. SemanticPolicy.observe stores
    actual history without generation. Both observe every actual transition.
    """
    if policy_type not in FACTORIES.values():
        raise ValueError(f"Unsupported takeover policy: {policy_type}")
    if takeover_steps <= 0:
        raise ValueError("takeover_steps must be positive")
    if first_refinements is not None and (first_refinements < 0 or not hasattr(agent, "refinements")):
        raise ValueError("first_refinements requires a neural policy and a nonnegative depth")
    prefix = load_prefix(source, prefix_steps)
    output = Path(output).resolve()
    if any((output / name).exists() for name in ("trajectory.jsonl", "run.json", "summary.json", "decisions.jsonl", "initial_rng.json.gz", "instance_updates.jsonl")):
        raise ValueError(f"Choose a new takeover output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    prime_forward = policy_type in FORWARD_PRIMED_POLICIES
    if prime_forward:
        agent.reset()
    else:
        agent.reset(trace_path=output / "decisions.jsonl")
    env = DiscoveryWorldEnv(output, prefix_steps + takeover_steps)
    observation = env.reset(**prefix["identity"])
    rng_snapshot = prefix["initial_rng_snapshot"]
    rng_replay = {"initial_rng_restored": False,
                  "status": "legacy_seed_only; unseeded object RNGs make exact replay unguaranteed"}
    local_snapshot = None
    if rng_snapshot is not None:
        source_snapshot = Path(prefix["trajectory"]).parent / rng_snapshot["path"]
        restored = env.restore_initial_rng(source_snapshot, expected_sha256=rng_snapshot.get("sha256"))
        filename = "initial_rng.json.gz"
        shutil.copyfile(source_snapshot, output / filename)
        local_snapshot = {"path": filename, **restored}
        rng_replay = {"initial_rng_restored": True, "status": "initial_native_python_rng_restored", **restored}
    else:
        warnings.warn("Source has no initial RNG snapshot: strict replay still checks every public result, "
                      "but scenario seed alone does not reproduce native instrument noise.", RuntimeWarning)
    initial_progress = _progress(env.metrics)
    metadata = {**env.configuration, "policy": "prefix_takeover_evaluation",
                "takeover_policy": policy_type, "policy_config": policy_metadata or {},
                "evaluation_kind": "externally_initialized_prefix_takeover", "is_end_to_end_evaluation": False,
                "prefix_steps": prefix_steps, "takeover_budget": takeover_steps,
                "first_action_intervention": first_action,
                "first_refinement_intervention": first_refinements,
                "prefix_source": {key: value for key, value in prefix.items() if key != "rows"},
                "rng_replay": rng_replay,
                "prefix_priming": ("source_forward_then_actual_feedback" if hasattr(agent, "prime") else
                                   "forward_discard_proposal_then_observe") if prime_forward else "observe_only_no_generation",
                "replay_comparison": "exact_public_ui_and_native_action_result; no image or wall_clock comparison",
                "action_descriptions": env.actions, "teleport_locations": env.teleports,
                "initial_task_metrics": deepcopy(env.metrics),
                "source_sha256": sha256(Path(__file__).read_bytes()).hexdigest()}
    if hasattr(agent, "evaluation_metadata"):
        metadata["policy_mechanism"] = agent.evaluation_metadata
    if local_snapshot is not None:
        metadata["initial_rng_snapshot"] = local_snapshot
    costs = {phase: {"policy_calls": 0, "policy_seconds": 0., "observe_seconds": 0., "environment_seconds": 0.}
             for phase in ("external_prefix", "policy_takeover")}
    activations, calls = _Activations(observation), []
    prefix_curve, takeover_curve = [initial_progress], []
    mismatches, error, prefix_executed, takeover_executed = [], None, 0, 0
    prefix_elapsed = takeover_elapsed = 0.
    start = perf_counter()

    def propose(phase):
        cost = costs[phase]
        cost["policy_calls"] += 1
        begin = perf_counter()
        call = {"control": phase, "index": env.steps, "proposal_executed": False}
        override = phase == "policy_takeover" and takeover_executed == 0 and first_refinements is not None
        original_depth = getattr(agent, "refinements", None)
        try:
            supplied_action = phase == "external_prefix" or (first_action is not None and takeover_executed == 0)
            proposer = getattr(agent, "prime", agent) if supplied_action else agent
            if override:
                agent.refinements = first_refinements
                call["refinement_intervention"] = first_refinements
            action = proposer(observation)
            call["action"] = deepcopy(action)
            return action
        except Exception as failure:
            call["error"] = {"type": type(failure).__name__, "message": str(failure)}
            raise
        finally:
            if override:
                agent.refinements = original_depth
            call["seconds"] = perf_counter() - begin
            cost["policy_seconds"] += call["seconds"]
            calls.append(call)

    def execute(action, phase):
        begin = perf_counter()
        transition = env.step(action)
        costs[phase]["environment_seconds"] += perf_counter() - begin
        if phase == "policy_takeover":
            calls[-1]["proposal_executed"] = action == calls[-1]["action"]
            calls[-1]["executed_action"] = deepcopy(action)
        transition["control"] = phase
        transition["takeover_index"] = env.steps - prefix_steps - 1 if phase == "policy_takeover" else None
        return transition

    def receive(transition, phase):
        begin = perf_counter()
        try:
            activations.observe(transition, phase)
            agent.observe(_public_transition(transition))
        finally:
            costs[phase]["observe_seconds"] += perf_counter() - begin

    with TrajectoryWriter(output, metadata) as writer:
        phase_start = perf_counter()
        try:
            for expected in prefix["rows"]:
                index = expected["index"]
                if env.ended:
                    mismatches.append({"index": index, "stage": "before_action", "reason": "native_episode_ended_before_prefix"})
                    break
                differences = _differences(expected["before_ui"], observation["ui"], "observation.ui")
                if differences:
                    mismatches.append({"index": index, "stage": "before_action", "differences": differences})
                    break
                proposal = propose("external_prefix") if prime_forward else None
                transition = execute(deepcopy(expected["action"]), "external_prefix")
                prefix_executed += 1
                prefix_curve.append(_progress(env.metrics))
                differences = _differences(expected["action_result"], transition["action_result"], "action_result")
                differences += _differences(expected["after_ui"], transition["next_observation"]["ui"], "next_observation.ui")
                transition["prefix_source_index"] = index
                transition["prefix_proposal_discarded"] = proposal
                transition["prefix_replay_matched"] = not differences
                if differences:
                    mismatches.append({"index": index, "stage": "after_action", "differences": differences})
                writer.append(transition)
                receive(transition, "external_prefix")
                observation = env.policy_input()
                if differences:
                    break
            prefix_elapsed = perf_counter() - phase_start
            if prefix_executed == prefix_steps and not mismatches:
                takeover_curve = [_progress(env.metrics)]
                phase_start = perf_counter()
                while not env.ended and takeover_executed < takeover_steps:
                    if getattr(agent, "stop_requested", False):
                        break
                    action = propose("policy_takeover")
                    if getattr(agent, "stop_requested", False):
                        break
                    if first_action is not None and takeover_executed == 0:
                        action = deepcopy(first_action)
                    transition = execute(action, "policy_takeover")
                    takeover_executed += 1
                    takeover_curve.append(_progress(env.metrics))
                    writer.append(transition)
                    receive(transition, "policy_takeover")
                    observation = env.policy_input()
                takeover_elapsed = perf_counter() - phase_start
        except Exception as failure:
            error = {"type": type(failure).__name__, "message": str(failure)}
            if takeover_curve:
                takeover_elapsed = perf_counter() - phase_start
            else:
                prefix_elapsed = perf_counter() - phase_start
    matched = prefix_executed == prefix_steps and not mismatches
    complete = all(row["completed"] for row in env.metrics)
    successful = all(row["completed_successfully"] for row in env.metrics)
    summary = {**prefix["identity"], "policy": metadata["policy"], "takeover_policy": policy_type,
               "evaluation_kind": metadata["evaluation_kind"], "is_end_to_end_evaluation": False,
               "prefix_steps_requested": prefix_steps, "prefix_steps_executed": prefix_executed,
               "first_action_intervention": first_action,
               "first_refinement_intervention": first_refinements,
               "rng_replay": rng_replay,
               "prefix_replay_matched": matched, "prefix_mismatches": mismatches,
               "takeover_budget": takeover_steps, "takeover_steps": takeover_executed, "steps": env.steps,
               "initial_progress": initial_progress, "prefix_final_progress": prefix_curve[-1],
               "takeover_start_progress": takeover_curve[0] if takeover_curve else None,
               "progress": _progress(env.metrics), "task_metrics": env.metrics,
               "prefix_progress_by_action": prefix_curve, "takeover_progress_by_action": takeover_curve,
               "completed": complete, "completed_successfully": successful,
               "takeover_completed_successfully": bool(matched and takeover_executed and successful),
               "truncated": env.steps >= env.max_steps and not complete,
               "activation_events": activations.events,
               "prefix_new_activations": sum(len(row["newly_activated_crystals"]) for row in activations.events if row["control"] == "external_prefix"),
               "takeover_new_activations": sum(len(row["newly_activated_crystals"]) for row in activations.events if row["control"] == "policy_takeover"),
               "prefix_elapsed_seconds": prefix_elapsed, "takeover_elapsed_seconds": takeover_elapsed,
               "costs": costs, "error": error, "elapsed_seconds": perf_counter() - start,
               "protocol": "official_step_based", "discovery_knowledge_score": None}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (output / "policy_calls.jsonl").write_text("".join(json.dumps(row) + "\n" for row in calls))
    if prime_forward and hasattr(agent, "decisions"):
        (output / "decisions.jsonl").write_text("".join(json.dumps(row) + "\n" for row in agent.decisions))
    if hasattr(agent, "update_events"):
        (output / "instance_updates.jsonl").write_text("".join(json.dumps(row) + "\n" for row in agent.update_events))
    if hasattr(agent, "trials"):
        (output / "experimental_trials.json").write_text(json.dumps(agent.trials) + "\n")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Source episode directory or trajectory.jsonl")
    parser.add_argument("--prefix-steps", type=int, required=True)
    parser.add_argument("--takeover-steps", type=int, default=100)
    parser.add_argument("--policy", choices=FACTORIES, required=True)
    parser.add_argument("--policy-config", type=Path, required=True, help="Existing factory's JSON keyword configuration")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--first-action", type=Path, help="JSON action for a controlled first-decision intervention after the prefix")
    parser.add_argument("--first-refinements", type=int, help="Override internal refinement count only at the first takeover decision")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args(argv)
    if args.prefix_steps < 0 or min(args.takeover_steps, args.threads) <= 0:
        parser.error("Prefix length must be nonnegative; takeover budget and thread count positive")
    import torch
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.random_seed)
    config = json.loads(args.policy_config.read_text())
    module, factory = args.policy.split(":")
    agent = getattr(import_module(module), factory)(config)
    policy_metadata = {"factory": args.policy, "factory_config": config, "random_seed": args.random_seed}
    if hasattr(agent, "configuration"):
        policy_metadata["effective_configuration"] = agent.configuration
    if hasattr(agent, "features"):
        policy_metadata["features"] = asdict(agent.features.config)
        policy_metadata["effective_configuration"] = {
            "mode": agent.mode, "persistent": agent.persistent, "restart_workspace": agent.restart_workspace,
            "refinements": agent.refinements if agent.refinements is not None else agent.model.config.refinements,
            "temperature": agent.temperature, "training_epoch": agent.checkpoint["epoch"]}
        digest = sha256()
        with Path(config["checkpoint"]).open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        policy_metadata["checkpoint_sha256"] = digest.hexdigest()
    try:
        summary = evaluate_prefix(agent, args.source, args.output, prefix_steps=args.prefix_steps,
                                  takeover_steps=args.takeover_steps, policy_type=FACTORIES[args.policy],
                                  policy_metadata=policy_metadata,
                                  first_action=json.loads(args.first_action.read_text()) if args.first_action else None,
                                  first_refinements=args.first_refinements)
    finally:
        if hasattr(agent, "features"):
            agent.features.close()
    print(json.dumps(summary, indent=2))
    if not summary["prefix_replay_matched"] or summary["error"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
