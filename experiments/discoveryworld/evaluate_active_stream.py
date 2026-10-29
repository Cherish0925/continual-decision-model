"""Run ordered native task instances with a continuing neural state.

Each instance has a fresh public measurement ledger and numerical posterior.
The default carries GRU state and Transformer workspace across instances.
With --continual, experience memory and per-stream adapters also persist.
The reset control clears the model state between instances, not between actions.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stdout
import fcntl
import gc
import json
import math
import multiprocessing
from pathlib import Path
from time import perf_counter

import torch

from cdm.active_experiment import load_active_checkpoint
from cdm.continual_experiment import load_continual_checkpoint
from .active_native import NativeActiveEpisode
from .adapter import DiscoveryWorldEnv, TrajectoryWriter
from .calibration_mixture import load_mixture_checkpoint
from .evaluate_active_native import _event_from_result, _public_test_mask
from .structured_observation import StructuredDiscoveryWorldEnv
from .train_active_core import make_observation, visible_beliefs


# Official Reactor Lab Normal keyDimension is randomSeed % 5. These labels
# describe the evaluator's schedule; they never enter a model observation.
NATIVE_RELATIONS = ("density", "temperature", "quantum_gap", "radiation", "spectrum_channel_5")


def observation(episode, event, mixture, step, max_steps, device, *, close_completed_task=False, belief_source=None):
    tensors = episode.tensors(device)
    if close_completed_task and bool(tensors["known"].all()) and len(episode.completed_targets) == 2:
        # The table environment's empty remaining-target set uses query zero.
        # This convention is confined to the last state update. Public records
        # and the event's actually tested object/hypothesis remain unchanged.
        tensors["query"] = torch.zeros_like(tensors["query"])
        event = event.clone()
        event[:, 32:36] = 0
        event[:, 32] = 1
    beliefs = (visible_beliefs(tensors["values"], tensors["seen"], tensors["known_hz"],
                              tensors["known"], tensors["query"], mixture)
               if belief_source is None else belief_source(tensors))
    override = {} if belief_source is None else {"beliefs_override": beliefs}
    obs = make_observation(tensors["values"], tensors["counts"], tensors["known_hz"],
                           tensors["known"], tensors["query"], tensors["failed"],
                           event, mixture, step, max_steps, **override)
    obs["action_mask"][:, 20:] &= _public_test_mask(episode, beliefs)[None]
    return obs, beliefs


def feedback_targets(decision, device, *, cost_mode="native"):
    """Adaptation feedback is separate from the inherited core's event input."""
    cost = decision["native_actions"] if cost_mode == "native" else (1 if decision["action_index"] < 20 else 5)
    return torch.tensor([[float(decision["success"]), math.log1p(cost)]], device=device)


def selected_prediction(model, result, action, previous_activity):
    """Record the prediction made before an action returns its actual outcome."""
    total = result["predictions"][0, action]
    shared = model.predictor(result["features"][0, action])
    state = result["state"]
    return {"total": total.tolist(), "shared": shared.tolist(), "adaptation": (total - shared).tolist(),
            "activity_counts": (state["activity_counts"] - previous_activity)[0].tolist(),
            "fast_adapter_norm": float(state["fast"].norm()), "slow_adapter_norm": float(state["slow"].norm())}


def consume_final_observation(model, state, obs, refinements, *, feedback=None):
    """Consume an executed outcome once without choosing or executing an action."""
    extra = {"feedback_targets": feedback} if feedback is not None else {}
    return model.advance(obs["event"], obs["evidence"], obs["evidence_mask"], state,
                         refinements=refinements, **extra)


def copy_state(state):
    return {key: value.detach().cpu().clone() for key, value in state.items()}


def state_distance(left, right):
    return {key: float((left[key].cpu().float() - right[key].cpu().float()).norm()) for key in left}


def public_object_order(episode):
    return [{"slot": slot, "public_handle": handle,
             "public_name": episode.operator.objects[handle]["name"],
             "role": "initial_support" if slot < 2 else "query_target",
             "reactor_handle": episode.reactors[slot]}
            for slot, handle in enumerate(episode.objects)]


def schedule_position(stream, index):
    """Analysis groups only; no schedule information is passed to the policy."""
    offset = 0
    for segment, length in enumerate(stream.get("segment_lengths", [])):
        if index < offset + length:
            return {"segment_index": segment, "segment_role": ("initial", "switch", "return")[segment],
                    "position_in_segment": index - offset, "first_in_segment": index == offset}
        offset += length
    return {}


def run_stream(job):
    options, stream = job
    torch.set_num_threads(options["threads"])
    device = torch.device(options["device"])
    continual = options.get("continual", False)
    loader = load_continual_checkpoint if continual else load_active_checkpoint
    model, _ = loader(options["checkpoint"], device)
    model.eval()
    if continual:
        model.retain_adaptation = not options.get("discard_adaptation", False)
    activity_generator = None
    if continual and options.get("activity_selection", "greedy") == "sample":
        activity_generator = torch.Generator(device=device).manual_seed(stream["activity_seed"])
    if model.config.mode != "persistent":
        raise ValueError("A stream comparison requires a core that persists within each instance")
    mixture, _ = load_mixture_checkpoint(options["mixture_checkpoint"], device)
    mixture.eval()
    belief_source = None
    if options.get("belief_checkpoint"):
        from experiments.physical_belief.native import PhysicalBelief
        belief_source = PhysicalBelief(options["belief_checkpoint"], device,
                                       mode=options.get("belief_mode", "carry"))
    root = Path(options["output"]) / stream["name"]
    root.mkdir(parents=True)
    state = model.initial(1, device)
    initial = copy_state(state)
    previous = None
    rows, boundaries = [], []
    started = perf_counter()
    for index, seed in enumerate(stream["seeds"]):
        if options["state_mode"] == "reset":
            state = model.initial(1, device)
        if belief_source is not None:
            belief_source.begin_episode()
        entry_memory = None if belief_source is None else belief_source.summary()
        entry_state = copy_state(state)
        output = root / f"episode{index:03d}_seed{seed}"
        output.mkdir()
        episode_start = perf_counter()
        event = torch.zeros(1, 36, device=device)
        feedback = None
        error, ending = None, "decision_budget"
        outcome_pending = False
        terminal_consumed = False
        final_state_change = None
        final_feedback_query = None
        final_feedback_object = None
        bootstrap_record = None
        step = 0
        with (output / "native.log").open("w") as log, redirect_stdout(log), torch.inference_mode():
            env_type = DiscoveryWorldEnv if options["observation_mode"] == "png" else StructuredDiscoveryWorldEnv
            env = env_type(output, options["max_actions"])
            env.reset("Reactor Lab", "Normal", seed)
            rng_root = Path(options["initial_rng_root"])
            rng_root.mkdir(parents=True, exist_ok=True)
            snapshot = rng_root / f"seed{seed}.json.gz"
            with (rng_root / f"seed{seed}.lock").open("a") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                rng = env.restore_initial_rng(snapshot) if snapshot.exists() else env.save_initial_rng(snapshot)
            (output / "initial_rng.json.gz").write_bytes(snapshot.read_bytes())
            metadata = {**env.configuration, "policy": "active_stream",
                        "policy_inputs": "public_structured_measurements_and_native_feedback",
                        "state_mode": options["state_mode"],
                        "carried_state": list(state) if options["state_mode"] == "carry" else [],
                        "continual_activities": continual,
                        "event_cost": "measurement=1,test=5",
                        "adaptation_feedback_cost": options.get("feedback_cost", "native") if continual else None,
                        "retain_adaptation": model.retain_adaptation if continual else None,
                        "activity_selection": options.get("activity_selection", "greedy") if continual else None,
                        "activity_seed": stream.get("activity_seed") if continual else None,
                        **({"prediction_target_order": ["test_success", "log1p_cost"]} if continual else {}),
                        "ledger": "fresh_for_each_instance",
                        "numerical_posterior": ("recomputed_from_current_instance_public_measurements"
                                                if belief_source is None else "public_ledger_conditioned_on_learned_cross_task_belief"),
                        **({"physical_belief_checkpoint": options["belief_checkpoint"],
                            "physical_belief_mode": options.get("belief_mode", "carry")}
                           if belief_source is not None else {}),
                        "initial_rng_snapshot": {"path": "initial_rng.json.gz", **rng},
                        "action_descriptions": env.actions, "teleport_locations": env.teleports,
                        "checkpoint": options["checkpoint"],
                        "mixture_checkpoint": options["mixture_checkpoint"],
                        "max_decisions": options["max_decisions"],
                        "failed_test_retention": "until_its_own_observed_support_or_query_changes"}
            with TrajectoryWriter(output, metadata) as writer:
                episode = NativeActiveEpisode(env, writer)
                try:
                    episode.bootstrap()
                    event[0, 32 + int(episode.query[0])] = 1
                    bootstrap_record = {
                        "measurement_samples": int(episode.counts.sum()),
                        "known_support_slots": torch.where(episode.known[0])[0].tolist(),
                        "known_support_hz": episode.known_hz[0, :2].tolist(),
                        "failed_hypotheses": int(episode.failed.sum()),
                        "public_object_order": public_object_order(episode)}
                    for step in range(options["max_decisions"]):
                        if env.ended:
                            ending = "native_termination" if all(m["completed"] for m in env.metrics) else "native_action_budget"
                            break
                        obs, beliefs = observation(episode, event, mixture, step,
                                                  options["max_decisions"], device, belief_source=belief_source)
                        extra = {"feedback_targets": feedback, "activity_generator": activity_generator} if continual else {}
                        previous_activity = state["activity_counts"].clone() if continual else None
                        result = model(**obs, state=state, refinements=options["refinements"], **extra)
                        state = result["state"]
                        outcome_pending = False
                        feedback = None
                        action = int(result["logits"][0].argmax())
                        if not obs["action_mask"][0, action]:
                            ending = "no_available_action"
                            break
                        mean = float(beliefs["means"][0, action - 20]) if action >= 20 else None
                        spread = float(beliefs["std"][0, action - 20]) if action >= 20 else None
                        prediction = selected_prediction(model, result, action, previous_activity) if continual else None
                        belief_prediction = None
                        if belief_source is not None:
                            belief_prediction = {key: beliefs[key][0].tolist()
                                                 for key in ("means", "std", "weights", "support_counts", "eligible")}
                        decision = episode.apply(action, mean_hz=mean, std_hz=spread)
                        if belief_source is not None:
                            decision["physical_prediction"] = belief_prediction
                        if continual:
                            decision["model_prediction"] = prediction
                        event = _event_from_result(decision, episode, mixture, step=step + 1,
                                                   max_steps=options["max_decisions"], device=device)
                        if continual:
                            feedback = feedback_targets(decision, device, cost_mode=options.get("feedback_cost", "native"))
                        outcome_pending = True
                    else:
                        if env.ended:
                            ending = "native_termination" if all(m["completed"] for m in env.metrics) else "native_action_budget"
                    if outcome_pending:
                        final_observation, _ = observation(episode, event, mixture, len(episode.decisions),
                                                           options["max_decisions"], device, close_completed_task=True,
                                                           belief_source=belief_source)
                        final_feedback_query = int(final_observation["event"][0, 32:36].argmax())
                        final_feedback_object = int(final_observation["event"][0, :4].argmax())
                        before_feedback = copy_state(state)
                        state = consume_final_observation(model, state, final_observation, options["refinements"],
                                                          feedback=feedback)
                        final_state_change = state_distance(copy_state(state), before_feedback)
                        terminal_consumed = True
                except (ValueError, RuntimeError) as exc:
                    error = f"{type(exc).__name__}: {exc}"
                    ending = "execution_error"
            if belief_source is not None and bootstrap_record is not None:
                belief_source.end_episode(episode.tensors(device))
            final_state = copy_state(state)
            row = {"stream": stream["name"], "episode_index": index, "seed": seed,
                   "relation": NATIVE_RELATIONS[seed % 5], "state_mode": options["state_mode"],
                   **schedule_position(stream, index),
                   "ending": ending, "error": error, "seconds": perf_counter() - episode_start,
                   "bootstrap": bootstrap_record,
                   "boundary": {"initial_state_distance": state_distance(entry_state, initial),
                                "previous_final_state_distance": None if previous is None else state_distance(entry_state, previous),
                                "final_outcome_consumed": terminal_consumed,
                                "final_feedback_query": final_feedback_query,
                                "final_feedback_object": final_feedback_object,
                                "final_outcome_state_change": final_state_change}, **episode.summary()}
            if belief_source is not None:
                row["physical_belief"] = {"mode": options.get("belief_mode", "carry"),
                                          "before": entry_memory, "after": belief_source.summary()}
            if continual:
                row["internal_activity"] = {
                    "counts": (final_state["activity_counts"] - entry_state["activity_counts"]).sum(0).tolist(),
                    "fast_adapter_norm": float(final_state["fast"].norm()),
                    "slow_adapter_norm": float(final_state["slow"].norm()),
                    "memory_entries": int(final_state["memory_valid"].sum())}
            (output / "summary.json").write_text(json.dumps(row, indent=2) + "\n")
        previous = final_state
        if options["save_boundary_states"]:
            boundaries.append({"seed": seed, "input": entry_state, "output": final_state})
        rows.append({key: value for key, value in row.items() if key != "decisions"})
        (root / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")
        print(json.dumps({key: row[key] for key in ("stream", "episode_index", "seed", "state_mode",
                                                   "completed_successfully", "native_actions", "ending")}), flush=True)
        del episode, env
        gc.collect()
        if error is not None:
            break
    if options["save_boundary_states"]:
        torch.save({"initial": initial, "episodes": boundaries}, root / "boundary_states.pt")
    return {"stream": stream["name"], "state_mode": options["state_mode"],
            "instances": len(rows), "completed": sum(row["completed_successfully"] for row in rows),
            "native_actions": sum(row["native_actions"] for row in rows),
            "seconds": perf_counter() - started, "episodes": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--manifest", type=Path, help="JSON object with streams: [{name, seeds}]")
    source.add_argument("--seeds", type=int, nargs="+", help="One ordered sequence of native instances")
    parser.add_argument("--state-mode", choices=("carry", "reset"), required=True,
                        help="Carry all model state or reset it between instances")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--mixture-checkpoint", required=True)
    parser.add_argument("--initial-rng-root", required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-actions", type=int, default=400)
    parser.add_argument("--max-decisions", type=int, default=160)
    parser.add_argument("--refinements", type=int, default=1)
    parser.add_argument("--observation-mode", choices=("structured", "png"), default="structured")
    parser.add_argument("--save-boundary-states", action="store_true")
    parser.add_argument("--continual", action="store_true",
                        help="Use the integrated activity policy and runtime adapter updates")
    parser.add_argument("--feedback-cost", choices=("native", "replay"), default="native",
                        help="Adapter target cost: actual native actions or the replay measurement=1/test=5 proxy; core events stay unchanged")
    parser.add_argument("--discard-adaptation", action="store_true",
                        help="Compute LEARN/CONSOLIDATE updates but discard their changes to fast/slow adapters")
    parser.add_argument("--activity-selection", choices=("greedy", "sample"), default="greedy",
                        help="Internal activity policy; external actions remain greedy")
    parser.add_argument("--activity-seed", type=int,
                        help="Independent activity RNG base seed; stream i uses seed+i and retains it across tasks")
    args = parser.parse_args()
    if args.discard_adaptation and not args.continual:
        parser.error("--discard-adaptation requires --continual")
    if args.activity_selection == "sample" and (not args.continual or args.activity_seed is None):
        parser.error("Activity sampling requires --continual and an explicit --activity-seed")
    if args.manifest:
        streams = json.loads(args.manifest.read_text())["streams"]
        args.manifest = str(args.manifest)
    else:
        streams = [{"name": "stream", "seeds": args.seeds}]
    if args.activity_selection == "sample":
        streams = [{**stream, "activity_seed": args.activity_seed + index} for index, stream in enumerate(streams)]
    names = [stream["name"] for stream in streams]
    if len(set(names)) != len(names) or any(Path(name).name != name for name in names):
        parser.error("Stream names must be unique directory names")
    for stream in streams:
        if "segment_lengths" in stream and (len(stream["segment_lengths"]) != 3 or
                                              sum(stream["segment_lengths"]) != len(stream["seeds"])):
            parser.error("An A-B-A stream needs three segment lengths covering its ordered seeds")
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    options = vars(args)
    (root / "configuration.json").write_text(json.dumps({**options, "streams": streams}, indent=2) + "\n")
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = [pool.submit(run_stream, (options, stream)) for stream in streams]
        for future in as_completed(futures):
            rows.append(future.result())
            (root / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
