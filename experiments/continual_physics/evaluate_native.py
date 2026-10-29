"""Execute a continual physical model on new ordered native task streams."""
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

from experiments.discoveryworld.active_native import NativeActiveEpisode
from experiments.discoveryworld.adapter import TrajectoryWriter
from experiments.discoveryworld.evaluate_active_native import _event_from_result
from experiments.discoveryworld.evaluate_active_stream import (
    NATIVE_RELATIONS, copy_state, feedback_targets, public_object_order,
    schedule_position,
)
from experiments.discoveryworld.structured_observation import StructuredDiscoveryWorldEnv
from experiments.discoveryworld.train_active_core import make_observation
from experiments.physical_belief.observations import add_physical_context
from .runtime import ContinualPhysicsSession, physical_feedback, setting_outcomes


def state_distance(left, right):
    """Record tensor changes, including an explicitly resized evidence history."""
    result = {}
    for key, value in left.items():
        previous = right[key]
        if value.shape == previous.shape:
            same = value == previous
            difference = torch.where(same, 0., value.float()-previous.float())
            finite = difference.isfinite()
            result[key] = (float(difference.norm()) if finite.all() else
                           {"finite_difference_norm": float(difference[finite].norm()),
                            "changed_nonfinite_entries": int((~finite).sum())})
        else:
            result[key] = {"before_shape": list(previous.shape), "after_shape": list(value.shape),
                           "before_norm": float(previous.float().norm()), "after_norm": float(value.float().norm())}
    return result


def native_observation(model, state, episode, event, step, max_steps, device, *, terminal=False,
                       task_boundary=False, tolerance_hz=2.):
    tensors = episode.tensors(device)
    if terminal and bool(tensors["known"].all()) and len(episode.completed_targets) == 2:
        tensors["query"] = torch.zeros_like(tensors["query"])
        event = event.clone()
        event[:, 32:36] = 0
        event[:, 32] = 1
    beliefs, posterior = model.belief.observe_predict(
        tensors["values"], tensors["seen"], tensors["known_hz"], tensors["known"],
        tensors["query"], model.belief_memory(state))
    obs = make_observation(tensors["values"], tensors["counts"], tensors["known_hz"],
                           tensors["known"], tensors["query"], torch.zeros_like(tensors["failed"]), event,
                           model.belief, step, max_steps, beliefs_override=beliefs)
    obs = add_physical_context(obs, beliefs, model.belief)
    obs.update({f"physical_{name}": beliefs[name] for name in ("means", "std", "weights", "eligible")})
    control = episode.controls[int(episode.query[0])]
    obs["physical_lower_hz"] = torch.tensor([control["lower_hz"]], device=device)
    obs["physical_upper_hz"] = torch.tensor([control["upper_hz"]], device=device)
    obs["physical_current_hz"] = tensors["current_hz"][:, int(episode.query[0])]
    obs["physical_minimum_step_hz"] = torch.tensor([control["minimum_step_hz"]], device=device)
    obs["task_boundary"] = torch.tensor([task_boundary], device=device)
    failed = sorted({value for decision in episode.decisions
                     if decision["query_index"] == int(tensors["query"][0])
                     for value, activated in setting_outcomes(decision) if not activated})
    obs["physical_exclusions"] = torch.tensor(
        [[[value-tolerance_hz, value+tolerance_hz] for value in failed]],
        device=device, dtype=torch.float64).reshape(1, -1, 2)
    obs["physical_exclusion_mask"] = torch.ones(1, len(failed), device=device, dtype=torch.bool)
    return obs, posterior


def next_request(session):
    """Activity decisions have their own budget; one return means one tool call."""
    activities = []
    for _ in range(session.model.continual_config.max_internal_steps + 1):
        result = session.tick()
        activities.append(result["activity"])
        if "action" in result:
            return result, activities
        if result["activity"] == "WAIT":
            return None, activities
    raise RuntimeError("The internal activity budget ended without ACT or WAIT")


def run_stream(job):
    from cdm.continual_physics import load_continual_physics_checkpoint

    options, stream = job
    torch.set_num_threads(options["threads"])
    device = torch.device(options["device"])
    model, _ = load_continual_physics_checkpoint(options["checkpoint"], device)
    model.eval()
    if options["discard_physical_updates"]:
        model.retain_physical_adaptation = False
    session = ContinualPhysicsSession(model, device)
    initial = copy_state(session.state)
    root = Path(options["output"]) / stream["name"]
    root.mkdir(parents=True)
    rows = []
    previous = None
    started = perf_counter()
    for index, seed in enumerate(stream["seeds"]):
        if options["state_mode"] == "reset":
            session.state = model.initial(1, device)
        entry_state = copy_state(session.state)
        start_deliveries, start_ticks = session.delivered_results, session.pending_ticks
        output = root / f"episode{index:03d}_seed{seed}"
        output.mkdir()
        start = perf_counter()
        error, ending, bootstrap = None, "decision_budget", None
        with (output / "native.log").open("w") as log, redirect_stdout(log), torch.no_grad():
            env = StructuredDiscoveryWorldEnv(output, options["max_actions"])
            env.reset("Reactor Lab", "Normal", seed)
            rng_root = Path(options["initial_rng_root"])
            rng_root.mkdir(parents=True, exist_ok=True)
            snapshot = rng_root / f"seed{seed}.json.gz"
            with (rng_root / f"seed{seed}.lock").open("a") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                rng = env.restore_initial_rng(snapshot) if snapshot.exists() else env.save_initial_rng(snapshot)
            (output / "initial_rng.json.gz").write_bytes(snapshot.read_bytes())
            metadata = {**env.configuration, "policy": "continual_physics",
                        "checkpoint": options["checkpoint"],
                        "policy_inputs": "public_structured_measurements_and_actual_setting_feedback",
                        "state_mode": options["state_mode"],
                        "retain_physical_updates": model.retain_physical_adaptation,
                        "retain_physical_memory": not options["discard_physical_memory"],
                        "feedback_delay_ticks": options["feedback_delay_ticks"],
                        "delay_meaning": "delivery delay; native world does not advance during internal ticks",
                        "activation_tolerance_hz": options["tolerance_hz"],
                        "initial_rng_snapshot": {"path": "initial_rng.json.gz", **rng},
                        "action_descriptions": env.actions, "teleport_locations": env.teleports,
                        "max_decisions": options["max_decisions"]}
            with TrajectoryWriter(output, metadata) as writer:
                episode = NativeActiveEpisode(env, writer)
                try:
                    episode.bootstrap()
                    bootstrap_seconds = perf_counter()-start
                    bootstrap = {"known_support_hz": episode.known_hz[0, :2].tolist(),
                                 "public_object_order": public_object_order(episode),
                                 "measurement_samples": int(episode.counts.sum())}
                    event = torch.zeros(1, 36, device=device)
                    event[0, 32 + int(episode.query[0])] = 1
                    obs, posterior = native_observation(model, session.state, episode, event, 0,
                                                        options["max_decisions"], device, task_boundary=True,
                                                        tolerance_hz=options["tolerance_hz"])
                    session.receive(session.last_event + 1, obs)
                    for step in range(options["max_decisions"]):
                        if env.ended:
                            ending = "native_termination" if all(m["completed"] for m in env.metrics) else "native_action_budget"
                            break
                        inference_start = perf_counter()
                        request, activities = next_request(session)
                        inference_seconds = perf_counter() - inference_start
                        if request is None:
                            ending = "no_available_action"
                            break
                        action, prediction = request["action"], request["prediction"]
                        if not session.observation["action_mask"][0, action]:
                            raise RuntimeError("Selected action is unavailable in the delivered observation")
                        mean = float(prediction["physical_settings"][0, action - 20]) if action >= 20 else None
                        spread = float(prediction["physical_std"][0, action - 20]) if action >= 20 else None
                        execution_start = perf_counter()
                        decision = episode.apply(action, mean_hz=mean, std_hz=spread)
                        decision["execution_seconds"] = perf_counter() - execution_start
                        decision["decision_seconds"] = inference_seconds
                        decision["internal_activities"] = activities
                        decision["physical_prediction"] = {
                            name: prediction[f"physical_{name}"][0].tolist()
                            for name in ("means", "std", "weights")}
                        decision["physical_settings"] = prediction["physical_settings"][0].tolist()
                        decision["bayes_physical_prediction"] = {
                            name: session.observation[f"physical_{name}"][0].tolist()
                            for name in ("means", "std", "weights")}
                        feedback_start = perf_counter()
                        event = _event_from_result(decision, episode, model.belief, step=step+1,
                                                   max_steps=options["max_decisions"], device=device)
                        obs, posterior = native_observation(model, session.state, episode, event, step+1,
                                                            options["max_decisions"], device, terminal=env.ended,
                                                            tolerance_hz=options["tolerance_hz"])
                        returned = physical_feedback(decision, device, tolerance_hz=options["tolerance_hz"])
                        decision["physical_feedback"] = {name: value[0].tolist() for name, value in returned.items()}
                        session.queue_result(session.last_event + 1, request["request_id"], obs,
                                             feedback_targets(decision, device), returned,
                                             delay_ticks=options["feedback_delay_ticks"])
                        before = copy_state(session.state)
                        waiting = []
                        while session.delivery is not None:
                            delivery = session.poll_result()
                            if not delivery["delivered"]:
                                waiting.append(delivery["activity"])
                        decision["pending_activities"] = waiting
                        decision["result_state_change"] = state_distance(copy_state(session.state), before)
                        decision["feedback_seconds"] = perf_counter() - feedback_start
                    else:
                        if env.ended:
                            ending = "native_termination" if all(m["completed"] for m in env.metrics) else "native_action_budget"
                    session.state = model.remember_belief(session.state, posterior,
                                                          retain=not options["discard_physical_memory"])
                except (ValueError, RuntimeError) as exc:
                    error, ending = f"{type(exc).__name__}: {exc}", "execution_error"
            final = copy_state(session.state)
            tested = [decision for decision in episode.decisions if decision["action_index"] >= 20
                      and "physical_prediction" in decision]
            corrections = [abs(decision["physical_prediction"]["means"][decision["action_index"]-20]
                               - decision["bayes_physical_prediction"]["means"][decision["action_index"]-20])
                           for decision in tested]
            row = {"stream": stream["name"], "episode_index": index, "seed": seed,
                   "relation": NATIVE_RELATIONS[seed % 5], **schedule_position(stream, index),
                   "ending": ending, "error": error, "seconds": perf_counter()-start,
                   "bootstrap": bootstrap, **episode.summary(),
                   "boundary": {"initial_state_distance": state_distance(entry_state, initial),
                                "previous_final_state_distance": None if previous is None else state_distance(entry_state, previous)},
                   "runtime": {"delivered_results": session.delivered_results-start_deliveries,
                               "pending_internal_ticks": session.pending_ticks-start_ticks,
                               "pending_request_at_end": session.pending_request},
                   "timing": {"initialization_and_bootstrap_seconds": bootstrap_seconds if bootstrap else None,
                              **{name: sum(decision.get(source, 0.) for decision in episode.decisions)
                                 for name, source in (("inference_seconds", "decision_seconds"),
                                                      ("execution_seconds", "execution_seconds"),
                                                      ("feedback_seconds", "feedback_seconds"))}},
                   "physical_learning": {"setting_decisions": len(tested),
                                         "sum_absolute_mean_correction_hz": sum(corrections),
                                         "max_absolute_mean_correction_hz": max(corrections, default=0.),
                                         "fast_correction_norm": float(final["physics_fast"].norm()),
                                         "slow_correction_norm": float(final["physics_slow"].norm()),
                                         "received_setting_events": int((final["physics_memory_cursor"]-entry_state["physics_memory_cursor"]).sum()),
                                         "retained_physical_memories": int(final["belief_valid"].sum())},
                   "internal_activity_counts": (final["activity_counts"]-entry_state["activity_counts"])[0].tolist()}
            (output / "summary.json").write_text(json.dumps(row, indent=2)+"\n")
        rows.append({key: value for key, value in row.items() if key != "decisions"})
        (root / "summary.json").write_text(json.dumps(rows, indent=2)+"\n")
        print(json.dumps({key: row[key] for key in ("stream", "seed", "completed_successfully", "native_actions", "ending")}), flush=True)
        previous = final
        del episode, env
        gc.collect()
        if error is not None:
            break
    return {"stream": stream["name"], "instances": len(rows),
            "completed": sum(row["completed_successfully"] for row in rows),
            "native_actions": sum(row["native_actions"] for row in rows),
            "seconds": perf_counter()-started, "episodes": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--initial-rng-root", required=True)
    parser.add_argument("--state-mode", choices=("carry", "reset"), default="carry")
    parser.add_argument("--discard-physical-updates", action="store_true")
    parser.add_argument("--discard-physical-memory", action="store_true")
    parser.add_argument("--feedback-delay-ticks", type=int, default=0)
    parser.add_argument("--tolerance-hz", type=float, default=2.)
    parser.add_argument("--max-actions", type=int, default=400)
    parser.add_argument("--max-decisions", type=int, default=160)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    streams = json.loads(args.manifest.read_text())["streams"]
    options = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "configuration.json").write_text(json.dumps({**options, "streams": streams}, indent=2)+"\n")
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = [pool.submit(run_stream, (options, stream)) for stream in streams]
        for future in as_completed(futures):
            rows.append(future.result())
            (args.output / "summary.json").write_text(json.dumps(rows, indent=2)+"\n")
    if any(episode["error"] for stream in rows for episode in stream["episodes"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
