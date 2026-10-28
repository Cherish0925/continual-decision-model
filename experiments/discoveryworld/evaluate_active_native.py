"""Evaluate active measurement policies and controls from native task resets."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stdout
import json
import fcntl
import math
import multiprocessing
from pathlib import Path
from time import perf_counter

import torch

from .active_native import NativeActiveEpisode
from .adapter import DiscoveryWorldEnv, TrajectoryWriter
from .calibration import FEATURE_QUANTITIES
from .calibration_mixture import load_mixture_checkpoint
from .experimental_plans import PlanValue
from .structured_observation import StructuredDiscoveryWorldEnv


def _full_measurement_actions(radiation_samples):
    for obj in range(4):
        for group in (0, 1, 3, 4, 2):
            yield from [obj * 5 + group] * (radiation_samples if group == 2 else 1)


def _plan_scores(plan, episode, beliefs, mixture):
    means, std, probs = (beliefs[key][0] for key in ("means", "std", "weights"))
    if plan is None:
        return probs
    query = int(episode.query[0])
    current = episode.current_hz[0, query].to(means.device)
    scale, center = mixture.y_scale, mixture.y_mean
    entropy = -(probs * probs.clamp_min(1e-30).log()).sum() / math.log(9)
    support = beliefs["support_counts"][0].to(means.dtype)
    features = torch.stack(((means - center) / scale, std / scale, probs,
                            (means - current) / scale, (means - current).abs() / scale,
                            torch.full_like(means, (current - center) / scale),
                            torch.full_like(means, entropy), support / 4,
                            torch.full_like(means, float(episode.failed.sum()) / 9),
                            torch.full_like(means, episode.controls[query]["minimum_step_hz"] / scale)), -1)
    state = torch.zeros(1, plan.state_dim, device=means.device)
    result = plan(state, features.float().unsqueeze(0))[0]
    return result[:, 0].sigmoid() - .001 * result[:, 1].clamp_min(0) * 64


def _public_test_mask(episode, beliefs):
    query = int(episode.query[0])
    control = episode.controls[query]
    means = beliefs["means"][0]
    return (beliefs["eligible"][0] & ~episode.failed[0].to(means.device)
            & (means >= control["lower_hz"]) & (means <= control["upper_hz"]))


def _event_from_result(decision, episode, mixture, *, step, max_steps, device):
    from .train_active_core import measurement_event, test_event
    query = torch.tensor([decision["query_index"]], device=device)
    action = decision["action_index"]
    if action >= 20:
        event = test_event(query, torch.tensor([action - 20], device=device),
                           torch.tensor([decision["success"]], device=device), step, max_steps, device,
                           cost=5.)
        event[:, 32:36] = 0
        event[0, 32 + int(episode.query[0])] = 1
        return event
    obj, group = divmod(action, 5)
    returned = torch.zeros(1, 9, device=device)
    present = torch.zeros(1, 9, device=device, dtype=torch.bool)
    for row in decision["result"]["measurements"]:
        if (row["kind"] == "instrument" and row["target_handle"] == episode.objects[obj]
                and row["quantity"] in FEATURE_QUANTITIES):
            quantity = FEATURE_QUANTITIES.index(row["quantity"])
            returned[0, quantity] = row["value"]
            present[0, quantity] = True
    return measurement_event(torch.tensor([obj], device=device), torch.tensor([group], device=device),
                             returned, present, query, step, max_steps, mixture,
                             cost=1.)


def run(job):
    from .train_active_core import make_observation, visible_beliefs
    from cdm.active_experiment import load_active_checkpoint
    options, seed = job
    torch.set_num_threads(options["threads"])
    output = Path(options["output"]) / f"seed{seed}"
    output.mkdir(parents=True)
    start = perf_counter()
    device = torch.device(options["device"])
    mixture, _ = load_mixture_checkpoint(options["mixture_checkpoint"], device)
    mixture.eval()
    model = state = plan = None
    if options["strategy"] == "learned":
        model, saved = load_active_checkpoint(options["checkpoint"], device)
        model.eval()
        state = model.initial(1, device)
    elif options["strategy"] == "full_plan":
        saved = torch.load(options["plan_checkpoint"], map_location="cpu", weights_only=True)
        plan = PlanValue(**saved["config"]).to(device).eval()
        plan.load_state_dict(saved["model"])
        if plan.use_state:
            raise ValueError("The same-input baseline requires the numerical plan readout")
    event = torch.zeros(1, 36, device=device)
    with (output / "native.log").open("w") as log, redirect_stdout(log), torch.inference_mode():
        env_type = DiscoveryWorldEnv if options["observation_mode"] == "png" else StructuredDiscoveryWorldEnv
        env = env_type(output, options["max_actions"])
        env.reset("Reactor Lab", "Normal", seed)
        rng_root = Path(options["initial_rng_root"])
        rng_root.mkdir(parents=True, exist_ok=True)
        snapshot = rng_root / f"seed{seed}.json.gz"
        with (rng_root / f"seed{seed}.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if snapshot.exists():
                rng = env.restore_initial_rng(snapshot)
            else:
                rng = env.save_initial_rng(snapshot)
        # Each arm also carries its own copy so the public trajectory is portable.
        (output / "initial_rng.json.gz").write_bytes(snapshot.read_bytes())
        metadata = {**env.configuration, "policy": options["strategy"],
                    "policy_inputs": "public_structured_measurements_and_native_feedback",
                    "initial_rng_snapshot": {"path": "initial_rng.json.gz", **rng},
                    "action_descriptions": env.actions, "teleport_locations": env.teleports,
                    "checkpoint": options.get("checkpoint"),
                    "plan_checkpoint": options.get("plan_checkpoint"),
                    "mixture_checkpoint": options["mixture_checkpoint"],
                    "max_decisions": options["max_decisions"],
                    "failed_test_retention": "until_its_own_observed_support_or_query_changes",
                    "reset_state_each_decision": options["reset_state"]}
        error, ending = None, "decision_budget"
        with TrajectoryWriter(output, metadata) as writer:
            episode = NativeActiveEpisode(env, writer)
            try:
                episode.bootstrap()
                event[0, 32 + int(episode.query[0])] = 1
                scheduled = iter(_full_measurement_actions(options["radiation_samples"]))
                for step in range(options["max_decisions"]):
                    if env.ended:
                        ending = "native_termination" if all(m["completed"] for m in env.metrics) else "native_action_budget"
                        break
                    tensors = episode.tensors(device)
                    beliefs = visible_beliefs(tensors["values"], tensors["seen"], tensors["known_hz"],
                                             tensors["known"], tensors["query"], mixture)
                    mean = spread = None
                    if model is not None:
                        observation = make_observation(tensors["values"], tensors["counts"],
                                                       tensors["known_hz"], tensors["known"], tensors["query"],
                                                       tensors["failed"], event, mixture,
                                                       step, options["max_decisions"])
                        observation["action_mask"][:, 20:] &= _public_test_mask(episode, beliefs)[None]
                        if options["reset_state"]:
                            state = model.initial(1, device)
                        result = model(**observation, state=state, refinements=options["refinements"])
                        state = result["state"]
                        action = int(result["logits"][0].argmax())
                        if not observation["action_mask"][0, action]:
                            ending = "no_available_action"
                            break
                    else:
                        action = next(scheduled, None)
                        if action is None:
                            valid = _public_test_mask(episode, beliefs)
                            if not bool(valid.any()):
                                ending = "available_hypotheses_exhausted"
                                break
                            scores = _plan_scores(plan, episode, beliefs, mixture).masked_fill(~valid, -torch.inf)
                            action = 20 + int(scores.argmax())
                    if action >= 20:
                        mean = float(beliefs["means"][0, action - 20])
                        spread = float(beliefs["std"][0, action - 20])
                    decision = episode.apply(action, mean_hz=mean, std_hz=spread)
                    if model is not None:
                        event = _event_from_result(decision, episode, mixture, step=step + 1,
                                                  max_steps=options["max_decisions"], device=device)
                else:
                    if env.ended:
                        ending = "native_termination" if all(m["completed"] for m in env.metrics) else "native_action_budget"
            except (ValueError, RuntimeError) as exc:
                error = f"{type(exc).__name__}: {exc}"
                ending = "execution_error"
        result = {"seed": seed, "strategy": options["strategy"], "ending": ending,
                  "rule_split": "heldout_rule" if seed % 5 == 4 else "new_instance",
                  "error": error, "seconds": perf_counter() - start, **episode.summary()}
        (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return {key: value for key, value in result.items() if key != "decisions"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    seeds = parser.add_mutually_exclusive_group(required=True)
    seeds.add_argument("--seeds", type=int, nargs="+")
    seeds.add_argument("--manifest", type=Path,
                       help="A JSON list of seeds, or an object with a seeds list")
    parser.add_argument("--strategy", choices=("learned", "full_posterior", "full_plan"), required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--mixture-checkpoint", required=True)
    parser.add_argument("--plan-checkpoint")
    parser.add_argument("--initial-rng-root", required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-actions", type=int, default=400)
    parser.add_argument("--max-decisions", type=int, default=160)
    parser.add_argument("--radiation-samples", type=int, default=32)
    parser.add_argument("--refinements", type=int, default=1)
    parser.add_argument("--reset-state", action="store_true")
    parser.add_argument("--observation-mode", choices=("structured", "png"), default="structured")
    args = parser.parse_args()
    if args.manifest is not None:
        manifest = json.loads(args.manifest.read_text())
        args.seeds = manifest if isinstance(manifest, list) else manifest["seeds"]
        args.manifest = str(args.manifest)
    if args.strategy == "learned" and not args.checkpoint:
        parser.error("A learned policy needs --checkpoint")
    if args.strategy == "full_plan" and not args.plan_checkpoint:
        parser.error("The learned numerical control needs --plan-checkpoint")
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    rows, options = [], vars(args)
    (root / "configuration.json").write_text(json.dumps(options, indent=2) + "\n")
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = [pool.submit(run, (options, seed)) for seed in args.seeds]
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            (root / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")
            print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
