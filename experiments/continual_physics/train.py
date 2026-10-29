"""Train continuous scientific activity at a fixed external-request budget.

Every optimizer window executes the same number of external requests per stream.
Internal computation, learning and waiting are additional measured activities.
Only returned public feedback changes the online learning buffer; exact physical
labels appear solely in the simulator transition and outer training loss.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import time

import torch
import torch.nn.functional as F

from cdm.continual_experiment import Activity, ContinualExperimentConfig, detach_state, select_rows
from cdm.continual_physics import load_continual_physics_checkpoint
from cdm.recurrent_feedback import RecurrentFeedbackConfig, configure_recurrent_feedback
from experiments.discoveryworld.calibration_mixture import load_mixture_checkpoint
from experiments.discoveryworld.train_active_lifetime import CalibrationTaskPool
from experiments.physical_belief.task_streams import MixedTaskStreamPlan
from experiments.physical_belief.train_policy import physical_observation
from .delayed import DelayedExperiments, ExactPhysicalTasks


def observe(model, environment, state, boundary=None, delayed=None):
    obs, belief, posterior = physical_observation(environment.table, model.belief,
                                                 model.belief_memory(state))
    bounds = belief["means"].new_zeros(belief["means"].shape[0])
    obs.update(physical_lower_hz=bounds, physical_upper_hz=bounds + 10000.,
               physical_minimum_step_hz=bounds,
               physical_means=belief["means"], physical_std=belief["std"],
               physical_weights=belief["weights"], physical_eligible=belief["eligible"])
    if boundary is not None:
        obs["task_boundary"] = boundary
    if delayed is not None:
        obs.update(delayed.query_exclusions())
    return obs, posterior


def physical_nll(output, target):
    mean, std = output["physical_means"].double(), output["physical_std"].double().clamp_min(.1)
    weights = output["physical_weights"].double().clamp_min(1e-20)
    density = -.5 * ((target[:, None] - mean) / std).square() - std.log() - .9189385332046727
    return -(weights.log() - weights.sum(-1, keepdim=True).log() + density).logsumexp(-1)


def receive_returns(model, delayed, state, receipt, mode, reference=None, reference_state=None):
    arrived, done = receipt["arrived"], receipt["task_done"]
    if not arrived.any():
        return state, reference_state
    state = model.feedback(state, receipt["targets"], arrived,
                           physical_feedback=receipt["physical_feedback"])
    obs, posterior = observe(model, delayed.env, state, delayed=delayed)
    state = model.receive(state, obs, arrived)
    if reference is not None:
        reference_state = reference_receive(reference, reference_state, obs, arrived)
    if done.any():
        state = model.remember_belief(state, posterior, done)
        reset = done if mode == "task_reset" else receipt["lifetime_done"]
        state = model.reset_rows(state, reset)
        delayed.env.next_tasks(done)
        delayed.reset_history(done)
        obs, _ = observe(model, delayed.env, state, done, delayed=delayed)
        state = model.receive(state, obs, done)
        if reference is not None:
            reference_state = reference.reset_rows(reference_state, done)
            reference_state = reference_receive(reference, reference_state, obs, done)
    return state, reference_state


@torch.no_grad()
def reference_receive(reference, state, observation, selected):
    rows = selected.nonzero().flatten()
    if not len(rows):
        return state
    revised = reference.advance(observation["event"][rows], observation["evidence"][rows],
        observation["evidence_mask"][rows], {k: v[rows] for k, v in state.items()})
    return {key: value.index_copy(0, rows, revised[key]) for key, value in state.items()}


def train_window(model, delayed, state, *, external_per_stream, mode="full",
                 compute_cost=.001, latency_cost=.0002, physical_weight=.05, gamma=.99,
                 activity_entropy_weight=.01, action_entropy_weight=.005,
                 reference=None, reference_state=None, anchor_weight=.02):
    batch, device = delayed.batch, delayed.device
    issued = torch.zeros(batch, dtype=torch.long, device=device)
    logps, values, rewards, ends, discounts, valid = [], [], [], [], [], []
    activity_entropies, action_entropies = [], []
    anchors = []
    physical_losses, physical_masks = [], []
    counts = torch.zeros(len(Activity), dtype=torch.long, device=device)
    crossings = torch.zeros(batch, dtype=torch.long, device=device)
    lifetime_crossings = torch.zeros_like(crossings)
    physical_updates = torch.zeros(2, dtype=torch.long, device=device)
    setting_shift_sum = 0.
    setting_requests = 0
    latency_ticks, tick = 0, 0
    before = delayed.dispatched.clone()
    while bool(((issued < external_per_stream) | delayed.pending).any()):
        active = (issued < external_per_stream) | delayed.pending
        obs, _ = observe(model, delayed.env, state, delayed=delayed)
        output = model.readout(state, obs["candidates"], obs["action_mask"])
        logits = output["activity_logits"].clone()
        logits[:, Activity.ACT] = logits[:, Activity.ACT].masked_fill(
            (issued >= external_per_stream) | delayed.pending, -torch.inf)
        logits[~active] = -torch.inf
        logits[~active, Activity.WAIT] = 0.
        activity_dist = torch.distributions.Categorical(logits=logits.float())
        activity = activity_dist.sample()
        external = (activity == Activity.ACT) & active
        action_logits = torch.where(obs["action_mask"].any(-1, keepdim=True),
                                    output["logits"], torch.zeros_like(output["logits"]))
        action_dist = torch.distributions.Categorical(logits=action_logits.float())
        action = action_dist.sample()
        rows = torch.arange(batch, device=device)
        target = delayed.env.table.targets[rows, delayed.env.table.query].clone()
        physical_losses.append(physical_nll(output, target))
        physical_masks.append(external)
        if reference is not None:
            available = torch.isfinite(output["logits"])
            with torch.no_grad():
                teacher = reference.readout(reference_state, obs["candidates"], obs["action_mask"])
                probabilities = teacher["logits"].float().masked_fill(~available, -torch.inf).softmax(-1)
            log_probability = output["logits"].float().log_softmax(-1).masked_fill(~available, 0.)
            anchors.append(-(probabilities * log_probability).sum(-1) * external)
        previous = state
        has_physical_records = state["physics_memory_valid"].any(-1)
        physical_updates[0] += ((activity == Activity.LEARN) & active & has_physical_records).sum()
        physical_updates[1] += ((activity == Activity.CONSOLIDATE) & active & has_physical_records).sum()
        state = model.internal(state, activity, obs)
        state = select_rows(previous, state, active)
        state = model.begin_action(state, output["features"], action, external, physical=output)
        predictions = {"means": output.get("physical_settings", output["physical_means"]), "std": output["physical_std"],
                       "weights": output["physical_weights"]}
        test = external & (action >= 20)
        if test.any():
            selected, quantity = rows[test], action[test] - 20
            setting_shift_sum += float((predictions["means"][selected, quantity] -
                                       output["physical_base_means"][selected, quantity]).abs().sum().detach())
            setting_requests += int(test.sum())
        delayed.dispatch(action, predictions, external)
        issued += external.long()
        pending = delayed.pending.clone()
        receipt = delayed.advance()
        crossings += receipt["task_done"].long()
        lifetime_crossings += receipt["lifetime_done"].long()
        state, reference_state = receive_returns(model, delayed, state, receipt, mode, reference, reference_state)
        internal = (activity > Activity.ACT) & (activity < Activity.WAIT) & active
        reward = receipt["reward"] - compute_cost * internal.float() - latency_cost * pending.float()
        logps.append(activity_dist.log_prob(activity) + action_dist.log_prob(action) * external)
        values.append(output["value"])
        rewards.append(reward)
        ends.append(receipt["lifetime_done"])
        discounts.append(torch.where(external, gamma, 1.))
        valid.append(active)
        activity_entropies.append(activity_dist.entropy())
        action_entropies.append(action_dist.entropy() * external)
        counts += torch.bincount(activity[active], minlength=len(Activity))
        latency_ticks += int(pending.sum())
        tick += 1
    if not bool((delayed.dispatched - before == external_per_stream).all()):
        raise RuntimeError("External-request budget differed between streams")
    with torch.no_grad():
        obs, _ = observe(model, delayed.env, state, delayed=delayed)
        target = model.readout(state, obs["candidates"], obs["action_mask"])["value"]
    returns = []
    for reward, end, discount, mask in zip(reversed(rewards), reversed(ends), reversed(discounts), reversed(valid)):
        target = torch.where(mask, reward + discount * target * ~end, target)
        returns.append(target)
    returns = torch.stack(returns[::-1])
    value, mask = torch.stack(values), torch.stack(valid).float()
    # One fixed unit of outer normalization per external request. Spending more
    # internal ticks does not dilute task return or physical prediction training.
    denominator = batch * external_per_stream
    pg = -(torch.stack(logps) * (returns - value).detach() * mask).sum() / denominator
    vl = (F.smooth_l1_loss(value, returns, reduction="none") * mask).sum() / denominator
    # Mean activity entropy has a fixed maximum independent of how long the
    # policy chooses to compute. Extra ticks earn no extra entropy bonus.
    activity_entropy = (torch.stack(activity_entropies) * mask).sum() / mask.sum()
    action_entropy = torch.stack(action_entropies).sum() / denominator
    physical = (torch.stack(physical_losses) * torch.stack(physical_masks)).sum() / denominator
    anchor = torch.stack(anchors).sum() / denominator if anchors else pg.new_zeros(())
    loss = (pg + .5 * vl - action_entropy_weight * action_entropy
            - activity_entropy_weight * activity_entropy + physical_weight * physical + anchor_weight * anchor)
    statistics = {"policy_loss": float(pg.detach()), "value_loss": float(vl.detach()),
                  "physical_nll": float(physical.detach()),
                  "reference_cross_entropy": float(anchor.detach()),
                  "activity_entropy": float(activity_entropy.detach()),
                  "action_entropy": float(action_entropy.detach()),
                  "activity_entropy_weight": activity_entropy_weight,
                  "action_entropy_weight": action_entropy_weight,
                  "mean_external_reward": float(torch.stack(rewards).sum() / denominator),
                  "external_requests": denominator, "ticks": tick,
                  "active_ticks": int(mask.sum()), "pending_ticks": latency_ticks,
                  "internal_activities": int(counts[Activity.REFINE:Activity.WAIT].sum()),
                  "activity_counts": counts.tolist(), "training_tasks_ended": len(delayed.env.task_rows),
                  "streams_crossing_task_fraction": float((crossings > 0).float().mean()),
                  "streams_crossing_lifetime_fraction": float((lifetime_crossings > 0).float().mean()),
                  "physical_learn_computations": int(physical_updates[0]),
                  "physical_consolidation_computations": int(physical_updates[1]),
                  "test_requests": setting_requests,
                  "mean_setting_shift_hz": setting_shift_sum / max(1, setting_requests),
                  "physical_fast_norm": float(state["physics_fast"].detach().flatten(1).norm(dim=-1).mean()),
                  "physical_slow_norm": float(state["physics_slow"].detach().flatten(1).norm(dim=-1).mean())}
    return loss, state, reference_state, statistics


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--belief-checkpoint", type=Path)
    p.add_argument("--mixture-checkpoint", type=Path, required=True)
    p.add_argument("--environment-truth", type=Path)
    p.add_argument("--allow-proxy", action="store_true", help="Explicitly label a pilot using public working settings as resonance")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=231)
    p.add_argument("--mode", choices=("full", "discard_physical", "task_reset", "fixed_schedule", "recurrent_reference"), default="full")
    p.add_argument("--recurrent-feedback", choices=("workspace", "evidence", "both"),
                   help="Candidate GRU context input; omitted keeps the parent architecture")
    p.add_argument("--updates", type=int, default=200)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--external-per-stream", type=int, default=32)
    p.add_argument("--max-steps", type=int, default=160)
    p.add_argument("--max-internal-steps", type=int, default=4)
    p.add_argument("--memory-size", type=int, default=64)
    p.add_argument("--max-delay", type=int, default=4)
    p.add_argument("--native-fraction", type=float, default=.5)
    p.add_argument("--compute-cost", type=float, default=.001)
    p.add_argument("--latency-cost", type=float, default=.0002)
    p.add_argument("--physical-weight", type=float, default=.05)
    p.add_argument("--anchor-weight", type=float, default=.02)
    p.add_argument("--activity-entropy-initial", type=float, default=.01)
    p.add_argument("--activity-entropy-final", type=float, default=0.)
    p.add_argument("--activity-entropy-anneal-fraction", type=float, default=.5)
    p.add_argument("--action-entropy-weight", type=float, default=.005)
    p.add_argument("--core-lr", type=float, default=3e-6)
    p.add_argument("--head-lr", type=float, default=3e-5)
    p.add_argument("--new-lr", type=float, default=1e-4)
    p.add_argument("--belief-lr", type=float, default=1e-4)
    p.add_argument("--save-every", type=int, default=50)
    args = p.parse_args()
    if args.environment_truth is None and not args.allow_proxy:
        p.error("Use --environment-truth for real activation training; --allow-proxy is for labelled pilots only")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    mixture, _ = load_mixture_checkpoint(args.mixture_checkpoint, args.device)
    mixture.requires_grad_(False)
    pool = CalibrationTaskPool(args.data / "train.npz", mixture, args.device)
    plan = MixedTaskStreamPlan(pool, source_path=args.data / "train.npz", streams=args.batch,
        lifetimes=(args.updates * args.external_per_stream + 19) // 20 + 2,
        seed=args.seed, device=args.device, native_fraction=args.native_fraction)
    environment = ExactPhysicalTasks(pool, plan, mixture, truth_path=args.environment_truth, max_steps=args.max_steps)
    delayed = DelayedExperiments(environment, seed=args.seed, max_delay=args.max_delay)
    recurrent = args.mode == "recurrent_reference"
    activity_mode = "fixed" if recurrent else ("fixed_schedule" if args.mode == "fixed_schedule" else "learned")
    if recurrent:
        args.max_internal_steps = 0
    model, _ = load_continual_physics_checkpoint(args.checkpoint, args.device,
        belief_checkpoint=args.belief_checkpoint,
        config=ContinualExperimentConfig(memory_size=args.memory_size,
            max_internal_steps=args.max_internal_steps, activity_mode=activity_mode,
            fixed_refinements=0 if recurrent else 1))
    if args.recurrent_feedback is not None:
        configure_recurrent_feedback(model, RecurrentFeedbackConfig(args.recurrent_feedback))
    model.retain_adaptation = not recurrent
    model.retain_physical_adaptation = args.mode not in {"discard_physical", "recurrent_reference"}
    reference = deepcopy(model.base).eval().requires_grad_(False)
    reference.core.gru.flatten_parameters()
    parameters = [(n, x) for n, x in model.named_parameters() if x.requires_grad]
    groups = []
    for prefix, lr in (("base.core.", args.core_lr), ("base.", args.head_lr), ("belief.", args.belief_lr), ("", args.new_lr)):
        selected = [(n, x) for n, x in parameters if n.startswith(prefix)]
        groups.append({"params": [x for _, x in selected], "lr": lr})
        names = {n for n, _ in selected}
        parameters = [(n, x) for n, x in parameters if n not in names]
    optimizer = torch.optim.AdamW(groups, weight_decay=.001)
    settings = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    settings.update(method="recurrent policy with shared physical tools" if recurrent else args.mode,
        retain_adaptation=model.retain_adaptation,
        retain_physical_adaptation=model.retain_physical_adaptation,
        task_source=pool.description(), schedule=plan.description(),
        activation="strict absolute error below 2 Hz against private environment resonance" if args.environment_truth else "proxy: strict error below 2 Hz against recorded working setting",
        feedback="actual executed frequency and binary activation; never hidden resonance or target distance",
        delay="1 to max_delay ticks, indexed independently by stream and external-request number",
        budget="equal external requests per stream and optimizer update; internal ticks recorded separately",
        state=("neural state and shared physical posterior memory persist through A-B-A; neural adapters disabled"
               if recurrent else "neural state, physical posterior memory and online adapters persist through A-B-A"),
        physical_online_learning=("disabled" if recurrent else
            "interval likelihood on received activation and non-activation feedback"),
        cost_target="log1p(1) for one executed request; queue delay is a separate latency cost",
        execution="single scalar setting per request; native operator may run multiple local probes",
        selection="fixed final update")
    (args.output / "config.json").write_text(json.dumps(settings, indent=2) + "\n")
    torch.save({**model.configuration(), "model": model.state_dict(), "training": settings, "update": 0}, args.output / "initial.pt")
    state = model.initial(args.batch, args.device)
    obs, _ = observe(model, environment, state, delayed=delayed)
    state = model.receive(state, obs)
    reference_state = reference_receive(reference, reference.initial(args.batch, args.device), obs,
        torch.ones(args.batch, dtype=torch.bool, device=args.device))
    started = time.monotonic()
    with (args.output / "training.jsonl").open("w") as log, (args.output / "tasks.jsonl").open("w") as tasks:
        for update in range(1, args.updates + 1):
            update_started = time.monotonic()
            progress = min(1., (update - 1) / max(1., args.updates * args.activity_entropy_anneal_fraction))
            activity_entropy_weight = args.activity_entropy_initial + progress * (
                args.activity_entropy_final - args.activity_entropy_initial)
            loss, state, reference_state, row = train_window(model, delayed, detach_state(state),
                external_per_stream=args.external_per_stream, mode=args.mode,
                compute_cost=args.compute_cost, latency_cost=args.latency_cost, physical_weight=args.physical_weight,
                activity_entropy_weight=activity_entropy_weight, action_entropy_weight=args.action_entropy_weight,
                reference=reference, reference_state=reference_state, anchor_weight=args.anchor_weight)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_([x for group in groups for x in group["params"]], 1.)
            if not torch.isfinite(grad):
                raise FloatingPointError(f"Nonfinite gradient at update {update}")
            optimizer.step()
            row.update(update=update, loss=float(loss.detach()), gradient_norm=float(grad),
                       seconds=time.monotonic() - started, update_seconds=time.monotonic() - update_started,
                       cumulative_external_requests=int(delayed.dispatched.sum()))
            if torch.device(args.device).type == "cuda":
                row["peak_allocated_gib"] = torch.cuda.max_memory_allocated(args.device) / 2**30
            log.write(json.dumps(row) + "\n"); log.flush()
            for task in environment.task_rows:
                tasks.write(json.dumps(task) + "\n")
            tasks.flush(); environment.task_rows.clear()
            if update == 1 or update % 10 == 0:
                print(json.dumps(row), flush=True)
            if update % args.save_every == 0 or update == args.updates:
                torch.save({**model.configuration(), "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                            "training": settings, "update": update}, args.output / "last.pt")
    (args.output / "finished.json").write_text(json.dumps({"updates": args.updates,
        "external_requests": int(delayed.dispatched.sum()), "external_requests_per_stream": delayed.dispatched.tolist(),
        "returned_requests": int(delayed.returned.sum()), "ticks": delayed.tick,
        "seconds": time.monotonic() - started}, indent=2) + "\n")


if __name__ == "__main__":
    main()
