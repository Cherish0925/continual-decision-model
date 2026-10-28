"""Jointly train actions, internal activity and replay updates on v0.9 task streams.

The recorded simulator supplies proxy execution costs. Native action counts are
measured separately by evaluate_active_stream --continual; they are not inferred
from these training costs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import torch
import torch.nn.functional as F

from cdm.continual_experiment import (Activity, ContinualExperimentConfig,
                                    detach_state, load_continual_checkpoint)
from experiments.discoveryworld.calibration_mixture import load_mixture_checkpoint
from experiments.discoveryworld.train_active_lifetime import (CalibrationTaskPool, TaskStreamPlan,
                                                               ActiveLifetimeEnvironment)


def step_selected(env, actions, selected):
    """Only externally acting streams advance the recorded environment."""
    alive = env.table.alive.clone()
    env.table.alive &= selected
    reward, task_done, lifetime_done = env.step(actions)
    env.table.alive = torch.where(selected, env.table.alive, alive)
    return reward, task_done & selected, lifetime_done & selected


def activity_distribution(model, out, state, mode, fixed_refinements):
    logits = out["activity_logits"].clone()
    if mode == "no_adaptation":
        logits[:, Activity.LEARN:Activity.CONSOLIDATE + 1] = -torch.inf
    if mode == "fixed":
        choice = torch.where(state["internal_steps"] < fixed_refinements,
                             int(Activity.REFINE), int(Activity.ACT))
        logits = torch.full_like(logits, -torch.inf).scatter(1, choice[:, None], 0.)
    return torch.distributions.Categorical(logits=logits.float())


def train_window(model, env, state, *, unroll, mode="full", fixed_refinements=1,
                 compute_cost=.001, gamma=.99):
    log_probs, values, rewards, ends, entropies, auxiliary = [], [], [], [], [], []
    counts = torch.zeros(len(Activity), dtype=torch.long, device=state["h"].device)
    for _ in range(unroll):
        obs = env.observation()
        out = model.readout(state, obs["candidates"], obs["action_mask"])
        activity_dist = activity_distribution(model, out, state, mode, fixed_refinements)
        activity = activity_dist.sample()
        counts += torch.bincount(activity, minlength=len(Activity))
        external = activity == Activity.ACT
        # A row with no executable action may wait; give it a finite sampling distribution.
        action_logits = torch.where(obs["action_mask"].any(-1, keepdim=True), out["logits"],
                                    torch.zeros_like(out["logits"]))
        action_dist = torch.distributions.Categorical(logits=action_logits.float())
        actions = action_dist.sample()
        state = model.internal(state, activity, obs)
        state = model.begin_action(state, out["features"], actions, external)
        reward, task_done, lifetime_done = step_selected(env, actions, external)
        returned = env.observation()
        event = returned["event"]
        targets = torch.stack((event[:, 29], (event[:, 30] * 20).clamp_min(0).log1p()), -1)
        chosen_prediction = out["predictions"][torch.arange(len(actions), device=actions.device), actions]
        auxiliary.append((F.smooth_l1_loss(chosen_prediction, targets, reduction="none").mean(-1)
                          * external).sum() / external.sum().clamp_min(1))
        state = model.feedback(state, targets, external)
        # Includes terminal feedback, before a task's public ledger is cleared.
        state = model.receive(state, returned, external)
        state = model.reset_rows(state, task_done if mode == "task_reset" else lifetime_done)
        env.next_tasks(task_done)
        state = model.receive(state, env.observation(), task_done)
        internal = (activity > Activity.ACT) & (activity < Activity.WAIT)
        rewards.append(reward - compute_cost * internal.float())
        log_probs.append(activity_dist.log_prob(activity) + action_dist.log_prob(actions) * external)
        entropies.append(activity_dist.entropy() + action_dist.entropy() * external)
        values.append(out["value"])
        ends.append(lifetime_done)
    with torch.no_grad():
        obs = env.observation()
        target = model.readout(state, obs["candidates"], obs["action_mask"])["value"]
    returns = []
    for reward, end in zip(reversed(rewards), reversed(ends)):
        target = reward + gamma * target * ~end
        returns.append(target)
    returns = torch.stack(returns[::-1])
    value = torch.stack(values)
    policy_loss = -(torch.stack(log_probs) * (returns - value).detach()).mean()
    value_loss = F.smooth_l1_loss(value, returns)
    prediction_loss = torch.stack(auxiliary).mean()
    entropy = torch.stack(entropies).mean()
    loss = policy_loss + .5 * value_loss + .1 * prediction_loss - .01 * entropy
    return loss, state, {"policy_loss": float(policy_loss.detach()), "value_loss": float(value_loss.detach()),
                         "prediction_loss": float(prediction_loss.detach()), "entropy": float(entropy.detach()),
                         "mean_tick_reward": float(torch.stack(rewards).mean()),
                         "activity_counts_this_update": counts.tolist()}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True, help="v0.9 data directory containing train.npz")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--mixture-checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=201)
    p.add_argument("--updates", type=int, default=1200)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--unroll", type=int, default=32)
    p.add_argument("--max-steps", type=int, default=160)
    p.add_argument("--max-internal-steps", type=int, default=4)
    p.add_argument("--mode", choices=("full", "fixed", "no_adaptation", "task_reset"), default="full")
    p.add_argument("--fixed-refinements", type=int, default=1)
    p.add_argument("--compute-cost", type=float, default=.001)
    p.add_argument("--core-lr", type=float, default=3e-5)
    p.add_argument("--new-lr", type=float, default=3e-4)
    p.add_argument("--save-every", type=int, default=100)
    args = p.parse_args()
    if min(args.batch, args.updates, args.unroll, args.max_steps, args.save_every) <= 0:
        p.error("Batch, updates, unroll, max-steps and save-every must be positive")
    if not 0 <= args.fixed_refinements <= args.max_internal_steps:
        p.error("fixed-refinements must fit within max-internal-steps")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    mixture, _ = load_mixture_checkpoint(args.mixture_checkpoint, args.device)
    mixture.requires_grad_(False)
    pool = CalibrationTaskPool(args.data / "train.npz", mixture, args.device)
    plan = TaskStreamPlan(pool, streams=args.batch, lifetimes=(args.updates * args.unroll + 19) // 20 + 2,
                          seed=args.seed, device=args.device)
    env = ActiveLifetimeEnvironment(pool, plan, mixture, max_steps=args.max_steps)
    model, _ = load_continual_checkpoint(args.checkpoint, args.device,
                    ContinualExperimentConfig(max_internal_steps=args.max_internal_steps,
                        activity_mode=args.mode if args.mode in {"fixed", "no_adaptation"} else "learned",
                        fixed_refinements=args.fixed_refinements))
    if model.continual_config.max_internal_steps != args.max_internal_steps:
        p.error("max-internal-steps must match a resumed continual checkpoint")
    expected_mode = args.mode if args.mode in {"fixed", "no_adaptation"} else "learned"
    if model.continual_config.activity_mode != expected_mode:
        p.error("Training mode must match the loaded continual checkpoint")
    parameters = [(name, x) for name, x in model.named_parameters() if x.requires_grad]
    optimizer = torch.optim.AdamW([
        {"params": [x for name, x in parameters if name.startswith("base.")], "lr": args.core_lr},
        {"params": [x for name, x in parameters if not name.startswith("base.")], "lr": args.new_lr}], weight_decay=.01)
    settings = {key: str(x) if isinstance(x, Path) else x for key, x in vars(args).items()}
    settings.update(training_pool=pool.description(), stream_plan=plan.description(),
                    feedback="received test success and log1p of replay proxy execution cost",
                    online_update="per-stream fast and slow prediction adapters; shared weights fixed at deployment",
                    modalities="structured observations only in this dataset",
                    return_boundary="lifetime; includes later tasks and internal activity costs",
                    initialization="model weights loaded; optimizer starts fresh",
                    checkpoint_selection="fixed final update; no test-set selection")
    (args.output / "config.json").write_text(json.dumps(settings, indent=2) + "\n")
    state = model.receive(model.initial(args.batch, args.device), env.observation())
    started = time.monotonic()
    with (args.output / "training.jsonl").open("w") as log, (args.output / "tasks.jsonl").open("w") as tasks:
        for update in range(1, args.updates + 1):
            model.train()
            loss, state, row = train_window(model, env, detach_state(state), unroll=args.unroll,
                                           mode=args.mode, fixed_refinements=args.fixed_refinements,
                                           compute_cost=args.compute_cost)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_([x for _, x in parameters], 1.)
            optimizer.step()
            row.update(update=update, loss=float(loss.detach()), gradient_norm=float(norm),
                       seconds=time.monotonic() - started)
            log.write(json.dumps(row) + "\n"); log.flush()
            for completed in env.task_rows:
                tasks.write(json.dumps(completed) + "\n")
            env.task_rows.clear()
            tasks.flush()
            if update == 1 or update % 10 == 0:
                print(json.dumps(row), flush=True)
            if update % args.save_every == 0 or update == args.updates:
                torch.save({**model.configuration(), "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                            "training": settings, "update": update}, args.output / "last.pt")


if __name__ == "__main__":
    main()
