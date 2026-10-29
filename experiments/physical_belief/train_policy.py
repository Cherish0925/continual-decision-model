"""Train the continuing action core with the physical predictions it executes.

The world holds unrevealed measurements and target settings. The belief sees
only the public ledger; later target labels supervise pre-action predictions.
Every model executes its own predicted setting, rather than receiving rewards
for an unrelated frozen numerical fit. Native evaluation is separate.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import torch
import torch.nn.functional as F

from cdm.active_experiment import load_active_checkpoint
from cdm.experimental_belief import load_experimental_belief_checkpoint, save_experimental_belief_checkpoint
from experiments.discoveryworld.calibration_mixture import load_mixture_checkpoint
from experiments.discoveryworld.train_active_core import make_observation, visible_beliefs
from experiments.discoveryworld.train_active_lifetime import (
    ActiveLifetimeEnvironment, CalibrationTaskPool, TaskStreamPlan, flush_terminal_state,
)
from .observations import add_physical_context
from .task_streams import MixedTaskStreamPlan, PhysicalTaskEnvironment


def detach(state):
    return {key: value.detach() for key, value in state.items()}


def replace_rows(old, new, rows):
    return {key: value.index_copy(0, rows, new[key]) for key, value in old.items()}


def physical_observation(table, belief_model, memory, *, physical_context=True):
    """One current posterior supplies both policy features and setting values."""
    if belief_model is None:
        belief = visible_beliefs(table.values, table.counts > 0, table.known_hz,
                                table.known, table.query, table.mixture)
        posterior = None
    else:
        belief, posterior = belief_model.observe_predict(
            table.values.clone(), table.counts > 0, table.known_hz.clone(),
            table.known.clone(), table.query.clone(), memory)
    obs = make_observation(table.values, table.counts, table.known_hz, table.known,
                           table.query, table.failed, table.event, table.mixture,
                           table.steps, table.max_steps,
                           table.group_counts < table.sample_count, beliefs_override=belief)
    # Ledger construction is non-differentiable. Reattach the three learned
    # physical quantities so policy loss also trains the belief's use in action.
    candidates = obs["candidates"].clone()
    eligible = belief["eligible"]
    mean = (belief["means"] - table.mixture.y_mean) / table.mixture.y_scale
    std = belief["std"] / table.mixture.y_scale
    candidates[:, 20:, 24] = torch.where(eligible, mean, 0.).float()
    candidates[:, 20:, 25] = torch.where(eligible, std, 0.).float()
    candidates[:, 20:, 26] = torch.where(eligible, belief["weights"], 0.).float()
    obs["candidates"] = candidates
    obs["action_mask"][~table.alive, 0] = True
    return (add_physical_context(obs, belief, table.mixture) if physical_context else obs), belief, posterior


def prediction_loss(belief, target):
    """Future physical label is an outer training target, never a policy input."""
    std = belief["std"].clamp_min(2.)
    z = (target[:, None] - belief["means"]) / std
    log_density = -.5 * z.square() - std.log() - .9189385332046727
    # A still unmeasured quantity can be the relevant relation. Its predictive
    # uncertainty must remain in the mixture: masking it would incorrectly force
    # already measured, irrelevant quantities to explain the future outcome.
    log_weights = belief["weights"].clamp_min(1e-20).log()
    log_weights = log_weights - log_weights.logsumexp(-1, keepdim=True)
    nll = -(log_weights + log_density).logsumexp(-1)
    return nll.mean()


def train_window(policy, belief_model, env, neural, memory, *, unroll, memory_mode,
                 physical_weight, reference=None, reference_state=None, anchor_weight=.02):
    logps, vals, rewards, ends, entropies, losses, anchors = [], [], [], [], [], [], []
    for _ in range(unroll):
        obs, belief, _ = physical_observation(env.table, belief_model, memory)
        out = policy(**obs, state=neural)
        dist = torch.distributions.Categorical(logits=out["logits"].float())
        action = dist.sample()
        rows = torch.arange(env.batch, device=env.device)
        if belief_model is not None:
            # Clone before environment mutation; targets do not enter observation.
            future = env.table.targets[rows, env.table.query].clone()
            losses.append(prediction_loss(belief, future))
        if reference is not None:
            with torch.no_grad():
                original_obs, _, _ = physical_observation(env.table, None, None, physical_context=False)
                teacher = reference(**original_obs, state=reference_state)
                reference_state = teacher["state"]
                p = teacher["logits"].float().softmax(-1)
            logp = out["logits"].float().log_softmax(-1).masked_fill(~obs["action_mask"], 0.)
            anchors.append(-(p * logp).sum(-1).mean())
        execution_belief = {k: v.detach().float() if v.is_floating_point() else v.detach()
                            for k, v in belief.items()}
        reward, task_done = env.table.step(action, beliefs=execution_belief)
        env.task_rewards += reward
        length = env.plan.lengths[rows, env.lifetime]
        lifetime_done = task_done & (env.task + 1 == length)
        final_obs, _, final_post = physical_observation(env.table, belief_model, memory)
        neural = flush_terminal_state(policy, out["state"], final_obs, task_done)
        # Within-task persistent computation is kept. Task boundaries match the
        # strongest rehearsal reference; physical memory is the continued state.
        neural = policy.reset_rows(neural, task_done)
        if reference is not None:
            reference_state = reference.reset_rows(reference_state, task_done)
        if belief_model is not None and task_done.any():
            selected = task_done.nonzero().squeeze(1)
            updated = belief_model.remember(
                {k: v[selected] for k, v in memory.items()},
                {k: v[selected] for k, v in final_post.items()})
            if memory_mode == "carry":
                memory = replace_rows(memory, updated, selected)
            reset = lifetime_done if memory_mode == "carry" else task_done
            selected = reset.nonzero().squeeze(1)
            if len(selected):
                memory = replace_rows(memory, belief_model.initial(len(selected), device=env.device), selected)
        env.next_tasks(task_done)
        logps.append(dist.log_prob(action)); vals.append(out["value"])
        rewards.append(reward); ends.append(lifetime_done); entropies.append(dist.entropy())
    with torch.no_grad():
        obs, _, _ = physical_observation(env.table, belief_model, memory)
        value = policy(**obs, state=neural)["value"]
    returns = []
    for reward, done in zip(reversed(rewards), reversed(ends)):
        value = reward + .99 * value * ~done
        returns.append(value)
    returns = torch.stack(returns[::-1])
    values = torch.stack(vals)
    pg = -(torch.stack(logps) * (returns - values).detach()).mean()
    vl = F.smooth_l1_loss(values, returns)
    entropy = torch.stack(entropies).mean()
    physical = torch.stack(losses).mean() if losses else pg.new_zeros(())
    anchor = torch.stack(anchors).mean() if anchors else pg.new_zeros(())
    loss = pg + .5 * vl - .01 * entropy + physical_weight * physical + anchor_weight * anchor
    stats = {"policy_loss": float(pg.detach()), "value_loss": float(vl.detach()),
             "physical_nll": float(physical.detach()), "reference_cross_entropy": float(anchor.detach()),
             "entropy": float(entropy.detach()), "mean_step_reward": float(torch.stack(rewards).mean()),
             "completed_tasks": len(env.task_rows)}
    return loss, neural, memory, reference_state, stats


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--belief-checkpoint", type=Path)
    p.add_argument("--mixture-checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--mode", choices=("ordinary", "carry", "reset"), default="carry")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=221)
    p.add_argument("--updates", type=int, default=400)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--unroll", type=int, default=16)
    p.add_argument("--max-steps", type=int, default=160)
    p.add_argument("--core-lr", type=float, default=3e-6)
    p.add_argument("--head-lr", type=float, default=3e-5)
    p.add_argument("--belief-lr", type=float, default=1e-4)
    p.add_argument("--physical-weight", type=float, default=.05)
    p.add_argument("--native-fraction", type=float, default=.5)
    p.add_argument("--anchor-weight", type=float, default=.02)
    p.add_argument("--save-every", type=int, default=100)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    mixture, _ = load_mixture_checkpoint(args.mixture_checkpoint, args.device)
    mixture.requires_grad_(False)
    belief_model = None
    if args.mode != "ordinary":
        belief_model, _ = load_experimental_belief_checkpoint(args.belief_checkpoint, args.device)
    pool = CalibrationTaskPool(args.data / "train.npz", mixture, args.device)
    plan = MixedTaskStreamPlan(pool, source_path=args.data / "train.npz", streams=args.batch,
        lifetimes=(args.updates * args.unroll + 19) // 20 + 2, seed=args.seed,
        device=args.device, native_fraction=args.native_fraction)
    env = PhysicalTaskEnvironment(pool, plan, mixture, max_steps=args.max_steps)
    policy, _ = load_active_checkpoint(args.checkpoint, args.device)
    reference, _ = load_active_checkpoint(args.checkpoint, args.device)
    reference.eval().requires_grad_(False)
    parameters = [(name, par) for name, par in policy.named_parameters() if par.requires_grad]
    groups = [{"params": [p for n, p in parameters if n.startswith("core.")], "lr": args.core_lr},
              {"params": [p for n, p in parameters if not n.startswith("core.")], "lr": args.head_lr}]
    if belief_model is not None:
        groups.append({"params": list(belief_model.parameters()), "lr": args.belief_lr})
    optimizer = torch.optim.AdamW(groups, weight_decay=.001)
    settings = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    settings.update(task_source=pool.description(), schedule=plan.description(),
                    physical_context_tokens=True,
                    neural_state="persistent within task; reset at task boundary in every arm",
                    online_learning="public-ledger coefficient updates and physical posterior memory",
                    outer_targets="unrevealed physical labels used only as supervised training targets",
                    selection="fixed final update", native_validation="separate actual tool execution",
                    trainable_policy_parameters=sum(p.numel() for _, p in parameters))
    (args.output / "config.json").write_text(json.dumps(settings, indent=2) + "\n")
    neural = policy.initial(args.batch, args.device)
    reference_state = reference.initial(args.batch, args.device)
    memory = belief_model.initial(args.batch, device=args.device) if belief_model else None
    started = time.monotonic()
    with (args.output / "training.jsonl").open("w") as log:
        for update in range(1, args.updates + 1):
            loss, neural, memory, reference_state, stats = train_window(
                policy, belief_model, env, detach(neural), detach(memory) if memory else None,
                unroll=args.unroll, memory_mode=args.mode, physical_weight=args.physical_weight,
                reference=reference, reference_state=detach(reference_state), anchor_weight=args.anchor_weight)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_([p for g in optimizer.param_groups for p in g["params"]], 1.)
            optimizer.step()
            stats.update(update=update, loss=float(loss.detach()), gradient_norm=float(norm),
                         seconds=time.monotonic() - started)
            if torch.device(args.device).type == "cuda":
                stats["peak_allocated_gib"] = torch.cuda.max_memory_allocated(args.device) / 2 ** 30
            log.write(json.dumps(stats) + "\n"); log.flush()
            if update == 1 or update % 10 == 0:
                print(json.dumps(stats), flush=True)
            env.task_rows.clear()
            if update % args.save_every == 0 or update == args.updates:
                torch.save({**policy.configuration(), "model": policy.state_dict(), "training": settings,
                            "optimizer": optimizer.state_dict(), "update": update}, args.output / "policy.pt")
                if belief_model is not None:
                    save_experimental_belief_checkpoint(belief_model, args.output / "belief.pt",
                                                       training={**settings, "update": update})
    (args.output / "finished.json").write_text(json.dumps({"updates": args.updates,
        "seconds": time.monotonic() - started, "decisions": args.updates * args.batch * args.unroll}, indent=2) + "\n")


if __name__ == "__main__":
    main()
