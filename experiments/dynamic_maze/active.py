#!/usr/bin/env python3
"""Choosing when to move, think, or survey, learned from branched rollouts.

The agent sees the true map only in a window around itself; elsewhere it keeps
the last map it saw. Each tick the planner reads this belief and the current
goal and refines once. The agent then chooses one activity:

``move``    step by the planner's action at its cell (a closed door it did not see stops it)
``think``   stay one tick and give the planner ``--think-extra`` further refinements
``survey``  stay one tick and receive the whole current map at the next tick

Targets for the activity values come from branching: from the same state, the
three first activities are each followed by a fixed continuation policy for
``H`` ticks of the same exogenous world, and their returns are compared. A
small head on the planner's own state predicts these returns; acting greedily
on it is one step of policy improvement over the continuation policy.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, fields, replace
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cdm.maze import move  # noqa: E402
from cdm.planner import PlannerState  # noqa: E402
from evaluate import GOAL_REWARD, STEP_COST, episodes, load  # noqa: E402
from train_partial import age_plane  # noqa: E402

MOVE, THINK, SURVEY = 0, 1, 2
ACTIVITIES = ("move", "think", "survey")
REGIME_ORDER = ("full_return", "partial_return", "continuous")   # episode seeds as in evaluate.py
MARGINS = (0.0, 0.01, 0.02, 0.05)   # required predicted gain over moving, chosen on training-distribution worlds


@dataclass
class World:
    blocked: torch.Tensor   # [N, T+1, H, W] true maps
    goal: torch.Tensor      # [N, T+1]
    start: torch.Tensor     # [N]


@dataclass
class Sim:
    """Everything that differs between branches of the same world."""
    planner: PlannerState
    belief: torch.Tensor     # [N, H, W] last seen map
    age: torch.Tensor        # [N, H, W] ticks since each cell was last seen
    pos: torch.Tensor        # [N]
    done: torch.Tensor       # [N] arrived at the current goal
    logits: torch.Tensor     # [N, 5, H, W] readout after the latest refinement
    distance: torch.Tensor   # [N, H, W] predicted distance to the goal (scaled by 1/32)
    before: torch.Tensor     # [N, 5] action distribution at the agent's cell before that refinement
    pending: torch.Tensor    # [N] survey result arrives at the next tick
    since_survey: torch.Tensor
    since_surprise: torch.Tensor
    surprise: torch.Tensor   # [N] cells whose seen state differed from the belief this tick
    since_goal: torch.Tensor


def take(x, idx):
    if isinstance(x, torch.Tensor) and x.dim() > 0:
        return x[idx]
    if isinstance(x, PlannerState):
        return replace(x, **{f.name: take(getattr(x, f.name), idx) for f in fields(x)
                             if isinstance(getattr(x, f.name), torch.Tensor)})
    return x


def take_sim(sim: Sim, idx) -> Sim:
    return Sim(**{f.name: take(getattr(sim, f.name), idx) for f in fields(sim)})


def at(maps, pos):
    n, c = maps.shape[:2]
    return maps.reshape(n, c, -1).gather(2, pos.view(n, 1, 1).expand(n, c, 1)).squeeze(2)


def window(pos, h, w, radius):
    r, c = pos // w, pos % w
    rows = torch.arange(h, device=pos.device)
    cols = torch.arange(w, device=pos.device)
    return (((rows[None, :] - r[:, None]).abs() <= radius)[:, :, None]
            & ((cols[None, :] - c[:, None]).abs() <= radius)[:, None, :])


@torch.no_grad()
def begin(model, world: World, radius, full):
    blocked = world.blocked[:, 0]
    age = torch.zeros(blocked.shape, device=blocked.device)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        planner = model.start(blocked, world.goal[:, 0], age_plane(age))
        logits, distance = model.readout_state(planner)
    logits, distance = logits.float(), distance.float()
    n = blocked.shape[0]
    zeros = torch.zeros(n, device=blocked.device)
    return Sim(planner=planner, belief=blocked.clone(), age=age, pos=world.start.clone(),
               done=torch.zeros(n, dtype=torch.bool, device=blocked.device), logits=logits, distance=distance,
               before=torch.softmax(at(logits, world.start), 1), pending=torch.zeros_like(zeros, dtype=torch.bool),
               since_survey=zeros.clone(), since_surprise=zeros + 32, surprise=zeros.clone(), since_goal=zeros + 32)


@torch.no_grad()
def perceive_and_think(model, sim: Sim, world: World, t, radius, full):
    """Start of tick t: update the belief, then one planner refinement."""
    true = world.blocked[:, t]
    goal = world.goal[:, t]
    n, h, w = true.shape
    seen = window(sim.pos, h, w, radius) | sim.pending[:, None, None] | full
    surprise = ((sim.belief != true) & seen).flatten(1).sum(1).float()
    belief = torch.where(seen, true, sim.belief)
    age = torch.where(seen, torch.zeros_like(sim.age), sim.age + 1)
    goal_moved = goal != world.goal[:, t - 1]
    before = torch.softmax(at(sim.logits, sim.pos), 1)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        planner = model.observe(sim.planner, belief, goal, age_plane(age))
        planner = model.advance(planner, 1)
        logits, distance = model.readout_state(planner)
    return replace(sim, planner=planner, belief=belief, age=age, logits=logits.float(), distance=distance.float(),
                   before=before,
                   pending=torch.zeros_like(sim.pending),
                   since_survey=torch.where(sim.pending, torch.zeros_like(sim.since_survey), sim.since_survey + 1),
                   surprise=surprise,
                   since_surprise=torch.where(surprise > 0, torch.zeros_like(surprise), sim.since_surprise + 1),
                   since_goal=torch.where(goal_moved, torch.zeros_like(sim.since_goal), sim.since_goal + 1),
                   done=sim.done & ~goal_moved)


@torch.no_grad()
def act(model, sim: Sim, world: World, t, activity, extra):
    """Execute the activities chosen at tick t; return the new state and the reward."""
    true, goal = world.blocked[:, t], world.goal[:, t]
    action = at(sim.logits, sim.pos).argmax(1)
    moved = move(true, sim.pos, action)
    pos = torch.where(activity == MOVE, moved, sim.pos)
    arrived = (pos == goal) & ~sim.done
    reward = torch.where(arrived, torch.full_like(sim.since_goal, GOAL_REWARD),
                         torch.where(sim.done, torch.zeros_like(sim.since_goal), torch.full_like(sim.since_goal, -STEP_COST)))
    sim = replace(sim, pos=pos, done=sim.done | arrived, pending=activity == SURVEY)
    idx = (activity == THINK).nonzero().squeeze(1)
    if extra and len(idx):
        p = sim.planner
        sub = replace(p, z=p.z[idx], x=p.x[idx], goal=p.goal[idx], blocked=p.blocked[idx])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            sub = model.advance(sub, extra)
            logits, distance = model.readout_state(sub)
        z, all_logits, all_distance = p.z.clone(), sim.logits.clone(), sim.distance.clone()
        z[idx], all_logits[idx], all_distance[idx] = sub.z, logits.float(), distance.float()
        sim = replace(sim, planner=replace(p, z=z), logits=all_logits, distance=all_distance)
    return sim, reward


def features(sim: Sim):
    probs = torch.softmax(at(sim.logits, sim.pos), 1)
    settle = 0.5 * (probs - sim.before).abs().sum(1)
    entropy = -(probs * probs.clamp_min(1e-8).log()).sum(1)
    z = at(sim.planner.z.float(), sim.pos)
    distance = at(sim.distance[:, None], sim.pos).squeeze(1)
    scalars = torch.stack((probs.max(1).values, entropy, settle, distance, (sim.since_survey / 32).clamp(max=2),
                           (sim.since_surprise / 32).clamp(max=1), (sim.surprise / 4).clamp(max=2),
                           (sim.since_goal / 32).clamp(max=1), sim.done.float()), 1)
    return torch.cat((probs, scalars, z), 1)


class ActivityHead(nn.Module):
    """Predicted return advantage of move, think and survey from standardised features."""

    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)
        self.net = nn.Sequential(nn.Linear(len(mean), 128), nn.GELU(), nn.Linear(128, 128), nn.GELU(),
                                 nn.Linear(128, 3))

    def forward(self, x):
        return self.net((x - self.mean) / self.std)


def choose(policy, sim: Sim, t, head=None, every=8, threshold=0.3, margin=0.0):
    n = sim.pos.shape[0]
    zeros = torch.zeros(n, dtype=torch.long, device=sim.pos.device)
    if policy == "move" or policy == "full":
        return zeros
    if policy.startswith("survey"):
        return torch.where(sim.since_survey >= every, torch.full_like(zeros, SURVEY), zeros)
    if policy == "think":       # think while the decision at the agent's cell still moves
        probs = torch.softmax(at(sim.logits, sim.pos), 1)
        settle = 0.5 * (probs - sim.before).abs().sum(1)
        return torch.where(settle > threshold, torch.full_like(zeros, THINK), zeros)
    with torch.no_grad():
        q = head(features(sim))
    best = q.argmax(1)
    gain = q.gather(1, best[:, None]).squeeze(1) - q[:, MOVE]
    return torch.where(gain > margin, best, torch.zeros_like(best))


@torch.no_grad()
def run_policy(model, world, policy, radius, extra, head=None, every=8, threshold=0.3, margin=0.0):
    full = policy == "full"
    sim = begin(model, world, radius, full)
    t1 = world.blocked.shape[1]
    ret = torch.zeros_like(sim.since_goal)
    counts = torch.zeros(3, device=ret.device)
    for t in range(1, t1):
        sim = perceive_and_think(model, sim, world, t, radius, full)
        activity = choose(policy, sim, t, head, every, threshold, margin)
        counts += torch.bincount(activity, minlength=3).float()
        sim, reward = act(model, sim, world, t, activity, extra)
        ret += reward
    return {"return": ret.cpu().numpy(), "share": (counts / counts.sum()).cpu().tolist()}


@torch.no_grad()
def branch_data(model, world, radius, extra, horizon, every, stride, head=None, continuation="survey", margin=0.0):
    """Features and returns of the three first activities, branched from states visited by the continuation."""
    sim = begin(model, world, radius, False)
    t1 = world.blocked.shape[1]
    n = sim.pos.shape[0]
    xs, ys = [], []
    for t in range(1, t1 - horizon):
        sim = perceive_and_think(model, sim, world, t, radius, False)
        if t % stride == 0:
            branches = take_sim(sim, torch.arange(n, device=sim.pos.device).repeat(3))
            sub = World(world.blocked.repeat(3, 1, 1, 1), world.goal.repeat(3, 1), world.start.repeat(3))
            first = torch.arange(3, device=sim.pos.device).repeat_interleave(n)
            branches, reward = act(model, branches, sub, t, first, extra)
            total = reward.clone()
            for u in range(t + 1, t + horizon):
                branches = perceive_and_think(model, branches, sub, u, radius, False)
                a = choose(continuation, branches, u, head, every, margin=margin)
                branches, reward = act(model, branches, sub, u, a, extra)
                total += reward
            xs.append(features(sim).cpu())
            ys.append(total.view(3, n).T.cpu())
        a = choose(continuation, sim, t, head, every, margin=margin)
        sim, _ = act(model, sim, world, t, a, extra)
    return torch.cat(xs), torch.cat(ys)


def fit(x, y, epochs, device, seed):
    torch.manual_seed(seed)
    head = ActivityHead(x.mean(0), x.std(0).clamp_min(1e-4)).to(device)
    target = y - y.mean(1, keepdim=True)            # advantages within each state
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
    data = torch.utils.data.TensorDataset(x, target)
    loader = torch.utils.data.DataLoader(data, batch_size=4096, shuffle=True)
    for _ in range(epochs):
        for xb, yb in loader:
            loss = F.smooth_l1_loss(head(xb.to(device)), yb.to(device), beta=0.05)
            opt.zero_grad()
            loss.backward()
            opt.step()
    return head.eval()


def make_world(rooms, regime, count, ticks, seed, phase_range, device):
    b = episodes(rooms, regime, count, ticks, seed, phase_range)
    return World(torch.from_numpy(b["blocked"]).to(device), torch.from_numpy(b["goal"]).to(device),
                 torch.from_numpy(b["start"]).to(device))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, nargs="+", required=True)
    p.add_argument("--rooms", type=int, default=5)
    p.add_argument("--regimes", nargs="+", default=["partial_return", "continuous"])
    p.add_argument("--radius", type=int, default=2)
    p.add_argument("--think-extra", type=int, default=3)
    p.add_argument("--horizon", type=int, default=24)
    p.add_argument("--every", type=int, default=8)
    p.add_argument("--stride", type=int, default=4)
    p.add_argument("--train-batches", type=int, default=8)
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--iterations", type=int, default=2)
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--episodes", type=int, default=1024)
    p.add_argument("--ticks", type=int, default=96)
    p.add_argument("--seed", type=int, default=424242)
    p.add_argument("--phase-range", type=int, nargs=2, default=[12, 24])
    p.add_argument("--device", default="cuda")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args(argv)
    device = torch.device(args.device)
    torch.backends.cudnn.benchmark = False   # subsets of varying size would re-trigger tuning
    results = {"args": {k: (str(v) if isinstance(v, Path) else [str(x) for x in v] if k == "model" else v)
                        for k, v in vars(args).items()}, "runs": []}
    for path in args.model:
        model = load(path, device)
        if model.config.mode in ("carry", "restart", "goal_restart"):
            model.config = replace(model.config, memory_size=0)
        heads, margins = {}, {}
        for regime_index, regime in enumerate(args.regimes):
            head, continuation, margin = None, "survey", 0.0
            validation = make_world(args.rooms, regime, args.batch, args.ticks, 8_000_000 + regime_index,
                                    (6, 12), device)
            for it in range(args.iterations):
                xs, ys = [], []
                for b in range(args.train_batches):
                    world = make_world(args.rooms, regime, args.batch, args.ticks,
                                       7_000_000 + 1000 * it + 10 * b + regime_index, (6, 12), device)
                    x, y = branch_data(model, world, args.radius, args.think_extra, args.horizon, args.every,
                                       args.stride, head, continuation, margin)
                    xs.append(x)
                    ys.append(y)
                x, y = torch.cat(xs), torch.cat(ys)
                head = fit(x, y, args.epochs, device, seed=it)
                scores = {m: float(run_policy(model, validation, "learned", args.radius, args.think_extra, head,
                                              args.every, margin=m)["return"].mean()) for m in MARGINS}
                margin = max(scores, key=scores.get)
                continuation = "learned"
                print(json.dumps({"model": str(path), "regime": regime, "iteration": it, "samples": len(x),
                                  "mean_branch_return": [round(float(v), 4) for v in y.mean(0)],
                                  "validation": {str(k): round(v, 4) for k, v in scores.items()}, "margin": margin}),
                      flush=True)
            heads[regime], margins[regime] = head, margin
        for regime_index, regime in enumerate(args.regimes):
            world = make_world(args.rooms, regime, args.episodes, args.ticks,
                               args.seed + 1000 * args.rooms + 100 * REGIME_ORDER.index(regime),
                               tuple(args.phase_range), device)
            for policy in ("full", "move", "survey8", "survey16", "survey32", "think", "learned"):
                every = int(policy[6:]) if policy.startswith("survey") else args.every
                out = run_policy(model, world, policy, args.radius, args.think_extra, heads[regime], every,
                                 margin=margins[regime])
                row = {"model": str(path), "rooms": args.rooms, "regime": regime, "policy": policy,
                       "return": float(out["return"].mean()),
                       "return_sem": float(out["return"].std(ddof=1) / np.sqrt(len(out["return"]))),
                       "move": out["share"][0], "think": out["share"][1], "survey": out["share"][2]}
                results["runs"].append(row)
                print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
