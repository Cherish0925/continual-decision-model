#!/usr/bin/env python3
"""Who should get the next refinement, and who should look? Learned values under shared budgets.

``N`` agents in independent changing mazes share one planner. Each agent sees
the true map in a window around itself and keeps the last map it saw elsewhere.
Every tick each agent gets one refinement, a further ``(b - 1) * N`` refinements
go to the agents with the highest computation priority, ``s * N`` agents receive
the whole current map at the next tick, and every agent moves. Priorities:

computation  ``uniform`` (b refinements each), ``settling`` (largest change of the
             action distribution at the agent's cell in its latest refinement),
             ``learned`` (predicted gain of one more refinement), ``oracle``
             (actual gain, from the labels)
observation  ``none``, ``cycle`` (fixed rotation), ``oldest`` (longest since the
             last survey), ``learned`` (predicted return gain of surveying now)

The computation value is trained on the one-step change in correctness of the
agent's action; the observation value on the return difference of two branches
of the same world over ``H`` ticks, with and without a survey.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from active import (MOVE, REGIME_ORDER, SURVEY, ActivityHead, World, at, begin, features, make_world,  # noqa: E402
                    perceive_and_think, take_sim)
from cdm.maze import distance_maps, move, optimal_actions  # noqa: E402
from evaluate import GOAL_REWARD, STEP_COST, load  # noqa: E402


def labels_for(world: World):
    b, t1, h, w = world.blocked.shape
    dist = distance_maps(world.blocked.flatten(0, 1), world.goal.flatten(0, 1))
    return optimal_actions(dist, world.blocked.flatten(0, 1)).view(b, t1, 5, h, w)


@torch.no_grad()
def refine_subset(model, sim, idx):
    p = sim.planner
    sub = replace(p, z=p.z[idx], x=p.x[idx], goal=p.goal[idx], blocked=p.blocked[idx])
    with torch.autocast("cuda", dtype=torch.bfloat16):
        sub = model.advance(sub, 1)
        logits, distance = model.readout_state(sub)
    z, all_logits, all_distance = p.z.clone(), sim.logits.clone(), sim.distance.clone()
    z[idx], all_logits[idx], all_distance[idx] = sub.z, logits.float(), distance.float()
    before = sim.before.clone()
    before[idx] = torch.softmax(at(sim.logits[idx], sim.pos[idx]), 1)
    return replace(sim, planner=replace(p, z=z), logits=all_logits, distance=all_distance, before=before)


def correct_at(sim, opt_t):
    action = at(sim.logits, sim.pos).argmax(1)
    return at(opt_t.float(), sim.pos).gather(1, action[:, None]).squeeze(1)


@torch.no_grad()
def step_move(sim, world, t, survey):
    true, goal = world.blocked[:, t], world.goal[:, t]
    action = at(sim.logits, sim.pos).argmax(1)
    pos = move(true, sim.pos, action)
    arrived = (pos == goal) & ~sim.done
    reward = torch.where(arrived, torch.full_like(sim.since_goal, GOAL_REWARD),
                         torch.where(sim.done, torch.zeros_like(sim.since_goal), torch.full_like(sim.since_goal, -STEP_COST)))
    return replace(sim, pos=pos, done=sim.done | arrived, pending=survey), reward


@torch.no_grad()
def run(model, world, opt, radius, budget, comp, survey_share, obs, heads, rounds=4, record=None):
    """One shared-budget rollout; ``record`` collects computation-gain training pairs when given."""
    n, t1 = world.goal.shape
    sim = begin(model, world, radius, False)
    extra = int(round((budget - 1) * n)) if comp != "uniform" else 0
    per_round = extra // rounds
    surveys = int(round(survey_share * n))
    ret = torch.zeros(n, device=world.goal.device)
    agent_ok = torch.zeros(n, device=world.goal.device)
    phase = torch.arange(n, device=world.goal.device)
    for t in range(1, t1):
        sim = perceive_and_think(model, sim, world, t, radius, False)
        if record is not None:   # unbiased sample of one-step gains for the computation value
            trial = refine_subset(model, sim, torch.arange(n, device=world.goal.device))
            gain = correct_at(trial, opt[:, t]) - correct_at(sim, opt[:, t])
            pick = torch.randperm(n, device=world.goal.device)[:64]
            record.append((features(sim)[pick].cpu(), gain[pick].cpu()))
        if comp == "uniform":
            for _ in range(int(budget) - 1):
                sim = refine_subset(model, sim, torch.arange(n, device=world.goal.device))
        for _ in range(rounds if per_round else 0):
            if comp == "settling":
                probs = torch.softmax(at(sim.logits, sim.pos), 1)
                priority = 0.5 * (probs - sim.before).abs().sum(1)
            elif comp == "learned":
                priority = heads["comp"](features(sim)).squeeze(1)
            else:   # oracle: actual one-step gain
                trial = refine_subset(model, sim, torch.arange(n, device=world.goal.device))
                priority = correct_at(trial, opt[:, t]) - correct_at(sim, opt[:, t])
            idx = priority.topk(per_round).indices
            sim = refine_subset(model, sim, idx)
        survey = torch.zeros(n, dtype=torch.bool, device=world.goal.device)
        if surveys:
            if obs == "cycle":
                survey = (phase + t) % max(1, round(1 / survey_share)) == 0
            elif obs in ("oldest", "learned"):
                if obs == "oldest":
                    priority = sim.since_survey + 1e-3 * torch.rand(n, device=world.goal.device)
                else:
                    priority = heads["obs"](features(sim)).squeeze(1)
                survey[priority.topk(surveys).indices] = True
        agent_ok += correct_at(sim, opt[:, t])
        sim, reward = step_move(sim, world, t, survey)
        ret += reward
    return {"return": ret.cpu().numpy(), "agent_optimal": (agent_ok / (t1 - 1)).cpu().numpy()}


@torch.no_grad()
def survey_data(model, world, radius, horizon, stride):
    """Features and return gain of surveying now versus not, from paired branches with no further surveys."""
    n, t1 = world.goal.shape
    sim = begin(model, world, radius, False)
    no = torch.zeros(n, dtype=torch.bool, device=world.goal.device)
    xs, ys = [], []
    for t in range(1, t1 - horizon):
        sim = perceive_and_think(model, sim, world, t, radius, False)
        if t % stride == 0:
            idx = torch.arange(n, device=world.goal.device).repeat(2)
            pair = take_sim(sim, idx)
            sub = World(world.blocked[idx], world.goal[idx], world.start[idx])
            first = torch.cat((torch.ones(n, dtype=torch.bool, device=no.device), no))
            pair, total = step_move(pair, sub, t, first)
            for u in range(t + 1, t + horizon):
                pair = perceive_and_think(model, pair, sub, u, radius, False)
                pair, reward = step_move(pair, sub, u, torch.zeros(2 * n, dtype=torch.bool, device=no.device))
                total += reward
            xs.append(features(sim).cpu())
            ys.append((total[:n] - total[n:]).cpu())
        sim, _ = step_move(sim, world, t, no)
    return torch.cat(xs), torch.cat(ys)


def fit(x, y, device, epochs=8, seed=0):
    torch.manual_seed(seed)
    head = ActivityHead(x.mean(0), x.std(0).clamp_min(1e-4)).to(device)
    head.net[-1] = torch.nn.Linear(128, 1).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x, y[:, None]), batch_size=4096, shuffle=True)
    for _ in range(epochs):
        for xb, yb in loader:
            loss = F.smooth_l1_loss(head(xb.to(device)), yb.to(device), beta=0.05)
            opt.zero_grad()
            loss.backward()
            opt.step()
    return head.eval()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, nargs="+", required=True)
    p.add_argument("--rooms", type=int, default=5)
    p.add_argument("--regimes", nargs="+", default=["partial_return", "continuous"])
    p.add_argument("--radius", type=int, default=2)
    p.add_argument("--budgets", type=float, nargs="+", default=[1.0, 2.0])
    p.add_argument("--survey-shares", type=float, nargs="+", default=[0.0, 1 / 16])
    p.add_argument("--horizon", type=int, default=24)
    p.add_argument("--stride", type=int, default=4)
    p.add_argument("--train-batches", type=int, default=6)
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--episodes", type=int, default=1024)
    p.add_argument("--ticks", type=int, default=96)
    p.add_argument("--seed", type=int, default=424242)
    p.add_argument("--phase-range", type=int, nargs=2, default=[12, 24])
    p.add_argument("--device", default="cuda")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args(argv)
    device = torch.device(args.device)
    torch.backends.cudnn.benchmark = True
    results = {"args": {k: (str(v) if isinstance(v, Path) else [str(x) for x in v] if k == "model" else v)
                        for k, v in vars(args).items()}, "runs": []}
    for path in args.model:
        model = load(path, device)
        if model.config.mode in ("carry", "restart", "goal_restart"):
            model.config = replace(model.config, memory_size=0)
        for regime in args.regimes:
            r_index = REGIME_ORDER.index(regime)
            # training worlds come from the training distribution with their own seeds
            comp_pairs, sx, sy = [], [], []
            for b in range(args.train_batches):
                world = make_world(args.rooms, regime, args.batch, args.ticks, 9_000_000 + 10 * b + r_index, (6, 12), device)
                opt = labels_for(world)
                run(model, world, opt, args.radius, 2.0, "settling", 1 / 16, "cycle", {}, record=comp_pairs)
                x, y = survey_data(model, world, args.radius, args.horizon, args.stride)
                sx.append(x)
                sy.append(y)
            cx = torch.cat([c[0] for c in comp_pairs])
            cy = torch.cat([c[1] for c in comp_pairs])
            heads = {"comp": fit(cx, cy, device), "obs": fit(torch.cat(sx), torch.cat(sy), device)}
            print(json.dumps({"model": str(path), "regime": regime, "comp_samples": len(cx),
                              "comp_gain_mean": round(float(cy.mean()), 4), "obs_samples": len(torch.cat(sy)),
                              "obs_gain_mean": round(float(torch.cat(sy).mean()), 4)}), flush=True)
            world = make_world(args.rooms, regime, args.episodes, args.ticks,
                               args.seed + 1000 * args.rooms + 100 * r_index, tuple(args.phase_range), device)
            opt = labels_for(world)
            settings = [("uniform", "none", 1.0, 0.0)]
            for budget in args.budgets:
                for comp in (["uniform", "settling", "learned", "oracle"] if budget > 1 else []):
                    settings.append((comp, "none", budget, 0.0))
            for share in args.survey_shares:
                for obs in (["cycle", "oldest", "learned"] if share > 0 else []):
                    settings.append(("uniform", obs, 1.0, share))
            for comp, obs, budget, share in settings:
                out = run(model, world, opt, args.radius, budget, comp, share, obs, heads)
                row = {"model": str(path), "rooms": args.rooms, "regime": regime, "budget": budget, "computation": comp,
                       "survey_share": share, "observation": obs}
                for key in ("return", "agent_optimal"):
                    row[key] = float(out[key].mean())
                    row[key + "_sem"] = float(out[key].std(ddof=1) / np.sqrt(len(out[key])))
                results["runs"].append(row)
                print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()
                                  if not k.endswith("_sem")}), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
