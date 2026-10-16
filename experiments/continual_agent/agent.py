#!/usr/bin/env python3
"""A continually running agent that learns how much to compute, when to look, and when to move.

World: a stream of room-maze sites. Doors toggle and obstacles move all the time;
requests (goals) arrive every 20-50 ticks; after reaching a request the agent is
idle until the next one. The agent sees the true map within two cells and keeps
the last map it saw elsewhere. A survey returns the whole map after ``--delay``
ticks and costs ``--survey-cost``.

Agent: a trained planner (workspace, goal read; optionally a change gate and an
age input) and an activity head. Each tick the head chooses

    refinements n in {0, 1, 2, 4}, each costing ``--compute-cost``,
    whether to survey,
    whether to move along the planner's action or stay.

Reward: +1 on reaching a request, -0.02 per tick while a request is pending,
minus compute and survey costs. The head is trained with PPO over parallel
streams; the planner's parameters stay fixed.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "dynamic_maze"))
from cdm.maze import MazeSpec, make_batch, move  # noqa: E402
from cdm.planner import PlannerState  # noqa: E402
from evaluate import load  # noqa: E402
from train_partial import age_plane, window  # noqa: E402

REFINEMENTS = (0, 1, 2, 4)
ACTIONS = [(n, s, m) for n in range(4) for s in (0, 1) for m in (0, 1)]   # index -> (refinement index, survey, move)
GOAL_REWARD, STEP_COST = 1.0, 0.02


@dataclass
class Stream:
    planner: PlannerState
    belief: torch.Tensor
    age: torch.Tensor
    pos: torch.Tensor
    done: torch.Tensor
    timer: torch.Tensor          # ticks until a requested survey arrives (0: none pending)
    logits: torch.Tensor
    distance: torch.Tensor
    before: torch.Tensor         # action distribution at the agent's cell before the latest refinement
    since_survey: torch.Tensor
    since_goal: torch.Tensor
    surprise: torch.Tensor
    wait: torch.Tensor           # ticks the current request has been pending


def at(maps, pos):
    n, c = maps.shape[:2]
    return maps.reshape(n, c, -1).gather(2, pos.view(n, 1, 1).expand(n, c, 1)).squeeze(2)


def make_sites(n, rooms, ticks, seed, device, door_rate=0.3):
    """Exogenous maps and requests for ``n`` sites: continuous change, requests every 20-50 ticks."""
    b = make_batch(np.random.default_rng(seed), MazeSpec(rooms), ["continuous"] * n, ticks, (10, 25), door_rate)
    return (torch.from_numpy(b["blocked"]).to(device), torch.from_numpy(b["goal"]).to(device),
            torch.from_numpy(b["start"]).to(device))


@torch.no_grad()
def readout(model, planner):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits, distance = model.readout_state(planner)
    return logits.float(), distance.float()


@torch.no_grad()
def begin(model, blocked, goal, start):
    age = torch.zeros(blocked[:, 0].shape, device=blocked.device)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        planner = model.start(blocked[:, 0], goal[:, 0], age_plane(age))
    logits, distance = readout(model, planner)
    n = blocked.shape[0]
    z = torch.zeros(n, device=blocked.device)
    return Stream(planner=planner, belief=blocked[:, 0].clone(), age=age, pos=start.clone(),
                  done=torch.zeros(n, dtype=torch.bool, device=blocked.device), timer=z.long(), logits=logits,
                  distance=distance, before=torch.softmax(at(logits, start), 1), since_survey=z.clone(),
                  since_goal=z.clone(), surprise=z.clone(), wait=z.clone())


@torch.no_grad()
def perceive(model, s: Stream, blocked, goal, t, radius):
    """Start of tick t: arriving survey results and the local view update the belief; the planner reads it."""
    true = blocked[:, t]
    n, h, w = true.shape
    arrived = s.timer == 1
    seen = window(s.pos, h, w, radius) | arrived[:, None, None]
    surprise = ((s.belief != true) & seen).flatten(1).sum(1).float()
    belief = torch.where(seen, true, s.belief)
    age = torch.where(seen, torch.zeros_like(s.age), s.age + 1)
    new_goal = goal[:, t] != goal[:, t - 1]
    with torch.autocast("cuda", dtype=torch.bfloat16):
        planner = model.observe(s.planner, belief, goal[:, t], age_plane(age))
    logits, distance = readout(model, planner)
    return replace(s, planner=planner, belief=belief, age=age, logits=logits, distance=distance,
                   before=torch.softmax(at(logits, s.pos), 1),
                   timer=(s.timer - 1).clamp(min=0), surprise=surprise,
                   since_survey=torch.where(arrived, torch.zeros_like(s.since_survey), s.since_survey + 1),
                   since_goal=torch.where(new_goal, torch.zeros_like(s.since_goal), s.since_goal + 1),
                   done=s.done & ~new_goal, wait=torch.where(new_goal, torch.zeros_like(s.wait), s.wait))


def features(s: Stream):
    probs = torch.softmax(at(s.logits, s.pos), 1)
    entropy = -(probs * probs.clamp_min(1e-8).log()).sum(1)
    distance = at(s.distance[:, None], s.pos).squeeze(1)
    mean_age = (s.age.flatten(1).mean(1) / 32).clamp(max=2)
    scalars = torch.stack((probs.max(1).values, entropy, distance, (s.since_survey / 32).clamp(max=2),
                           (s.since_goal / 32).clamp(max=2), (s.surprise / 4).clamp(max=2), s.done.float(),
                           (s.timer > 0).float(), (s.wait / 32).clamp(max=2), mean_age), 1)
    return torch.cat((probs, scalars, at(s.planner.z.float(), s.pos)), 1)


@torch.no_grad()
def act(model, s: Stream, blocked, goal, t, choice, args):
    """Execute joint choices; return the new stream state, reward and compute used."""
    n_index, survey, step = (torch.tensor([a[i] for a in ACTIONS], device=choice.device)[choice] for i in range(3))
    counts = torch.tensor(REFINEMENTS, device=choice.device)[n_index]
    planner, logits, distance = s.planner, s.logits, s.distance
    z = planner.z
    for steps in (1, 2, 4):                          # refine each group of agents by its chosen number of steps
        idx = (counts == steps).nonzero().squeeze(1)
        if len(idx):
            sub = replace(planner, z=z[idx], x=planner.x[idx], goal=planner.goal[idx], blocked=planner.blocked[idx])
            with torch.autocast("cuda", dtype=torch.bfloat16):
                sub = model.advance(sub, steps)
                l, d = model.readout_state(sub)
            z = z.clone()
            z[idx] = sub.z
            logits, distance = logits.clone(), distance.clone()
            logits[idx], distance[idx] = l.float(), d.float()
    s = replace(s, planner=replace(planner, z=z), logits=logits, distance=distance)
    action = at(s.logits, s.pos).argmax(1)
    pos = torch.where(step.bool(), move(blocked[:, t], s.pos, action), s.pos)
    arrived = (pos == goal[:, t]) & ~s.done
    pending = ~s.done & ~arrived
    reward = arrived.float() * GOAL_REWARD - pending.float() * STEP_COST
    reward = reward - args.compute_cost * counts.float() - args.survey_cost * survey.float()
    timer = torch.where(survey.bool() & (s.timer == 0), torch.full_like(s.timer, args.delay), s.timer)
    s = replace(s, pos=pos, done=s.done | arrived, timer=timer, wait=torch.where(pending, s.wait + 1, s.wait))
    return s, reward, counts.float(), survey.float(), arrived.float()


class Policy(nn.Module):
    def __init__(self, size, hidden=256):
        super().__init__()
        self.norm = nn.LayerNorm(size)
        self.body = nn.Sequential(nn.Linear(size, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU())
        self.pi = nn.Linear(hidden, len(ACTIONS))
        self.v = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self.body(self.norm(x))
        return self.pi(h), self.v(h).squeeze(-1)


def fixed(name, s: Stream, t):
    """Hand-written schedules used as baselines."""
    n = s.pos.shape[0]
    pending = ~s.done
    probs = torch.softmax(at(s.logits, s.pos), 1)
    settle = 0.5 * (probs - s.before).abs().sum(1)
    refine = torch.ones(n, dtype=torch.long, device=s.pos.device)          # index of n=1
    survey = torch.zeros(n, dtype=torch.long, device=s.pos.device)
    step = torch.ones(n, dtype=torch.long, device=s.pos.device)
    if name == "one":                       # one refinement and a step every tick, no surveys
        pass
    elif name == "four":                    # four refinements every tick
        refine = torch.full_like(refine, 3)
    elif name == "survey16":                # one refinement, a survey every 16 ticks
        survey = ((s.since_survey >= 16) & (s.timer == 0)).long()
    elif name == "scaled":                  # compute and look only for a pending request; more right after it arrives
        refine = torch.where(pending, torch.where((s.since_goal < 2) | (settle > 0.3), 3, 1), 0)
        survey = (pending & (s.since_survey >= 16) & (s.timer == 0)).long()
        step = pending.long()
    index = refine * 4 + survey * 2 + step
    return index


def run(model, policy_fn, args, sites, ticks, seed, device, learn=None):
    """Roll out ``sites`` batches of ``args.streams`` streams; ``learn`` collects PPO transitions."""
    totals = {"reward": 0.0, "compute": 0.0, "survey": 0.0, "arrivals": 0.0, "requests": 0.0, "ticks": 0}
    for site in range(sites):
        blocked, goal, start = make_sites(args.streams, args.rooms, ticks, seed + site, device)
        s = begin(model, blocked, goal, start)
        totals["requests"] += float((goal[:, 1:] != goal[:, :-1]).sum() + args.streams)
        for t in range(1, ticks + 1):
            s = perceive(model, s, blocked, goal, t, args.radius)
            x = features(s)
            choice, extra = policy_fn(x, s, t)
            s, reward, compute, survey, arrived = act(model, s, blocked, goal, t, choice, args)
            if learn is not None:
                learn(x, choice, reward, extra, t == ticks)
            totals["reward"] += float(reward.sum())
            totals["compute"] += float(compute.sum())
            totals["survey"] += float(survey.sum())
            totals["arrivals"] += float(arrived.sum())
            totals["ticks"] += args.streams
    per = totals["ticks"]
    return {"reward_per_100_ticks": 100 * totals["reward"] / per, "refinements_per_tick": totals["compute"] / per,
            "surveys_per_100_ticks": 100 * totals["survey"] / per,
            "arrivals_per_request": totals["arrivals"] / totals["requests"]}


def ppo(model, args, device):
    torch.manual_seed(args.seed)
    size = features(begin(model, *make_sites(2, args.rooms, 4, 0, device))).shape[1]
    policy = Policy(size).to(device)
    opt = torch.optim.Adam(policy.parameters(), lr=args.lr)
    buffer = []
    log = []

    def policy_fn(x, s, t):
        with torch.no_grad():
            logits, value = policy(x)
            dist = torch.distributions.Categorical(logits=logits)
            choice = dist.sample()
            return choice, (dist.log_prob(choice), value)

    def learn(x, choice, reward, extra, last):
        if len(buffer) == args.horizon:      # the segment ended one tick ago: bootstrap from the state reached
            update(x)
        buffer.append((x, choice, reward, extra[0], extra[1]))
        if last:                             # a site ends the episode
            update(None)

    def update(next_x):
        xs = torch.stack([b[0] for b in buffer])            # [L, N, F]
        acts = torch.stack([b[1] for b in buffer])
        rews = torch.stack([b[2] for b in buffer])
        logp = torch.stack([b[3] for b in buffer])
        vals = torch.stack([b[4] for b in buffer])
        with torch.no_grad():
            next_v = torch.zeros_like(vals[0]) if next_x is None else policy(next_x)[1]
        adv = torch.zeros_like(rews)
        gae = torch.zeros_like(rews[0])
        for i in reversed(range(len(buffer))):
            nv = next_v if i == len(buffer) - 1 else vals[i + 1]
            delta = rews[i] + args.gamma * nv - vals[i]
            gae = delta + args.gamma * args.lam * gae
            adv[i] = gae
        ret = adv + vals
        flat = lambda a: a.reshape(-1, *a.shape[2:])
        X, A, LP, ADV, RET = flat(xs), flat(acts), flat(logp), flat(adv), flat(ret)
        ADV = (ADV - ADV.mean()) / (ADV.std() + 1e-8)
        for _ in range(args.epochs):
            perm = torch.randperm(len(X), device=device)
            for i in range(0, len(X), args.minibatch):
                j = perm[i:i + args.minibatch]
                logits, v = policy(X[j])
                dist = torch.distributions.Categorical(logits=logits)
                ratio = (dist.log_prob(A[j]) - LP[j]).exp()
                pg = -torch.min(ratio * ADV[j], ratio.clamp(1 - args.clip, 1 + args.clip) * ADV[j]).mean()
                loss = pg + 0.5 * F.mse_loss(v, RET[j]) - args.entropy * dist.entropy().mean()
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
                opt.step()
        buffer.clear()

    start = time.perf_counter()
    for it in range(args.iterations):
        stats = run(model, policy_fn, args, 1, args.site_ticks, 1_000_000 + 10_000 * args.seed + it, device, learn)
        stats.update(iteration=it, seconds=round(time.perf_counter() - start, 1))
        log.append(stats)
        if it % 5 == 0 or it == args.iterations - 1:
            print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in stats.items()}), flush=True)
    return policy, log


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--rooms", type=int, default=5)
    p.add_argument("--radius", type=int, default=2)
    p.add_argument("--delay", type=int, default=2)
    p.add_argument("--survey-cost", type=float, default=0.01)
    p.add_argument("--compute-cost", type=float, default=0.002)
    p.add_argument("--streams", type=int, default=512)
    p.add_argument("--site-ticks", type=int, default=256)
    p.add_argument("--iterations", type=int, default=150)
    p.add_argument("--horizon", type=int, default=32)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--minibatch", type=int, default=4096)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lam", type=float, default=0.95)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--entropy", type=float, default=0.01)
    p.add_argument("--eval-sites", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args(argv)
    device = torch.device(args.device)
    torch.backends.cudnn.benchmark = False   # subsets of varying size would re-trigger tuning
    model = load(args.model, device)
    if model.config.mode in ("carry", "restart", "goal_restart"):
        model.config = replace(model.config, memory_size=0)
    policy, log = ppo(model, args, device)
    results = {"args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}, "training": log,
               "evaluation": {}}
    seed = 5_000_000 + 1000 * args.rooms
    for name in ("one", "four", "survey16", "scaled"):
        results["evaluation"][name] = run(model, lambda x, s, t, name=name: (fixed(name, s, t), None), args,
                                          args.eval_sites, args.site_ticks, seed, device)
    def greedy(x, s, t):
        with torch.no_grad():
            return policy(x)[0].argmax(1), None
    results["evaluation"]["learned"] = run(model, greedy, args, args.eval_sites, args.site_ticks, seed, device)
    for name, r in results["evaluation"].items():
        print(json.dumps({"policy": name, **{k: round(v, 4) for k, v in r.items()}}), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=1))
    torch.save({"policy": policy.state_dict(), "args": results["args"]}, args.output.with_suffix(".pt"))


if __name__ == "__main__":
    main()
