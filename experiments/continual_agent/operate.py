#!/usr/bin/env python3
"""One continuous run: act, look, think and learn from one shared compute budget.

The agent of ``agent.py`` runs a long stream of maze sites in phases that change
the maze size or the rate at which doors change (``rooms:door_rate``). Every tick
the fleet has ``--budget`` refinements per stream. Refinements the activity head
does not use go into a learning account; each time the account covers one
learning step, the planner takes a gradient step on its own recent experience.
Two targets are available:

``distill``    on replayed sequences of beliefs and goals, every one-refinement
               readout moves toward the same network's readout ``--depth``
               refinements deeper (faster use of what the network already computes)
``hindsight``  when a survey returns the true map, the belief, ages and goal at the
               moment the survey was requested are paired with the optimal actions
               on the returned map (what the agent should have done given what it
               later saw); this carries the world's actual behaviour, including a
               change in how fast doors change

Conditions: ``frozen`` (no learning), ``spare`` (learning paid by unused
refinements), ``fixed`` (one learning step every ``--fixed-every`` ticks,
whatever the activity head does).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from agent import Policy, act, begin, features, fixed, make_sites, perceive  # noqa: E402
from replay import sample_rows  # noqa: E402
from cdm.maze import distance_maps, optimal_actions  # noqa: E402
from evaluate import load  # noqa: E402
from train import step_losses  # noqa: E402
from train_partial import age_plane  # noqa: E402


def replay_batch(buf, picks, batch_size, rng):
    """Sample across the selected records and all of their recorded streams."""
    records = [buf[int(i)] for i in picks]
    selected = [(i, torch.as_tensor(rows, device=records[i][0].device))
                for i, rows in sample_rows([len(r[0]) for r in records], batch_size, rng)]
    return tuple(torch.cat([records[i][j].index_select(0, rows) for i, rows in selected])
                 for j in range(len(records[0])))


def hindsight_loss(model, belief, age, goal, true):
    """From the belief held when a survey was requested, the optimal actions on the map it returned."""
    dist = distance_maps(true, goal)
    opt = optimal_actions(dist, true)
    state = model.start(belief, goal, age_plane(age))
    logits, distance = model.readout_state(state)
    action, d, _ = step_losses(logits, distance, opt, dist, ~true)
    return action + 0.5 * d


def distill_loss(model, belief, age, goal, depth, teacher=None):
    """Distillation over replayed sequences [B, L, H, W]: shallow readouts toward deeper ones.

    Without ``teacher`` the deeper readout comes from the learner itself; with a frozen ``teacher`` it
    comes from that network running the same sequence.
    """
    state = model.start(belief[:, 0], goal[:, 0], age_plane(age[:, 0]))
    t_state = teacher.start(belief[:, 0], goal[:, 0], age_plane(age[:, 0])) if teacher is not None else None
    total, count = 0.0, 0
    for t in range(1, belief.shape[1]):
        state = model.observe(state, belief[:, t], goal[:, t], age_plane(age[:, t]))
        state, outputs = model.think(state, 1)
        with torch.no_grad():
            if teacher is None:
                deep_logits, deep_distance = model.readout_state(model.advance(state.detach(), depth))
            else:
                t_state = teacher.advance(teacher.observe(t_state, belief[:, t], goal[:, t], age_plane(age[:, t])), 1)
                deep_logits, deep_distance = teacher.readout_state(teacher.advance(t_state, depth))
            target = torch.softmax(deep_logits.float(), 1)
        free = ~belief[:, t]
        n = free.sum().clamp_min(1)
        logits, distance = outputs[-1]
        total = total + (-(target * F.log_softmax(logits.float(), 1)).sum(1) * free).sum() / n
        total = total + 0.5 * (F.smooth_l1_loss(distance.float(), deep_distance.float(), reduction="none") * free).sum() / n
        count += 1
    return total / count


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--policy", type=Path, help="activity head saved by agent.py; otherwise --schedule")
    p.add_argument("--schedule", default="scaled", choices=("one", "four", "survey16", "scaled"))
    p.add_argument("--condition", choices=("frozen", "spare", "fixed"), required=True)
    p.add_argument("--phases", nargs="+", default=["5:0.3", "5:1.0", "9:0.3", "5:0.3"], help="rooms:door_rate")
    p.add_argument("--targets", nargs="+", default=["distill", "hindsight"], choices=("distill", "hindsight"))
    p.add_argument("--teacher", choices=("self", "frozen"), default="frozen",
                   help="distill toward the learner's own deeper readout or a frozen copy of the deployed planner")
    p.add_argument("--sites-per-phase", type=int, default=6)
    p.add_argument("--site-ticks", type=int, default=128)
    p.add_argument("--streams", type=int, default=512)
    p.add_argument("--budget", type=float, default=2.0, help="refinements per stream per tick for the whole fleet")
    p.add_argument("--step-cost", type=float, default=4096.0, help="refinement-equivalents per learning step")
    p.add_argument("--fixed-every", type=int, default=8)
    p.add_argument("--record", type=int, default=64, help="streams whose experience enters the replay buffer")
    p.add_argument("--sequence", type=int, default=8)
    p.add_argument("--replay", type=int, default=64, help="sequences kept")
    p.add_argument("--learn-batch", type=int, default=32)
    p.add_argument("--hindsight-batch", type=int, default=128, help="returned surveys per learning step")
    p.add_argument("--hindsight-keep", type=int, default=256, help="ticks of returned surveys kept")
    p.add_argument("--depth", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--replay-old", action="store_true", help="each learning step also replays sequences from earlier maze sizes")
    p.add_argument("--radius", type=int, default=2)
    p.add_argument("--delay", type=int, default=2)
    p.add_argument("--survey-cost", type=float, default=0.01)
    p.add_argument("--compute-cost", type=float, default=0.002)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args(argv)
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False   # subsets of varying size would re-trigger tuning
    model = load(args.model, device)
    if model.config.mode in ("carry", "restart", "goal_restart"):
        model.config = replace(model.config, memory_size=0)
    policy = None
    if args.policy:
        blob = torch.load(args.policy, map_location=device, weights_only=False)
        size = blob["policy"]["norm.weight"].shape[0]
        policy = Policy(size).to(device)
        policy.load_state_dict(blob["policy"])
        policy.eval()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    teacher = None
    if args.teacher == "frozen":
        teacher = load(args.model, device).requires_grad_(False).eval()
        if teacher.config.mode in ("carry", "restart", "goal_restart"):
            teacher.config = replace(teacher.config, memory_size=0)
    replay, account, steps = {}, 0.0, 0     # maze side -> recent sequences
    hindsight = {}                          # maze side -> (belief, age, goal, returned map) samples
    curve = []
    start_time = time.perf_counter()
    tick = 0
    for phase, spec in enumerate(args.phases):
        rooms, door_rate = int(spec.split(":")[0]), float(spec.split(":")[1])
        for site in range(args.sites_per_phase):
            blocked, goal, start = make_sites(args.streams, rooms, args.site_ticks,
                                              8_000_000 + 1000 * phase + site, device, door_rate)
            model.eval()
            s = begin(model, blocked, goal, start)
            r = args.record
            asked = (s.belief[:r].clone(), s.age[:r].clone(), goal[:r, 0].clone())
            history = []
            sums = {"reward": 0.0, "compute": 0.0, "survey": 0.0, "arrived": 0.0}
            for t in range(1, args.site_ticks + 1):
                returning = s.timer[:r] == 1
                s = perceive(model, s, blocked, goal, t, args.radius)
                if returning.any():
                    side = s.belief.shape[-1]
                    i = returning.nonzero().squeeze(1)
                    sample = (asked[0][i], asked[1][i], asked[2][i], blocked[:r][i, t])
                    hindsight[side] = (hindsight.get(side, []) + [sample])[-args.hindsight_keep:]
                x = features(s)
                if policy is None:
                    choice = fixed(args.schedule, s, t)
                else:
                    with torch.no_grad():
                        choice = policy(x)[0].argmax(1)
                before_timer = s.timer[:r].clone()
                s, reward, compute, survey, arrived = act(model, s, blocked, goal, t, choice, args)
                requested = (before_timer == 0) & (s.timer[:r] == args.delay)
                if requested.any():
                    asked = tuple(torch.where(requested.view(-1, *([1] * (a.dim() - 1))), now, a)
                                  for a, now in zip(asked, (s.belief[:r], s.age[:r], goal[:r, t])))
                for key, value in (("reward", reward), ("compute", compute), ("survey", survey), ("arrived", arrived)):
                    sums[key] += float(value.sum())
                history.append((s.belief[:args.record].clone(), s.age[:args.record].clone(), goal[:args.record, t]))
                if len(history) == args.sequence:
                    side = s.belief.shape[-1]
                    replay[side] = (replay.get(side, []) + [tuple(torch.stack(v, 1) for v in zip(*history))])[-args.replay:]
                    history = []
                tick += 1
                if args.condition == "spare":
                    account += args.budget * args.streams - float(compute.sum())
                due = (args.condition == "spare" and account >= args.step_cost) or \
                      (args.condition == "fixed" and tick % args.fixed_every == 0)
                side = s.belief.shape[-1]
                if due and len(replay.get(side, [])) >= 4:
                    rng = np.random.default_rng(tick)
                    sides = [side] + ([int(rng.choice([k for k in replay if k != side]))]
                                      if args.replay_old and len(replay) > 1 else [])
                    model.train()
                    loss = 0.0
                    for k in sides:
                        if "distill" in args.targets:
                            buf = replay[k]
                            picks = rng.choice(len(buf), size=min(len(buf), 4), replace=False)
                            belief, age, goals = replay_batch(buf, picks, args.learn_batch, rng)
                            with torch.autocast("cuda", dtype=torch.bfloat16):
                                loss = loss + distill_loss(model, belief, age, goals, args.depth, teacher) / len(sides)
                        if "hindsight" in args.targets and len(hindsight.get(k, [])) >= 2:
                            buf = hindsight[k]
                            picks = rng.choice(len(buf), size=min(len(buf), 16), replace=False)
                            belief, age, goals, true = replay_batch(buf, picks, args.hindsight_batch, rng)
                            with torch.autocast("cuda", dtype=torch.bfloat16):
                                loss = loss + hindsight_loss(model, belief, age, goals, true) / len(sides)
                    if not torch.is_tensor(loss):
                        model.eval()
                        continue
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    model.eval()
                    steps += 1
                    if args.condition == "spare":
                        account -= args.step_cost
            n = args.streams * args.site_ticks
            row = {"phase": phase, "rooms": rooms, "door_rate": door_rate, "site": site,
                   "reward_per_100_ticks": 100 * sums["reward"] / n,
                   "refinements_per_tick": sums["compute"] / n, "surveys_per_100_ticks": 100 * sums["survey"] / n,
                   "learning_steps": steps, "seconds": round(time.perf_counter() - start_time, 1)}
            curve.append(row)
            print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
                                       "curve": curve}, indent=1))


if __name__ == "__main__":
    main()

