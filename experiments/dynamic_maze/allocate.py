#!/usr/bin/env python3
"""One planner, many agents, a fixed number of refinements per tick: where should they go?

``N`` agents act in independent changing mazes and share one trained planner.
Every tick each agent receives one refinement. A further ``(budget - 1) * N``
refinements are handed out in rounds to the agents with the highest priority;
the rules differ only in the priority signal:

``uniform``     every agent receives ``budget`` refinements (integer budgets)
``random``      random agents
``change``      agents with the fewest refinements since their map or goal last changed
``confidence``  agents whose action probability at their own cell is lowest
``settling``    agents whose action distribution at their own cell moved most in their latest refinement
``oracle``      agents whose current action is not optimal (reads the labels; a reference)

Every rule except ``oracle`` uses only what the planner computes itself.
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

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cdm.maze import move  # noqa: E402
from evaluate import GOAL_REWARD, STEP_COST, episodes, load  # noqa: E402
from train import labels  # noqa: E402

RULES = ("uniform", "random", "change", "confidence", "settling", "oracle")


def at(maps: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
    """Values of [N, C, H, W] maps at flat positions: [N, C]."""
    n, c = maps.shape[:2]
    return maps.reshape(n, c, -1).gather(2, pos.view(n, 1, 1).expand(n, c, 1)).squeeze(2)


@torch.no_grad()
def run(model, world, rule, budget, rounds, device, seed=0):
    blocked, goal, pos, opt = world
    n, t1 = blocked.shape[:2]
    gen = torch.Generator(device=device).manual_seed(seed)
    extra = int(round((budget - 1) * n))
    per_round = extra // rounds if rule != "uniform" else 0
    if rule == "uniform" and abs(budget - round(budget)) > 1e-9:
        raise ValueError("uniform needs an integer budget")
    ret = torch.zeros(n, device=device)
    reached, phases = torch.zeros(n, device=device), torch.ones(n, device=device)
    done = torch.zeros(n, dtype=torch.bool, device=device)
    agent_ok = torch.zeros(n, device=device)
    since = torch.zeros(n, device=device)               # refinements since the last change
    spent = torch.zeros(n, device=device)
    seconds = 0.0
    amp = dict(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda")
    with torch.autocast(**amp):
        state = model.start(blocked[:, 0], goal[:, 0])
        logits = model.readout_state(state)[0].float()
    for t in range(1, t1):
        changed = (blocked[:, t] != blocked[:, t - 1]).flatten(1).any(1) | (goal[:, t] != goal[:, t - 1])
        since = torch.where(changed, torch.zeros_like(since), since)
        torch.cuda.synchronize(device)
        begin = time.perf_counter()
        before = torch.softmax(at(logits, pos), 1)
        with torch.autocast(**amp):
            state = model.observe(state, blocked[:, t], goal[:, t])
            state, outputs = model.think(state, int(budget) if rule == "uniform" else 1)
        logits = outputs[-1][0].float()
        since += int(budget) if rule == "uniform" else 1
        spent += int(budget) if rule == "uniform" else 1
        probs = torch.softmax(at(logits, pos), 1)
        moved = 0.5 * (probs - before).abs().sum(1)
        for _ in range(rounds if per_round else 0):
            if rule == "random":
                priority = torch.rand(n, device=device, generator=gen)
            elif rule == "change":
                priority = -since + 1e-3 * torch.rand(n, device=device, generator=gen)
            elif rule == "confidence":
                priority = -probs.max(1).values
            elif rule == "settling":
                priority = moved
            else:
                wrong = ~at(opt[:, t].float(), pos).bool().gather(1, probs.argmax(1, keepdim=True)).squeeze(1)
                priority = wrong.float() + 1e-3 * torch.rand(n, device=device, generator=gen)
            idx = priority.topk(per_round).indices
            sub = replace(state, z=state.z[idx], x=state.x[idx], goal=state.goal[idx], blocked=state.blocked[idx])
            with torch.autocast(**amp):
                sub, out = model.think(sub, 1)
            z = state.z.clone()
            z[idx] = sub.z
            state = replace(state, z=z)
            logits[idx] = out[-1][0].float()
            new = torch.softmax(at(logits[idx], pos[idx]), 1)
            moved[idx] = 0.5 * (new - probs[idx]).abs().sum(1)
            probs[idx] = new
            since[idx] += 1
            spent[idx] += 1
        action = probs.argmax(1)
        torch.cuda.synchronize(device)
        seconds += time.perf_counter() - begin
        agent_ok += at(opt[:, t].float(), pos).gather(1, action[:, None]).squeeze(1)
        goal_changed = goal[:, t] != goal[:, t - 1]
        phases += goal_changed.float()
        done &= ~goal_changed
        pos = move(blocked[:, t], pos, action)
        arrived = (pos == goal[:, t]) & ~done
        ret += torch.where(arrived, torch.full_like(ret, GOAL_REWARD),
                           torch.where(done, torch.zeros_like(ret), torch.full_like(ret, -STEP_COST)))
        reached += arrived.float()
        done |= arrived
    ticks = t1 - 1
    return {"return": ret.cpu().numpy(), "success": (reached / phases).cpu().numpy(),
            "agent_optimal": (agent_ok / ticks).cpu().numpy(),
            "refinements_per_tick": float(spent.mean() / ticks), "ms_per_tick": 1000 * seconds / ticks}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, nargs="+", required=True)
    p.add_argument("--rules", nargs="+", default=list(RULES), choices=RULES)
    p.add_argument("--budgets", type=float, nargs="+", default=[1.0, 1.5, 2.0, 3.0, 4.0])
    p.add_argument("--rounds", type=int, default=4)
    p.add_argument("--rooms", type=int, nargs="+", default=[5])
    p.add_argument("--regimes", nargs="+", default=["full_return", "partial_return", "continuous"])
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
        for rooms in args.rooms:
            for r_index, regime in enumerate(args.regimes):
                # same episodes as evaluate.py with --batch equal to --episodes
                batch = episodes(rooms, regime, args.episodes, args.ticks, args.seed + 1000 * rooms + 100 * r_index,
                                 args.phase_range)
                blocked = torch.from_numpy(batch["blocked"]).to(device)
                goal = torch.from_numpy(batch["goal"]).to(device)
                world = (blocked, goal, torch.from_numpy(batch["start"]).to(device), labels(blocked, goal)[1])
                for budget in args.budgets:
                    for rule in args.rules:
                        if rule == "uniform" and abs(budget - round(budget)) > 1e-9:
                            continue
                        out = run(model, world, rule, budget, args.rounds, device)
                        row = {"model": str(path), "rooms": rooms, "regime": regime, "budget": budget, "rule": rule,
                               "refinements_per_tick": out["refinements_per_tick"], "ms_per_tick": out["ms_per_tick"]}
                        for key in ("return", "success", "agent_optimal"):
                            row[key] = float(out[key].mean())
                            row[key + "_sem"] = float(out[key].std(ddof=1) / np.sqrt(len(out[key])))
                        results["runs"].append(row)
                        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()
                                          if not k.endswith("_sem")}), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
