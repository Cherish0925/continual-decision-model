#!/usr/bin/env python3
"""Closed-loop evaluation, recovery after changes, size transfer and latency.

The agent acts every tick from the planner's readout at its own cell after
``k`` refinements. All models see identical episodes (fixed seeds). The
optimal policy on the current map provides the reference return.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cdm.maze import MazeSpec, make_batch, move  # noqa: E402
from cdm.planner import PlannerConfig, build_planner  # noqa: E402
from train import labels  # noqa: E402

STEP_COST, GOAL_REWARD = 0.02, 1.0
RECOVERY_WINDOW = 24  # the last bucket holds every later tick


def load(path: Path, device) -> torch.nn.Module:
    blob = torch.load(path, map_location=device, weights_only=False)
    model = build_planner(PlannerConfig(**blob["config"])).to(device)
    model.load_state_dict(blob["state_dict"])
    return model.eval()


def episodes(rooms, regime, count, ticks, seed, phase_range=(12, 24)):
    rng = np.random.default_rng(seed)
    return make_batch(rng, MazeSpec(rooms), [regime] * count, ticks, tuple(phase_range))


KINDS = ("initial", "goal", "door", "obstacle", "door+obstacle", "goal+map")


def change_kinds(blocked, goal):
    """Kind of change at every tick and whether the new context occurred earlier."""
    b, t1, h, w = blocked.shape
    rows = torch.arange(h, device=blocked.device)[:, None] % 4 == 0
    cols = torch.arange(w, device=blocked.device)[None, :] % 4 == 0
    wall_line = rows | cols
    diff = blocked[:, 1:] != blocked[:, :-1]
    door = (diff & wall_line).flatten(2).any(2)
    obstacle = (diff & ~wall_line).flatten(2).any(2)
    goal_changed = goal[:, 1:] != goal[:, :-1]
    kind = torch.zeros(b, t1, dtype=torch.long, device=blocked.device)
    k = torch.zeros_like(goal_changed, dtype=torch.long)
    k = torch.where(door & ~obstacle, torch.full_like(k, 2), k)
    k = torch.where(obstacle & ~door, torch.full_like(k, 3), k)
    k = torch.where(obstacle & door, torch.full_like(k, 4), k)
    k = torch.where(goal_changed & ~(door | obstacle), torch.full_like(k, 1), k)
    k = torch.where(goal_changed & (door | obstacle), torch.full_like(k, 5), k)
    kind[:, 1:] = k
    weights = torch.randn(h * w, dtype=torch.float64, device=blocked.device)
    key = (blocked.flatten(2).double() @ weights) + goal.double() * 1e3
    returned = torch.zeros(b, t1, dtype=torch.bool, device=blocked.device)
    for t in range(1, t1):
        returned[:, t] = (key[:, :t] == key[:, t:t + 1]).any(1)
    return kind, returned


@torch.no_grad()
def prepare(batch, device):
    """Episode tensors with labels and change kinds, shared by every refinement budget."""
    blocked = torch.from_numpy(batch["blocked"]).to(device)
    goal = torch.from_numpy(batch["goal"]).to(device)
    kinds, returned = change_kinds(blocked, goal)
    return {"blocked": blocked, "goal": goal, "start": torch.from_numpy(batch["start"]).to(device),
            "opt": labels(blocked, goal)[1], "kinds": kinds, "returned": returned}


@torch.no_grad()
def rollout(model, world, think, device, policy="model"):
    blocked, goal, pos, opt = world["blocked"], world["goal"], world["start"], world["opt"]
    kinds, returned = world["kinds"], world["returned"]
    b, t1, h, w = blocked.shape
    free = ~blocked
    changed = torch.zeros(b, t1, dtype=torch.bool, device=device)
    changed[:, 1:] = (blocked[:, 1:] != blocked[:, :-1]).flatten(2).any(2) | (goal[:, 1:] != goal[:, :-1])
    since = torch.full((b,), RECOVERY_WINDOW, dtype=torch.long, device=device)
    last_kind = torch.zeros(b, dtype=torch.long, device=device)
    last_return = torch.zeros(b, dtype=torch.bool, device=device)
    done = torch.zeros(b, dtype=torch.bool, device=device)
    ret = torch.zeros(b, device=device)
    phases = torch.ones(b, device=device)
    reached = torch.zeros(b, device=device)
    log = {key: torch.zeros(b, t1 - 1, device=device) for key in ("map", "agent")}
    log.update({key: torch.zeros(b, t1 - 1, dtype=torch.long, device=device)
                for key in ("since", "kind", "affected", "affected_ok")})
    log["returned"] = torch.zeros(b, t1 - 1, dtype=torch.bool, device=device)
    reference = opt[:, 0].clone()   # optimal actions just before the last change
    state = None
    if policy == "model":
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            state = model.start(blocked[:, 0], goal[:, 0])
    for t in range(1, t1):
        c = changed[:, t]
        since = torch.where(c, torch.zeros_like(since), (since + 1).clamp(max=RECOVERY_WINDOW))
        last_kind = torch.where(c, kinds[:, t], last_kind)
        last_return = torch.where(c, returned[:, t], last_return)
        reference = torch.where(c[:, None, None, None], opt[:, t - 1], reference)
        if policy == "model":
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                state = model.observe(state, blocked[:, t], goal[:, t])
                state = model.advance(state, think)
                logits = model.readout_state(state)[0]
            pred = logits.float().argmax(1)                                  # [B, H, W]
        else:  # optimal policy on the current map; ties resolved by action order
            pred = opt[:, t].float().argmax(1)
        correct = opt[:, t].gather(1, pred[:, None]).squeeze(1) & free[:, t]
        log["map"][:, t - 1] = correct.flatten(1).sum(1) / free[:, t].flatten(1).sum(1)
        affected = free[:, t] & ~(opt[:, t] & reference).any(1)   # no earlier optimal action is still optimal
        log["affected"][:, t - 1] = affected.flatten(1).sum(1)
        log["affected_ok"][:, t - 1] = (correct & affected).flatten(1).sum(1)
        action = pred.view(b, -1).gather(1, pos[:, None]).squeeze(1)
        opt_here = opt[:, t].view(b, 5, -1).gather(2, pos.view(b, 1, 1).expand(b, 5, 1)).squeeze(2)
        log["agent"][:, t - 1] = opt_here.gather(1, action[:, None]).squeeze(1).float()
        log["since"][:, t - 1], log["kind"][:, t - 1], log["returned"][:, t - 1] = since, last_kind, last_return
        goal_changed = goal[:, t] != goal[:, t - 1]
        phases += goal_changed.float()
        done &= ~goal_changed
        pos = move(blocked[:, t], pos, action)
        arrived = (pos == goal[:, t]) & ~done
        ret += torch.where(arrived, torch.full_like(ret, GOAL_REWARD),
                           torch.where(done, torch.zeros_like(ret), torch.full_like(ret, -STEP_COST)))
        reached += arrived.float()
        done |= arrived
    groups = {"all": torch.ones_like(log["returned"])}
    groups.update({f"kind:{name}": log["kind"] == i for i, name in enumerate(KINDS) if i})
    groups["returned"] = log["returned"] & (log["kind"] > 0)
    groups["novel"] = ~log["returned"] & (log["kind"] > 0)
    curves = {}
    for name, mask in groups.items():
        rows = []
        for s in range(RECOVERY_WINDOW + 1):
            m = mask & (log["since"] == s)
            rows.append([float(log["map"][m].sum()), float(log["agent"][m].sum()), int(m.sum()),
                         int(log["affected_ok"][m].sum()), int(log["affected"][m].sum())])
        curves[name] = rows
    return {
        "return": ret.cpu().numpy(), "success": (reached / phases).cpu().numpy(),
        "agent_optimal": log["agent"].mean(1).cpu().numpy(),
        "map_accuracy": log["map"].mean(1).cpu().numpy(), "curves": curves,
        "changes_per_episode": float(changed.float().sum(1).mean()),
    }


def summarise(chunks):
    out = {}
    for key in ("return", "success", "agent_optimal", "map_accuracy"):
        values = np.concatenate([c[key] for c in chunks])
        out[key] = float(values.mean())
        out[key + "_sem"] = float(values.std(ddof=1) / np.sqrt(len(values)))
    curves = {}
    for name in chunks[0]["curves"]:
        total = np.sum([c["curves"][name] for c in chunks], 0)        # [S, 5]
        with np.errstate(invalid="ignore", divide="ignore"):
            curves[name] = {"map": (total[:, 0] / total[:, 2]).tolist(),
                            "agent": (total[:, 1] / total[:, 2]).tolist(),
                            "affected": (total[:, 3] / total[:, 4]).tolist(),
                            "affected_cells": (total[:, 4] / total[:, 2]).tolist(),
                            "count": total[:, 2].astype(int).tolist()}
    out["curves"] = curves
    out["changes_per_episode"] = float(np.mean([c["changes_per_episode"] for c in chunks]))
    out["episodes"] = int(sum(len(c["return"]) for c in chunks))
    return out


@torch.no_grad()
def latency(model, rooms, think, batch_size, device, ticks=200, seed=5):
    data = episodes(rooms, "continuous", batch_size, ticks, seed)
    blocked = torch.from_numpy(data["blocked"]).to(device)
    goal = torch.from_numpy(data["goal"]).to(device)
    pos = torch.from_numpy(data["start"]).to(device)
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        state = model.start(blocked[:, 0], goal[:, 0])
        times = []
        for t in range(1, ticks + 1):
            torch.cuda.synchronize(device)
            begin = time.perf_counter()
            state = model.observe(state, blocked[:, t], goal[:, t])
            state = model.advance(state, think)
            logits = model.readout_state(state)[0]
            b = logits.shape[0]
            action = logits.view(b, 5, -1).gather(2, pos.view(b, 1, 1).expand(b, 5, 1)).squeeze(2).argmax(1)
            pos = move(blocked[:, t], pos, action)
            torch.cuda.synchronize(device)
            times.append(time.perf_counter() - begin)
    ms = np.array(times[20:]) * 1000
    return {"mean_ms": float(ms.mean()), "p50_ms": float(np.percentile(ms, 50)),
            "p95_ms": float(np.percentile(ms, 95)), "batch": batch_size, "think": think}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, nargs="*", default=[])
    p.add_argument("--oracle", action="store_true", help="evaluate the optimal policy instead")
    p.add_argument("--rooms", type=int, nargs="+", default=[5])
    p.add_argument("--regimes", nargs="+", default=["full_return", "partial_return", "continuous"])
    p.add_argument("--think", type=int, nargs="+", default=[1, 2, 4, 8])
    p.add_argument("--episodes", type=int, default=1024)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--ticks", type=int, default=96)
    p.add_argument("--seed", type=int, default=424242)
    p.add_argument("--phase-range", type=int, nargs=2, default=[12, 24])
    p.add_argument("--latency", action="store_true", help="also time decisions; run on an otherwise idle GPU")
    p.add_argument("--latency-only", action="store_true")
    p.add_argument("--device", default="cuda")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args(argv)
    device = torch.device(args.device)
    torch.backends.cudnn.benchmark = True
    def plain(v):
        return [plain(x) for x in v] if isinstance(v, list) else (str(v) if isinstance(v, Path) else v)
    results = {"args": {k: plain(v) for k, v in vars(args).items()}, "runs": []}
    targets = [("oracle", None)] if args.oracle else [(str(m), load(m, device)) for m in args.model]
    for name, model in targets:
        for rooms in args.rooms:
            for r_index, regime in enumerate([] if args.latency_only else args.regimes):
                worlds = [prepare(episodes(rooms, regime, min(args.batch, args.episodes - c), args.ticks,
                                           args.seed + 1000 * rooms + 100 * r_index + c, args.phase_range), device)
                          for c in range(0, args.episodes, args.batch)]
                for think in ([0] if model is None else args.think):
                    chunks = [rollout(model, world, think, device, "oracle" if model is None else "model")
                              for world in worlds]
                    row = {"model": name, "rooms": rooms, "regime": regime, "think": think, **summarise(chunks)}
                    results["runs"].append(row)
                    print(json.dumps({k: row[k] for k in ("model", "rooms", "regime", "think", "return", "success", "map_accuracy")}), flush=True)
            if (args.latency or args.latency_only) and model is not None:
                for think in args.think:
                    for bs in (1, 256):
                        lat = latency(model, rooms, think, bs, device)
                        results.setdefault("latency", []).append({"model": name, "rooms": rooms, **lat})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
