#!/usr/bin/env python3
"""Refinements needed after one controlled change, starting from converged computation.

Each planner refines on a static maze until converged (``--settle`` refinements),
then exactly one premise changes: the goal moves, three shortcut doors toggle,
or both. The probe scores the cells whose optimal action the change altered,
before any further refinement (r = 0) and after each of ``--refinements`` more.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cdm.maze import Config, MazeSpec, distance_maps, make_layout, optimal_actions, render, sample_config  # noqa: E402
from evaluate import load  # noqa: E402

CHANGES = ("goal", "doors", "goal+doors")


def worlds(rooms, count, seed, doors=3):
    """Before/after maps and goals for every change type: dict name -> 4 tensors."""
    rng = np.random.default_rng(seed)
    spec = MazeSpec(rooms)
    out = {name: ([], [], [], []) for name in CHANGES}
    for _ in range(count):
        layout = make_layout(rng, spec)
        config = sample_config(rng, layout)
        goal_b = int(rng.choice(layout.goal_cells[layout.goal_cells != config.goal]))
        toggled = config.doors_open.copy()
        flip = rng.choice(len(toggled), size=min(doors, len(toggled)), replace=False)
        toggled[flip] = ~toggled[flip]
        before = render(layout, config)
        after = render(layout, Config(toggled, config.obstacles, config.goal))
        for name, (b1, g1) in {"goal": (before, goal_b), "doors": (after, config.goal),
                                "goal+doors": (after, goal_b)}.items():
            for lst, v in zip(out[name], (before, config.goal, b1, g1)):
                lst.append(v)
    return {name: tuple(torch.from_numpy(np.stack(v)) for v in lists) for name, lists in out.items()}


def accuracy(logits, opt, mask):
    pred = logits.float().argmax(1)
    ok = opt.gather(1, pred[:, None]).squeeze(1) & mask
    return float(ok.sum() / mask.sum().clamp_min(1))


@torch.no_grad()
def probe(model, world, settle, refinements, device, batch=256):
    b0, g0, b1, g1 = (t.to(device) for t in world)
    opt0 = optimal_actions(distance_maps(b0, g0), b0)
    opt1 = optimal_actions(distance_maps(b1, g1), b1)
    affected = ~b1 & ~(opt0 & opt1).any(1)
    free = ~b1
    sums = np.zeros((2, refinements + 1))
    for i in range(0, len(b0), batch):
        s = slice(i, i + batch)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            state = model.start(b0[s], g0[s])
            state = model.advance(state, settle)
            state = model.observe(state, b1[s], g1[s])
            outputs = [model.readout_state(state)] + model.think(state, refinements)[1]
        n = s.stop - i if s.stop <= len(b0) else len(b0) - i
        for r, (logits, _) in enumerate(outputs):
            sums[0, r] += accuracy(logits, opt1[s], affected[s]) * n
            sums[1, r] += accuracy(logits, opt1[s], free[s]) * n
    return {"affected": (sums[0] / len(b0)).tolist(), "map": (sums[1] / len(b0)).tolist(),
            "affected_cells": float(affected.sum(dim=(1, 2)).float().mean())}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, nargs="+", required=True)
    p.add_argument("--rooms", type=int, nargs="+", default=[5, 9])
    p.add_argument("--mazes", type=int, default=1024)
    p.add_argument("--settle", type=int, default=32)
    p.add_argument("--refinements", type=int, default=8)
    p.add_argument("--seed", type=int, default=515151)
    p.add_argument("--device", default="cuda")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args(argv)
    device = torch.device(args.device)
    data = {rooms: worlds(rooms, args.mazes, args.seed + rooms) for rooms in args.rooms}
    results = {"args": {k: (str(v) if isinstance(v, Path) else [str(x) for x in v] if k == "model" else v)
                        for k, v in vars(args).items()}, "runs": []}
    for path in args.model:
        model = load(path, device)
        for rooms in args.rooms:
            for name in CHANGES:
                row = {"model": str(path), "rooms": rooms, "change": name,
                       **probe(model, data[rooms][name], args.settle, args.refinements, device)}
                results["runs"].append(row)
                print(json.dumps({"model": str(path), "rooms": rooms, "change": name,
                                  "affected": [round(x, 3) for x in row["affected"]]}), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
