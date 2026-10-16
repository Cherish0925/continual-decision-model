#!/usr/bin/env python3
"""How much of a maze's optimal policy does not depend on the goal?

For random mazes, find for every free cell the single action that is optimal
for the largest share of admissible goals, and report that share averaged over
free cells. Map accuracy below this level carries no information about the goal.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from cdm.maze import MazeSpec, distance_maps, make_layout, optimal_actions, render, sample_config  # noqa: E402


@torch.no_grad()
def goal_independent_share(rng, rooms, device):
    layout = make_layout(rng, MazeSpec(rooms))
    blocked = torch.from_numpy(render(layout, sample_config(rng, layout))).to(device)
    goals = torch.tensor(layout.goal_cells, device=device)
    goals = goals[~blocked.flatten()[goals]]
    maps = blocked[None].expand(len(goals), -1, -1).contiguous()
    opt = optimal_actions(distance_maps(maps, goals), maps).float()   # [G, 5, H, W]
    return float(opt.mean(0).max(0).values[~blocked].mean())


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rooms", type=int, nargs="+", default=[5, 7, 9])
    p.add_argument("--mazes", type=int, default=64)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args(argv)
    for rooms in args.rooms:
        rng = np.random.default_rng(rooms)
        shares = [goal_independent_share(rng, rooms, torch.device(args.device)) for _ in range(args.mazes)]
        print(f"{rooms}x{rooms} rooms: {100 * np.mean(shares):.1f}% of free cells "
              f"(sd {100 * np.std(shares):.1f}, {args.mazes} mazes)")


if __name__ == "__main__":
    main()
