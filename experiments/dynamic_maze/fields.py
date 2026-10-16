#!/usr/bin/env python3
"""Action fields of several planners after one goal move and after one door closure.

Each planner converges on a static maze, then one premise changes. With one
refinement per tick, the figure shows which cells hold an optimal action at
selected ticks after the change.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cdm.maze import MazeSpec, distance_maps, make_layout, optimal_actions, render, sample_config  # noqa: E402
from evaluate import load  # noqa: E402

WALL, RIGHT, WRONG, INK, MUTED = "#383835", "#f0efec", "#e34948", "#0b0b0b", "#52514e"
LABELS = {"restart": "Restart", "goal_restart": "Restart on goal moves", "carry": "Carry",
          "snapshot": "Exact snapshot", "nearest": "Nearest context", "memory_local": "Learned memory, per cell",
          "memory_global": "Learned memory, whole map", "carry_static": "Carry, static training",
          "factorized": "Factorized by premise", "factorized_local": "Factorized, no goal read",
          "factorized_carry": "Factorized, no rebuild", "carry_read": "Carry with goal read",
          "carry_gate": "Goal read and learned rebuild gate", "factorized_memory": "Factorized with map records"}


def scenarios(rng, rooms, before=24, after=12):
    """Two exogenous schedules on one maze: the goal moves, or the most used open door closes."""
    spec = MazeSpec(rooms)
    layout = make_layout(rng, spec)
    config = sample_config(rng, layout)
    blocked = torch.from_numpy(render(layout, config))
    side = spec.side
    # goal move: the admissible goal farthest from the current one
    d0 = distance_maps(blocked[None], torch.tensor([config.goal]))[0].flatten()
    far = layout.goal_cells[np.argmax(np.where(np.isfinite(d0[layout.goal_cells].numpy()),
                                               d0[layout.goal_cells].numpy(), -1))]
    goal_move = (torch.stack([blocked] * (before + after)),
                 torch.tensor([config.goal] * before + [int(far)] * after))
    # door closure: the open shortcut whose closure changes the most optimal actions
    opt0 = optimal_actions(d0.view(1, side, side), blocked[None])
    best, best_count = None, -1
    for k in np.flatnonzero(config.doors_open):
        doors = config.doors_open.copy()
        doors[k] = False
        closed = torch.from_numpy(render(layout, type(config)(doors, config.obstacles, config.goal)))
        d1 = distance_maps(closed[None], torch.tensor([config.goal]))
        opt1 = optimal_actions(d1, closed[None])
        count = int(((opt0 & opt1).sum(1) == 0)[0][~closed].sum())
        if count > best_count:
            best, best_count = closed, count
    door_close = (torch.stack([blocked] * before + [best] * after),
                  torch.tensor([config.goal] * (before + after)))
    return {"Goal moves": goal_move, "Door closes": door_close}, before


@torch.no_grad()
def correctness(model, blocked, goal, change, show, device):
    """Optimal-action mask over free cells at the requested ticks after the change."""
    blocked, goal = blocked[None].to(device), goal[None].to(device)
    dist = distance_maps(blocked[0], goal[0])
    opt = optimal_actions(dist, blocked[0])
    state = model.start(blocked[:, 0], goal[:, 0])
    frames = {}
    for t in range(1, blocked.shape[1]):
        state = model.observe(state, blocked[:, t], goal[:, t])
        state, outputs = model.think(state, 1)
        s = t - change
        if s in show:
            pred = outputs[-1][0].float().argmax(1)[0]
            frames[s] = opt[t].gather(0, pred[None]).squeeze(0).cpu().numpy()
    return frames


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", nargs="+", required=True, help="variant=path/to/model.pt")
    p.add_argument("--rooms", type=int, default=5)
    p.add_argument("--seed", type=int, default=3)
    p.add_argument("--show", type=int, nargs="+", default=[0, 1, 2, 4, 8])
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)
    device = torch.device(args.device)
    models = [(name, load(Path(path), device)) for name, path in (m.split("=", 1) for m in args.model)]
    worlds, change = scenarios(np.random.default_rng(args.seed), args.rooms)
    cmap = ListedColormap([WRONG, RIGHT, WALL])
    rows, cols = len(models), len(args.show)
    fig = plt.figure(figsize=(1.25 * (2 * cols + 0.35) + 1.4, 1.45 * rows + 0.5))
    grid = fig.add_gridspec(rows, 2 * cols + 1, width_ratios=[1] * cols + [0.35] + [1] * cols,
                            wspace=0.06, hspace=0.3)
    axes = np.array([[fig.add_subplot(grid[r, c if c < cols else c + 1]) for c in range(2 * cols)]
                     for r in range(rows)])
    for w_index, (title, (blocked, goal)) in enumerate(worlds.items()):
        for r, (name, model) in enumerate(models):
            frames = correctness(model, blocked, goal, change, set(args.show), device)
            for c, s in enumerate(args.show):
                ax = axes[r, w_index * cols + c]
                t = change + s
                walls = blocked[t].numpy()
                image = np.where(walls, 2, frames[s].astype(int))
                ax.imshow(image, cmap=cmap, vmin=0, vmax=2, interpolation="nearest")
                h = walls.shape[1]
                g_new, g_old = int(goal[t]), int(goal[change - 1])
                ax.plot(g_new % h, g_new // h, marker="*", color=INK, markersize=6.5, markeredgewidth=0)
                if g_old != g_new:
                    ax.plot(g_old % h, g_old // h, marker="*", markerfacecolor="none", markeredgecolor=INK,
                            markersize=6.5, markeredgewidth=0.7)
                wrong = int((~frames[s].astype(bool) & ~walls).sum())
                ax.text(0.98, -0.03, f"{wrong}", transform=ax.transAxes, ha="right", va="top", fontsize=6.5,
                        color=MUTED)
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False)
                if r == 0:
                    ax.set_title(f"+{s}" if s else "change", fontsize=7.5, color=INK, pad=3)
                if c == 0 and w_index == 0:
                    ax.set_ylabel(LABELS.get(name, name), fontsize=7.5, color=INK, rotation=0, ha="right", va="center",
                                  labelpad=4)
        x0 = axes[0, w_index * cols].get_position().x0
        x1 = axes[0, (w_index + 1) * cols - 1].get_position().x1
        fig.text((x0 + x1) / 2, axes[0, 0].get_position().y1 + 0.045, title, ha="center", fontsize=9, color=INK)
    fig.text(0.5, axes[-1, 0].get_position().y0 - 0.055,
             "Ticks after the change, one refinement per tick. Red: cells whose action is not optimal on the current map; "
             "number: count of such cells; filled star: goal; open star: previous goal", ha="center", fontsize=7,
             color=MUTED)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(args.out.with_suffix(".pdf"), bbox_inches="tight")


if __name__ == "__main__":
    main()
