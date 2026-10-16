#!/usr/bin/env python3
"""What does a learned keep-or-rebuild gate rebuild after each kind of change?

For planners with a change gate, start from converged computation, apply one
change (goal moves, three doors toggle, both), and record the rebuilt share of
every channel averaged over free cells, and the rebuilt share of every cell as
a function of its distance to the nearest changed cell.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import load  # noqa: E402
from probe import CHANGES, worlds  # noqa: E402
from report import INK, MUTED, save, style_axes  # noqa: E402

COLOURS = {"goal": "#4a3aa7", "doors": "#1baf7a", "goal+doors": "#eda100"}
TITLES = {"goal": "goal moves", "doors": "three doors toggle", "goal+doors": "goal moves and doors toggle"}


@torch.no_grad()
def gate_shares(model, world, settle, device, batch=256):
    b0, g0, b1, g1 = (t.to(device) for t in world)
    channel, by_distance = [], []
    for i in range(0, len(b0), batch):
        s = slice(i, i + batch)
        state = model.start(b0[s], g0[s])
        state = model.advance(state, settle)
        after = model.observe(state, b1[s], g1[s])
        r = after.gates.float()                                    # [B, C, H, W]
        free = ~b1[s]
        channel.append((r * free[:, None]).sum((2, 3)) / free.sum((1, 2))[:, None])
        n, c, h, w = r.shape
        changed = (b0[s] != b1[s])
        changed.view(n, -1).scatter_(1, g0[s][:, None], True)
        changed.view(n, -1).scatter_(1, g1[s][:, None], True)
        rows, cols = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing="ij")
        cells = torch.stack((rows.flatten(), cols.flatten()), 1).float()               # [HW, 2]
        dist = []
        for k in range(n):
            where = changed[k].flatten().nonzero().squeeze(1)
            src = torch.stack((where // w, where % w), 1).float()
            dist.append(torch.cdist(cells, src, p=1).min(1).values.view(h, w))
        dist = torch.stack(dist)
        mean_r = r.mean(1)
        rows_out = []
        for d in range(0, 21):
            m = free & (dist.round() == d)
            rows_out.append([float(mean_r[m].sum()), int(m.sum())])
        by_distance.append(np.array(rows_out))
    channel = torch.cat(channel).mean(0).cpu().numpy()
    total = np.sum(by_distance, 0)
    with np.errstate(invalid="ignore"):
        profile = total[:, 0] / total[:, 1]
    return channel, profile


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, nargs="+", required=True)
    p.add_argument("--rooms", type=int, default=5)
    p.add_argument("--mazes", type=int, default=512)
    p.add_argument("--settle", type=int, default=32)
    p.add_argument("--seed", type=int, default=616161)
    p.add_argument("--device", default="cuda")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--figure", type=Path)
    args = p.parse_args(argv)
    device = torch.device(args.device)
    data = worlds(args.rooms, args.mazes, args.seed)
    results = {"args": {k: (str(v) if isinstance(v, Path) else [str(x) for x in v] if k == "model" else v)
                        for k, v in vars(args).items()}, "runs": []}
    for path in args.model:
        model = load(path, device)
        for name in CHANGES:
            channel, profile = gate_shares(model, data[name], args.settle, device)
            results["runs"].append({"model": str(path), "change": name, "channel": channel.tolist(),
                                    "by_distance": [None if np.isnan(x) else float(x) for x in profile]})
            print(json.dumps({"model": str(path), "change": name, "mean": round(float(channel.mean()), 3),
                              "channels_over_half": int((channel > 0.5).sum()), "near": round(float(profile[0]), 3),
                              "far": round(float(np.nanmean(profile[8:])), 3)}), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=1))
    if args.figure:
        first = [r for r in results["runs"] if r["model"] == str(args.model[0])]
        order = np.argsort(-np.array(next(r for r in first if r["change"] == "goal")["channel"]))
        fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.0))
        for r in first:
            axes[0].plot(np.array(r["channel"])[order], color=COLOURS[r["change"]], linewidth=1.5,
                         label=TITLES[r["change"]])
            axes[1].plot(range(len(r["by_distance"])), r["by_distance"], color=COLOURS[r["change"]], marker="o",
                         markersize=3, linewidth=1.5, label=TITLES[r["change"]])
        axes[0].set_xlabel("channel, sorted by its rebuilt share after a goal move")
        axes[0].set_ylabel("rebuilt share, mean over free cells")
        axes[1].set_xlabel("distance to the nearest changed cell (moves, ignoring walls)")
        axes[1].set_ylabel("rebuilt share, mean over channels")
        for ax in axes:
            style_axes(ax)
        axes[0].legend(frameon=False, fontsize=7.5)
        fig.suptitle("What the learned gate rebuilds after one change", fontsize=9, color=INK)
        args.figure.parent.mkdir(parents=True, exist_ok=True)
        save(fig, args.figure.parent, args.figure.name)


if __name__ == "__main__":
    main()
