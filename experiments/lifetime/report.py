#!/usr/bin/env python3
"""Aggregate lifetime evaluations and training logs across seeds into a table and figures."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# Fixed variant -> (label, colour, marker), in the validated categorical order.
STYLE = {
    "full": ("CDM", "#2a78d6", "o"),
    "fixed_learn": ("CDM, fixed learning schedule", "#eb6834", "s"),
    "frozen": ("CDM without learning", "#1baf7a", "^"),
    "world_model": ("World model, fixed schedules", "#eda100", "D"),
    "recurrent": ("Recurrent policy", "#e87ba4", "v"),
    "full_meta": ("CDM trained through its learning step", "#008300", "*"),
    "predict_only": ("Predictions without workspace computation or learning", "#4a3aa7", "h"),
}
METRICS = ("error_while_active", "reward_per_100_ticks", "response_ticks_median", "prediction_error",
           "shared_prediction_error", "refinements_per_tick", "measurements_per_100_ticks", "learning_per_100_ticks")
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e0"
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "legend.fontsize": 7.5,
                     "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "axes.titlecolor": INK,
                     "axes.labelcolor": MUTED, "pdf.fonttype": 42})


def style_axes(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
        ax.spines[side].set_linewidth(0.6)
    ax.tick_params(colors=MUTED, width=0.6)
    ax.grid(True, color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)


def mean_se(values):
    v = np.asarray([x for x in values if x is not None], dtype=float)
    if len(v) == 0:
        return None, None, 0
    return float(v.mean()), (float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0), len(v)


def collect(paths):
    """rows[(variant, setting)] = evaluation rows, one per seed; setting is "own" or the mode imposed."""
    rows = defaultdict(list)
    for path in paths:
        for r in json.loads(Path(path).read_text())["runs"]:
            rows[(r["variant"], "own" if r["mode"] == r["trained"] else r["mode"])].append(r)
    return rows


def label(variant, setting):
    if setting == "own":
        return STYLE[variant][0]
    if setting == "frozen":
        return f"{STYLE[variant][0]}, learning switched off"
    return f"{STYLE[variant][0]}, run in mode {setting}"


def table(rows):
    order = [k for v in STYLE for k in sorted(rows, key=lambda k: (k[1] != "own", k[1])) if k[0] == v]
    stats = {f"{t}/{m}": {metric: mean_se([r[metric] for r in rows[(t, m)]]) for metric in METRICS} for t, m in order}
    head = ["Model", "seeds", "error", "reward/100", "response", "prediction", "shared alone", "refine/tick",
            "measure/100", "learn/100"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for t, m in order:
        s = stats[f"{t}/{m}"]
        cells = [label(t, m), str(s["error_while_active"][2])]
        for metric, digits in zip(METRICS, (4, 2, 1, 4, 4, 2, 1, 1)):
            mean, se, _ = s[metric]
            cells.append("—" if mean is None else f"{mean:.{digits}f} ± {se:.{digits}f}")
        lines.append("| " + " | ".join(cells) + " |")
    return stats, "\n".join(lines)


def training_curves(run_root: Path, out: Path, smooth: int = 5):
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    for variant, (name, colour, _) in STYLE.items():
        curves = []
        for log in sorted((run_root / variant).glob("seed*/train_log.jsonl")):
            rows = [json.loads(line) for line in log.read_text().splitlines()]
            if len(rows) < smooth:
                continue
            x = np.array([r["transitions"] for r in rows]) / 1e6
            y = np.convolve([r["error_while_active"] for r in rows], np.ones(smooth) / smooth, mode="valid")
            curves.append((x[smooth - 1:], y))
        if not curves:
            continue
        n = min(len(c[1]) for c in curves)
        x, ys = curves[0][0][:n], np.array([c[1][:n] for c in curves])
        ax.plot(x, ys.mean(0), color=colour, linewidth=1.5, label=f"{name} ({len(curves)} seeds)")
        ax.fill_between(x, ys.min(0), ys.max(0), color=colour, alpha=0.12, linewidth=0)
    style_axes(ax)
    ax.set_xlabel("interactions (millions)")
    ax.set_ylabel("tracking error while a request is active")
    if ax.get_legend_handles_labels()[0]:
        ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def after_switch(rows, field, ylabel, out, keys, groups=("new", "returned")):
    bins = rows[keys[0]][0]["switch_bins"]
    ticks = [f"{a}–{b}" if b < 10 ** 8 else f"{a}+" for a, b in zip(bins[:-1], bins[1:])]
    fig, axes = plt.subplots(1, len(groups), figsize=(3.4 * len(groups), 2.9), sharey=True)
    for ax, group in zip(np.atleast_1d(axes), groups):
        for variant, setting in keys:
            ys = np.array([r[f"{field}_after_switch_{group}"] for r in rows[(variant, setting)]])
            colour, marker = STYLE[variant][1], STYLE[variant][2]
            hollow = setting != "own"
            ax.errorbar(range(len(ticks)), ys.mean(0),
                        yerr=ys.std(0, ddof=1) / np.sqrt(len(ys)) if len(ys) > 1 else None,
                        color=colour, marker=marker, markersize=3.6, linewidth=1.5, capsize=0,
                        linestyle=(0, (4, 2)) if hollow else "-", markerfacecolor="white" if hollow else colour,
                        label=label(variant, setting))
        style_axes(ax)
        ax.set_xticks(range(len(ticks)), ticks, rotation=35, ha="right")
        ax.set_xlabel("ticks since the operating condition changed")
        ax.set_title("new condition" if group == "new" else "condition met before in the lifetime")
    np.atleast_1d(axes)[0].set_ylabel(ylabel)
    np.atleast_1d(axes)[-1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--evaluations", type=Path, nargs="+", required=True)
    p.add_argument("--runs", type=Path, help="training output root with <mode>/seed<k>/train_log.jsonl")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = collect(args.evaluations)
    stats, md = table(rows)
    (args.output / "summary.json").write_text(json.dumps(stats, indent=1))
    (args.output / "table.md").write_text(md + "\n")
    print(md)
    own = [(v, "own") for v in STYLE if (v, "own") in rows]
    after_switch(rows, "error", "tracking error while a request is active", args.output / "error_after_switch.png", own)
    learners = [k for v in ("full", "full_meta") for k in sorted(rows, key=lambda k: k[1] != "own") if k[0] == v]
    if learners:
        after_switch(rows, "prediction_error", "prediction error of the next reading",
                     args.output / "prediction_after_switch.png", learners)
        after_switch(rows, "learning_rate", "learning steps per tick", args.output / "learning_after_switch.png",
                     [k for k in learners if k[1] == "own"])
    if args.runs:
        training_curves(args.runs, args.output / "training.png")


if __name__ == "__main__":
    main()
