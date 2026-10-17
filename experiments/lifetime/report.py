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
    "predict_hidden": ("Recurrent policy trained to predict, reading no prediction", "#52514e", "X"),
    "meta_step": ("CDM trained through a learned step size", "#e34948", "P"),
    # the two-family runs are reported on their own, so they reuse the first hues of the same order
    "mixed_full": ("CDM on both families", "#2a78d6", "o"),
    "mixed_frozen": ("CDM on both families, without learning", "#1baf7a", "^"),
    "mixed_recurrent": ("Recurrent policy on both families", "#e87ba4", "v"),
    "thermal_full": ("CDM on thermal processes only", "#eb6834", "s"),
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
            if r["trained"] == "world_model" and r["mode"] == "frozen":
                continue                             # its activity heads were never trained; discard is the comparison
            rows[(r["variant"], "own" if r["mode"] == r["trained"] else r["mode"])].append(r)
    return rows


def label(variant, setting):
    if setting == "own":
        return STYLE[variant][0]
    if setting == "frozen":
        return f"{STYLE[variant][0]}, learning switched off"
    if setting == "discard":
        return f"{STYLE[variant][0]}, learning steps kept, updates discarded"
    return f"{STYLE[variant][0]}, run in mode {setting}"


OTHER = {"frozen": "learning switched off", "discard": "learning steps kept, updates discarded"}


def compute_cost(latency_file: Path | None):
    """GPU microseconds per instance for the control path, one refinement and one learning step, measured at the
    largest batch in the latency file."""
    if latency_file is None or not latency_file.exists():
        return None
    row = max(json.loads(latency_file.read_text())["timing"], key=lambda r: r["instances"])
    return {k: 1000 * row[f"{k}_ms"] / row["instances"] for k in ("control", "refinement", "learning")}


def compute_per_tick(r, cost):
    return cost["control"] + cost["refinement"] * r["refinements_per_tick"] + \
        cost["learning"] * r["learning_per_100_ticks"] / 100


def paired(rows, variant, a, b, metric):
    """Mean and standard error over seeds of metric(setting a) - metric(setting b) for one trained variant."""
    first = {r["seed"]: r[metric] for r in rows.get((variant, a), [])}
    second = {r["seed"]: r[metric] for r in rows.get((variant, b), [])}
    return mean_se([first[k] - second[k] for k in first if k in second])


def table(rows, cost=None):
    order = [k for v in STYLE for k in sorted(rows, key=lambda k: (k[1] != "own", k[1])) if k[0] == v]
    stats = {f"{t}/{m}": {metric: mean_se([r[metric] for r in rows[(t, m)]]) for metric in METRICS} for t, m in order}
    if cost:
        for t, m in order:
            stats[f"{t}/{m}"]["compute_us_per_tick"] = mean_se([compute_per_tick(r, cost) for r in rows[(t, m)]])
    head = ["Model", "seeds", "error", "reward/100", "response", "prediction", "shared alone", "refine/tick",
            "measure/100", "learn/100"] + (["GPU µs/tick"] if cost else [])
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for t, m in order:
        s = stats[f"{t}/{m}"]
        cells = [label(t, m), str(s["error_while_active"][2])]
        for metric, digits in zip(METRICS + (("compute_us_per_tick",) if cost else ()), (4, 2, 1, 4, 4, 2, 1, 1, 2)):
            mean, se, _ = s[metric]
            cells.append("—" if mean is None else f"{mean:.{digits}f} ± {se:.{digits}f}")
        lines.append("| " + " | ".join(cells) + " |")
    learners = [v for v in STYLE if (v, "frozen") in rows or (v, "discard") in rows]
    if learners:
        lines += ["", "Paired differences over seeds, own setting minus the other (negative: lower with learning):", "",
                  "| Model | other setting | error | reward/100 |", "|---|---|---|---|"]
        for v in learners:
            for other in ("frozen", "discard"):
                if (v, other) in rows:
                    e, r = paired(rows, v, "own", other, "error_while_active"), paired(rows, v, "own", other,
                                                                                         "reward_per_100_ticks")
                    lines.append(f"| {STYLE[v][0]} | {OTHER[other]} | "
                                 f"{e[0]:+.4f} ± {e[1]:.4f} (n={e[2]}) | {r[0]:+.2f} ± {r[1]:.2f} |")
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
    handles, labels = np.atleast_1d(axes)[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(3, len(labels)), frameon=False)
    fig.tight_layout(rect=(0, 0.07 * ((len(labels) + 2) // 3), 1, 1))
    fig.savefig(out, dpi=200)
    plt.close(fig)


def early_error(r, ticks):
    """Tracking error over the first ``ticks`` of the evaluated lifetime, from its thousand-tick bins."""
    bins = r["error_by_thousand_ticks"][:max(1, ticks // 1000)]
    return float(np.mean(bins))


def families_table(groups, ticks=None):
    """Own-setting error of every variant on every family: groups maps a family name to its rows. With ``ticks``,
    every evaluation is compared over the same first part of its lifetime."""
    variants = [v for v in STYLE if any((v, "own") in rows for rows in groups.values())]
    lines = ["| Model | " + " | ".join(groups) + " |", "|" + "---|" * (len(groups) + 1)]
    for v in variants:
        cells = []
        for rows in groups.values():
            values = [early_error(r, ticks) if ticks else r["error_while_active"] for r in rows.get((v, "own"), [])]
            mean, se, n = mean_se(values)
            cells.append("—" if mean is None else f"{mean:.4f} ± {se:.4f} ({n})")
        lines.append(f"| {STYLE[v][0]} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


LADDER = ("recurrent", "predict_hidden", "predict_only", "frozen", "fixed_learn", "full", "world_model", "full_meta", "meta_step")


def ladder(rows, out):
    """Error of every setting in its own mode: one point per seed, mean and standard error."""
    names = [v for v in LADDER if (v, "own") in rows]
    if not names:
        return
    fig, ax = plt.subplots(figsize=(5.6, 3.0))
    for i, v in enumerate(names):
        ys = np.array([r["error_while_active"] for r in rows[(v, "own")]])
        colour = STYLE[v][1]
        ax.scatter(np.full(len(ys), i) + np.linspace(-0.12, 0.12, len(ys)), ys, s=14, color=colour, alpha=0.55,
                   linewidths=0)
        mean, se, _ = mean_se(ys)
        ax.errorbar(i + 0.28, mean, yerr=se, color=colour, marker=STYLE[v][2], markersize=5, capsize=0, linewidth=1.5)
    style_axes(ax)
    ax.set_xticks(range(len(names)), [STYLE[v][0] for v in names], rotation=30, ha="right")
    ax.set_ylabel("tracking error while active")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def compute_figure(rows, cost, out):
    """Measured GPU time per instance and tick against error, for every setting evaluated."""
    keys = [k for v in STYLE for k in sorted(rows, key=lambda k: (k[1] != "own", k[1])) if k[0] == v]
    if not keys or not cost:
        return
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    for v, setting in keys:
        xs = [compute_per_tick(r, cost) for r in rows[(v, setting)]]
        ys = [r["error_while_active"] for r in rows[(v, setting)]]
        (mx, sx, _), (my, sy, _) = mean_se(xs), mean_se(ys)
        hollow = setting != "own"
        ax.errorbar(mx, my, xerr=sx, yerr=sy, color=STYLE[v][1], marker=STYLE[v][2], markersize=5, capsize=0,
                    linewidth=1.0, markerfacecolor="white" if hollow else STYLE[v][1], label=label(v, setting))
    style_axes(ax)
    ax.set_xlabel("GPU time per instance and tick (µs, batch 1,024)")
    ax.set_ylabel("tracking error while active")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False, fontsize=6.5)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--evaluations", type=Path, nargs="+", required=True)
    p.add_argument("--families", nargs="*", default=[], metavar="NAME=DIR[,DIR]",
                   help="also tabulate own-setting error per family from evaluation directories")
    p.add_argument("--family-ticks", type=int, help="compare families over the first ticks of each lifetime")
    p.add_argument("--runs", type=Path, help="training output root with <mode>/seed<k>/train_log.jsonl")
    p.add_argument("--latency", type=Path, help="latency.py output, to express activity in GPU time")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = collect(args.evaluations)
    cost = compute_cost(args.latency)
    stats, md = table(rows, cost)
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
    if args.families:
        groups = {name: collect(sorted(p for d in dirs.split(",") for p in Path(d).glob("*.json")))
                  for name, dirs in (f.split("=") for f in args.families)}
        md_families = families_table(groups, args.family_ticks)
        (args.output / "families.md").write_text(md_families + "\n")
        print(md_families)
    ladder(rows, args.output / "ladder.png")
    compute_figure(rows, cost, args.output / "compute.png")
    if args.runs:
        training_curves(args.runs, args.output / "training.png")


if __name__ == "__main__":
    main()
