#!/usr/bin/env python3
"""Aggregate evaluation files across seeds into tables and figures."""
from __future__ import annotations

import argparse
import json
import warnings
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

warnings.filterwarnings("ignore", message="Mean of empty slice")
warnings.filterwarnings("ignore", message="Degrees of freedom")

# Fixed entity -> (label, colour, marker, dash). Hue marks a family (restart, carry, single workspace with
# a goal read, stored workspaces, learned memory, factorized); members differ by dash and hollow markers.
# The hues follow the validated categorical order.
DASHED, DOTTED, DASHDOT = (0, (4, 2)), (0, (1, 1.5)), (0, (5, 1.5, 1, 1.5))
STYLE = {
    "restart": ("Restart", "#eb6834", "s", None),
    "goal_restart": ("Restart on goal moves", "#eb6834", "s", DASHED),
    "carry": ("Carry", "#2a78d6", "o", None),
    "carry_static": ("Carry, static training", "#2a78d6", "o", DASHED),
    "carry_read": ("Carry with goal read", "#008300", "d", None),
    "carry_gate": ("Goal read and learned rebuild gate", "#008300", "d", DASHED),
    "snapshot": ("Exact snapshot", "#1baf7a", "^", None),
    "nearest": ("Nearest context", "#eda100", "D", None),
    "memory_global": ("Learned memory, whole map", "#e87ba4", "v", None),
    "memory_local": ("Learned memory, per cell", "#e87ba4", "v", DASHED),
    "factorized": ("Factorized by premise", "#4a3aa7", "*", None),
    "factorized_local": ("Factorized, no goal read", "#4a3aa7", "*", DASHED),
    "factorized_carry": ("Factorized, no rebuild", "#4a3aa7", "*", DOTTED),
    "factorized_memory": ("Factorized with map records", "#e34948", "h", None),
    "partial_read": ("Goal read, partial view", "#008300", "d", DOTTED),
    "partial_read_age": ("Goal read, partial view, age input", "#008300", "d", DASHDOT),
    "partial_factorized_age": ("Factorized, partial view, age input", "#4a3aa7", "*", DASHDOT),
}
MAIN = ("restart", "goal_restart", "carry", "carry_read", "carry_gate", "factorized", "factorized_local",
        "factorized_carry")
STORED = ("restart", "carry", "snapshot", "nearest", "memory_global", "memory_local", "factorized", "factorized_memory")
REGIME_NAMES = {"full_return": "Full return", "partial_return": "Partial return", "continuous": "Continuous change"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e0"
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "legend.fontsize": 7.5,
                     "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "axes.titlecolor": INK,
                     "axes.labelcolor": MUTED, "pdf.fonttype": 42})


def load(root: Path):
    data = defaultdict(list)   # variant -> list of seed result dicts
    for path in sorted(root.glob("*/seed*.json")):
        data[path.parent.name].append(json.loads(path.read_text()))
    oracle = json.loads((root / "oracle.json").read_text()) if (root / "oracle.json").exists() else None
    return data, oracle


def mean_se(values):
    v = np.asarray(values, dtype=float)
    return float(np.mean(v)), (float(np.std(v, ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0), len(v)


def table(data):
    """Per-seed means aggregated across seeds.

    stats[(variant, rooms, regime, think)][metric] = (mean, se, seeds)
    curve[(variant, rooms, regime, think, group, field)] = (mean[S], se[S], cells[S])
    """
    rows = defaultdict(lambda: defaultdict(list))
    curves = defaultdict(list)
    for variant, seeds in data.items():
        for result in seeds:
            for r in result["runs"]:
                key = (variant, r["rooms"], r["regime"], r["think"])
                for m in ("agent_optimal", "map_accuracy", "success", "return"):
                    rows[key][m].append(r[m])
                for group, c in r["curves"].items():
                    for field in ("map", "agent", "affected", "affected_cells"):
                        if field in c:
                            curves[key + (group, field)].append(np.array(c[field], dtype=float))
    stats = {k: {m: mean_se(v) for m, v in metrics.items()} for k, metrics in rows.items()}
    curve = {}
    for k, v in curves.items():
        v = np.array(v)
        se = np.nanstd(v, 0, ddof=1) / np.sqrt(len(v)) if len(v) > 1 else np.zeros(v.shape[1])
        curve[k] = (np.nanmean(v, 0), se)
    return stats, curve


def style_axes(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
        ax.spines[side].set_linewidth(0.6)
    ax.tick_params(colors=MUTED, width=0.6)
    ax.grid(True, color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)


ERROR_FLOOR = 0.002   # percent; drawn for measurements with no non-optimal action


def log_axis(ax):
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(matplotlib.ticker.LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda y, _: f"{y:g}"))
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())


def line_kw(v, size=3.6):
    name, colour, marker, dash = STYLE[v]
    return dict(color=colour, marker=marker, markersize=size * (1.35 if marker == "*" else 1.0),
                linewidth=1.5, linestyle=dash or "-",
                markerfacecolor="white" if dash else colour, markeredgecolor=colour, markeredgewidth=0.9,
                label=name)


def legend_handles(variants, lines=True):
    handles = []
    for v in variants:
        kw = line_kw(v)
        if not lines:
            kw["linestyle"] = "none"
        handles.append(Line2D([], [], **kw))
    return handles


def save(fig, out: Path, name: str):
    fig.savefig(out / f"{name}.png", dpi=220, bbox_inches="tight")
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def present(variants, order=tuple(STYLE)):
    return [v for v in order if v in variants]


def figure_premise(curve, variants, out, rooms=5, think=1, horizon=13):
    """Recovery of the cells whose optimal action changed, by the premise that changed."""
    panels = [("partial_return", "kind:goal", "Goal moves, map unchanged"),
              ("partial_return", "kind:door", "Doors change, goal unchanged"),
              ("partial_return", "kind:goal+map", "Goal and map change together")]
    vs = present(variants, MAIN)
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.0), sharey=True)
    for ax, (regime, group, title) in zip(axes, panels):
        cells = None
        for v in vs:
            key = (v, rooms, regime, think, group, "affected")
            if key not in curve:
                continue
            mean, se = curve[key]
            x = np.arange(horizon)
            ok = np.isfinite(mean[:horizon])
            ax.plot(x[ok], 100 * mean[:horizon][ok], **line_kw(v))
            ax.fill_between(x[ok], 100 * (mean - se)[:horizon][ok], 100 * (mean + se)[:horizon][ok],
                            color=STYLE[v][1], alpha=0.12, linewidth=0)
            cells = curve.get((v, rooms, regime, think, group, "affected_cells"), (None,))[0]
        subtitle = f"\n{cells[0]:.0f} cells change their optimal action" if cells is not None else ""
        ax.set_title(title + subtitle, fontsize=8.5)
        ax.set_xticks(range(0, horizon, 2))
        ax.set_xlabel("ticks after the change")
        style_axes(ax)
    axes[0].set_ylabel("affected cells with an optimal action (%)")
    fig.legend(handles=legend_handles(vs), loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.98))
    save(fig, out, "premise")


def figure_stored(curve, variants, out, rooms=5, think=1):
    """Accuracy on affected cells at the change tick, for contexts seen before and new ones."""
    rows = [("full_return", "returned", "full return, seen before"),
            ("partial_return", "returned", "partial return, seen before"),
            ("partial_return", "novel", "partial return, new"),
            ("continuous", "returned", "continuous change, seen before"),
            ("continuous", "novel", "continuous change, new")]
    vs = present(variants, STORED)
    fig, ax = plt.subplots(figsize=(7.0, 3.2))
    for ri, (regime, group, label) in enumerate(rows):
        for vi, v in enumerate(vs):
            key = (v, rooms, regime, think, group, "affected")
            if key not in curve:
                continue
            mean, se = curve[key]
            y = ri + (vi - (len(vs) - 1) / 2) * 0.1
            kw = line_kw(v, size=4.6)
            kw.pop("linestyle")
            ax.errorbar(100 * mean[0], y, xerr=100 * se[0], linestyle="none", capsize=1.5, elinewidth=0.8, **kw)
    ax.set_yticks(range(len(rows)), [r[2] for r in rows])
    ax.invert_yaxis()
    ax.set_xlabel("affected cells with an optimal action at the change tick (%)")
    style_axes(ax)
    ax.grid(False, axis="y")
    ax.legend(handles=legend_handles(vs, lines=False), frameon=False, loc="center left", bbox_to_anchor=(1.0, 0.5))
    save(fig, out, "stored")


def figure_budget(stats, variants, out, order=MAIN, rooms_list=(5, 9), thinks=(1, 2, 4), name="budget"):
    vs = present(variants, order)
    fig, axes = plt.subplots(len(rooms_list), 3, figsize=(10.2, 2.7 * len(rooms_list)), sharex=True)
    axes = np.atleast_2d(axes)
    for i, rooms in enumerate(rooms_list):
        for j, regime in enumerate(REGIME_NAMES):
            ax = axes[i, j]
            for v in vs:
                pts = [(k, stats[(v, rooms, regime, k)]["agent_optimal"]) for k in thinks if (v, rooms, regime, k) in stats]
                if not pts:
                    continue
                ax.errorbar([p[0] for p in pts], [max(ERROR_FLOOR, 100 * (1 - p[1][0])) for p in pts],
                            yerr=[100 * p[1][1] for p in pts], capsize=1.5, elinewidth=0.8, **line_kw(v))
            log_axis(ax)
            ax.set_xscale("log", base=2)
            ax.set_xticks(thinks, [str(k) for k in thinks])
            ax.minorticks_off()
            ax.set_title(f"{REGIME_NAMES[regime]}, {rooms}x{rooms} rooms")
            style_axes(ax)
            if j == 0:
                ax.set_ylabel("non-optimal agent actions (%)")
            if i == len(rooms_list) - 1:
                ax.set_xlabel("refinements per tick")
    fig.tight_layout()
    fig.legend(handles=legend_handles(vs), loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.0))
    save(fig, out, name)


def figure_transfer(stats, variants, out, think=1, rooms_list=(5, 7, 9), order=MAIN):
    vs = present(variants, order)
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 2.9), sharey=True)
    for ax, regime in zip(axes, REGIME_NAMES):
        for v in vs:
            pts = [(r, stats[(v, r, regime, think)]["agent_optimal"]) for r in rooms_list if (v, r, regime, think) in stats]
            if not pts:
                continue
            ax.errorbar([p[0] for p in pts], [max(ERROR_FLOOR, 100 * (1 - p[1][0])) for p in pts],
                        yerr=[100 * p[1][1] for p in pts], capsize=1.5, elinewidth=0.8, **line_kw(v))
        log_axis(ax)
        ax.set_xticks(rooms_list, [f"{r}x{r}" for r in rooms_list])
        ax.set_title(REGIME_NAMES[regime])
        ax.set_xlabel("rooms (trained on 5x5)")
        style_axes(ax)
    axes[0].set_ylabel("non-optimal agent actions (%)")
    fig.legend(handles=legend_handles(vs), loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.98))
    save(fig, out, f"transfer_k{think}")


def fmt(v, scale=100, digits=1):
    m, se, _ = v
    return f"{scale * m:.{digits}f} ± {scale * se:.{digits}f}"


PROBE_TITLES = {"goal": "Goal moves", "doors": "Three doors toggle", "goal+doors": "Goal moves and three doors toggle"}


def figure_probe(files, out, order=MAIN):
    """Controlled single changes from converged computation; returns table lines."""
    rows = defaultdict(list)
    cells = {}
    for path in files:
        for r in json.loads(path.read_text())["runs"]:
            v = Path(r["model"]).parent.parent.name
            key = (v, r.get("rooms", 5), r["change"])
            rows[key].append(np.array(r["affected"]))
            cells[key[1:]] = r["affected_cells"]
    if not rows:
        return []
    vs = present({k[0] for k in rows}, order)
    sizes = sorted({k[1] for k in rows})
    fig, axes = plt.subplots(len(sizes), 3, figsize=(10.2, 2.9 * len(sizes)), sharey=True, squeeze=False)
    for i, rooms in enumerate(sizes):
        for ax, change in zip(axes[i], PROBE_TITLES):
            for v in vs:
                if (v, rooms, change) not in rows:
                    continue
                a = np.array(rows[(v, rooms, change)])
                mean = a.mean(0)
                se = a.std(0, ddof=1) / np.sqrt(len(a)) if len(a) > 1 else np.zeros_like(mean)
                x = np.arange(len(mean))
                ax.plot(x, 100 * mean, **line_kw(v))
                ax.fill_between(x, 100 * (mean - se), 100 * (mean + se), color=STYLE[v][1], alpha=0.12, linewidth=0)
            ax.set_title(f"{PROBE_TITLES[change]}, {rooms}x{rooms} rooms\n"
                         f"{cells.get((rooms, change), float('nan')):.0f} cells change their optimal action")
            ax.set_xticks(range(0, 9, 2))
            style_axes(ax)
            if i == len(sizes) - 1:
                ax.set_xlabel("refinements after the change")
        axes[i, 0].set_ylabel("affected cells with an optimal action (%)")
    fig.tight_layout()
    fig.legend(handles=legend_handles(vs), loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.0))
    save(fig, out / "figures", "probe")
    lines = []
    for rooms in sizes:
        lines += [f"### One change from converged computation: affected cells with an optimal action (%), {rooms}x{rooms} rooms",
                  "", "| Method | " + " | ".join(f"{PROBE_TITLES[c].lower()}, r={r}" for c in PROBE_TITLES for r in (0, 1, 2, 4))
                  + " |", "|---|" + "---:|" * 12]
        for v in vs:
            out_cells = []
            for c in PROBE_TITLES:
                a = np.array(rows.get((v, rooms, c), []))
                for r in (0, 1, 2, 4):
                    if len(a):
                        se = a[:, r].std(ddof=1) / np.sqrt(len(a)) if len(a) > 1 else 0.0
                        out_cells.append(f"{100 * a[:, r].mean():.1f} ± {100 * se:.1f}")
                    else:
                        out_cells.append("")
            lines.append(f"| {STYLE[v][0]} | " + " | ".join(out_cells) + " |")
        lines.append("")
    return lines


RULES = {  # rule -> (label, colour, marker, linestyle)
    "uniform": ("Same for every agent", "#52514e", "o", "-"),
    "random": ("Random agents", "#898781", "o", (0, (4, 2))),
    "change": ("Most recent change", "#1baf7a", "^", "-"),
    "confidence": ("Lowest confidence", "#eda100", "D", "-"),
    "settling": ("Largest last step", "#e34948", "*", "-"),
    "oracle": ("Wrong action (labels)", "#0b0b0b", "x", ":"),
}


def allocation(files, out, rooms=5):
    """Shared-budget allocation: tables for every model and a figure for the first two."""
    rows = defaultdict(lambda: defaultdict(list))
    for path in files:
        for r in json.loads(path.read_text())["runs"]:
            variant = Path(r["model"]).parent.parent.name
            key = (variant, r["rooms"], r["regime"], r["budget"], r["rule"])
            for m in ("agent_optimal", "return", "success", "ms_per_tick"):
                rows[key][m].append(r[m])
    stats = {k: {m: mean_se(v) for m, v in d.items()} for k, d in rows.items()}
    variants = [v for v in STYLE if any(k[0] == v for k in stats)]
    if not variants:
        return []
    shown = variants[:2]
    fig, axes = plt.subplots(len(shown), 3, figsize=(10.2, 2.7 * len(shown)), squeeze=False)
    for i, v in enumerate(shown):
        for j, regime in enumerate(REGIME_NAMES):
            ax = axes[i, j]
            for rule, (label, colour, marker, ls) in RULES.items():
                pts = sorted((k[3], val["agent_optimal"]) for k, val in stats.items()
                             if k[0] == v and k[1] == rooms and k[2] == regime and k[4] == rule)
                if not pts:
                    continue
                ax.errorbar([b for b, _ in pts], [max(ERROR_FLOOR, 100 * (1 - m[0])) for _, m in pts],
                            yerr=[100 * m[1] for _, m in pts], color=colour, marker=marker, linestyle=ls,
                            markersize=4.5 if marker == "*" else 3.6, linewidth=1.4, capsize=1.5, elinewidth=0.8,
                            markerfacecolor="white" if rule == "random" else colour, markeredgecolor=colour, label=label)
            log_axis(ax)
            ax.set_title(f"{STYLE[v][0]}, {REGIME_NAMES[regime].lower()}")
            style_axes(ax)
            if j == 0:
                ax.set_ylabel("non-optimal agent actions (%)")
            if i == len(shown) - 1:
                ax.set_xlabel("mean refinements per agent per tick")
    handles = [Line2D([], [], color=c, marker=m, linestyle=ls, markerfacecolor="white" if r == "random" else c,
                      markeredgecolor=c, label=l) for r, (l, c, m, ls) in RULES.items()]
    fig.tight_layout()
    fig.legend(handles=handles, loc="lower center", ncol=len(RULES), frameon=False, bbox_to_anchor=(0.5, 1.0))
    save(fig, out / "figures", "allocation")
    lines = []
    for v in variants:
        for rm in sorted({k[1] for k in stats if k[0] == v}):
            budgets = sorted({k[3] for k in stats if k[0] == v and k[1] == rm})
            lines += [f"### Shared budget, {STYLE[v][0].lower()}, {rm}x{rm} rooms: optimal agent actions (%)", "",
                      "| Rule | " + " | ".join(f"{REGIME_NAMES[r]} {b:g}" for r in REGIME_NAMES for b in budgets) + " |",
                      "|---|" + "---:|" * (3 * len(budgets))]
            for rule in RULES:
                cells = [fmt(stats[(v, rm, r, b, rule)]["agent_optimal"], digits=2) if (v, rm, r, b, rule) in stats
                         else "" for r in REGIME_NAMES for b in budgets]
                if any(cells):
                    lines.append(f"| {RULES[rule][0]} | " + " | ".join(cells) + " |")
            lines.append("")
    return lines


def markdown(stats, curve, variants, oracle, latency, out):
    vs = present(variants)
    lines = []
    for rooms in (5, 7, 9):
        if not any((v, rooms, "continuous", 1) in stats for v in vs):
            continue
        lines += [f"### Optimal agent actions (%), {rooms}x{rooms} rooms", "",
                  "| Method | " + " | ".join(f"{REGIME_NAMES[r]} k={k}" for r in REGIME_NAMES for k in (1, 2, 4)) + " |",
                  "|---|" + "---:|" * 9]
        for v in vs:
            cells = [fmt(stats[(v, rooms, r, k)]["agent_optimal"]) if (v, rooms, r, k) in stats else ""
                     for r in REGIME_NAMES for k in (1, 2, 4)]
            lines.append(f"| {STYLE[v][0]} | " + " | ".join(cells) + " |")
        lines.append("")
    o = {(r["rooms"], r["regime"]): r for r in oracle["runs"]} if oracle else {}
    for rooms in (5, 9):
        lines += [f"### Return per episode, {rooms}x{rooms} rooms (optimal policy in the last row)", "",
                  "| Method | " + " | ".join(f"{REGIME_NAMES[r]} k={k}" for r in REGIME_NAMES for k in (1, 4)) + " |",
                  "|---|" + "---:|" * 6]
        for v in vs:
            cells = [fmt(stats[(v, rooms, r, k)]["return"], scale=1, digits=2) if (v, rooms, r, k) in stats else ""
                     for r in REGIME_NAMES for k in (1, 4)]
            lines.append(f"| {STYLE[v][0]} | " + " | ".join(cells) + " |")
        if o:
            lines.append("| Optimal policy | " + " | ".join(f"{o[(rooms, r)]['return']:.2f}" if (rooms, r) in o else ""
                                                          for r in REGIME_NAMES for _ in (1, 4)) + " |")
        lines.append("")
    groups = [("partial_return", "kind:goal", "goal moves"), ("partial_return", "kind:door", "doors change"),
              ("partial_return", "kind:goal+map", "goal and map"), ("full_return", "returned", "full return"),
              ("continuous", "kind:goal", "continuous, goal"), ("continuous", "kind:door", "continuous, door")]
    lines += ["### Affected cells with an optimal action at the change tick → two ticks later (%), 5x5 rooms, k=1", "",
              "| Method | " + " | ".join(g[2] for g in groups) + " |", "|---|" + "---:|" * len(groups)]
    for v in vs:
        cells = []
        for regime, g, _ in groups:
            key = (v, 5, regime, 1, g, "affected")
            cells.append(f"{100 * curve[key][0][0]:.1f} → {100 * curve[key][0][2]:.1f}" if key in curve else "")
        lines.append(f"| {STYLE[v][0]} | " + " | ".join(cells) + " |")
    any_v = next((v for v in vs if (v, 5, "partial_return", 1, "kind:goal", "affected_cells") in curve), None)
    if any_v:
        counts = [curve.get((any_v, 5, regime, 1, g, "affected_cells"), (np.array([np.nan]),))[0][0] for regime, g, _ in groups]
        lines.append("| Affected cells per change | " + " | ".join(f"{c:.1f}" for c in counts) + " |")
    lines.append("")
    if latency:
        lines += ["### Decision latency on one H100 (ms per tick, mean)", "",
                  "| Method | 1 agent, k=1 | 1 agent, k=4 | 256 agents, k=1 | 256 agents, k=4 |", "|---|---:|---:|---:|---:|"]
        for v in vs:
            if latency.get(v):
                L = latency[v]
                lines.append(f"| {STYLE[v][0]} | " + " | ".join(f"{L.get((b, k), float('nan')):.2f}"
                                                              for b, k in ((1, 1), (1, 4), (256, 1), (256, 4))) + " |")
        lines.append("")
    (out / "tables.md").write_text("\n".join(lines))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--eval-root", type=Path, required=True)
    p.add_argument("--latency", type=Path, nargs="*", default=[], help="files written by evaluate.py --latency")
    p.add_argument("--allocation", type=Path, nargs="*", default=[], help="files written by allocate.py")
    p.add_argument("--probe", type=Path, nargs="*", default=[], help="files written by probe.py")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)
    data, oracle = load(args.eval_root)
    stats, curve = table(data)
    figures = args.out / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    variants = set(data)
    latency = defaultdict(dict)
    acc = defaultdict(lambda: defaultdict(list))
    sources = [json.loads(p.read_text()) for p in args.latency] or [r for seeds in data.values() for r in seeds]
    for result in sources:
        for row in result.get("latency", []):
            v = Path(row["model"]).parent.parent.name
            if row["rooms"] == 5:
                acc[v][(row["batch"], row["think"])].append(row["mean_ms"])
    for v, d in acc.items():
        latency[v] = {k: float(np.mean(x)) for k, x in d.items()}
    figure_premise(curve, variants, figures)
    figure_stored(curve, variants, figures)
    figure_budget(stats, variants, figures)
    figure_budget(stats, variants, figures, order=STORED, name="budget_stored")
    for k in (1, 4):
        figure_transfer(stats, variants, figures, think=k)
    markdown(stats, curve, variants, oracle, latency, args.out)
    extra = figure_probe(args.probe, args.out) + allocation(args.allocation, args.out)
    if extra:
        with (args.out / "tables.md").open("a") as f:
            f.write("\n".join(extra) + "\n")
    summary = {"metrics": [{"variant": k[0], "rooms": k[1], "regime": k[2], "think": k[3],
                             **{m: {"mean": v[0], "se": v[1], "seeds": v[2]} for m, v in vals.items()}}
                            for k, vals in sorted(stats.items())],
               "recovery": [{"variant": k[0], "rooms": k[1], "regime": k[2], "think": k[3], "group": k[4],
                             "field": k[5], "mean": [None if not np.isfinite(x) else round(float(x), 5) for x in m]}
                            for k, (m, _) in sorted(curve.items()) if k[3] == 1 and k[1] == 5],
               "latency_ms": {v: {f"batch{b}_k{k}": t for (b, k), t in L.items()} for v, L in latency.items()},
               "oracle": oracle["runs"] if oracle else None}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1))
    seeds = [{"variant": v, "seed": i, "rooms": r["rooms"], "regime": r["regime"], "think": r["think"],
              **{m: round(r[m], 6) for m in ("agent_optimal", "map_accuracy", "success", "return")}}
             for v, results in sorted(data.items()) for i, result in enumerate(results) for r in result["runs"]]
    (args.out / "eval_seeds.json").write_text(json.dumps(seeds, indent=0))


if __name__ == "__main__":
    main()
