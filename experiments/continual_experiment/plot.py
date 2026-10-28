"""Plot native quality, action cost and same-weight interventions.

Read summary.json produced by experiments.continual_experiment.summarize.
Only completed conditions are plotted; missing conditions mark the figure as
partial. Save vector PDF/SVG and a PNG preview without pooling training seeds.

python -m experiments.continual_experiment.plot --summary path/summary.json \
    --output runs/continual-experiment/figures/native_comparison
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from matplotlib.transforms import blended_transform_factory


BASELINES = {
    "baseline_parent": "Frozen parent",
    "baseline_rehearsal": "Calibration rehearsal",
    "baseline_explicit": "Explicit relation memory",
}
MODES = {
    "full": ("Full", "#0072B2"),
    "no_adaptation": ("No adaptation", "#D55E00"),
    "task_reset": ("Task-reset training", "#8C6BB1"),
    "fixed_schedule": ("Fixed activity schedule", "#009E73"),
}
INTERVENTIONS = ("full", "full_reset", "full_discard_updates")
INTERVENTION_LABELS = ("Full", "State reset", "Updates discarded")
MARKERS = ("o", "s", "^", "D")


def draw(summary, output):
    totals = {name: row["groups"]["all"] for name, row in summary["native"].items()}
    seeds = sorted({row["seed"] for row in summary["training"].values()})
    main_names = [*BASELINES, *(f"{mode}_seed{seed}" for seed in seeds for mode in MODES)]
    same_names = [f"{mode}_seed{seed}" for seed in seeds for mode in INTERVENTIONS]
    main = [name for name in main_names if name in totals]
    same = [name for name in same_names if name in totals]
    if not main:
        raise ValueError("The summary has no completed main comparison conditions")
    first = totals[main[0]]
    instances, streams = first["instances"], first["streams"]
    partial = len(main) < len(main_names) or len(same) < len(same_names)
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10, "axes.labelsize": 10,
        "xtick.labelsize": 9.5, "ytick.labelsize": 9.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#8A9096", "axes.linewidth": .7,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    })
    fig, (quality, intervention) = plt.subplots(
        1, 2, figsize=(12.8, 6.4), gridspec_kw={"width_ratios": [1.15, 1]})
    fig.subplots_adjust(left=.075, right=.89, top=.79, bottom=.31, wspace=.68)
    for ax in (quality, intervention):
        ax.set_axisbelow(True)
        ax.grid(axis="x", color="#E7E9EC", linewidth=.7)
        ax.xaxis.set_major_locator(MaxNLocator(5))
    offsets = {"baseline_parent": (16, 15), "baseline_rehearsal": (0, -19),
               "baseline_explicit": (0, 22)}
    for name in BASELINES:
        if name not in totals:
            continue
        row = totals[name]
        point = row["mean_native_actions"], row["completed"]
        quality.scatter(*point, marker="D", s=44, color="#50565C", zorder=3)
        quality.annotate(BASELINES[name], point, xytext=offsets[name],
                         textcoords="offset points",
                         ha="left" if name == "baseline_parent" else "center", fontsize=9,
                         color="#42474C", arrowprops={"arrowstyle": "-", "color": "#92989E", "lw": .6})
    for index, seed in enumerate(seeds):
        for mode, (_, color) in MODES.items():
            name = f"{mode}_seed{seed}"
            if name not in totals:
                continue
            row = totals[name]
            quality.scatter(row["mean_native_actions"], row["completed"],
                            marker=MARKERS[index % len(MARKERS)], s=64 if index == 0 else 85,
                            facecolors=color if index == 0 else "none",
                            edgecolors=color, linewidths=1.5, zorder=4)
    costs = [totals[name]["mean_native_actions"] for name in main]
    quality.set(xlim=(max(0, min(costs) - 20), max(costs) + 20),
                ylim=(min(totals[name]["completed"] for name in main) - 1.5, instances + .5),
                xlabel="Mean native actions per instance",
                ylabel=f"Tasks completed (of {instances})")
    quality.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
    quality.axhline(instances, color="#BFC4C9", linestyle=(0, (3, 3)), linewidth=.8)
    quality.set_title("a   Task quality and execution cost", loc="left", fontsize=12, pad=17)

    positions, labels = [], []
    for index, seed in enumerate(seeds):
        present = [(mode, label) for mode, label in zip(INTERVENTIONS, INTERVENTION_LABELS)
                   if f"{mode}_seed{seed}" in totals]
        if not present:
            continue
        start = (positions[-1] + 1.5) if positions else 0
        group_positions = [start + j for j in range(len(present))]
        group_costs = [totals[f"{mode}_seed{seed}"]["mean_native_actions"] for mode, _ in present]
        intervention.plot(group_costs, group_positions, color="#CBD1D6", linewidth=1, zorder=1)
        for y, (mode, label), cost in zip(group_positions, present, group_costs):
            row = totals[f"{mode}_seed{seed}"]
            color = "#0072B2" if mode == "full" else "#626970"
            intervention.scatter(cost, y, s=52, color=color, zorder=3)
            intervention.annotate(f"{cost:.1f}", (cost, y), xytext=(7, 0),
                                  textcoords="offset points", va="center", fontsize=9.5)
            intervention.text(1.065, y, f"{row['completed']} / {row['instances']}",
                              transform=blended_transform_factory(intervention.transAxes, intervention.transData),
                              va="center", fontsize=9.5)
            positions.append(y)
            labels.append(f"{label} · {seed}")
    intervention.set(yticks=positions, yticklabels=labels,
                     ylim=((positions[-1] + .75) if positions else 1, -.75),
                     xlim=(0, max([totals[name]["mean_native_actions"] for name in same], default=1) * 1.23),
                     xlabel="Mean native actions per instance")
    intervention.spines["left"].set_visible(False)
    intervention.tick_params(axis="y", length=0)
    intervention.text(1.065, 1.015, "Completed", transform=intervention.transAxes,
                      ha="left", va="bottom", fontsize=9.5)
    intervention.set_title("b   Same-weight interventions", loc="left", fontsize=12, pad=17)
    if not same:
        intervention.text(.5, .5, "No completed interventions", transform=intervention.transAxes,
                          ha="center", fontsize=10, color="#626970")

    handles = [Line2D([], [], color=color, marker="o", linestyle="none", markersize=6,
                      label=label) for label, color in MODES.values()]
    handles += [Line2D([], [], color="#50565C", marker=MARKERS[i % len(MARKERS)],
                       markerfacecolor="#50565C" if i == 0 else "none", linestyle="none",
                       markersize=6, label=f"Training seed {seed}") for i, seed in enumerate(seeds)]
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(.067, .105), ncol=3,
               frameon=False, fontsize=9.5, columnspacing=2.6, handletextpad=.7)
    fig.suptitle("Continual experimental decisions", x=.075, y=.965,
                 ha="left", fontsize=14, fontweight="bold")
    status = "Partial results · " if partial else ""
    fig.text(.075, .9, f"{status}{instances} instances · {streams} ordered task streams · "
             f"{len(seeds)} training repetitions", fontsize=10, color="#626970")
    fig.text(.075, .065, f"{instances} instances in {streams} ordered streams; training repetitions use the same instances and are shown separately.", fontsize=9)
    fig.text(.075, .03, "Action means include successes and failures. Points show observed means; paired uncertainty is reported in the result tables.", fontsize=9)
    output.parent.mkdir(parents=True, exist_ok=True)
    for extension in ("pdf", "svg", "png"):
        fig.savefig(output.with_suffix(f".{extension}"), dpi=250, bbox_inches="tight", pad_inches=.15)
    plt.close(fig)
    print(json.dumps({"main_conditions": len(main), "same_weight_conditions": len(same),
                      "partial": partial, "output": str(output)}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Common output prefix")
    args = parser.parse_args()
    draw(json.loads(args.summary.read_text()), args.output)


if __name__ == "__main__":
    main()
