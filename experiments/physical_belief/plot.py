"""Plot complete native comparisons, keeping independent parents separate."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.transforms import blended_transform_factory


def draw(summary, output):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.spines.left": False, "pdf.fonttype": 42, "svg.fonttype": "none"})
    conditions = summary["conditions"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4))
    fig.subplots_adjust(left=.17, right=.91, bottom=.21, top=.77, wspace=.85)
    labels = ["Frozen parent", "Ordinary continuation", "Task-reset physical learning",
              "Physical memory", "Same weights: memory reset", "Same weights: updates discarded"]
    colors = ["#626970", "#D55E00", "#8C6BB1", "#0072B2", "#626970", "#626970"]
    for ax, seed, parent, letter in zip(axes, (221, 222), (183, 184), ("A", "B")):
        names = [f"baseline_rehearsal_{parent}", f"ordinary_seed{seed}", f"reset_seed{seed}",
                 f"carry_seed{seed}", f"carry_seed{seed}_memory_reset", f"carry_seed{seed}_memory_discard"]
        for y, (name, color) in enumerate(zip(names, colors)):
            row = conditions[name]
            if y == 0:
                delta, lower, upper = 0., 0., 0.
            else:
                comparison = summary["specified_pairs"].get(f"{name}:{names[0]}")
                if comparison is None:
                    if summary["reference"] != names[0]:
                        raise ValueError(f"Missing matched parent comparison for {name}")
                    comparison = summary["comparisons"][name]
                stat = comparison["all"]
                delta = stat["mean_action_difference"]
                lower, upper = stat["mean_action_difference_95ci"]
            ax.errorbar(delta, y, xerr=[[delta-lower], [upper-delta]], fmt="o", color=color,
                        capsize=3, markersize=5, linewidth=1.4, zorder=3)
            ax.text(1.04, y, f"{row['completed']}/{row['instances']}", va="center",
                    transform=blended_transform_factory(ax.transAxes, ax.transData))
        ax.axvline(0, color="#BFC4C9", lw=1, zorder=0)
        ax.grid(axis="x", color="#E7E9EC", linewidth=.7)
        ax.set(yticks=range(len(labels)), yticklabels=labels, ylim=(5.65, -.65),
               xlabel="Native actions relative to frozen parent")
        ax.tick_params(axis="y", length=0)
        ax.set_title(f"Parent {letter} · independent training", loc="left", fontsize=11, pad=17)
        ax.text(1.04, 1.04, "Completed", transform=ax.transAxes, fontsize=9)
    limits = [ax.get_xlim() for ax in axes]
    common = min(x[0] for x in limits), max(x[1] for x in limits)
    for ax in axes: ax.set_xlim(*common)
    fig.suptitle("Physical knowledge and experimental decisions", x=.17, ha="left", fontsize=15)
    fig.text(.17, .095, "120 new native instances; all action costs include failed tasks.", fontsize=9)
    fig.text(.17, .045, "Intervals resample 20 complete task streams. Negative differences indicate fewer actions.", fontsize=9)
    output.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("svg", "pdf", "png"):
        fig.savefig(output.with_suffix('.'+ext), dpi=220, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--summary", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    draw(json.loads(args.summary.read_text()), args.output)
