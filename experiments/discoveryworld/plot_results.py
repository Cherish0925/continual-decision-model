"""Plot actual native-task progress from a compact results file."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Output prefix")
    args = parser.parse_args()
    data = json.loads(args.results.read_text())
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    labels = {"cdm": "CDM", "gru": "Recurrent policy",
              "cdm_reset_workspace": "CDM: workspace reset"}
    colors = {"cdm": "#23689b", "gru": "#60666d", "cdm_reset_workspace": "#9cb9cb"}
    palette = ["#23689b", "#60666d", "#4a8b7d", "#b17a39", "#887fa3", "#9cb9cb"]
    labels.update({key: model.get("label", labels.get(key, key))
                   for key, model in data["models"].items()})
    for index, (key, model) in enumerate(data["models"].items()):
        colors[key] = model.get("color", colors.get(key, palette[index % len(palette)]))
    fig, axes = plt.subplots(1, 2, figsize=(9.2, max(3.1, .45 * len(data["models"]) + 1.2)),
                             gridspec_kw={"width_ratios": [1.3, 1]})
    for index, (key, model) in enumerate(data["models"].items()):
        # Completed trajectories may end early. Retain their terminal score
        # for the remaining action budget when displaying paired mean curves.
        curves = np.asarray([np.pad(row, (0, data["max_actions"] + 1 - len(row)), mode="edge")
                             for row in model["progress_by_action"]])
        axes[0].plot(np.arange(curves.shape[1]), curves.mean(0), color=colors[key],
                     label=labels[key], lw=1.7)
        points = model["endpoint_progress"]
        axes[1].scatter(points, index + np.linspace(-.1, .1, len(points)),
                        color=colors[key], s=28, edgecolors="white", linewidths=.4)
    axes[0].set(xlabel="Executed actions", ylabel="Official task progress",
                xlim=(0, data["max_actions"]), ylim=(0, 1))
    axes[0].legend(frameon=False, loc="upper left", fontsize=8)
    axes[0].set_title("Mean progress during execution", loc="left", fontsize=10)
    axes[1].set(xlabel=f"Official task progress at {data['max_actions']} actions", xlim=(0, 1),
                yticks=range(len(data["models"])),
                yticklabels=[labels[k] for k in data["models"]])
    axes[1].invert_yaxis()
    axes[1].set_title(f"{len(data['seeds'])} paired new instances", loc="left", fontsize=10)
    for ax in axes:
        ax.grid(axis="x", color="#e8ebee", linewidth=.6)
        ax.set_axisbelow(True)
    fig.tight_layout(w_pad=2)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(args.output.with_suffix(".png"), dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
