"""Illustrate calibration rehearsal using recorded training measurements.

Example (from the repository root):
    python -m experiments.discoveryworld.plot_calibration_rehearsal

The default example is world 0, quantity axes 0 and 1, and the first target
object. Selection does not inspect native query outcomes or evaluation data.
The original training function fits each proposed relation and checks validity.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, StrMethodFormatter
import numpy as np
import torch

from .calibration import FEATURE_QUANTITIES
from .train_active_core import counterfactual_calibration


DATA = "runs/discoveryworld/active_experiments/data_large/train.npz"
QUANTITY_LABELS = {
    "density_g_cm3": ("Density", r"Density (g cm$^{-3}$)"),
    "temperature_c": ("Temperature", "Temperature (°C)"),
    "quantum_gap_nm": ("Quantum gap", "Quantum gap (nm)"),
}
BLUE, ORANGE, GRAY = "#0072B2", "#D55E00", "#747C84"


def make_example(path, source, world, quantities):
    with np.load(path, allow_pickle=False) as data:
        samples = torch.from_numpy(data["sample_values"][world:world + 1])
        present = torch.from_numpy(data["sample_present"][world:world + 1])
        values = (torch.where(present, samples, 0.).sum(3)
                  / present.sum(3).clamp_min(1)).sum(2)
        values[:, :, 2] = torch.where(present[:, :, 2, :, 2],
                                     samples[:, :, 2, :, 2], torch.inf).min(-1).values
        available = present.any(3).any(2)
        settings = torch.from_numpy(data["working_frequency_hz"][world:world + 1])
        supports = torch.from_numpy(data["initial_support_mask"][world:world + 1])
        targets = torch.from_numpy((data["target_mask"] & data["label_present"])[world:world + 1])
        query = int(torch.where(targets[0])[0][0])
        example = {
            "source_npz": source,
            "source_trajectory": str(data["source"][world]),
            "world_index": world,
            "selection": "Fixed training world and quantity axes; first target object. "
                         "Native query outcomes are not used to fit or select relations.",
            "support_slots": torch.where(supports[0])[0].tolist(),
            "target_slots": torch.where(targets[0])[0].tolist(),
            "plotted_query_slot": query,
            "original_working_frequency_hz": settings[0].tolist(),
            "original_query_settings_role": "Recorded for provenance only; not used in the fit.",
            "reading_summary": "Exact training summary: mean of available readings per instrument; "
                               "radiation uses the minimum of its recorded samples.",
            "formula": "a = (f_s2 - f_s1) / (x_s2 - x_s1); b = f_s1 - a*x_s1; "
                       "generated query label = a*x_query + b",
            "support_and_samples_changed": False,
            "relations": [],
        }
        for q in quantities:
            labels, accepted, details = counterfactual_calibration(
                values, available, settings, supports, targets, torch.tensor([q]))
            if not accepted.item():
                raise ValueError(f"World {world}, quantity {q} does not pass the training validity rule")
            example["relations"].append({
                "quantity_index": q,
                "quantity": FEATURE_QUANTITIES[q],
                "first_recorded_readings": data["x"][world, :, q].tolist(),
                "recorded_samples_by_object": [samples[0, obj, :, :, q][
                    present[0, obj, :, :, q]].tolist() for obj in range(4)],
                "training_reading_summary": values[0, :, q].tolist(),
                "slope": details["slope"].item(),
                "intercept": details["intercept"].item(),
                "accepted_by_training_function": bool(accepted.item()),
                "generated_settings_hz": labels[0].tolist(),
            })
    return example


def draw(example, output):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.labelsize": 10, "axes.titlesize": 11,
                         "xtick.labelsize": 9, "ytick.labelsize": 9,
                         "pdf.fonttype": 42, "svg.fonttype": "none"})
    fig, axes = plt.subplots(1, 2, figsize=(7.8, 4.7), sharey=True)
    fig.subplots_adjust(left=.105, right=.975, bottom=.27, top=.75, wspace=.23)
    supports = example["support_slots"]
    query = example["plotted_query_slot"]
    f1, f2 = [example["original_working_frequency_hz"][i] for i in supports]
    fig.suptitle("Alternative calibrations from the same observed supports", y=.975,
                 fontsize=13, fontweight="bold", color="#24292E")
    fig.text(.5, .89, f"Observed settings: S1 = {f1:,.2f} Hz; S2 = {f2:,.2f} Hz",
             ha="center", fontsize=10, color="#444B52")
    plotted_y = [f1, f2] + [r["generated_settings_hz"][query] for r in example["relations"]]
    ymin, ymax = min(plotted_y), max(plotted_y)
    padding = max(500., (ymax - ymin) * .16)
    for panel, (ax, relation) in enumerate(zip(axes, example["relations"])):
        quantity = relation["quantity"]
        title, xlabel = QUANTITY_LABELS.get(
            quantity, (quantity.replace("_", " ").capitalize(), "Recorded reading"))
        x = np.asarray(relation["training_reading_summary"])
        y = np.asarray(relation["generated_settings_hz"])
        used = supports + [query]
        lo, hi = x[used].min(), x[used].max()
        margin = (hi - lo) * .14
        line_x = np.linspace(lo - margin, hi + margin, 100)
        ax.plot(line_x, relation["slope"] * line_x + relation["intercept"],
                color=GRAY, linewidth=1.4, zorder=1)
        for index in supports:
            ax.axhline(y[index], color="#D4DADF", linewidth=.8, linestyle=(0, (3, 3)), zorder=0)
        ax.scatter(x[supports], y[supports], s=42, color=BLUE, edgecolor="white", linewidth=.6, zorder=3)
        ax.scatter(x[query], y[query], s=73, marker="D", facecolors="white",
                   edgecolors=ORANGE, linewidth=1.7, zorder=4)
        for number, index in enumerate(supports, start=1):
            ax.annotate(f"S{number}", (x[index], y[index]), xytext=(-8, 13),
                        textcoords="offset points", ha="center", fontsize=10, color=BLUE)
        query_offset = (24, 32) if x[query] < np.median(x[used]) else (-35, -34)
        ax.annotate("Query", (x[query], y[query]), xytext=query_offset,
                    textcoords="offset points", ha="center", fontsize=10, color=ORANGE,
                    arrowprops={"arrowstyle": "-", "color": ORANGE, "linewidth": .8})
        ax.text(.96, .07, f"Query setting\n{y[query]:,.2f} Hz", transform=ax.transAxes,
                ha="right", va="bottom", color=ORANGE, fontsize=10,
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 2})
        ax.set(title=f"{'AB'[panel]}  {title} hypothesis", xlabel=xlabel,
               xlim=(lo - margin, hi + margin), ylim=(max(0., ymin - padding), ymax + padding))
        ax.xaxis.set_major_locator(MaxNLocator(5))
        ax.yaxis.set_major_locator(MaxNLocator(5))
        ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color("#A4ABB2")
        ax.tick_params(color="#A4ABB2")
    axes[0].set_ylabel("Setting frequency (Hz)")
    legend = [Line2D([], [], marker="o", color="none", markerfacecolor=BLUE,
                     markeredgecolor=BLUE, markersize=6, label="Observed support setting"),
              Line2D([], [], marker="D", color="none", markerfacecolor="white",
                     markeredgecolor=ORANGE, markersize=6, label="Generated query setting"),
              Line2D([], [], color=GRAY, linewidth=1.4, label="Affine fit to supports")]
    fig.legend(handles=legend, loc="lower center", bbox_to_anchor=(.5, .105),
               ncol=3, frameon=False, fontsize=9, columnspacing=1.4, handlelength=1.6)
    fig.text(.5, .045, "Recorded measurements and support settings are retained; only query labels are generated.",
             ha="center", fontsize=9, color="#444B52")
    output.parent.mkdir(parents=True, exist_ok=True)
    for extension in ("pdf", "svg", "png"):
        fig.savefig(output.with_suffix(f".{extension}"), dpi=300, facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path(DATA))
    parser.add_argument("--source", default=DATA, help="Relative data source recorded in the example JSON")
    parser.add_argument("--world-index", type=int, default=0)
    parser.add_argument("--quantities", type=int, nargs=2, default=[0, 1])
    parser.add_argument("--output", type=Path, default=Path("experiments/discoveryworld/figures/calibration_rehearsal"))
    args = parser.parse_args()
    example = make_example(args.data, args.source, args.world_index, args.quantities)
    draw(example, args.output)
    args.output.with_suffix(".json").write_text(json.dumps(example, indent=2) + "\n")
    print(f"Wrote {args.output}.{{pdf,svg,png,json}}")


if __name__ == "__main__":
    main()
