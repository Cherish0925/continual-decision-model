"""Plot completed native task-stream comparisons from their three summary CSVs.

Counts and paired instances are read from the supplied data. Use --preview to
label layout work with completed subsets. Intervals are read from the paired
whole-stream bootstrap, not recomputed from individual episodes. Outputs are
editable SVG, vector PDF and 300-dpi PNG.
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.transforms import blended_transform_factory
import numpy as np


METHODS = ("carry", "taskreset", "resettrained", "parent", "relation_memory", "relation_stateless")
LABELS = {
    "carry": "CDM: carried state", "taskreset": "Same weights: task reset",
    "resettrained": "CDM: reset-trained", "parent": "Frozen parent",
    "relation_memory": "Explicit relation memory", "relation_stateless": "Explicit relation reset",
}
COLORS = {
    "carry": "#0072B2", "taskreset": "#56B4E9", "resettrained": "#626870",
    "parent": "#A8ADB3", "relation_memory": "#D55E00", "relation_stateless": "#E69F00",
}
GROUPS = ("initial_first", "initial_subsequent", "switch_first", "switch_subsequent",
          "return_first", "return_subsequent")
GROUP_LABELS = ("Initial A: first", "Initial A: later", "Switch B: first",
                "Switch B: later", "Return A: first", "Return A: later")
GROUP_COLORS = ("#0072B2", "#0072B2", "#D55E00", "#D55E00", "#009E73", "#009E73")


def read_csv(path):
    with path.open(newline="") as source:
        return list(csv.DictReader(source))


def load_comparison(directory, arms, candidate, reference):
    instances = read_csv(directory / "instances.csv")
    summaries = read_csv(directory / "summary.csv")
    pairs = read_csv(directory / "paired.csv")
    totals = {row["arm"]: row for row in summaries if row["group"] == "all"}
    if arms is None:
        arms = [arm for arm in METHODS if arm in totals] + [arm for arm in totals if arm not in METHODS]
    missing = set(arms) - totals.keys()
    if missing:
        raise ValueError(f"{directory}: missing completed methods {sorted(missing)}")
    identities = None
    for arm in arms:
        selected = [row for row in instances if row["arm"] == arm]
        keys = {(row["stream"], row["episode_index"], row["scenario_seed"]) for row in selected}
        total = totals[arm]
        if len(keys) != len(selected) or len(selected) != int(total["instances"]):
            raise ValueError(f"{directory}/{arm}: instance CSV and summary disagree")
        if sum(int(row["completed"]) for row in selected) != int(total["completed"]):
            raise ValueError(f"{directory}/{arm}: completion counts disagree")
        actual_mean = np.mean([float(row["native_actions"]) for row in selected])
        if not math.isclose(actual_mean, float(total["mean_native_actions"]), abs_tol=1e-8):
            raise ValueError(f"{directory}/{arm}: action counts disagree")
        if identities is not None and keys != identities:
            raise ValueError(f"{directory}: plotted arms use different scenarios")
        identities = keys
        if len({row["stream"] for row in selected}) != int(total["streams"]):
            raise ValueError(f"{directory}/{arm}: stream count disagrees")
    effects = {row["group"]: row for row in pairs
               if row["candidate"] == candidate and row["reference"] == reference}
    if set(GROUPS) - effects.keys():
        raise ValueError(f"{directory}: missing grouped {candidate} versus {reference} comparison")
    repetitions = {row["repetition"] for row in summaries}
    if len(repetitions) != 1:
        raise ValueError("Each comparison directory must contain one training repetition")
    return {"totals": totals, "effects": effects, "arms": arms,
            "identities": identities, "repetition": repetitions.pop()}


def method_panel(ax, data, labels, title, xmax):
    arms = data["arms"]
    ys = np.arange(len(arms))
    for y, arm in zip(ys, arms):
        row = data["totals"][arm]
        mean = float(row["mean_native_actions"])
        color = COLORS.get(arm, "#626870")
        ax.hlines(y, 0, mean, color=color, alpha=.35, linewidth=1.5)
        ax.scatter(mean, y, s=47, color=color, edgecolors="white", linewidth=.7, zorder=3)
        ax.annotate(f"{mean:.1f}", (mean, y), xytext=(7, 0), textcoords="offset points",
                    va="center", ha="left", fontsize=9.5, color="#24292E")
        ax.text(1.055, y, f"{int(row['completed'])} / {int(row['instances'])}",
                transform=blended_transform_factory(ax.transAxes, ax.transData),
                va="center", ha="left", fontsize=10,
                fontweight="bold" if arm == "carry" else "normal")
    ax.text(1.055, 1.025, "Completed", transform=ax.transAxes, ha="left", va="bottom", fontsize=9.5)
    ax.set(yticks=ys, yticklabels=[labels.get(arm, arm) for arm in arms],
           ylim=(len(arms) - .45, -.65), xlim=(0, xmax),
           xlabel="Mean native actions per instance")
    ax.set_title(title, loc="left", fontsize=11, fontweight="bold", pad=23)
    ax.xaxis.set_major_locator(MaxNLocator(5, steps=[1, 2, 2.5, 5, 10]))
    ax.grid(axis="x", color="#E4E7EA", linewidth=.65)
    ax.set_axisbelow(True)


def effect_panel(ax, data, title, limits):
    positions = np.array([0, 1, 2.5, 3.5, 5, 6])
    ax.axvline(0, color="#70757A", linewidth=1, linestyle=(0, (3, 3)), zorder=0)
    for i, (group, y, color) in enumerate(zip(GROUPS, positions, GROUP_COLORS)):
        row = data["effects"][group]
        value = float(row["all_native_actions_difference"])
        low, high = row["all_native_actions_ci_low"], row["all_native_actions_ci_high"]
        if low and high:
            low, high = float(low), float(high)
            ax.hlines(y, low, high, color=color, linewidth=1.7)
            ax.vlines([low, high], y - .10, y + .10, color=color, linewidth=1.15)
        ax.scatter(value, y, s=43, marker="o" if i % 2 == 0 else "D", color=color,
                   edgecolors="white", linewidth=.65, zorder=3)
        count = int(row["instances"])
        completion = f"{int(row['candidate_completed'])}/{count} | {int(row['reference_completed'])}/{count}"
        ax.text(1.045, y, completion,
                transform=blended_transform_factory(ax.transAxes, ax.transData),
                va="center", ha="left", fontsize=9)
    for y in (1.75, 4.25):
        ax.axhline(y, color="#E4E7EA", linewidth=.65)
    ax.text(1.045, 1.025, "Completed\ncarry | reset", transform=ax.transAxes,
            ha="left", va="bottom", fontsize=9)
    ax.set(yticks=positions, yticklabels=GROUP_LABELS, ylim=(6.6, -.7), xlim=limits,
           xlabel="Action difference (carry − reset)")
    ax.set_title(title, loc="left", fontsize=11, fontweight="bold", pad=23)
    ax.xaxis.set_major_locator(MaxNLocator(5, steps=[1, 2, 2.5, 5, 10]))
    ax.grid(axis="x", color="#E4E7EA", linewidth=.65)
    ax.set_axisbelow(True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison", action="append", required=True, metavar="LABEL=DIRECTORY")
    parser.add_argument("--output", type=Path, required=True, help="Common output prefix")
    parser.add_argument("--candidate", default="carry")
    parser.add_argument("--reset-arm", default="taskreset")
    parser.add_argument("--arms", nargs="+", help="Ordered method identifiers; default includes all supplied methods")
    parser.add_argument("--label", action="append", default=[], metavar="ARM=TEXT")
    parser.add_argument("--preview", action="store_true", help="Use completed subsets for a clearly labelled layout preview")
    args = parser.parse_args()
    labels = {**LABELS, **dict(item.split("=", 1) for item in args.label)}
    comparisons = []
    for specification in args.comparison:
        label, directory = specification.split("=", 1)
        data = load_comparison(Path(directory), args.arms, args.candidate, args.reset_arm)
        comparisons.append((label, data))
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10, "axes.labelsize": 10,
        "xtick.labelsize": 9.5, "ytick.labelsize": 9.5,
        "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
        "axes.edgecolor": "#8A9096", "axes.linewidth": .7,
        "ytick.major.size": 0, "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    })
    xmax = max(float(data["totals"][arm]["mean_native_actions"])
               for _, data in comparisons for arm in data["arms"])
    xmax = math.ceil((xmax + 30) / 50) * 50
    endpoints = [0.]
    for _, data in comparisons:
        for group in GROUPS:
            row = data["effects"][group]
            endpoints.extend(float(row[key]) for key in ("all_native_actions_difference",
                             "all_native_actions_ci_low", "all_native_actions_ci_high") if row[key])
    extent = max(5, math.ceil(max(abs(value) for value in endpoints) / 5) * 5)
    limits = (-extent * 1.10, extent * 1.10)
    nrows = len(comparisons)
    fig, axes = plt.subplots(nrows, 2, figsize=(12.8, 3.15 * nrows + 1.25), squeeze=False,
                             gridspec_kw={"width_ratios": [1, 1.18], "wspace": .9})
    fig.subplots_adjust(left=.17, right=.88, top=.80 if nrows == 1 else .87,
                        bottom=.22 if nrows == 1 else .135, hspace=.63)
    for i, (label, data) in enumerate(comparisons):
        method_panel(axes[i, 0], data, labels, f"{chr(97 + 2*i)}   {label}", xmax)
        effect_panel(axes[i, 1], data, f"{chr(98 + 2*i)}   Paired effect of carried state", limits)
    title = "Completed reference runs — layout preview" if args.preview else "Continuing state across native task instances"
    fig.suptitle(title, x=.17, ha="left", fontsize=13, fontweight="bold", y=.985)
    stream_counts = [int(data["totals"][data["arms"][0]]["streams"]) for _, data in comparisons]
    interval_note = "95% intervals resample complete streams." if max(stream_counts) > 1 else "One development stream; intervals are omitted."
    fig.text(.17, .055 if nrows == 1 else .047,
             "Mean action counts include every outcome. " + interval_note + " Negative differences favour carrying state.",
             ha="left", fontsize=9)
    same_instances = all(data["identities"] == comparisons[0][1]["identities"] for _, data in comparisons)
    instance_counts = [len(data["identities"]) for _, data in comparisons]
    if same_instances and len(set(stream_counts)) == 1:
        note = f"{instance_counts[0]} evaluation instances in {stream_counts[0]} stream"
        note += "s" if stream_counts[0] != 1 else ""
        note += "; shared instances, rows reported separately." if nrows > 1 else "."
    else:
        note = "; ".join(f"{label}: {n} instances, {s} stream{'s' if s != 1 else ''}"
                         for (label, _), n, s in zip(comparisons, instance_counts, stream_counts)) + "."
    fig.text(.17, .020, note, ha="left", fontsize=9)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for extension in ("pdf", "svg", "png"):
        fig.savefig(args.output.with_suffix(f".{extension}"), dpi=300, bbox_inches="tight", pad_inches=.18)
    plt.close(fig)
    print(f"Saved PDF, SVG and PNG to {args.output}")


if __name__ == "__main__":
    main()
