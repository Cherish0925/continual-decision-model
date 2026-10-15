"""Run paired dynamic-grid computational-memory comparisons on a CPU."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import csv
import gzip
import io
import json
import platform
from pathlib import Path
from time import perf_counter

import numpy as np

from grid import GridWorld, action_values, choose_action, make_trace, reference_values
from planners import METHODS, Planner

REGIMES = ("full_return", "partial_return", "continuous")


def run_episode(trace, method, fraction, reference_cache):
    world = GridWorld(trace)
    planner = None if method == "current_map_reference" else Planner(method, trace.side, fraction)
    reward_sum, action_gap_sum = 0., 0.
    decisions_seconds = []
    phases = {}
    raw = []
    for iteration in range(len(trace.maps) - 1):
        before = world.observe()
        started = perf_counter()
        if planner is not None:
            values = planner.compute(before)
        # Planning never freezes the environment. All methods see the new
        # observation in the action head, including goal and obstacle changes.
        fresh = world.advance_time()
        if planner is None:
            values = reference_values(fresh, trace.side)
        action = choose_action(values, fresh, trace.side)
        decisions_seconds.append(perf_counter() - started)
        # The reference is isolated from the tested planners and their timings.
        exact_values = reference_cache[fresh.tick]
        reference_q = action_values(exact_values, fresh, trace.side)[fresh.position]
        gap = max(0., float(reference_q.max() - reference_q[action]))
        action_gap_sum += gap
        after, reward, hit = world.move(action)
        reward_sum += reward
        entry = phases.setdefault(fresh.phase_id, {"ticks": 0, "hit": False, "arrival_tick": None,
                                                    "return": 0., "action_gap": 0.})
        entry["ticks"] += 1
        entry["return"] += reward
        entry["action_gap"] += gap
        if hit:
            entry["hit"] = True
            entry["arrival_tick"] = entry["ticks"]
        raw.append({"seed": trace.seed, "regime": trace.regime, "method": method,
                    "budget_fraction": fraction, "tick": fresh.tick,
                    "phase": fresh.phase_id, "position_before": fresh.position,
                    "position_after": after.position, "goal": fresh.goal,
                    "action": action, "reward": reward, "reached": int(hit),
                    "current_map_action_gap": gap,
                    "decision_ms": decisions_seconds[-1] * 1000})
    total_ticks = len(raw)
    row = {"seed": trace.seed, "regime": trace.regime, "method": method,
           "budget_fraction": fraction, "return": reward_sum,
           "goal_phases": len(phases), "goal_hits": sum(v["hit"] for v in phases.values()),
           "goal_completion": np.mean([v["hit"] for v in phases.values()]),
           # Failed phases receive their full duration, rather than disappearing
           # from a successful-arrivals-only average.
           "capped_goal_ticks": np.mean([v["arrival_tick"] if v["hit"] else v["ticks"] for v in phases.values()]),
           "mean_action_gap": action_gap_sum / total_ticks,
           "decision_ms_mean": np.mean(decisions_seconds) * 1000,
           "decision_ms_p95": np.quantile(decisions_seconds, .95) * 1000,
           "cell_backups": planner.total_backups if planner else 0,
           "memory_read_ms": planner.read_seconds * 1000 if planner else 0,
           "memory_write_ms": planner.write_seconds * 1000 if planner else 0,
           "bellman_ms": planner.backup_seconds * 1000 if planner else 0,
           "context_changes": planner.context_changes if planner else 0,
           "retrievals": planner.retrievals if planner else 0,
           "exact_reads": planner.exact_reads if planner else 0,
           "local_cell_reads": planner.local_reads if planner else 0}
    phase_rows = [{"seed": trace.seed, "regime": trace.regime, "method": method,
                   "budget_fraction": fraction, "phase": phase, **entry}
                  for phase, entry in phases.items()]
    return row, phase_rows, raw


def write_csv(path, rows):
    with ExitStack() as stack:
        if path.suffix == ".gz":
            raw = stack.enter_context(path.open("wb"))
            binary = stack.enter_context(gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0))
            stream = stack.enter_context(io.TextIOWrapper(binary, newline=""))
        else:
            stream = stack.enter_context(path.open("w", newline=""))
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/continual-grid"))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(5)))
    parser.add_argument("--ticks", type=int, default=192)
    parser.add_argument("--budgets", type=float, nargs="+", default=[.25, 1.])
    args = parser.parse_args()
    if args.ticks < 1 or any(not 0 < x <= 1 for x in args.budgets):
        parser.error("ticks must be positive and budgets in (0,1]")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = {"seeds": args.seeds, "ticks": args.ticks, "budget_fractions": args.budgets,
              "regimes": list(REGIMES), "methods": list(METHODS), "grid_side": 9,
              "phase_length": 24, "memory_capacity": 32, "local_radius": 2,
              "gamma": .97, "step_cost": .02, "python": platform.python_version(),
              "numpy": np.__version__, "platform": platform.platform(),
              "clock": "One bounded cell-backup block advances the world by one tick; one action follows.",
              "timing": "CPU wall time, including context preparation, memory, backup and action head; reference metrics excluded.",
              "reference": "Exact stationary current-map planner; extra planning work, no future context.",
              "parameters_learned": False}
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    rows, phases, traces = [], [], []
    start = perf_counter()
    for seed in args.seeds:
        for regime in REGIMES:
            trace = make_trace(seed, regime, args.ticks)
            cache = {}
            # Current-map reference depends on map and goal, not agent location.
            reference_world = GridWorld(trace)
            for tick in range(1, args.ticks + 1):
                cache[tick] = reference_values(reference_world.advance_time(), 9)
            for fraction in args.budgets:
                for method in (*METHODS, "current_map_reference"):
                    row, phase, raw = run_episode(trace, method, fraction, cache)
                    rows.append(row)
                    phases.extend(phase)
                    traces.extend(raw)
            print(f"seed={seed} regime={regime} done", flush=True)
    write_csv(args.output_dir / "episodes.csv", rows)
    write_csv(args.output_dir / "phases.csv", phases)
    write_csv(args.output_dir / "decisions.csv.gz", traces)
    summary = []
    metrics = ("return", "goal_completion", "capped_goal_ticks", "mean_action_gap",
               "decision_ms_mean", "decision_ms_p95", "cell_backups", "memory_read_ms", "memory_write_ms")
    for regime in REGIMES:
        for fraction in args.budgets:
            for method in (*METHODS, "current_map_reference"):
                subset = [r for r in rows if r["regime"] == regime and r["budget_fraction"] == fraction and r["method"] == method]
                item = {"regime": regime, "budget_fraction": fraction, "method": method, "seeds": len(subset)}
                for metric in metrics:
                    values = np.asarray([r[metric] for r in subset])
                    item[metric + "_mean"] = float(values.mean())
                    item[metric + "_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.
                summary.append(item)
    write_csv(args.output_dir / "summary.csv", summary)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Completed {len(rows)} paired episodes in {perf_counter()-start:.2f}s; {args.output_dir}")


if __name__ == "__main__":
    main()
