"""Measure computation where one trained policy actually changes its proposal.

Selection uses training histories and current public observations only. Native
branches replay the same history, change the first prediction-cycle budget,
then use the same model and budget for subsequent decisions.
"""
from __future__ import annotations

import argparse
import base64
from concurrent.futures import ProcessPoolExecutor
from contextlib import redirect_stdout
import json
import multiprocessing
from pathlib import Path

import torch

from .predictive_actions import PredictivePolicy
from .prefix_evaluate import evaluate_prefix
from .train import load_episodes, public_observation

AGENT = None


def initialize(checkpoint, assignments):
    global AGENT
    torch.set_num_threads(2)
    torch.backends.cuda.enable_cudnn_sdp(False)
    AGENT = PredictivePolicy(checkpoint, "runs/discoveryworld/features", device=assignments.get(),
                             encoder_path="data/cache/siglip2", local_probes=True)


def selection(source, minimum, maximum):
    episode, = load_episodes([source])
    AGENT.reset()
    for i, row in enumerate(episode.transitions[:maximum]):
        observation = public_observation(episode, row)
        observation["vision"] = {key: value if isinstance(value, str) else
                                 "data:image/png;base64," + base64.b64encode((episode.path.parent / value["path"]).read_bytes()).decode()
                                 for key, value in observation.get("vision", {}).items()}
        AGENT.prime(observation)
        choices = AGENT.decisions[-1]["selected_by_cycle"]
        if i >= minimum and choices[0] != choices[-1] and AGENT.calibration.current_dialog() is None:
            return i, choices
        AGENT.observe(row)
    return None, None


def execute(job):
    source, output, minimum, maximum, total_budget = job
    output.mkdir(parents=True, exist_ok=True)
    with (output / "execution.log").open("w") as stream, redirect_stdout(stream):
        prefix, choices = selection(source, minimum, maximum)
        result = {"source": str(source), "prefix": prefix, "selected_by_cycle": choices, "branches": {}}
        if prefix is not None:
            for cycles in (0, 2):
                summary = evaluate_prefix(AGENT, source, output / f"cycles{cycles}", prefix_steps=prefix,
                                          takeover_steps=total_budget - prefix,
                                          policy_type=AGENT.policy_type, first_refinements=cycles)
                result["branches"][str(cycles)] = {key: summary[key] for key in
                    ("prefix_replay_matched", "progress", "steps", "takeover_steps",
                     "takeover_completed_successfully", "error")}
        (output / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--minimum-prefix", type=int, default=48)
    parser.add_argument("--maximum-prefix", type=int, default=240)
    parser.add_argument("--total-budget", type=int, default=400)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--devices", nargs="+", default=["cuda:0", "cuda:1"])
    args = parser.parse_args()
    jobs = [(args.source / f"seed{s}", args.output / f"seed{s}", args.minimum_prefix,
             args.maximum_prefix, args.total_budget) for s in args.seeds]
    if any((job[1] / "comparison.json").exists() for job in jobs):
        parser.error("Choose a fresh branch output")
    context = multiprocessing.get_context("spawn")
    assignments = context.Queue()
    for worker in range(args.workers):
        assignments.put(args.devices[worker % len(args.devices)])
    with ProcessPoolExecutor(args.workers, mp_context=context, initializer=initialize,
                             initargs=(args.checkpoint, assignments)) as pool:
        results = []
        for result in pool.map(execute, jobs):
            results.append(result)
            print(json.dumps(result), flush=True)
    (args.output / "comparison.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
