"""Collect paired consequences of alternative actions in ambiguous training states.

Each prefix is replayed with the same received history and simulator randomness.
Only the first takeover action differs; the same deployed policy continues both
branches. These are training interventions, not autonomous test achievements.
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


def select(source, output):
    episode, = load_episodes([source])
    AGENT.reset()
    windows = ((48, 112), (112, 176), (176, 240))
    selected = [None] * len(windows)
    for index, row in enumerate(episode.transitions[:windows[-1][1]]):
        observation = public_observation(episode, row)
        observation["vision"] = {key: value if isinstance(value, str) else
                                 "data:image/png;base64," + base64.b64encode((episode.path.parent / value["path"]).read_bytes()).decode()
                                 for key, value in observation.get("vision", {}).items()}
        AGENT.prime(observation)
        # Numeric dialogues are handled by the explicit execution tool, not the
        # neural proposal whose uncertainty selects these experimental states.
        if AGENT.calibration.current_dialog() is None:
            scores, indices = AGENT._last_scores[-1].topk(2)
            margin = float(scores[0] - scores[1])
            for window, (begin, end) in enumerate(windows):
                if begin <= index < end and (selected[window] is None or margin < selected[window]["margin"]):
                    selected[window] = {"prefix": index, "margin": margin,
                                        "actions": [dict(AGENT._last_candidates[int(i)].action) for i in indices],
                                        "candidate_indices": indices.tolist()}
        AGENT.observe(row)
    result = {"source": str(source), "states": [s for s in selected if s is not None]}
    output.mkdir(parents=True, exist_ok=True)
    (output / "selection.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def selection_job(job):
    source, output = job
    output.mkdir(parents=True, exist_ok=True)
    with (output / "selection.log").open("w") as stream, redirect_stdout(stream):
        return select(source, output)


def branch_job(job):
    source, output, state, rank, budget = job
    output.mkdir(parents=True, exist_ok=True)
    with (output / "execution.log").open("w") as stream, redirect_stdout(stream):
        summary = evaluate_prefix(AGENT, source, output, prefix_steps=state["prefix"],
                                  takeover_steps=budget - state["prefix"], policy_type=AGENT.policy_type,
                                  first_action=state["actions"][rank])
    return {"source": str(source), "prefix": state["prefix"], "rank": rank,
            "action": state["actions"][rank], "output": str(output),
            **{key: summary[key] for key in ("prefix_replay_matched", "progress", "steps", "takeover_steps",
                                             "takeover_completed_successfully", "error")}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--devices", nargs="+", default=["cuda:0", "cuda:1"])
    parser.add_argument("--total-budget", type=int, default=400)
    args = parser.parse_args()
    if any(seed < 100 or seed % 5 == 4 for seed in args.seeds):
        parser.error("Use training instances; unseen-rule instances are reserved")
    if args.output.exists():
        parser.error("Choose a fresh collection output")
    args.output.mkdir(parents=True)
    (args.output / "config.json").write_text(json.dumps({**{k: str(v) if isinstance(v, Path) else v
                                                          for k, v in vars(args).items()},
        "selection": "smallest top-two action margin in each of three preassigned history windows",
        "history_windows": [[48, 112], [112, 176], [176, 240]],
        "test_data_used": False}, indent=2) + "\n")
    context = multiprocessing.get_context("spawn")
    assignments = context.Queue()
    for worker in range(args.workers):
        assignments.put(args.devices[worker % len(args.devices)])
    with ProcessPoolExecutor(args.workers, mp_context=context, initializer=initialize,
                             initargs=(args.checkpoint, assignments)) as pool:
        selections = list(pool.map(selection_job, [(args.source / f"seed{s}", args.output / f"seed{s}")
                                                   for s in args.seeds]))
        jobs = [(Path(selection["source"]), args.output / Path(selection["source"]).name /
                 f"prefix{state['prefix']}/rank{rank}", state, rank, args.total_budget)
                for selection in selections for state in selection["states"] for rank in (0, 1)]
        print(json.dumps({"selected_states": len(jobs) // 2, "native_branches": len(jobs)}), flush=True)
        rows = []
        for row in pool.map(branch_job, jobs):
            rows.append(row)
            print(json.dumps(row), flush=True)
            (args.output / "comparison.json").write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
