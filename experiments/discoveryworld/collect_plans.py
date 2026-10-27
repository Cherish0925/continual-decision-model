"""Collect complete hypothesis tests from matched public experimental histories."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stdout
import json
import multiprocessing
import os
from pathlib import Path

import torch
import numpy as np

from .calibration import CalibrationLog
from .calibration_mixture import load_mixture_checkpoint
from .experimental_plans import ExperimentalPolicy
from .numeric_reference import quantity_vector
from .prefix_evaluate import evaluate_prefix

AGENT = None


def first_experiment(source, mixture):
    log = CalibrationLog()
    for line in (source / "trajectory.jsonl").open():
        transition = json.loads(line)
        log.append_transition(transition)
        dialog = log.current_dialog()
        if dialog is None or dialog["activated"] is not False or len(log.support()) < 2:
            continue
        if any(x.object_handle == dialog["object_handle"] for x in log.labels):
            continue
        query = log.query(dialog["object_handle"])
        if not all(query.presence.get(q, False) for q in log.required_quantities):
            continue
        support = log.support()
        posterior = mixture.fit_posterior(np.asarray([quantity_vector(s.query) for s in support]),
                                         [s.label.frequency_hz for s in support])
        pred = mixture.predict(np.asarray([quantity_vector(query)]), posterior)
        means = pred["expert_mean_hz"][0].tolist()
        return {"source": str(source), "prefix": transition["index"] + 1,
                "hypotheses": [i for i,m in enumerate(means) if 0 <= m <= 10000],
                "predicted_settings_hz": means}
    return None


def initialize(checkpoint, devices):
    global AGENT
    torch.set_num_threads(2)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["RAYON_NUM_THREADS"] = "2"
    torch.backends.cuda.enable_cudnn_sdp(False)
    AGENT = ExperimentalPolicy(checkpoint, "runs/discoveryworld/features", device=devices.get(),
                               encoder_path="data/cache/siglip2", local_probes=True, one_trial=True)


def run(job):
    selection, index, root = job
    output = root / Path(selection["source"]).name / f"hypothesis{index}"
    output.mkdir(parents=True, exist_ok=True)
    AGENT.forced_plan = index
    with (output / "execution.log").open("w") as stream, redirect_stdout(stream):
        summary = evaluate_prefix(AGENT, Path(selection["source"]), output,
                                  prefix_steps=selection["prefix"], takeover_steps=64,
                                  policy_type=AGENT.policy_type)
    return result(job, summary, AGENT.trials)


def result(job, summary, trials):
    selection, index, root = job
    output = root / Path(selection["source"]).name / f"hypothesis{index}"
    if summary["error"] or not summary["prefix_replay_matched"]:
        raise RuntimeError(f"Experimental prefix failed: {output}")
    return {"source": selection["source"], "prefix": selection["prefix"], "hypothesis": index,
            "output": str(output), "finished": bool(trials),
            "success": bool(trials and trials[0]["success"]), "actions": summary["takeover_steps"],
            "progress": summary["progress"], "prefix_replay_matched": summary["prefix_replay_matched"]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, default=Path("runs/discoveryworld/predictive_actions/feedback/last.pt"))
    p.add_argument("--source", type=Path, default=Path("runs/discoveryworld/predictive_actions/reference/train"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seeds", type=int, nargs="+", required=True)
    p.add_argument("--hypotheses", type=int, nargs="+")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--devices", nargs="+", default=["cuda:0", "cuda:1"])
    p.add_argument("--resume", action="store_true", help="Reuse finished experiments and complete remaining branches")
    args = p.parse_args()
    if args.output.exists() and not args.resume:
        p.error("Choose a fresh collection directory")
    args.output.mkdir(parents=True, exist_ok=args.resume)
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    reader = torch.load(saved["base_checkpoint"], map_location="cpu", weights_only=True)
    mixture,_ = load_mixture_checkpoint(reader["mixture_checkpoint"], "cpu")
    torch.set_num_threads(2)
    with torch.no_grad():
        selected = [first_experiment(args.source / f"seed{s}", mixture) for s in args.seeds]
    (args.output / "selection.json").write_text(json.dumps(selected, indent=2) + "\n")
    (args.output / "config.json").write_text(json.dumps({k:str(v) if isinstance(v,Path) else v
                                                        for k,v in vars(args).items()}, indent=2)+"\n")
    jobs = [(s,i,args.output) for s in selected if s for i in s["hypotheses"]
            if args.hypotheses is None or i in args.hypotheses]
    print(json.dumps({"selected_histories":sum(s is not None for s in selected), "experiments":len(jobs)}),flush=True)
    records, pending = [], []
    for job in jobs:
        s, i, root = job
        output = root / Path(s["source"]).name / f"hypothesis{i}"
        summary, trials = output / "summary.json", output / "experimental_trials.json"
        if args.resume and summary.exists() and trials.exists():
            records.append(result(job, json.loads(summary.read_text()), json.loads(trials.read_text())))
        else:
            if args.resume and output.exists():
                interrupted = root / "interrupted" / output.parent.name
                interrupted.mkdir(parents=True, exist_ok=True)
                attempt = 1
                while (interrupted / f"hypothesis{i}-{attempt}").exists():
                    attempt += 1
                output.rename(interrupted / f"hypothesis{i}-{attempt}")
            pending.append(job)
    (args.output/"comparison.json").write_text(json.dumps(records,indent=2)+"\n")
    print(json.dumps({"reused":len(records), "remaining":len(pending), "workers":args.workers}),flush=True)
    if not pending:
        return
    context = multiprocessing.get_context("spawn")
    devices = context.Queue()
    for i in range(min(args.workers,len(pending))):
        devices.put(args.devices[i%len(args.devices)])
    with ProcessPoolExecutor(min(args.workers,len(pending)), mp_context=context,
                             initializer=initialize, initargs=(args.checkpoint,devices)) as pool:
        futures = [pool.submit(run,job) for job in pending]
        for future in as_completed(futures):
            try:
                row = future.result()
            except Exception as failure:
                print(f"Collection stopped: {type(failure).__name__}: {failure}", flush=True)
                for other in futures:
                    other.cancel()
                raise
            records.append(row)
            (args.output/"comparison.json").write_text(json.dumps(records,indent=2)+"\n")
            print(json.dumps(row),flush=True)


if __name__ == "__main__":
    main()
