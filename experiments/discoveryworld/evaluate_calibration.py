"""Execute retained/discarded numerical learners after public experiment prefixes."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
from pathlib import Path

import torch

from .prefix_evaluate import evaluate_prefix
from .quantitative_policy import QuantitativePolicy


_POLICY = None


def _initialize(configuration, threads):
    global _POLICY
    torch.set_num_threads(threads)
    configuration = dict(configuration)
    kind = configuration.pop("prediction_type")
    if kind == "mixture":
        from .mixture_policy import MixturePolicy
        _POLICY = MixturePolicy(**configuration)
    elif kind == "reference":
        from .affine_reference_policy import AffineReferencePolicy
        _POLICY = AffineReferencePolicy(**configuration)
    else:
        _POLICY = QuantitativePolicy(**configuration)


def _execute(job):
    entry, retain_updates, output, steps = job
    _POLICY.retain_updates = retain_updates
    result = evaluate_prefix(
        _POLICY, Path(entry["source"]), Path(output),
        prefix_steps=entry["prefix_steps"], takeover_steps=steps,
        policy_type=getattr(_POLICY, "policy_type", "learned_calibration_with_native_scalar_control"),
        policy_metadata=_POLICY.evaluation_metadata,
    )
    return {"seed": entry["seed"], "cohort": entry["cohort"],
            "support_count": entry["support_count"], "retain_updates": retain_updates,
            "output": output, **result}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prediction-type", choices=("core", "mixture", "reference"), default="core")
    parser.add_argument("--prefixes", type=Path, required=True)
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--encoder-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--steps", type=int, default=48)
    parser.add_argument("--local-probes", action="store_true")
    args = parser.parse_args(argv)
    if min(args.workers, args.threads, args.steps) < 1:
        parser.error("workers, threads and steps must be positive")
    if args.local_probes and args.prediction_type != "mixture":
        parser.error("Local probing currently requires a mixture predictive distribution")
    entries = json.loads(args.prefixes.read_text())["prefixes"]
    configuration = {"checkpoint": str(args.checkpoint), "feature_cache": str(args.feature_cache),
                     "encoder_path": str(args.encoder_path), "device": args.device,
                     "prediction_type": args.prediction_type}
    if args.local_probes:
        configuration["local_probes"] = True
    jobs = []
    for entry in entries:
        if not entry["public_before_trial_clean"]:
            raise ValueError("A numerical decision prefix must precede its first frequency trial")
        for retain in (True, False):
            condition = "retained" if retain else "computed_discarded"
            output = args.output / condition / f"support{entry['support_count']}" / f"seed{entry['seed']}"
            jobs.append((entry, retain, str(output), args.steps))
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "config.json").write_text(json.dumps({
        "checkpoint": str(args.checkpoint), "prefixes": str(args.prefixes),
        "device": args.device, "workers": args.workers, "takeover_steps": args.steps,
        "prediction_type": args.prediction_type,
        "local_probes": args.local_probes,
        "comparison": "Same frozen policy, replayed initial evidence and action budget. Both compute updates on their actual feedback and retain or discard them; later experience follows actual actions.",
        "control": "Explicit native scalar controller; not a learned action head or autonomous full-task score.",
    }, indent=2) + "\n")
    with ProcessPoolExecutor(args.workers, mp_context=multiprocessing.get_context("spawn"),
                             initializer=_initialize, initargs=(configuration, args.threads)) as pool:
        results = []
        for result in pool.map(_execute, jobs):
            results.append(result)
            print(json.dumps({key: result[key] for key in (
                "seed", "cohort", "support_count", "retain_updates", "takeover_steps",
                "takeover_start_progress", "progress", "takeover_new_activations", "error")}), flush=True)
    (args.output / "results.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
