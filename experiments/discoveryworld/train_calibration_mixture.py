"""Learn an explicit Gaussian affine-mixture prior from causal public queries.

The original policy/core is preserved by reference and SHA. This is a new task
readout, not an implementation of the original CDM A/B update. Validation is
logged without checkpoint selection; held-out predictions are made only after
the fixed training budget is complete.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from hashlib import sha256
import gzip
import json
import math
import os
from pathlib import Path
import random
from time import perf_counter

import numpy as np
import torch

from .calibration import CalibrationLog, FEATURE_QUANTITIES
from .calibration_mixture import AffineMixture, save_mixture_checkpoint, load_mixture_checkpoint
from .numeric_reference import quantity_vector


def paths(roots):
    found = set()
    for root in roots:
        root = Path(root).resolve()
        found.update([root] if root.is_file() else root.rglob("trajectory.jsonl"))
    return sorted(found)


def identity(path):
    metadata = json.loads((path.parent / "run.json").read_text())
    return tuple(metadata[key] for key in ("scenario", "difficulty", "seed"))


def compact_sample(sample, episode_identity):
    query, support = sample["query"], sample["support"]
    decision = sample["decision_sequence"]
    label = sample["evaluation_only_target"]
    if query["object_handle"] in {row["label"]["object_handle"] for row in support}:
        raise ValueError("A query's own target must not enter its support")
    if query["available_after_sequence"] >= decision or label["received_sequence"] < decision:
        raise ValueError("Query inputs must precede the prediction and its label")
    if any(max(row["available_after_sequence"], row["label"]["received_sequence"]) >= decision for row in support):
        raise ValueError("Support must be available before the query")
    return {"identity": list(episode_identity), "decision_sequence": decision,
            "object_handle": query["object_handle"], "x": quantity_vector(query).tolist(),
            "query_available": query["available_after_sequence"],
            "target_hz": label["frequency_hz"], "label_sequence": label["received_sequence"],
            "support_x": [quantity_vector(row["query"]).tolist() for row in support],
            "support_hz": [row["label"]["frequency_hz"] for row in support],
            "support_objects": [row["label"]["object_handle"] for row in support],
            "support_available": [row["query"]["available_after_sequence"] for row in support],
            "support_label_sequences": [row["label"]["received_sequence"] for row in support]}


def load_samples(trajectory_paths, public_cache=None):
    cached = {str(Path(row["source"]).resolve()): row for row in (public_cache or [])}
    samples, manifest = [], []
    for path in trajectory_paths:
        content = path.read_bytes()
        digest = sha256(content).hexdigest()
        source = cached.get(str(path))
        if source is not None and source["sha256"] == digest:
            raw = source["samples"]
        else:
            log = CalibrationLog()
            for line in content.splitlines():
                row = json.loads(line)
                transition = {key: row[key] for key in ("index", "action", "action_result")}
                transition.update(observation={"ui": row["observation"]["ui"]},
                                  next_observation={"ui": row["next_observation"]["ui"]})
                log.append_transition(transition)
            raw = [{"query": asdict(example.opportunity.query),
                    "support": [asdict(row) for row in example.opportunity.support],
                    "decision_sequence": example.opportunity.decision_sequence,
                    "evaluation_only_target": asdict(example.label)}
                   for example in log.query_examples if example.opportunity.clean]
        key = identity(path)
        examples = [compact_sample(row, key) for row in raw]
        samples.extend(examples)
        manifest.append({"path": str(path), "identity": list(key), "sha256": digest, "queries": len(examples)})
    return samples, manifest


def fit_normalization(samples):
    """Only training instances; each object's latest public feature vector once."""
    values, labels = {}, {}
    for row in samples:
        instance = tuple(row["identity"])
        objects = [(row["object_handle"], row["query_available"], row["x"], row["label_sequence"], row["target_hz"])]
        objects += list(zip(row["support_objects"], row["support_available"], row["support_x"],
                            row["support_label_sequences"], row["support_hz"]))
        for obj, available, vector, received, target in objects:
            key = instance + (obj,)
            if key not in values or available > values[key][0]:
                values[key] = (available, vector)
            if key not in labels or received < labels[key][0]:
                labels[key] = (received, target)
    if not values:
        raise ValueError("No clean training queries")
    X = np.asarray([row[1] for row in values.values()], dtype=np.float64)
    y = np.asarray([row[1] for row in labels.values()], dtype=np.float64)
    scale = X.std(axis=0)
    return {"x_mean": X.mean(axis=0).tolist(), "x_scale": np.where(scale > 0, scale, 1.).tolist(),
            "y_mean_hz": float(y.mean()), "y_scale_hz": max(float(y.std()), 2.),
            "training_objects": len(values), "training_labels": len(labels),
            "quantity_names": list(FEATURE_QUANTITIES), "scope": "training_instances_only"}


def tensors(row, device):
    make = lambda values: torch.as_tensor(values, dtype=torch.float64, device=device)
    return make(row["support_x"]).reshape(-1, 9), make(row["support_hz"]), make([row["x"]]), make([row["target_hz"]])


def summarize(rows):
    def group(selected):
        if not selected:
            return {"queries": 0}
        output = {"queries": len(selected), "nll": float(np.mean([row["nll"] for row in selected])),
                  "known_support_mae_hz": float(np.mean([row["known_support_mae_hz"] for row in selected]))}
        for key in ("mixture_mean", "map_expert"):
            errors = np.abs([row[key + "_hz"] - row["target_hz"] for row in selected])
            output[key] = {"mae_hz": float(errors.mean()), "median_absolute_error_hz": float(np.median(errors)),
                           "rmse_hz": float(np.sqrt(np.mean(errors ** 2))),
                           "within_2hz": int((errors < 2).sum())}
        return output
    return {"all": group(rows), "by_support_count": {str(count): group([r for r in rows if r["support_count"] == count])
            for count in sorted({r["support_count"] for r in rows})}}


def evaluate(model, samples):
    model.eval()
    results = {}
    with torch.no_grad():
        for retain in (True, False):
            rows = []
            for row in samples:
                X, y, x, target = tensors(row, model.device)
                computed = model.fit_posterior(X, y)  # Same computation in both arms.
                posterior = computed if retain else None
                pred = model.predict(x, posterior)
                support_fit = model.predict(X, posterior)
                nll = model.predictive_nll(x, target, posterior)
                rows.append({"identity": row["identity"], "decision_sequence": row["decision_sequence"],
                             "label_sequence": row["label_sequence"], "support_count": len(y),
                             "target_hz": float(target[0]), "nll": float(nll[0]),
                             "mixture_mean_hz": float(pred["mixture_mean_hz"][0]),
                             "map_expert_hz": float(pred["map_expert_mean_hz"][0]),
                             "map_expert_index": int(pred["map_expert_index"][0]),
                             "expert_posterior": pred["expert_posterior"][0].tolist(),
                             "expert_mean_hz": pred["expert_mean_hz"][0].tolist(),
                             "expert_std_hz": pred["expert_std_hz"][0].tolist(),
                             "known_support_mae_hz": float((support_fit["map_expert_mean_hz"] - y).abs().mean())})
            results["retained" if retain else "computed_discarded"] = {"metrics": summarize(rows),
                "posterior_computations": len(rows), "predictions": rows}
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-trajectories", type=Path, nargs="+", required=True)
    parser.add_argument("--validation-trajectories", type=Path, nargs="+", default=[])
    parser.add_argument("--heldout-trajectories", type=Path, nargs="+", default=[])
    parser.add_argument("--public-input-cache", type=Path)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--prior-mode", choices=("typed", "shared"), default="typed")
    parser.add_argument("--train-only", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args(argv)
    if not args.train_only and not args.prepare_only and not args.validation_trajectories:
        parser.error("Provide validation trajectories or mark the run --train-only")
    if args.train_only and (args.validation_trajectories or args.heldout_trajectories):
        parser.error("Train-only pilots cannot consume validation/heldout paths")
    if min(args.epochs, args.batch_size, args.threads) <= 0 or args.lr <= 0:
        parser.error("Training settings must be positive")
    if (args.output / "training.jsonl").exists():
        parser.error("Choose a fresh training output")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    split_paths = {name: paths(getattr(args, name + "_trajectories")) for name in ("train", "validation", "heldout")}
    identities = {name: {identity(path) for path in group} for name, group in split_paths.items()}
    if any(identities[a] & identities[b] for a, b in (("train", "validation"), ("train", "heldout"), ("validation", "heldout"))):
        raise ValueError("Instance overlap between data splits")
    public_cache = json.loads(gzip.decompress(args.public_input_cache.read_bytes())) if args.public_input_cache else []
    train, train_manifest = load_samples(split_paths["train"], public_cache)
    normalization = fit_normalization(train)
    model = AffineMixture(normalization, prior_mode=args.prior_mode).to(args.device)
    source = os.path.relpath(args.source_checkpoint.resolve(), args.project_root.resolve())
    source_hash = sha256(args.source_checkpoint.read_bytes()).hexdigest()
    args.output.mkdir(parents=True, exist_ok=True)
    settings = {key: str(value) if isinstance(value, Path) else [str(v) for v in value] if isinstance(value, list) else value
                for key, value in vars(args).items()}
    settings.update(normalization=normalization, train_manifest=train_manifest, dirichlet_concentration=1.1,
                    gradient_clip=10., seed_affects="minibatch_order; all candidates start with the same uniform prior",
                    basis="nine hand-specified named single-quantity affine candidates; prior parameters are learned")
    save = lambda destination, epoch: save_mixture_checkpoint(model, destination, source_checkpoint=source,
        source_sha256=source_hash, training={**settings, "epoch": epoch})
    save(args.output / "initial.pt", 0)
    (args.output / "config.json").write_text(json.dumps(settings, indent=2) + "\n")
    print(json.dumps({"initial_checkpoint": str(args.output / "initial.pt"), "training_queries": len(train),
                      "training_instances": len({tuple(row['identity']) for row in train})}), flush=True)
    if args.prepare_only:
        return
    validation, validation_manifest = load_samples(split_paths["validation"], public_cache)
    settings["validation_manifest"] = validation_manifest
    evaluation_samples = train if args.train_only else validation
    if not evaluation_samples:
        raise ValueError("No evaluation queries")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    for epoch in range(args.epochs + 1):
        started, losses = perf_counter(), []
        if epoch:
            model.train()
            order = list(train)
            rng.shuffle(order)
            for start in range(0, len(order), args.batch_size):
                batch = order[start:start + args.batch_size]
                optimizer.zero_grad(set_to_none=True)
                objectives = []
                for row in batch:
                    X, y, x, target = tensors(row, model.device)
                    posterior = model.fit_posterior(X, y)
                    objectives.append(model.predictive_nll(x, target, posterior).mean())
                loss = torch.stack(objectives).mean() + model.prior_penalty(examples=len(train))
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError("Non-finite future conditional mixture NLL")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10.)
                optimizer.step()
                losses.extend(float(value.detach()) for value in objectives)
        paired = evaluate(model, evaluation_samples)
        result = {"epoch": epoch, "train_nll": float(np.mean(losses)) if losses else None,
                  "evaluation_split": "train_in_sample" if args.train_only else "validation",
                  "evaluation": paired, "seconds": perf_counter() - started,
                  "expert_prior": model.mixture_logits.detach().softmax(-1).tolist(),
                  "noise_std_hz": model.noise_std_hz.detach().tolist()}
        with (args.output / "training.jsonl").open("a") as stream:
            stream.write(json.dumps(result) + "\n")
        print(json.dumps({key: value for key, value in result.items() if key != "evaluation"}
                         | {"evaluation": {key: value["metrics"] for key, value in paired.items()}}), flush=True)
        if epoch == 0:
            (args.output / "initial_evaluation.json").write_text(json.dumps(result, indent=2) + "\n")
    save(args.output / "last.pt", args.epochs)
    heldout, heldout_manifest = load_samples(split_paths["heldout"], public_cache)
    initial, _ = load_mixture_checkpoint(args.output / "initial.pt", args.device)
    final = {"evaluation": paired, "heldout": {"initial_uniform": evaluate(initial, heldout),
                                               "learned_final": evaluate(model, heldout)} if heldout else {},
             "heldout_manifest": heldout_manifest}
    (args.output / "final_evaluation.json").write_text(json.dumps(final, indent=2) + "\n")


if __name__ == "__main__":
    main()
