"""Train deployed CDM quantity updates from causal public calibration episodes.

Frequency labels score predictions made before intervention. Frozen observation
contexts retain the source actor's complete native history. Numerical accuracy
is not native activation or task completion.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import random
from time import perf_counter
from types import SimpleNamespace

import torch

from cdm.digital import PublicObjectMemory, public_candidates
from cdm.core import GaussianQuantityLikelihood
from .calibration import CalibrationLog
from .calibration_predictor import CalibrationPredictor, save_calibration_checkpoint
from .features import FeatureCache, FeatureConfig, batch_features, load_features
from .train import (load_episodes, load_checkpoint, public_observation,
                    create_measurement_log, with_measurement_records)

CACHE_SCHEMA = "calibration_context_v1_pre_feedback_all_support"
ADAPTER_INITIALIZATION_SEED = 0


def file_hash(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def trajectory_paths(roots):
    paths = set()
    for root in roots:
        root = Path(root)
        paths.update([root.resolve()] if root.is_file() else root.resolve().rglob("trajectory.jsonl"))
    return sorted(paths)


def support_revision(support):
    return tuple((item.label.object_handle, item.label.received_sequence,
                  max(record.sequence for record in item.query.records)) for item in support)


def build_program(episode):
    """Labels are attached later; inputs and support retain decision-time facts."""
    log, updates, queries, pool, keys = CalibrationLog(), [], [], [], {}
    log.observe_initial(episode.transitions[0]["observation"])

    def intern(query):
        value = {"object_handle": query.object_handle,
                 "available_after_sequence": query.available_after_sequence,
                 "features": {key: [asdict(row) for row in rows] for key, rows in query.features.items()}}
        key = sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
        if key not in keys:
            keys[key] = len(pool)
            pool.append(value)
        return keys[key]

    previous = ()
    for row in episode.transitions:
        log.append_transition(row)
        support = log.support()
        revision = support_revision(support)
        if revision and revision != previous:
            updates.append({"kind": "update", "sequence": row["index"],
                            "queries": [intern(item.query) for item in support],
                            "targets": [item.label.frequency_hz for item in support],
                            "label_sequences": [item.label.received_sequence for item in support]})
            previous = revision
    excluded = []
    decision_indices = {row["index"] for row in episode.transitions}
    for example in log.query_examples:
        opportunity, label = example.opportunity, example.label
        if not opportunity.clean:
            excluded.append({"sequence": opportunity.decision_sequence, "reason": opportunity.exclusion_reason})
            continue
        if opportunity.decision_sequence not in decision_indices:
            continue
        if label.received_sequence < opportunity.decision_sequence:
            raise ValueError("A future query label cannot precede the query")
        queries.append({"kind": "query", "sequence": opportunity.decision_sequence,
                        "query": intern(opportunity.query), "target": label.frequency_hz,
                        "label_sequence": label.received_sequence,
                        "support_count": len(opportunity.support),
                        "object_handle": opportunity.query.object_handle})
    last = max((row["sequence"] for row in queries), default=-1)
    events = [row for row in updates if row["sequence"] < last] + queries
    events.sort(key=lambda row: (row["sequence"], row["kind"] == "update"))
    return {"identity": list(episode.identity), "trajectory": str(episode.path),
            "queries": pool, "events": events, "excluded": excluded,
            "query_count": len(queries), "last_context": last}


def context_cache_path(directory, source_hash, trajectory_hash):
    key = sha256((CACHE_SCHEMA + source_hash + trajectory_hash).encode()).hexdigest()
    return Path(directory) / (key + ".pt")


def prepare_context(path, model, metadata, feature_cache, directory, source_hash):
    trajectory_hash = file_hash(path)
    destination = context_cache_path(directory, source_hash, trajectory_hash)
    if destination.exists():
        return destination
    episode = load_episodes([path])[0]
    program = build_program(episode)
    program.update(schema=CACHE_SCHEMA, trajectory_sha256=trajectory_hash, source_sha256=source_hash)
    contexts = {}
    if program["events"]:
        device = next(model.parameters()).device
        # The exact same adapter initialization is retained for paired tests.
        # It does not enter the frozen source observation recurrence.
        with torch.random.fork_rng(devices=[device] if device.type == "cuda" else []):
            torch.manual_seed(ADAPTER_INITIALIZATION_SEED)
            state = model.source.initial(1, device)
        program["initial_adapters"] = {key: state[key].cpu() for key in ("A", "B")}
        memory = PublicObjectMemory()
        measurements = create_measurement_log(episode.transitions[0]["observation"], feature_cache.config)
        needed = {event["sequence"] for event in program["events"]}
        settings = metadata["training_args"]
        for row in episode.transitions[:program["last_context"] + 1]:
            observation = with_measurement_records(public_observation(episode, row), measurements,
                                                   object_relations=feature_cache.config.object_relations)
            observation = memory.observe(observation)
            candidates = public_candidates(observation)
            features = feature_cache.encode(observation, candidates, episode.path.parent)
            obs, actions = batch_features([load_features(str(features))], device)
            with torch.no_grad():
                _, state, _ = model.source(obs, actions, state, mode=model.source_mode,
                                            persistent=not settings.get("restart_state", False),
                                            restart_workspace=settings.get("restart_workspace", False),
                                            predict_outcomes=False)
                if row["index"] in needed:
                    contexts[row["index"]] = model.context(state).cpu()
            if measurements is not None:
                measurements.append_transition(row)
    program["contexts"] = contexts
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(program, destination)
    return destination


def run_program(model, program, *, retain_updates=True, differentiable=False):
    device = next(model.parameters()).device
    adapters = {key: value.to(device) for key, value in program["initial_adapters"].items()}
    queries = [SimpleNamespace(**query) for query in program["queries"]]
    losses, predictions, update_count = [], [], 0
    for event in program["events"]:
        context = program["contexts"][event["sequence"]].to(device)
        if event["kind"] == "update":
            selected = [queries[index] for index in event["queries"]]
            updated, _ = model.update(selected, context.expand(len(selected), -1), event["targets"], adapters,
                                       differentiable=differentiable)
            update_count += 1
            if retain_updates:
                adapters = updated
        else:
            raw_mean, raw_logvar = model.raw_prediction([queries[event["query"]]], context, adapters)
            target = torch.tensor([[event["target"]]], device=device)
            loss = model.likelihood(raw_mean, raw_logvar, target).mean()
            losses.append(loss)
            mean, logvar = model.likelihood.distribution(raw_mean, raw_logvar)
            predictions.append({"sequence": event["sequence"], "label_sequence": event["label_sequence"],
                                "support_count": event["support_count"], "target_hz": event["target"],
                                "mean_hz": float(mean.detach().item()),
                                "std_hz": float((logvar.detach() / 2).exp().item()),
                                "nll": float(loss.detach()), "updates_before_query": update_count})
    return torch.stack(losses).mean(), {"identity": program["identity"], "predictions": predictions,
                                       "updates_computed": update_count, "retain_updates": retain_updates}


def summarize(results):
    rows = [row for result in results for row in result["predictions"]]

    def group(values):
        if not values:
            return {"queries": 0}
        errors = [abs(row["mean_hz"] - row["target_hz"]) for row in values]
        return {"queries": len(values), "nll": sum(row["nll"] for row in values) / len(values),
                "mae_hz": sum(errors) / len(errors),
                "rmse_hz": (sum(error ** 2 for error in errors) / len(errors)) ** .5,
                "within_2hz_of_public_working_setting": sum(error < 2 for error in errors) / len(errors),
                "mean_std_hz": sum(row["std_hz"] for row in values) / len(values)}
    return {"all": group(rows), "by_support_count": {
        str(count): group([row for row in rows if row["support_count"] == count])
        for count in sorted({row["support_count"] for row in rows})}}


def evaluate(model, programs):
    model.eval()
    result = {}
    for retain in (True, False):
        with torch.no_grad():
            episodes = [run_program(model, program, retain_updates=retain)[1] for program in programs]
        result["retained" if retain else "computed_discarded"] = {"metrics": summarize(episodes), "episodes": episodes}
    return result


def initialize_quantity_readout(model, training_programs, *, seed):
    """Adapt task units using training labels only, before any outer updates.

    Each crystal's first working return enters the prior once, irrespective of
    repeated support revisions. Query/validation contexts are not inspected.
    The inherited predictor hidden layers and frozen actor remain unchanged.
    """
    labels = {}
    for program in training_programs:
        identity = tuple(program["identity"])
        for event in program["events"]:
            if event["kind"] == "query":
                entries = [(event["object_handle"], event["label_sequence"], event["target"])]
            else:
                entries = [(program["queries"][index]["object_handle"], sequence, target)
                           for index, sequence, target in zip(event["queries"], event["label_sequences"], event["targets"])]
            for handle, sequence, target in entries:
                key = identity + (handle,)
                if key not in labels or sequence < labels[key][0]:
                    labels[key] = (sequence, float(target))
    values = torch.tensor([target for _, target in labels.values()], dtype=torch.float64)
    if not len(values) or not torch.isfinite(values).all():
        raise ValueError("Finite training working-frequency labels are required for task units")
    offset = float(values.mean())
    # Native display/activation resolution supplies a meaningful floor when a
    # development subset has constant labels. It is not fit on validation.
    scale = max(float(values.std(correction=0)), model.likelihood.noise_std, 1.)
    model.likelihood = GaussianQuantityLikelihood(offset=offset, scale=scale,
        noise_std=model.likelihood.noise_std, logvar_bounds=model.likelihood.logvar_bounds)
    readout = model.core.pred_out[-1]
    device = readout.weight.device
    with torch.random.fork_rng(devices=[device] if device.type == "cuda" else []), torch.no_grad():
        torch.manual_seed(seed)
        torch.nn.init.normal_(readout.weight, mean=0., std=1e-3)
        readout.bias.zero_()
    return {"method": "training_working_frequency_prior_and_new_task_final_readout",
            "unique_training_crystals": len(labels), "offset_hz": offset, "scale_hz": scale,
            "scale_floor_hz": max(model.likelihood.noise_std, 1.),
            "readout_weight_std": 1e-3, "readout_bias": [0., 0.], "seed": seed,
            "preserved": "predictor_hidden_layers_and_source_observation_action_path"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--train-trajectories", type=Path, nargs="+", default=[])
    parser.add_argument("--validation-trajectories", type=Path, nargs="+", default=[])
    parser.add_argument("--heldout-trajectories", type=Path, nargs="+", default=[])
    parser.add_argument("--prepare-trajectories", type=Path, nargs="+", default=[])
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--context-cache", type=Path, required=True)
    parser.add_argument("--encoder-path", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--feature-device")
    parser.add_argument("--feature-batch-size", type=int, default=64)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--inner-lr", type=float, default=.05)
    parser.add_argument("--context-mode", choices=("current", "object_only"), default="current")
    parser.add_argument("--query-encoding", choices=("late_binding", "typed_interaction"), default="late_binding")
    parser.add_argument("--seed", type=int, default=31)
    args = parser.parse_args(argv)
    if not args.prepare_only and (not args.train_trajectories or not args.validation_trajectories):
        parser.error("Training requires explicit train and validation trajectories")
    if args.prepare_trajectories and not args.prepare_only:
        parser.error("--prepare-trajectories is only for preparation shards")
    if (args.output / "training.jsonl").exists():
        parser.error("Choose a new output directory")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    source, metadata = load_checkpoint(args.source_checkpoint, args.device)
    model = CalibrationPredictor(source, source_mode=metadata["training_args"]["mode"],
                                 learning_rate=args.inner_lr, context_mode=args.context_mode,
                                 query_encoding=args.query_encoding).to(args.device)
    source_hash = file_hash(args.source_checkpoint)
    config = FeatureConfig(**{"schema": 1, "time_basis": "wall_clock", **metadata["features"]})
    cache = FeatureCache(args.feature_cache, config, args.feature_device or args.device,
                         args.encoder_path, args.feature_batch_size)
    args.output.mkdir(parents=True, exist_ok=True)
    splits = {name: trajectory_paths(getattr(args, name + "_trajectories"))
              for name in ("train", "validation", "heldout", "prepare")}
    programs, manifest = {}, {"source_sha256": source_hash, "cache_schema": CACHE_SCHEMA, "splits": {}}
    try:
        for split, paths in splits.items():
            programs[split], manifest["splits"][split] = [], []
            for path in paths:
                started = perf_counter()
                destination = prepare_context(path, model, metadata, cache, args.context_cache, source_hash)
                program = torch.load(destination, weights_only=True)
                manifest["splits"][split].append({"path": str(path), "cache": str(destination),
                                                  "identity": program["identity"], "queries": program["query_count"],
                                                  "excluded": program["excluded"]})
                if program["query_count"]:
                    programs[split].append(program)
                print(json.dumps({"prepared": str(path), "queries": program["query_count"],
                                  "seconds": round(perf_counter() - started, 2)}), flush=True)
    finally:
        cache.close()
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if args.prepare_only:
        return
    for first, second in (("train", "validation"), ("train", "heldout"), ("validation", "heldout")):
        identities = lambda name: {tuple(row["identity"]) for row in manifest["splits"][name]}
        if identities(first) & identities(second):
            raise ValueError(f"Instance overlap between {first} and {second}")
    if not programs["train"] or not programs["validation"]:
        raise ValueError("No clean future-labeled queries in the requested training/validation split")
    initial_adapters = programs["train"][0]["initial_adapters"]
    for split in ("train", "validation", "heldout"):
        for program in programs[split]:
            if any(not torch.equal(initial_adapters[key], program["initial_adapters"][key]) for key in ("A", "B")):
                raise ValueError("Prepared adapter initializations differ; exact shared initialization is required")
    initialization = initialize_quantity_readout(model, programs["train"], seed=args.seed)
    settings = {key: str(value) if isinstance(value, Path) else
                [str(item) for item in value] if isinstance(value, list) else value for key, value in vars(args).items()}
    settings.update(manifest=manifest, likelihood=asdict(model.likelihood), readout_initialization=initialization,
                    adapter_initialization_seed=ADAPTER_INITIALIZATION_SEED,
                    inner_update={"step_rule": model.core.config.step_rule,
                                  "fallback_learning_rate": args.inner_lr,
                                  "initial_log_step": float(model.core.log_step.detach()) if model.core.log_step is not None else None,
                                  "initial_log_kappa": float(model.core.log_kappa.detach()) if model.core.log_kappa is not None else None,
                                  "schedule": "all_support_on_each_public_support_revision",
                                  "context": "after_pre_action_observation_before_actual_feedback"},
                    trainable_parameters=[name for name, parameter in model.named_parameters() if parameter.requires_grad])
    (args.output / "config.json").write_text(json.dumps(settings, indent=2) + "\n")
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    best = float("inf")
    for epoch in range(args.epochs + 1):
        started, losses = perf_counter(), []
        if epoch:
            model.train()
            order = list(programs["train"])
            random.shuffle(order)
            for start in range(0, len(order), args.batch_size):
                batch = order[start:start + args.batch_size]
                optimizer.zero_grad(set_to_none=True)
                for program in batch:
                    loss, _ = run_program(model, program, differentiable=True)
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Non-finite future quantity loss")
                    (loss / len(batch)).backward()
                    losses.append(float(loss.detach()))
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.)
                optimizer.step()
        validation = evaluate(model, programs["validation"])
        result = {"epoch": epoch, "train_nll": sum(losses) / len(losses) if losses else None,
                  "validation": validation, "seconds": perf_counter() - started}
        with (args.output / "training.jsonl").open("a") as stream:
            stream.write(json.dumps(result) + "\n")
        print(json.dumps({key: value for key, value in result.items() if key != "validation"}
                         | {"validation": {key: value["metrics"] for key, value in validation.items()}}), flush=True)
        training = {**settings, "epoch": epoch}
        save_calibration_checkpoint(model, metadata, args.output / "last.pt", training_metadata=training,
                                    initial_adapters=initial_adapters)
        score = validation["retained"]["metrics"]["all"]["nll"]
        if score < best:
            best = score
            save_calibration_checkpoint(model, metadata, args.output / "best.pt", training_metadata=training,
                                        initial_adapters=initial_adapters)
    final = {split: evaluate(model, programs[split]) for split in ("validation", "heldout") if programs[split]}
    (args.output / "final_evaluation.json").write_text(json.dumps(final, indent=2) + "\n")


if __name__ == "__main__":
    main()
