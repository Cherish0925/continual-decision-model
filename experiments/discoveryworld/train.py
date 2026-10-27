"""Sequence imitation and received-outcome learning on real DiscoveryWorld traces.

The input split is explicit and occurs before window construction. Batches
contain episodes; every episode is visited in temporal order, with h/z carried
across truncated-backpropagation windows. Offline imitation metrics are not
scientific task completion; evaluate.py executes the learned policy.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
from hashlib import sha256
import json
import math
from pathlib import Path
import random
import subprocess
from time import perf_counter

import torch
import torch.nn.functional as F

from cdm.core import ContinualCore, CoreConfig
from cdm.digital import DigitalConfig, DigitalPolicy, PublicObjectMemory, observed_numbers, public_candidates
from .features import FeatureCache, FeatureConfig, batch_features, canonical_action, load_features


@dataclass
class Episode:
    path: Path
    metadata: dict
    transitions: list[dict]

    @property
    def identity(self):
        return tuple(self.metadata[key] for key in ("scenario", "difficulty", "seed"))


@dataclass
class PreparedStep:
    feature_path: str
    expert_target: int
    executed_target: int
    outcome: list[float]
    run_weight: float = 1.0
    repeated_run_exit: bool = False


@dataclass
class PreparedEpisode:
    episode: Episode
    steps: list[PreparedStep]


def load_episodes(roots: list[Path]) -> list[Episode]:
    paths = set()
    for root in roots:
        if root.is_file():
            paths.add(root.resolve())
        else:
            paths.update(path.resolve() for path in root.rglob("trajectory.jsonl"))
    episodes = []
    for path in sorted(paths):
        metadata = json.loads((path.parent / "run.json").read_text())
        for key in ("scenario", "difficulty", "seed", "action_descriptions", "teleport_locations"):
            if key not in metadata:
                raise ValueError(f"{path}: missing run.json field {key}")
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if not rows:
            raise ValueError(f"Empty episode: {path}")
        if any(row["index"] != i for i, row in enumerate(rows)):
            raise ValueError(f"Trajectory must be contiguous from index zero: {path}")
        if any(row.get("terminated") or row.get("truncated") for row in rows[:-1]):
            raise ValueError(f"Episode contains transitions after its boundary: {path}")
        episodes.append(Episode(path, metadata, rows))
    if not episodes:
        raise ValueError("No trajectory.jsonl files found")
    return episodes


def validate_split(train: list[Episode], validation: list[Episode]):
    overlap = {episode.identity for episode in train} & {episode.identity for episode in validation}
    hashes = lambda episodes: {sha256(episode.path.read_bytes()).hexdigest() for episode in episodes}
    if overlap or hashes(train) & hashes(validation):
        raise ValueError(f"Training/validation episode overlap: {sorted(overlap)}")


def public_observation(episode: Episode, row: dict) -> dict:
    return {**row["observation"], "actions": episode.metadata["action_descriptions"],
            "teleport_locations": episode.metadata["teleport_locations"]}


def create_measurement_log(initial_observation: dict, feature_config: FeatureConfig):
    """Initialize episode-owned public measurement history only when enabled."""
    if not getattr(feature_config, "measurement_memory", False):
        return None
    from .measurements import MeasurementLog
    log = MeasurementLog()
    log.observe_initial(initial_observation)
    return log


def with_measurement_records(observation: dict, measurement_log, *, object_relations: bool = False) -> dict:
    """Project records already received; querying this view consumes no event."""
    if measurement_log is None:
        return observation
    observation = {**observation, "measurement_records": [asdict(summary) for summary in measurement_log.summaries()]}
    if object_relations:
        observation["measurement_dialog_owner"] = measurement_log.dialog_owner
    return observation


def outcome_target(row: dict) -> list[float]:
    # Supervision is the returned action result and task feedback. These fields
    # are never passed to public_observation or to the feature encoder.
    result = row.get("action_result", {})
    ui = row.get("next_observation", {}).get("ui", {})
    messages = [str(ui.get(key, "")) for key in ("lastActionMessage", "extended_action_message")]
    if not any(messages):
        messages = [str(result.get(key, "")) for key in ("message", "successMessage", "errors")]
    # Preserve received order; explicit UUID annotations are excluded by the
    # shared parser. These are numeric contents, not pre-labeled physical fields.
    values = observed_numbers("\n".join(dict.fromkeys(messages)))[:5]
    transformed = [math.copysign(math.log1p(abs(value)), value) / 10.0 for value in values]
    presence = [1.0] * len(values) + [0.0] * (5 - len(values))
    return [float(bool(result.get("success", False))), float(row.get("training_reward", 0)),
            float(bool(row.get("terminated", False))), float(bool(values))] + \
        transformed + [0.0] * (5 - len(values)) + presence



def action_run_weights(transitions: list[dict]) -> tuple[list[float], list[bool]]:
    """One unit of imitation mass per consecutive expert-action run.

    Runs are determined over the complete episode, before any BPTT windows.
    Exit flags mark the first different action after a run of at least two;
    neither instruments nor a particular repetition count enter this rule.
    """
    actions = [canonical_action(row.get("expert_action", row["action"])) for row in transitions]
    weights, exits = [0.0] * len(actions), [False] * len(actions)
    start = 0
    while start < len(actions):
        stop = start + 1
        while stop < len(actions) and actions[stop] == actions[start]:
            stop += 1
        length = stop - start
        weights[start:stop] = [1.0 / length] * length
        if length > 1 and stop < len(actions):
            exits[stop] = True
        start = stop
    return weights, exits

def prepare_episodes(episodes: list[Episode], cache: FeatureCache) -> list[PreparedEpisode]:
    prepared = []
    for episode in episodes:
        steps = []
        memory = PublicObjectMemory()
        measurement_log = create_measurement_log(public_observation(episode, episode.transitions[0]), cache.config)
        run_weights, exits = action_run_weights(episode.transitions)
        for row_index, row in enumerate(episode.transitions):
            observation = with_measurement_records(memory.observe(public_observation(episode, row)), measurement_log,
                                                   object_relations=cache.config.object_relations)
            candidates = public_candidates(observation)
            def candidate_index(action: dict, label: str) -> int:
                target = canonical_action(action)
                matches = [i for i, candidate in enumerate(candidates) if canonical_action(candidate.action) == target]
                if len(matches) != 1:
                    raise ValueError(f"{episode.path}:{row['index']}: {label} action {target} has "
                                     f"{len(matches)} public candidates; fix the shared action generator")
                return matches[0]

            executed = candidate_index(row["action"], "executed")
            expert = candidate_index(row.get("expert_action", row["action"]), "expert")
            feature_path = cache.encode(observation, candidates, episode.path.parent)
            steps.append(PreparedStep(str(feature_path), expert, executed, outcome_target(row),
                                      run_weights[row_index], exits[row_index]))
            if measurement_log is not None:
                # This feedback belongs to the executed action, not the expert
                # label. It becomes available only to the next decision.
                measurement_log.observe(row["action"], row["action_result"], row["next_observation"],
                                        sequence=row["index"])
        prepared.append(PreparedEpisode(episode, steps))
        print(json.dumps({"prepared": str(episode.path), "transitions": len(steps),
                          "features": cache.config.mode}), flush=True)
    return prepared


def received_outcome_loss(predictions: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Action result, progress and actual returned numeric-content prediction.

    The five numeric values use signed log1p(value)/10 and are regressed only
    when received. Their presence is learned separately. Exact text and units
    remain in the source trajectory and frozen observation text features.
    """
    binary = F.binary_cross_entropy_with_logits(predictions[:, [0, 2, 3]], target[:, [0, 2, 3]],
                                                reduction="none").mean(-1)
    progress = F.mse_loss(predictions[:, 1], target[:, 1], reduction="none")
    present = target[:, 9:14]
    numeric = ((predictions[:, 4:9] - target[:, 4:9]).square() * present).sum(-1) / present.sum(-1).clamp_min(1)
    presence = F.binary_cross_entropy_with_logits(predictions[:, 9:14], present, reduction="none").mean(-1)
    return binary + progress + numeric + presence


def _detach(state):
    return {key: value.detach() for key, value in state.items()}


def run_epoch(model: DigitalPolicy, episodes: list[PreparedEpisode], *, batch_size: int,
              sequence_length: int, device: str | torch.device, mode: str = "cdm",
              persistent: bool = True, restart_workspace: bool = False, outcome_weight: float = .5, immediate_weight: float = .25,
              optimizer=None, shuffle_seed: int = 0, imitation_reduction: str = "event") -> dict:
    if imitation_reduction not in ("event", "action_run"):
        raise ValueError("imitation_reduction must be event or action_run")
    training = optimizer is not None
    model.train(training)
    order = list(range(len(episodes)))
    if training:
        random.Random(shuffle_seed).shuffle(order)
    totals = {key: 0.0 for key in ("imitation_loss", "outcome_loss", "immediate_loss", "correct", "count",
                                  "raw_imitation_loss", "run_correct", "run_mass", "exit_correct", "exit_count", "ce_mass")}
    device = torch.device(device)
    for start in range(0, len(order), batch_size):
        batch = [episodes[i] for i in order[start:start + batch_size]]
        state = model.initial(len(batch), device)
        max_length = max(len(episode.steps) for episode in batch)
        # Normalize each full episode's effective CE weights to mean one.
        # A fixed CE denominator across its windows avoids canceling a long
        # run's downweighting or overweighting a short final BPTT window.
        run_scales = torch.tensor([len(episode.steps) / sum(step.run_weight for step in episode.steps)
                                   for episode in batch], device=device)
        ce_window_count = sum(len(episode.steps) for episode in batch) / math.ceil(max_length / sequence_length)
        for window in range(0, max_length, sequence_length):
            stop = min(window + sequence_length, max_length)
            count = sum(min(stop, len(episode.steps)) - min(window, len(episode.steps)) for episode in batch)
            if training:
                optimizer.zero_grad(set_to_none=True)
            window_loss = None
            with torch.set_grad_enabled(training):
                for t in range(window, stop):
                    alive = torch.tensor([t < len(episode.steps) for episode in batch], device=device)
                    steps = [episode.steps[min(t, len(episode.steps) - 1)] for episode in batch]
                    obs, candidates = batch_features([load_features(step.feature_path) for step in steps], device)
                    targets = torch.tensor([step.expert_target for step in steps], device=device)
                    executed = torch.tensor([step.executed_target for step in steps], device=device)
                    outcomes = torch.tensor([step.outcome for step in steps], device=device)
                    run_weights = torch.tensor([step.run_weight for step in steps], device=device)
                    ce_weights = run_weights * run_scales if imitation_reduction == "action_run" else torch.ones_like(run_weights)
                    exit_mask = torch.tensor([step.repeated_run_exit for step in steps], device=device) & alive
                    logits, next_state, predicted = model(obs, candidates, state,
                                                          persistent=persistent, mode=mode, restart_workspace=restart_workspace,
                                                          outcome_indices=executed)
                    imitation = F.cross_entropy(logits, targets, reduction="none")
                    outcome = received_outcome_loss(predicted, outcomes)
                    immediate = torch.zeros_like(imitation)
                    if immediate_weight and mode == "cdm" and model.config.refinements:
                        early, _, _ = model(obs, candidates, state, persistent=persistent,
                                            refinements=0, mode=mode, restart_workspace=restart_workspace,
                                            predict_outcomes=False)
                        immediate = F.cross_entropy(early, targets, reduction="none")
                    if imitation_reduction == "event":
                        loss = ((imitation + outcome_weight * outcome + immediate_weight * immediate) * alive).sum() / count
                    else:
                        loss = ((imitation + immediate_weight * immediate) * ce_weights * alive).sum() / ce_window_count
                        loss = loss + outcome_weight * (outcome * alive).sum() / count
                    window_loss = loss if window_loss is None else window_loss + loss
                    # Finished episodes contribute neither loss nor artificial state updates.
                    state = {key: torch.where(alive.reshape((-1,) + (1,) * (value.ndim - 1)), value, state[key])
                             for key, value in next_state.items()}
                    totals["imitation_loss"] += float((imitation.detach() * ce_weights)[alive].sum())
                    totals["raw_imitation_loss"] += float(imitation.detach()[alive].sum())
                    totals["ce_mass"] += float(ce_weights[alive].sum())
                    totals["outcome_loss"] += float(outcome.detach()[alive].sum())
                    totals["immediate_loss"] += float((immediate.detach() * ce_weights)[alive].sum())
                    correct = logits.argmax(-1) == targets
                    totals["correct"] += int(correct[alive].sum())
                    totals["run_correct"] += float((correct * run_weights)[alive].sum())
                    totals["run_mass"] += float(run_weights[alive].sum())
                    totals["exit_correct"] += int(correct[exit_mask].sum())
                    totals["exit_count"] += int(exit_mask.sum())
                    totals["count"] += int(alive.sum())
                if training:
                    window_loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
            # This is truncated BPTT, not an episode reset: values of h/z survive.
            state = _detach(state)
    count = totals["count"]
    result = {"imitation_loss": totals["imitation_loss"] / totals["ce_mass"],
              "outcome_loss": totals["outcome_loss"] / count,
              "immediate_loss": totals["immediate_loss"] / totals["ce_mass"],
              "raw_imitation_loss": totals["raw_imitation_loss"] / count}
    return {**result, "loss": result["imitation_loss"] + outcome_weight * result["outcome_loss"]
                            + immediate_weight * result["immediate_loss"],
            "imitation_reduction": imitation_reduction,
            "imitation_accuracy": totals["correct"] / count,
            "raw_event_imitation_accuracy": totals["correct"] / count,
            "action_run_weighted_accuracy": totals["run_correct"] / totals["run_mass"],
            "repeated_run_exit_accuracy": totals["exit_correct"] / totals["exit_count"] if totals["exit_count"] else None,
            "repeated_run_exits": int(totals["exit_count"]), "transitions": int(count)}


def create_model(args, text_dim: int, image_dim: int):
    digital = DigitalConfig(width=args.width, text_dim=text_dim, image_dim=image_dim,
                            refinements=args.refinements, outcome_dim=14,
                            object_relations=args.object_relations)
    if args.initial_checkpoint:
        saved = torch.load(args.initial_checkpoint, map_location="cpu", weights_only=True)
        digital = DigitalConfig(**{**asdict(digital), "width": saved["config"]["width"]})
        model = DigitalPolicy.from_lifetime_checkpoint(saved, digital)
        if args.measurement_memory and model.core.config.record_dim != 8:
            raise ValueError("Measurement memory requires a source core with record_dim=8; checkpoint weights are unchanged")
        return model
    core_config = CoreConfig(event_dim=args.width, record_dim=8 if args.measurement_memory else args.width, actions=1,
                             width=args.width, slots=args.slots, heads=args.heads,
                             records=1, hidden=args.width, action_width=args.width)
    return DigitalPolicy(digital, ContinualCore(core_config, torch.zeros(1, 1)))


def load_checkpoint(path: str | Path, device: str | torch.device = "cpu"):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if bool(checkpoint["digital_config"].get("object_relations", False)) != bool(checkpoint["features"].get("object_relations", False)):
        raise ValueError("Checkpoint model and feature object_relations settings must agree")
    core = ContinualCore(CoreConfig(**checkpoint["core_config"]), checkpoint["model"]["core.action_values"])
    model = DigitalPolicy(DigitalConfig(**checkpoint["digital_config"]), core)
    model.load_state_dict(checkpoint["model"], strict=True)
    return model.to(device), checkpoint


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-trajectories", nargs="+", type=Path, required=True)
    parser.add_argument("--validation-trajectories", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--features", choices=("siglip", "lexical"), help="Defaults to checkpoint mode on resume, otherwise siglip")
    parser.add_argument("--measurement-memory", action=argparse.BooleanOptionalAction, default=None,
                        help="Include prior public measurement summaries; inherit checkpoint setting by default")
    parser.add_argument("--object-relations", action=argparse.BooleanOptionalAction, default=None,
                        help="Bind candidates to public measurement objects; requires measurement memory; inherit checkpoint setting")
    parser.add_argument("--mask-object-relations", action=argparse.BooleanOptionalAction, default=None,
                        help="Zero object identity relations with the same record reader; requires object relations; inherit checkpoint setting")
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--encoder-path", type=Path)
    parser.add_argument("--feature-device", default="cpu")
    parser.add_argument("--feature-batch-size", type=int, default=32)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--width", type=int)
    parser.add_argument("--heads", type=int)
    parser.add_argument("--slots", type=int)
    parser.add_argument("--refinements", type=int)
    parser.add_argument("--mode", choices=("cdm", "gru"))
    parser.add_argument("--restart-state", action=argparse.BooleanOptionalAction, default=None, help="Train a matched state-reset intervention")
    parser.add_argument("--restart-workspace", action=argparse.BooleanOptionalAction, default=None, help="Reset z only; keep recurrent h and public object memory")
    initialization = parser.add_mutually_exclusive_group()
    initialization.add_argument("--initial-checkpoint", type=Path, help="Released lifetime core; its width takes precedence")
    initialization.add_argument("--resume-checkpoint", type=Path, help="Continue saved DigitalPolicy weights and feature/config schema with a fresh optimizer")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--sequence-length", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--imitation-reduction", choices=("event", "action_run"), default="event")
    parser.add_argument("--outcome-weight", type=float, default=.5)
    parser.add_argument("--immediate-weight", type=float, default=.25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args(argv)
    if min(args.epochs, args.batch_size, args.sequence_length, args.threads, args.feature_batch_size) <= 0:
        parser.error("Epochs, batch sizes, sequence length and threads must be positive")
    if (args.output / "training.jsonl").exists():
        parser.error("Choose a new output directory; an existing training log would be overwritten")
    return args


def initialize_training(args):
    """Resolve inherited settings before loading data or computing features."""
    model = source = None
    defaults = {"width": 128, "heads": 4, "slots": 8, "refinements": 1,
                "mode": "cdm", "restart_state": False, "restart_workspace": False}
    if args.resume_checkpoint:
        model, source = load_checkpoint(args.resume_checkpoint, "cpu")
        feature_config = FeatureConfig(**{"schema": 1, "time_basis": "wall_clock", **source["features"]})
        if args.features is not None and args.features != feature_config.mode:
            raise ValueError("--features must match the resumed checkpoint feature configuration")
        for key, value in {"width": model.config.width, "heads": model.core.config.heads,
                           "slots": model.core.config.slots, "refinements": model.config.refinements}.items():
            requested = getattr(args, key)
            if requested is not None and requested != value:
                raise ValueError(f"--{key}={requested} conflicts with resumed checkpoint value {value}")
            defaults[key] = value
        for key in ("mode", "restart_state", "restart_workspace"):
            defaults[key] = source["training_args"].get(key, defaults[key])
        dimension = 768 if feature_config.mode == "siglip" else feature_config.lexical_dim
        expected = {"text_dim": dimension, "image_dim": dimension, "numeric_dim": 32,
                    "candidate_numeric_dim": 8, "outcome_dim": 14}
        incompatible = {key: (getattr(model.config, key), value) for key, value in expected.items()
                        if getattr(model.config, key) != value}
        if incompatible:
            raise ValueError(f"Resumed checkpoint dimensions do not match this feature/target schema: {incompatible}")
    else:
        feature_config = FeatureConfig(mode=args.features or "siglip")
    overrides = {key: getattr(args, key) for key in ("measurement_memory", "object_relations", "mask_object_relations")
                 if getattr(args, key) is not None}
    if overrides:
        feature_config = replace(feature_config, **overrides)
    args.measurement_memory = getattr(feature_config, "measurement_memory", False)
    args.object_relations = feature_config.object_relations
    args.mask_object_relations = feature_config.mask_object_relations
    if model is not None and args.measurement_memory and model.core.config.record_dim != 8:
        raise ValueError("Measurement memory requires a source core with record_dim=8; checkpoint weights are unchanged")
    if model is not None and model.config.object_relations != args.object_relations:
        model = DigitalPolicy.with_object_relations(model, enabled=args.object_relations)
    args.features = feature_config.mode
    for key, value in defaults.items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    return model, source, feature_config


def main(argv=None):
    args = parse_args(argv)
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    model, source, feature_config = initialize_training(args)
    train = load_episodes(args.train_trajectories)
    validation = load_episodes(args.validation_trajectories)
    validate_split(train, validation)
    historical_train = {tuple(identity) for identity in source.get("train_instances", [])} if source else set()
    historical_validation = {tuple(identity) for identity in source.get("validation_instances", [])} if source else set()
    if historical_train & {episode.identity for episode in validation}:
        raise ValueError("Validation instances overlap the resumed checkpoint's training history")
    if historical_validation & {episode.identity for episode in train}:
        raise ValueError("New training instances overlap the resumed checkpoint's held-out validation")
    args.output.mkdir(parents=True, exist_ok=True)
    cache = FeatureCache(args.feature_cache, feature_config, args.feature_device,
                         args.encoder_path, args.feature_batch_size)
    start = perf_counter()
    train_prepared, validation_prepared = prepare_episodes(train, cache), prepare_episodes(validation, cache)
    cache.close()
    model = (model if model is not None else create_model(args, cache.text_dim, cache.image_dim)).to(args.device)
    resume = None if source is None else {"path": str(args.resume_checkpoint.resolve()),
                                         "sha256": sha256(args.resume_checkpoint.read_bytes()).hexdigest(),
                                         "source_epoch": source["epoch"], "optimizer_state": "fresh",
                                         "source_training_phase": source.get("training_phase", "sequence_imitation")}
    manifest = lambda episodes: [
        {"path": str(episode.path), "sha256": sha256(episode.path.read_bytes()).hexdigest(),
         "scenario": episode.metadata["scenario"], "difficulty": episode.metadata["difficulty"],
         "seed": episode.metadata["seed"], "transitions": len(episode.transitions),
         "expert_labeled_transitions": sum("expert_action" in row for row in episode.transitions),
         "collection": episode.metadata} for episode in episodes]
    metadata = {"args": {key: str(value) if isinstance(value, Path) else
                           [str(item) for item in value] if isinstance(value, list) else value
                           for key, value in vars(args).items()},
                "features": asdict(feature_config), "resume": resume, "core_config": asdict(model.core.config),
                "digital_config": asdict(model.config), "train": manifest(train),
                "validation": manifest(validation), "split": "explicit_disjoint_instances",
                "rule_holdout": "not asserted; different seeds can share scientific rules",
                "outcome_targets": ["action_success", "progress_delta", "task_terminated", "numeric_result_present"]
                                   + [f"received_numeric_{i}_signed_log1p_div10" for i in range(5)]
                                   + [f"received_numeric_{i}_present" for i in range(5)],
                "state_training": "episode_ordered_truncated_bptt", "task_completion_metric": False,
                "imitation_target": "expert_action when provided, otherwise executed action",
                "outcome_action": "executed action only",
                "imitation_reduction": args.imitation_reduction,
                "action_run_weighting": "1/run_length over full episode; CE weights normalized to episode mean one",
                "action_run_window_normalization": "fixed batch event count divided by number of BPTT windows",
                "outcome_reduction": "unchanged equal weighting per real event",
                "repeated_run_exit_definition": "first changed expert action after a consecutive run of length greater than one",
                "revision": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                             text=True, check=True).stdout.strip(),
                "parameters": sum(parameter.numel() for parameter in model.parameters()),
                "feature_seconds": perf_counter() - start}
    (args.output / "config.json").write_text(json.dumps(metadata, indent=2) + "\n")
    if args.prepare_only:
        return
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    kwargs = dict(batch_size=args.batch_size, sequence_length=args.sequence_length, device=args.device,
                  mode=args.mode, persistent=not args.restart_state, restart_workspace=args.restart_workspace,
                  outcome_weight=args.outcome_weight,
                  immediate_weight=args.immediate_weight, imitation_reduction=args.imitation_reduction)
    best = float("inf")
    source_epoch = source["epoch"] if source else 0
    train_instances = historical_train | {episode.identity for episode in train}
    validation_instances = historical_validation | {episode.identity for episode in validation}
    with (args.output / "training.jsonl").open("x") as log:
        for epoch in range(args.epochs):
            training = run_epoch(model, train_prepared, optimizer=optimizer, shuffle_seed=args.seed + epoch, **kwargs)
            validation_stats = run_epoch(model, validation_prepared, **kwargs)
            row = {"epoch": source_epoch + epoch + 1, "stage_epoch": epoch + 1, "training": training, "validation": validation_stats,
                   "elapsed_seconds": perf_counter() - start}
            log.write(json.dumps(row) + "\n")
            log.flush()
            print(json.dumps(row), flush=True)
            checkpoint = {"model": model.state_dict(), "core_config": asdict(model.core.config),
                          "digital_config": asdict(model.config), "features": asdict(feature_config),
                          "training_args": metadata["args"], "epoch": row["epoch"], "metrics": row,
                          "resume": resume, "training_phase": "sequence_imitation",
                          "train_instances": [list(identity) for identity in sorted(train_instances)],
                          "validation_instances": [list(identity) for identity in sorted(validation_instances)]}
            torch.save(checkpoint, args.output / "last.pt")
            if validation_stats["loss"] < best:
                best = validation_stats["loss"]
                torch.save(checkpoint, args.output / "best.pt")


if __name__ == "__main__":
    main()
