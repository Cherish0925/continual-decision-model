"""Joint action/forecast learning from successful and unsuccessful executions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
from time import perf_counter

import torch
import torch.nn.functional as F

from cdm.digital import PublicObjectMemory, public_candidates
from .train_joint_policy import select_state
from .belief_actions import BELIEF_DIM, BeliefFeatures
from .calibration import CalibrationLog
from .calibration_mixture import load_mixture_checkpoint
from .features import FeatureCache, FeatureConfig, batch_features, canonical_action, load_features
from .predictive_actions import HORIZONS, build
from .train import action_run_weights, load_checkpoint, load_episodes, outcome_target, public_observation, with_measurement_records


def future_targets(transitions, gamma=.99, action_cost=.001):
    """Finite-horizon observed returns; time-limit tails are unknown, not zero."""
    rewards = [float(row.get("training_reward", 0)) - action_cost for row in transitions]
    terminal = bool(transitions[-1].get("terminated"))
    targets, masks = [], []
    for start in range(len(rewards)):
        target, mask = [], []
        for horizon in HORIZONS:
            stop = min(start + horizon, len(rewards))
            target.append(sum(gamma ** (i - start) * rewards[i] for i in range(start, stop)))
            mask.append(float(terminal or start + horizon <= len(rewards)))
        targets.append(target)
        masks.append(mask)
    return targets, masks


def supervised_rows(episode, cached_rows, *, gamma, action_cost):
    transitions = episode.transitions
    if len(cached_rows) != len(transitions):
        raise ValueError("Observation and execution histories differ")
    targets, masks = future_targets(transitions, gamma, action_cost)
    summary = json.loads((episode.path.parent / "summary.json").read_text())
    successful = all(m["completed_successfully"] for m in summary["task_metrics"])
    rows = []
    for cached, transition, returns, return_mask in zip(cached_rows, transitions, targets, masks):
        rows.append({**cached, "outcome": outcome_target(transition), "returns": returns,
                     "return_mask": return_mask, "imitate": successful})
    return {"identity": list(episode.identity), "source": str(episode.path), "successful": successful,
            "rows": rows}


@torch.no_grad()
def prepare(args):
    args.output.mkdir(parents=True, exist_ok=True)
    # Reuse received feature/evidence caches without replaying a different model.
    if args.base_manifest:
        existing = torch.load(args.base_manifest, map_location="cpu", weights_only=True)
        for split, episodes in existing.items():
            for i, cached in enumerate(episodes):
                if i % args.parts != args.part:
                    continue
                path = args.output / split / ("_".join(map(str, cached["identity"])).replace(" ", "_") + ".pt")
                if path.exists():
                    continue
                episode, = load_episodes([Path(cached["source"])])
                prepared = supervised_rows(episode, cached["rows"], gamma=args.gamma, action_cost=args.action_cost)
                path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(prepared, path)
    if not args.trajectories:
        return
    reader = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    # Nine tiny affine fits are faster on CPU than synchronizing dozens of
    # scalar CUDA reads per candidate. Perception remains on the chosen GPU.
    mixture, meta = load_mixture_checkpoint(reader["mixture_checkpoint"], "cpu")
    _, source = load_checkpoint(meta["source_checkpoint"], "cpu")
    cache = FeatureCache(args.feature_cache, FeatureConfig(**source["features"]), args.device, args.encoder_path)
    for i, episode in enumerate(load_episodes(args.trajectories)):
        if i % args.parts != args.part:
            continue
        name = "_".join(map(str, episode.identity)).replace(" ", "_")
        path = args.output / args.split / f"{name}.pt"
        if path.exists():
            continue
        started = perf_counter()
        memory, log = PublicObjectMemory(), CalibrationLog()
        beliefs = BeliefFeatures(log, mixture)
        weights, _ = action_run_weights(episode.transitions)
        rows = []
        for transition, weight in zip(episode.transitions, weights):
            observation = public_observation(episode, transition)
            log.observe_initial(observation)
            observation = with_measurement_records(memory.observe(observation), log.measurements,
                                                   object_relations=cache.config.object_relations)
            candidates = public_candidates(observation)
            feature_path = cache.encode(observation, candidates, episode.path.parent)
            chosen = canonical_action(transition["action"])
            target = [j for j, action in enumerate(candidates) if canonical_action(action.action) == chosen]
            if len(target) != 1:
                raise ValueError(f"Executed action must have one public candidate: {chosen}")
            rows.append({"feature_path": str(feature_path), "belief": beliefs.encode(observation, candidates).half(),
                         "target": target[0], "weight": weight})
            log.observe(transition["action"], transition["action_result"], transition["next_observation"],
                        sequence=transition["index"])
            beliefs.observe(transition)
        prepared = supervised_rows(episode, rows, gamma=args.gamma, action_cost=args.action_cost)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(prepared, path)
        print(json.dumps({"prepared": str(path), "actions": len(rows), "successful": prepared["successful"],
                          "seconds": perf_counter() - started}), flush=True)
    cache.close()


def forecast_loss(prediction, outcome, returns, mask):
    """Per-row received-result and future-return errors, with missing values masked."""
    binary = [0, 2, 3, 9, 10, 11, 12, 13]
    received = F.binary_cross_entropy_with_logits(prediction[..., binary].float(),
                                                 outcome[..., binary], reduction="none").mean(-1)
    values = F.smooth_l1_loss(prediction[..., 4:9].float(), outcome[..., 4:9], reduction="none")
    received = received + (values * outcome[..., 9:14]).sum(-1) / outcome[..., 9:14].sum(-1).clamp_min(1)
    # Reward is only a training target, never an acting input.
    received = received + F.smooth_l1_loss(prediction[..., 1].float() * 10, outcome[..., 1] * 10,
                                         reduction="none")
    future = F.smooth_l1_loss(prediction[..., 14:].float() * 10, returns * 10, reduction="none")
    future = (future * mask).sum(-1) / mask.sum(-1).clamp_min(1)
    return received, future


def epoch(model, episodes, args, *, optimizer=None, number=0):
    training = optimizer is not None
    model.train(training)
    order = list(range(len(episodes)))
    if training:
        random.Random(args.seed + number).shuffle(order)
    totals = {key: 0. for key in ("loss", "imitation", "received", "future", "count", "imitated")}
    correct = [0] * (args.cycles + 1)
    prediction_error = [0.] * (args.cycles + 1)
    changed = 0
    for offset in range(0, len(order), args.batch_size):
        selected = [episodes[i] for i in order[offset:offset + args.batch_size]]
        lengths = [len(e["rows"]) for e in selected]
        scales = [len(e["rows"]) / sum(r["weight"] for r in e["rows"]) for e in selected]
        state = model.digital.initial(len(selected), args.device)
        for start in range(0, max(lengths), args.sequence_length):
            losses = []
            for time in range(start, min(start + args.sequence_length, max(lengths))):
                rows = [e["rows"][min(time, length - 1)] for e, length in zip(selected, lengths)]
                observations, candidates = batch_features([load_features(r["feature_path"]) for r in rows], args.device)
                belief = torch.zeros(*candidates["present"].shape, BELIEF_DIM, device=args.device)
                for i, row in enumerate(rows):
                    belief[i, :len(row["belief"])] = row["belief"].to(args.device)
                target = torch.tensor([r["target"] for r in rows], device=args.device)
                valid = torch.tensor([time < length for length in lengths], device=args.device)
                imitation_weight = torch.tensor([r["weight"] * scale * r["imitate"]
                                                  for r, scale in zip(rows, scales)], device=args.device) * valid
                outcome = torch.tensor([r["outcome"] for r in rows], device=args.device)
                returns = torch.tensor([r["returns"] for r in rows], device=args.device)
                return_mask = torch.tensor([r["return_mask"] for r in rows], device=args.device)
                with torch.set_grad_enabled(training), torch.autocast(torch.device(args.device).type,
                        dtype=torch.bfloat16, enabled=args.bf16):
                    logits, forecasts, branches = model.proposals(observations, candidates, belief, state,
                                                                  cycles=args.cycles)
                    chosen_predictions = forecasts.gather(2, target[:, None, None, None].expand(
                        -1, args.cycles + 1, 1, forecasts.shape[-1])).squeeze(2)
                    ce = torch.stack([F.cross_entropy(logits[:, d].float(), target, reduction="none")
                                      for d in range(args.cycles + 1)], -1)
                    received, future = forecast_loss(chosen_predictions, outcome[:, None].expand_as(chosen_predictions[..., :14]),
                                                     returns[:, None].expand_as(chosen_predictions[..., 14:]),
                                                     return_mask[:, None].expand_as(chosen_predictions[..., 14:]))
                    per_row = ce.mean(-1) * imitation_weight + args.prediction_weight * (
                        received.mean(-1) + future.mean(-1)) * valid
                    losses.append(per_row.sum() / len(rows))
                    indices = torch.randint(args.cycles + 1, (len(rows),), device=args.device) if training else \
                        torch.full((len(rows),), args.cycles, device=args.device, dtype=torch.long)
                    state = select_state(branches, indices)
                totals["loss"] += float(per_row.detach().sum())
                totals["imitation"] += float((ce.detach().mean(-1) * imitation_weight).sum())
                totals["received"] += float((received.detach().mean(-1) * valid).sum())
                totals["future"] += float((future.detach().mean(-1) * valid).sum())
                totals["count"] += int(valid.sum())
                totals["imitated"] += int((imitation_weight > 0).sum())
                for depth in range(args.cycles + 1):
                    correct[depth] += int(((logits[:, depth].argmax(-1) == target) & (imitation_weight > 0)).sum())
                    prediction_error[depth] += float(((received[:, depth] + future[:, depth]).detach() * valid).sum())
                changed += int(((logits[:, 0].argmax(-1) != logits[:, -1].argmax(-1)) & valid).sum())
            if training:
                optimizer.zero_grad(set_to_none=True)
                torch.stack(losses).mean().backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                if not torch.isfinite(norm):
                    raise FloatingPointError("Non-finite prediction/action gradient")
                optimizer.step()
            state = {key: value.detach() for key, value in state.items()}
        print(json.dumps({"epoch": number, "training": training, "episodes": min(offset + args.batch_size, len(order)),
                          "actions": totals["count"], "loss": totals["loss"] / totals["count"]}), flush=True)
    return {**{key: value / totals["count"] for key, value in totals.items() if key not in ("count", "imitated")},
            "actions": int(totals["count"]), "imitation_actions": int(totals["imitated"]),
            "accuracy_by_cycle": [n / max(totals["imitated"], 1) for n in correct],
            "prediction_error_by_cycle": [n / totals["count"] for n in prediction_error],
            "fraction_action_changed": changed / totals["count"]}


def train(args):
    torch.manual_seed(args.seed)
    model, _, _ = build(args.checkpoint, args.device, feedback=args.feedback)
    episodes = {split: [torch.load(p, map_location="cpu", weights_only=True) for p in sorted((args.prepared / split).glob("*.pt"))]
                for split in ("train", "validation")}
    ids = {split: {tuple(e["identity"]) for e in values} for split, values in episodes.items()}
    if not all(ids.values()) or ids["train"] & ids["validation"]:
        raise ValueError("Need separate nonempty train and validation instances")
    args.output.mkdir(parents=True, exist_ok=True)
    settings = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    settings["trajectories"] = [str(p) for p in args.trajectories]
    settings.update(train_instances=sorted(ids["train"]), validation_instances=sorted(ids["validation"]),
                    trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                    data={split: {"episodes": len(values), "successful": sum(e["successful"] for e in values),
                                  "actions": sum(len(e["rows"]) for e in values)} for split, values in episodes.items()})
    (args.output / "config.json").write_text(json.dumps(settings, indent=2) + "\n")
    core = list(model.digital.parameters())
    core_ids = {id(p) for p in core}
    optimizer = torch.optim.AdamW([{"params": core, "lr": args.core_lr},
                                  {"params": [p for p in model.parameters() if id(p) not in core_ids], "lr": args.lr}],
                                 weight_decay=1e-4)
    for number in range(1, args.epochs + 1):
        started = perf_counter()
        trained = epoch(model, episodes["train"], args, optimizer=optimizer, number=number)
        validated = epoch(model, episodes["validation"], args, number=number)
        result = {"epoch": number, "training": trained, "validation": validated, "seconds": perf_counter() - started}
        with (args.output / "training.jsonl").open("a") as stream:
            stream.write(json.dumps(result) + "\n")
        saved = {"model": model.state_dict(), "base_checkpoint": str(args.checkpoint), "epoch": number,
                 "model_config": {"feedback": args.feedback, "candidates_per_cycle": model.candidates_per_cycle},
                 "training": settings, "metrics": result}
        torch.save(saved, args.output / f"epoch{number}.pt")
        torch.save(saved, args.output / "last.pt")
        print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--base-manifest", type=Path)
    parser.add_argument("--dataset", type=Path, help="JSON listing released train and validation trajectory sources")
    parser.add_argument("--trajectories", nargs="*", type=Path, default=[])
    parser.add_argument("--split", choices=("train", "validation"), default="train")
    parser.add_argument("--part", type=int, default=0)
    parser.add_argument("--parts", type=int, default=1)
    parser.add_argument("--prepared", type=Path)
    parser.add_argument("--feature-cache", type=Path, default=Path("runs/discoveryworld/features"))
    parser.add_argument("--encoder-path", type=Path, default=Path("data/cache/siglip2"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--feedback", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--sequence-length", type=int, default=8)
    parser.add_argument("--cycles", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--core-lr", type=float, default=1e-5)
    parser.add_argument("--prediction-weight", type=float, default=.5)
    parser.add_argument("--gamma", type=float, default=.99)
    parser.add_argument("--action-cost", type=float, default=.001)
    parser.add_argument("--seed", type=int, default=151)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.dataset:
        args.trajectories = [Path(row["source"]) for row in json.loads(args.dataset.read_text())[args.split]]
    if not args.prepare and (args.prepared is None or (args.output / "training.jsonl").exists()):
        parser.error("Specify prepared data and a fresh training output")
    torch.set_num_threads(args.threads)
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.enable_cudnn_sdp(False)
    prepare(args) if args.prepare else train(args)


if __name__ == "__main__":
    main()
