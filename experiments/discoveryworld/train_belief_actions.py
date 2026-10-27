"""Learn action choices from a frozen continuing CDM and its causal beliefs.

The first comparison holds the source recurrence and numerical model fixed.
Only the action readout learns. Prepared states follow executed trajectories;
expert labels never enter the evidence view. Native evaluation, not imitation
accuracy, determines whether the new reader enables autonomous experiments.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
from time import perf_counter

import torch
import torch.nn.functional as F

from cdm.digital import PublicObjectMemory, public_candidates
from .belief_actions import BELIEF_DIM, BeliefActionHead, BeliefFeatures
from .calibration import CalibrationLog
from .calibration_mixture import load_mixture_checkpoint
from .features import FeatureCache, FeatureConfig, batch_features, canonical_action, load_features
from .train import action_run_weights, load_checkpoint, load_episodes, public_observation, with_measurement_records


@torch.no_grad()
def prepare(args):
    mixture, mixture_metadata = load_mixture_checkpoint(args.mixture_checkpoint, args.device)
    model, source = load_checkpoint(mixture_metadata["source_checkpoint"], args.device)
    model.eval().requires_grad_(False)
    cache = FeatureCache(args.feature_cache, FeatureConfig(**source["features"]), args.device, args.encoder_path)
    episodes = load_episodes(args.trajectories)
    args.prepared.mkdir(parents=True, exist_ok=True)
    for index, episode in enumerate(episodes):
        if index % args.parts != args.part:
            continue
        name = "_".join(str(x).replace(" ", "_") for x in episode.identity) + ".pt"
        output = args.prepared / name
        if output.exists():
            continue
        log, memory = CalibrationLog(), PublicObjectMemory()
        beliefs = BeliefFeatures(log, mixture)
        state = model.initial(1, args.device)
        weights, exits = action_run_weights(episode.transitions)
        rows = []
        started = perf_counter()
        for row, weight, is_exit in zip(episode.transitions, weights, exits):
            observation = public_observation(episode, row)
            log.observe_initial(observation)
            observation = with_measurement_records(memory.observe(observation), log.measurements,
                                                   object_relations=cache.config.object_relations)
            candidates = public_candidates(observation)
            path = cache.encode(observation, candidates, episode.path.parent)
            obs, features = batch_features([load_features(str(path))], args.device)
            logits, state, _ = model(obs, features, state, mode=source["training_args"]["mode"],
                                    persistent=not source["training_args"].get("restart_state", False),
                                    restart_workspace=source["training_args"].get("restart_workspace", False),
                                    predict_outcomes=False)
            action = canonical_action(row.get("expert_action", row["action"]))
            target = [i for i, candidate in enumerate(candidates) if canonical_action(candidate.action) == action]
            if len(target) != 1:
                raise ValueError(f"Expert action has {len(target)} candidates: {action}")
            evidence = beliefs.encode(observation, candidates)
            assert evidence.shape == (len(candidates), BELIEF_DIM)
            rows.append({"state": model.core.state_features(state["h"], state["z"])[0].cpu().half(),
                         "text": features["text"][0].cpu().half(), "belief": evidence.half(),
                         "base_logits": logits[0].cpu().float(), "target": target[0],
                         "weight": weight, "run_exit": is_exit, "sequence": row["index"]})
            log.observe(row["action"], row["action_result"], row["next_observation"], sequence=row["index"])
            beliefs.observe(row)
        torch.save({"identity": list(episode.identity), "source": str(episode.path),
                    "mixture_checkpoint": str(args.mixture_checkpoint), "rows": rows}, output)
        print(json.dumps({"prepared": str(output), "steps": len(rows),
                          "seconds": perf_counter() - started}), flush=True)
    cache.close()


def read_prepared(root):
    episodes = [torch.load(path, map_location="cpu", weights_only=True) for path in sorted(root.glob("*.pt"))]
    # Successful reference traces provide action imitation. Unsuccessful
    # experiments remain in the raw collection for outcome/recovery learning.
    episodes = [episode for episode in episodes if all(row["completed_successfully"] for row in
                json.loads((Path(episode["source"]).parent / "summary.json").read_text())["task_metrics"])]
    if not episodes:
        raise ValueError(f"No prepared episodes in {root}")
    return episodes


def batch(rows, device):
    count = max(len(row["base_logits"]) for row in rows)
    state = torch.stack([row["state"] for row in rows]).to(device).float()
    text = torch.zeros(len(rows), count, rows[0]["text"].shape[-1], device=device)
    belief = torch.zeros(len(rows), count, BELIEF_DIM, device=device)
    logits = torch.full((len(rows), count), -torch.inf, device=device)
    for index, row in enumerate(rows):
        n = len(row["base_logits"])
        text[index, :n] = row["text"].to(device)
        belief[index, :n] = row["belief"].to(device)
        logits[index, :n] = row["base_logits"].to(device)
    target = torch.tensor([row["target"] for row in rows], device=device)
    weight = torch.tensor([row["weight"] for row in rows], device=device)
    return state, text, belief, logits, target, weight


def epoch(model, rows, device, batch_size, optimizer=None, seed=0):
    order = list(range(len(rows)))
    if optimizer is not None:
        random.Random(seed).shuffle(order)
    totals = dict(loss=0., weight=0., correct=0, exits=0, exits_correct=0, count=0)
    for start in range(0, len(rows), batch_size):
        chosen = [rows[i] for i in order[start:start + batch_size]]
        state, text, belief, logits, targets, weights = batch(chosen, device)
        with torch.set_grad_enabled(optimizer is not None):
            prediction = model(state, text, belief, logits)
            losses = F.cross_entropy(prediction, targets, reduction="none")
            loss = (losses * weights).sum() / weights.sum()
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step()
        correct = prediction.argmax(-1).eq(targets).cpu()
        totals["loss"] += float((losses * weights).sum().detach())
        totals["weight"] += float(weights.sum())
        totals["correct"] += int(correct.sum())
        totals["count"] += len(chosen)
        totals["exits"] += sum(row["run_exit"] for row in chosen)
        totals["exits_correct"] += sum(bool(ok) and row["run_exit"] for ok, row in zip(correct, chosen))
    return {"loss": totals["loss"] / totals["weight"], "accuracy": totals["correct"] / totals["count"],
            "exit_accuracy": totals["exits_correct"] / max(totals["exits"], 1), "steps": totals["count"]}


def train(args):
    training, validation = read_prepared(args.prepared), read_prepared(args.validation)
    train_ids = {tuple(episode["identity"]) for episode in training}
    val_ids = {tuple(episode["identity"]) for episode in validation}
    if train_ids & val_ids:
        raise ValueError("Prepared training and validation instances overlap")
    train_rows = [row for episode in training for row in episode["rows"]]
    val_rows = [row for episode in validation for row in episode["rows"]]
    config = dict(state_dim=train_rows[0]["state"].shape[-1], text_dim=train_rows[0]["text"].shape[-1],
                  width=args.width, belief_mode=args.belief_mode)
    torch.manual_seed(args.seed)
    model = BeliefActionHead(**config).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    args.output.mkdir(parents=True, exist_ok=True)
    settings = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    settings.update(train_instances=[list(x) for x in sorted(train_ids)],
                    validation_instances=[list(x) for x in sorted(val_ids)], parameters=sum(p.numel() for p in model.parameters()),
                    objective="action_run_weighted_imitation", source_core="frozen_continuing_state",
                    online_numeric_control="fixed_mixture_prior_and_explicit_native_primitive")
    (args.output / "config.json").write_text(json.dumps(settings, indent=2) + "\n")
    for number in range(args.epochs + 1):
        started = perf_counter()
        trained = epoch(model, train_rows, args.device, args.batch_size,
                        optimizer if number else None, args.seed + number)
        checked = epoch(model, val_rows, args.device, args.batch_size)
        row = dict(epoch=number, training=trained, validation=checked, seconds=perf_counter() - started)
        print(json.dumps(row), flush=True)
        with (args.output / "training.jsonl").open("a") as stream:
            stream.write(json.dumps(row) + "\n")
        torch.save({"model": model.state_dict(), "config": config,
                    "mixture_checkpoint": str(args.mixture_checkpoint), "epoch": number,
                    "train_instances": settings["train_instances"], "validation_instances": settings["validation_instances"],
                    "metrics": row, "training": settings}, args.output / ("initial.pt" if number == 0 else "last.pt"))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--mixture-checkpoint", type=Path, required=True)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--trajectories", type=Path, nargs="+")
    parser.add_argument("--validation", type=Path)
    parser.add_argument("--feature-cache", type=Path, default=Path("runs/discoveryworld/features"))
    parser.add_argument("--encoder-path", type=Path, default=Path("data/cache/siglip2"))
    parser.add_argument("--part", type=int, default=0)
    parser.add_argument("--parts", type=int, default=1)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=61)
    parser.add_argument("--belief-mode", choices=("full", "observations", "none"), default="full")
    args = parser.parse_args(argv)
    if args.prepare and not args.trajectories:
        parser.error("Preparation requires trajectories")
    if not args.prepare and (args.validation is None or args.output is None):
        parser.error("Training requires validation and output")
    if not args.prepare and (args.output / "training.jsonl").exists():
        parser.error("Choose a new training output")
    torch.set_num_threads(args.threads)
    prepare(args) if args.prepare else train(args)


if __name__ == "__main__":
    main()
