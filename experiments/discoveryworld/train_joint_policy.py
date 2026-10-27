"""Sequence training of CDM state, action readout and compute allocation.

Uses the same successful reference trajectories as the established action
reader, but replays observations through the current GRU/workspace. Cached
frozen perception and received numerical evidence are reusable; cached old
recurrent states are not. The adaptive objective is an imitation warm start,
not evidence of learned long-horizon computation value.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
from time import perf_counter

import torch
import torch.nn.functional as F

from .adaptive_compute import BELIEF_DIM, DEPTHS, from_reader
from .features import FeatureCache, FeatureConfig, batch_features, load_features
from .train import load_episodes, prepare_episodes
from .train_belief_actions import read_prepared


def prepare(args):
    _, source, _ = from_reader(args.checkpoint, "cpu")
    cache = FeatureCache(args.feature_cache, FeatureConfig(**source["features"]), args.device, args.encoder_path)
    result = {}
    for split, root in (("train", args.prepared), ("validation", args.validation)):
        result[split] = []
        for existing in read_prepared(root):
            episode = load_episodes([Path(existing["source"])])[0]
            replay = prepare_episodes([episode], cache)[0]
            if len(replay.steps) != len(existing["rows"]):
                raise ValueError("Evidence and observation sequences differ")
            rows = []
            for step, evidence in zip(replay.steps, existing["rows"]):
                if step.expert_target != evidence["target"]:
                    raise ValueError("Candidate ordering differs from prepared evidence")
                rows.append(dict(feature_path=step.feature_path, belief=evidence["belief"],
                                 target=step.expert_target, weight=step.run_weight))
            result[split].append(dict(identity=existing["identity"], source=existing["source"], rows=rows))
    train_ids = {tuple(e["identity"]) for e in result["train"]}
    if train_ids & {tuple(e["identity"]) for e in result["validation"]}:
        raise ValueError("Training and validation overlap")
    cache.close()
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, args.manifest)
    print(json.dumps({"manifest": str(args.manifest),
                      **{k: {"episodes": len(v), "actions": sum(len(e['rows']) for e in v)}
                         for k, v in result.items()}}), flush=True)


def select_state(states, indices):
    return {key: torch.stack([state[key] for state in states], 1)[
        torch.arange(len(indices), device=indices.device), indices] for key in states[0]}


def epoch(model, episodes, args, *, optimizer=None, number=0):
    model.train(optimizer is not None)
    order = list(range(len(episodes)))
    if optimizer is not None:
        random.Random(args.seed + number).shuffle(order)
    totals = dict(loss=0., mass=0., correct=0, count=0, refinements=0., gradient_norm=0., windows=0)
    histogram = [0] * len(DEPTHS)
    depths = DEPTHS if args.adaptive else (1,)
    for offset in range(0, len(order), args.batch_size):
        selected = [episodes[i] for i in order[offset:offset + args.batch_size]]
        lengths = [len(e["rows"]) for e in selected]
        state = model.digital.initial(len(selected), args.device)
        scales = [len(e["rows"]) / sum(r["weight"] for r in e["rows"]) for e in selected]
        for start in range(0, max(lengths), args.sequence_length):
            losses = []
            for time in range(start, min(start + args.sequence_length, max(lengths))):
                rows = [e["rows"][min(time, length - 1)] for e, length in zip(selected, lengths)]
                observations, candidates = batch_features([load_features(r["feature_path"]) for r in rows], args.device)
                belief = torch.zeros(*candidates["present"].shape, BELIEF_DIM, device=args.device)
                for i, row in enumerate(rows):
                    belief[i, :len(row["belief"])] = row["belief"].to(args.device)
                target = torch.tensor([r["target"] for r in rows], device=args.device)
                weight = torch.tensor([r["weight"] * scale * (time < length)
                                       for r, scale, length in zip(rows, scales, lengths)], device=args.device)
                with torch.set_grad_enabled(optimizer is not None), torch.autocast(
                        torch.device(args.device).type, dtype=torch.bfloat16, enabled=args.bf16):
                    allocation, logits, branches = model.proposals(observations, candidates, belief, state, depths=depths)
                    ce = torch.stack([F.cross_entropy(logits[:, i].float(), target, reduction="none")
                                      for i in range(len(depths))], -1)
                    if args.adaptive:
                        probabilities = allocation.float().softmax(-1)
                        cost = ce + args.compute_cost * ce.new_tensor(DEPTHS)[None]
                        # Every depth receives training, while the allocator
                        # learns conditional action accuracy versus compute.
                        per_row = .5 * ce.mean(-1) + .5 * (probabilities * cost).sum(-1)
                        index = torch.multinomial(probabilities.detach(), 1)[:, 0] if optimizer is not None else probabilities.argmax(-1)
                    else:
                        per_row = ce[:, 0]
                        index = torch.zeros(len(rows), dtype=torch.long, device=args.device)
                    state = select_state(branches, index)
                    losses.append((per_row * weight).sum() / len(rows))
                chosen = logits[torch.arange(len(rows), device=args.device), index].argmax(-1)
                valid = torch.tensor([time < n for n in lengths], device=args.device)
                totals["loss"] += float((per_row.detach() * weight).sum())
                totals["mass"] += float(weight.sum())
                totals["correct"] += int(((chosen == target) & valid).sum())
                totals["count"] += int(valid.sum())
                for i, n in enumerate(depths):
                    count = int(((index == i) & valid).sum())
                    histogram[DEPTHS.index(n)] += count
                    totals["refinements"] += n * count
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                torch.stack(losses).mean().backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                if not torch.isfinite(norm):
                    raise FloatingPointError("Non-finite joint gradient")
                optimizer.step()
                totals["gradient_norm"] += float(norm)
                totals["windows"] += 1
            state = {key: value.detach() for key, value in state.items()}
        print(json.dumps({"epoch": number, "split": "train" if optimizer else "validation",
                          "episodes_done": min(offset + args.batch_size, len(order)),
                          "actions": totals["count"], "accuracy": totals["correct"] / totals["count"]}), flush=True)
    return {"loss": totals["loss"] / totals["mass"], "accuracy": totals["correct"] / totals["count"],
            "actions": totals["count"], "mean_refinements": totals["refinements"] / totals["count"],
            "depth_counts": dict(zip(map(str, DEPTHS), histogram)),
            "gradient_norm": totals["gradient_norm"] / max(totals["windows"], 1)}


def train(args):
    torch.manual_seed(args.seed)
    model, source, reader = from_reader(args.checkpoint, args.device)
    data = torch.load(args.manifest, map_location="cpu", weights_only=True)
    if not args.adaptive:
        model.allocator.requires_grad_(False)
    optimizer = torch.optim.AdamW([
        {"params": model.digital.parameters(), "lr": args.core_lr},
        {"params": model.reader.parameters(), "lr": args.lr},
        {"params": model.allocator.parameters(), "lr": args.lr}], weight_decay=1e-4)
    args.output.mkdir(parents=True, exist_ok=True)
    settings = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    settings.update(objective="joint_sequence_imitation_with_compute_cost" if args.adaptive else "joint_fixed_compute_sequence_imitation",
                    attention_backends="flash_efficient_math; cudnn_sdp_disabled",
                    state_replay="current_parameters; no_cached_recurrent_states",
                    online_numeric_prior="fixed", trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                    train_instances=[e["identity"] for e in data["train"]],
                    validation_instances=[e["identity"] for e in data["validation"]])
    (args.output / "config.json").write_text(json.dumps(settings, indent=2) + "\n")
    def save(name, number, metrics):
        torch.save({"model": model.state_dict(), "base_checkpoint": str(args.checkpoint),
                    "epoch": number, "training": settings, "metrics": metrics}, args.output / name)
    save("initial.pt", 0, {})
    for number in range(1, args.epochs + 1):
        started = perf_counter()
        trained = epoch(model, data["train"], args, optimizer=optimizer, number=number)
        checked = epoch(model, data["validation"], args, number=number)
        result = dict(epoch=number, training=trained, validation=checked, seconds=perf_counter() - started)
        with (args.output / "training.jsonl").open("a") as stream:
            stream.write(json.dumps(result) + "\n")
        print(json.dumps(result), flush=True)
        save(f"epoch{number}.pt", number, result)
        save("last.pt", number, result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--prepared", type=Path, default=Path("runs/discoveryworld/belief_actions/prepared/train"))
    parser.add_argument("--validation", type=Path, default=Path("runs/discoveryworld/belief_actions/prepared/validation"))
    parser.add_argument("--feature-cache", type=Path, default=Path("runs/discoveryworld/features"))
    parser.add_argument("--encoder-path", type=Path, default=Path("data/cache/siglip2"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--adaptive", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--sequence-length", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--core-lr", type=float, default=1e-5)
    parser.add_argument("--compute-cost", type=float, default=.01)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if not args.prepare and (args.output is None or (args.output / "training.jsonl").exists()):
        parser.error("Choose a new training output")
    torch.set_num_threads(args.threads)
    torch.set_float32_matmul_precision("high")
    # Variable observation lengths otherwise trigger cuDNN attention execution
    # plan construction on almost every event. Keep cuDNN GRU, and allow the
    # flash/efficient/math attention implementations for the same operation.
    torch.backends.cuda.enable_cudnn_sdp(False)
    prepare(args) if args.prepare else train(args)


if __name__ == "__main__":
    main()
