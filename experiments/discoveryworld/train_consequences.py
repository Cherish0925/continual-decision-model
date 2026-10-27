"""Improve the continuing policy from paired native action consequences.

Cached incoming states come from the fixed parent on training histories. Each
update differentiates the next assimilation and workspace computations. Parent
action distributions on the same histories provide a replay term. The paired
arm additionally learns both executed outcomes and return-weighted preferences;
the matched replay arm uses the same states and optimizer schedule.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import random

import torch
import torch.nn.functional as F

from cdm.digital import PublicObjectMemory, public_candidates
from .belief_actions import BELIEF_DIM
from .features import batch_features, canonical_action, load_features
from .predictive_actions import build
from .train import load_episodes, outcome_target, public_observation
from .train_predictive_actions import forecast_loss, future_targets


def parent(checkpoint, device):
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model, _, _ = build(saved["base_checkpoint"], device, feedback=saved["model_config"]["feedback"])
    model.load_state_dict(saved["model"])
    return model, saved


def batch(rows, device):
    obs, candidates = batch_features([load_features(r["feature_path"]) for r in rows], device)
    belief = torch.zeros(*candidates["present"].shape, BELIEF_DIM, device=device)
    for i, row in enumerate(rows):
        belief[i, :len(row["belief"])] = row["belief"].to(device)
    state = {k: torch.stack([r["state"][k] for r in rows]).to(device) for k in rows[0]["state"]}
    return obs, candidates, belief, state


def branch_targets(root, gamma, action_cost):
    comparisons = json.loads((root / "comparison.json").read_text())
    grouped = defaultdict(dict)
    for row in comparisons:
        if row["error"] or not row["prefix_replay_matched"]:
            raise ValueError("Action supervision requires successfully replayed histories")
        grouped[(str(Path(row["source"]).resolve()), row["prefix"])][row["rank"]] = row
    result = {}
    for (source, prefix), alternatives in grouped.items():
        if set(alternatives) != {0, 1}:
            raise ValueError("The action pair is unfinished")
        selected = next(s for s in json.loads((root / Path(source).name / "selection.json").read_text())["states"]
                        if s["prefix"] == prefix)
        targets = []
        for rank in (0, 1):
            row = alternatives[rank]
            transitions = [json.loads(s) for s in (Path(row["output"]) / "trajectory.jsonl").read_text().splitlines()][prefix:]
            returns, masks = future_targets(transitions, gamma, action_cost)
            targets.append({"outcome": outcome_target(transitions[0]), "returns": returns[0], "return_mask": masks[0],
                            "utility": row["progress"] - action_cost * row["takeover_steps"]})
        result[(source, prefix)] = {"candidates": selected["candidate_indices"], "targets": targets}
    # Candidate positions refer to the public action order at the intervention,
    # not the rank order of a newly evaluated or padded policy batch.
    for source in {key[0] for key in result}:
        episode, = load_episodes([Path(source)])
        memory = PublicObjectMemory()
        for position, transition in enumerate(episode.transitions):
            observation = memory.observe(public_observation(episode, transition))
            if (source, position) not in result:
                continue
            candidates = public_candidates(observation)
            for rank, index in enumerate(result[(source, position)]["candidates"]):
                if canonical_action(candidates[index].action) != canonical_action(grouped[(source, position)][rank]["action"]):
                    raise ValueError("Counterfactual target and cached candidate differ")
    return result


@torch.no_grad()
def prepare(args):
    model, saved = parent(args.checkpoint, args.device)
    model.eval()
    targets = branch_targets(args.branches, saved["training"]["gamma"], args.action_cost)
    episodes = [torch.load(p, map_location="cpu", weights_only=True) for p in sorted((args.prepared / "train").glob("*.pt"))]
    contexts, found = [], set()
    for offset in range(0, len(episodes), args.batch_size):
        selected = episodes[offset:offset + args.batch_size]
        state = model.digital.initial(len(selected), args.device)
        lengths = [len(e["rows"]) for e in selected]
        for time in range(max(lengths)):
            rows = [e["rows"][min(time, n - 1)] for e, n in zip(selected, lengths)]
            obs, candidates = batch_features([load_features(r["feature_path"]) for r in rows], args.device)
            belief = torch.zeros(*candidates["present"].shape, BELIEF_DIM, device=args.device)
            for i, row in enumerate(rows):
                belief[i, :len(row["belief"])] = row["belief"].to(args.device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits, _, states = model.proposals(obs, candidates, belief, state, cycles=2)
            for i, (episode, row, length) in enumerate(zip(selected, rows, lengths)):
                key = (str(Path(episode["source"]).resolve().parent), time)
                if time >= length or (time % args.replay_stride and key not in targets):
                    continue
                count = len(row["belief"])
                item = {"feature_path": row["feature_path"], "belief": row["belief"],
                        "state": {k: v[i].detach().cpu().clone() for k, v in state.items()},
                        "parent_logits": logits[i, :, :count].float().cpu(),
                        "target": row["target"], "imitate": episode["successful"],
                        "source": key[0], "position": time, "pair": targets.get(key)}
                contexts.append(item)
                if key in targets:
                    found.add(key)
            state = states[-1]
        print(json.dumps({"prepared_episodes": offset + len(selected), "contexts": len(contexts), "pairs": len(found)}), flush=True)
    if found != targets.keys():
        raise ValueError("Some paired histories are absent from the training data")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"checkpoint": str(args.checkpoint), "rows": contexts, "pairs": len(found),
                "action_cost": args.action_cost, "replay_stride": args.replay_stride}, args.output)


def replay_loss(logits, rows):
    loss = logits.new_zeros(())
    for i, row in enumerate(rows):
        n = len(row["parent_logits"][0])
        teacher = row["parent_logits"].to(logits.device).softmax(-1)
        loss = loss + F.kl_div(logits[i, :, :n].float().log_softmax(-1), teacher, reduction="batchmean")
        if row["imitate"]:
            loss = loss + .1 * F.cross_entropy(logits[i, :, :n].float(),
                                                torch.full((logits.shape[1],), row["target"], device=logits.device))
    return loss / len(rows)


def pair_loss(logits, forecasts, rows, device):
    ranking, prediction = logits.new_zeros(()), logits.new_zeros(())
    count = 0
    for i, row in enumerate(rows):
        pair = row["pair"]
        if pair is None:
            continue
        a, b = pair["candidates"]
        targets = pair["targets"]
        difference = targets[0]["utility"] - targets[1]["utility"]
        if abs(difference) > 1e-8:
            preferred, other = (a, b) if difference > 0 else (b, a)
            ranking = ranking + abs(difference) * F.softplus(logits[i, :, other] - logits[i, :, preferred]).mean()
        for candidate, target in zip((a, b), targets):
            p = forecasts[i, :, candidate]
            received, future = forecast_loss(p, torch.tensor(target["outcome"], device=device).expand_as(p[:, :14]),
                                            torch.tensor(target["returns"], device=device).expand_as(p[:, 14:]),
                                            torch.tensor(target["return_mask"], device=device).expand_as(p[:, 14:]))
            prediction = prediction + (received + future).mean()
            count += 1
    # Preserve the scale of the measured benefit: a one-action saving must not
    # receive the same weight as rescuing a failed experiment.
    return ranking / max(count / 2, 1), prediction / max(count, 1)


def train(args):
    torch.manual_seed(args.seed)
    data = torch.load(args.contexts, map_location="cpu", weights_only=True)
    if data["checkpoint"] != str(args.checkpoint):
        raise ValueError("Context parent and training parent differ")
    rows = data["rows"]
    pairs = [r for r in rows if r["pair"] is not None]
    model, saved = parent(args.checkpoint, args.device)
    model.train()
    core = list(model.digital.parameters())
    core_ids = {id(p) for p in core}
    optimizer = torch.optim.AdamW([{"params": core, "lr": args.core_lr},
                                  {"params": [p for p in model.parameters() if id(p) not in core_ids], "lr": args.lr}],
                                 weight_decay=1e-4)
    args.output.mkdir(parents=True, exist_ok=True)
    settings = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    settings.update(contexts=len(rows), paired_states=len(pairs), gradient_history="one observed transition; parent incoming recurrent state",
                    utility=f"final task progress minus {data['action_cost']} times takeover actions",
                    replay="parent policy KL plus 0.1 successful-action imitation",
                    ranking_weight=.25, prediction_weight=.5)
    (args.output / "config.json").write_text(json.dumps(settings, indent=2) + "\n")
    for epoch in range(args.epochs):
        order = list(range(len(rows)))
        rng = random.Random(args.seed + epoch)
        rng.shuffle(order)
        totals = {"replay": 0., "ranking": 0., "prediction": 0., "updates": 0}
        for offset in range(0, len(rows), args.batch_size):
            group = [rows[i] for i in order[offset:offset + args.batch_size]] + rng.sample(pairs, min(2, len(pairs)))
            obs, candidates, belief, state = batch(group, args.device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits, forecasts, _ = model.proposals(obs, candidates, belief, state, cycles=2)
                replay = replay_loss(logits, group)
                ranking, prediction = pair_loss(logits, forecasts, group, args.device)
                loss = replay + (.25 * ranking + .5 * prediction if args.paired else 0.)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            for key, value in (("replay", replay), ("ranking", ranking), ("prediction", prediction)):
                totals[key] += float(value.detach())
            totals["updates"] += 1
        result = {"epoch": epoch + 1, **{k: v / totals["updates"] for k, v in totals.items() if k != "updates"},
                  "updates": totals["updates"]}
        print(json.dumps(result), flush=True)
        with (args.output / "training.jsonl").open("a") as stream:
            stream.write(json.dumps(result) + "\n")
        torch.save({**saved, "model": model.state_dict(), "consequence_training": settings,
                    "consequence_epoch": epoch + 1}, args.output / "last.pt")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--branches", type=Path)
    parser.add_argument("--prepared", type=Path)
    parser.add_argument("--contexts", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--paired", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--replay-stride", type=int, default=16)
    parser.add_argument("--action-cost", type=float, default=.001)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--core-lr", type=float, default=1e-6)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=161)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.enable_cudnn_sdp(False)
    if args.output.exists():
        parser.error("Choose a fresh output")
    prepare(args) if args.prepare else train(args)


if __name__ == "__main__":
    main()
