"""Learn experiment choices from paired, actually executed continuations.

Each pair starts at the same replayed training history and differs in its next
action. Subsequent actions come from the same fixed policy. The learning target
is the difference in final task progress minus native action cost. Evaluation
does not branch or receive the future returns used here.
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
from .belief_actions import BeliefActionHead
from .features import canonical_action
from .train import load_episodes, public_observation
from .train_belief_actions import batch, epoch, read_prepared


def load_pairs(root, checkpoint, action_cost):
    plans = json.loads((root / "comparison.json").read_text())
    groups = defaultdict(list)
    for plan in plans:
        groups[(plan["source"], plan["prefix_steps"])].append(plan)
    pairs, records = [], []
    for (source, position), alternatives in groups.items():
        episode, = load_episodes([Path(source)])
        cached = torch.load(Path(source) / "learning.pt", map_location="cpu", weights_only=True)
        if cached["summary"]["checkpoint"] != str(checkpoint):
            raise ValueError("Cached decisions must come from the frozen source reader")
        memory = PublicObjectMemory()
        for transition in episode.transitions[:position + 1]:
            observation = memory.observe(public_observation(episode, transition))
        candidates = public_candidates(observation)
        row = cached["rows"][position]
        if len(candidates) != len(row["base_logits"]) or row["sequence"] != position:
            raise ValueError("Branch decision does not match its cached candidate set")
        choices = [canonical_action(candidate.action) for candidate in candidates]
        if choices[row["target"]] != canonical_action(episode.transitions[position]["action"]):
            raise ValueError("Cached chosen action and public candidate order differ")
        outcomes = []
        for plan in alternatives:
            summary = json.loads((root / f"seed{plan['seed']}_{plan['arm']}" / "summary.json").read_text())
            if summary["error"] is not None or not summary["prefix_replay_matched"]:
                raise ValueError("Only matched completed branch executions supply a target")
            action = canonical_action(plan["supplied_first_action"])
            if canonical_action(summary["first_action_intervention"]) != action:
                raise ValueError("Requested and executed branch actions differ")
            outcomes.append({"arm": plan["arm"], "candidate": choices.index(action),
                             "return": summary["progress"] - action_cost * summary["steps"],
                             "progress": summary["progress"], "steps": summary["steps"]})
        outcomes.sort(key=lambda item: item["return"])
        gain = outcomes[-1]["return"] - outcomes[0]["return"]
        record = {"identity": list(episode.identity), "decision": position,
                  "outcomes": outcomes, "gain": gain}
        records.append(record)
        if gain > 1e-8:
            pairs.append({"row": row, "preferred": outcomes[-1]["candidate"],
                          "alternative": outcomes[0]["candidate"], "gain": gain})
    if not pairs:
        raise ValueError("The continuations do not distinguish any action choices")
    return pairs, records


def preference_loss(logits, preferred, alternative, gains):
    indices = torch.arange(logits.shape[0], device=logits.device)
    losses = F.softplus(logits[indices, alternative] - logits[indices, preferred])
    return (losses * gains).sum() / gains.sum()


def train(args):
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    pairs, records = load_pairs(args.branches, args.checkpoint, args.action_cost)
    branch_ids = {tuple(record["identity"]) for record in records}
    validation = read_prepared(args.validation)
    if branch_ids & {tuple(episode["identity"]) for episode in validation}:
        raise ValueError("Branch learning instances overlap validation")
    rows = [row for episode in read_prepared(args.prepared) for row in episode["rows"]]
    validation_rows = [row for episode in validation for row in episode["rows"]]
    actor = BeliefActionHead(**saved["config"]).to(args.device)
    actor.load_state_dict(saved["model"], strict=True)
    actor.train()
    optimizer = torch.optim.AdamW(actor.parameters(), lr=args.lr, weight_decay=1e-4)
    branch_batch = batch([pair["row"] for pair in pairs], args.device)[:4]
    preferred = torch.tensor([pair["preferred"] for pair in pairs], device=args.device)
    alternative = torch.tensor([pair["alternative"] for pair in pairs], device=args.device)
    gains = torch.tensor([pair["gain"] for pair in pairs], device=args.device)
    logs = []
    for number in range(args.epochs):
        order = list(range(len(rows)))
        random.Random(args.seed + number).shuffle(order)
        totals = {"replay_loss": 0., "preference_loss": 0., "batches": 0}
        for offset in range(0, len(rows), args.batch_size):
            group = [rows[index] for index in order[offset:offset + args.batch_size]]
            state, text, belief, base, target, weights = batch(group, args.device)
            scores = actor(state, text, belief, base)
            replay = (F.cross_entropy(scores, target, reduction="none") * weights).sum() / weights.sum()
            preference = preference_loss(actor(*branch_batch), preferred, alternative, gains)
            optimizer.zero_grad(set_to_none=True)
            (replay + args.branch_weight * preference).backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.)
            optimizer.step()
            totals["replay_loss"] += float(replay.detach())
            totals["preference_loss"] += float(preference.detach())
            totals["batches"] += 1
        with torch.no_grad():
            scores = actor(*branch_batch)
            indices = torch.arange(len(pairs), device=args.device)
            margins = (scores[indices, preferred] - scores[indices, alternative]).tolist()
        logs.append({"epoch": number + 1,
                     **{key: value / totals["batches"] for key, value in totals.items() if key != "batches"},
                     "training_pair_margins": margins,
                     "validation": epoch(actor, validation_rows, args.device, args.batch_size)})
        print(json.dumps(logs[-1]), flush=True)
    settings = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    settings.update(continuations=records, informative_pairs=len(pairs), replay_actions=len(rows),
                    objective="action-run-weighted replay plus return-weighted pairwise action ranking",
                    history=logs)
    identities = set(map(tuple, saved["train_instances"])) | branch_ids
    args.output.mkdir(parents=True, exist_ok=True)
    torch.save({**saved, "model": actor.state_dict(), "branch_training": settings,
                "train_instances": [list(identity) for identity in sorted(identities)]}, args.output / "last.pt")
    (args.output / "training.json").write_text(json.dumps(settings, indent=2) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--branches", type=Path, required=True)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--branch-weight", type=float, default=0.25)
    parser.add_argument("--action-cost", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=81)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args(argv)
    if (args.output / "last.pt").exists():
        parser.error("Choose a new training output")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    train(args)


if __name__ == "__main__":
    main()
