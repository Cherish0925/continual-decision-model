"""Train the belief action reader on actual scientific-task return and cost.

The continuing encoder and numerical inference are fixed. Actions chosen by
the learned reader receive policy gradients; movements executed by the explicit
scalar primitive consume time and contribute reward but are never relabelled
as sampled neural actions. Returns cover the actual finite episode, including
the cost of that primitive. No teacher action enters this training.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
from time import perf_counter

import torch
from torch import nn
import torch.nn.functional as F

from .adapter import DiscoveryWorldEnv, TrajectoryWriter
from .belief_actions import BELIEF_DIM, BeliefActionHead, BeliefPolicy
from .train_belief_actions import batch


class ReturnPredictor(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(state_dim + BELIEF_DIM, 128), nn.Tanh(), nn.Linear(128, 1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, state, belief, present):
        summary = (belief * present[..., None]).sum(1) / present.sum(1, keepdim=True).clamp_min(1)
        return self.net(torch.cat((state, summary), -1)).squeeze(-1)


class CollectingPolicy(BeliefPolicy):
    def action_logits(self, logits, observation, candidates, features):
        evidence = self.beliefs.encode(observation, candidates).to(self.device)
        state = self.model.core.state_features(self.state["h"], self.state["z"])
        result = self.actor(state, features["text"], evidence[None], logits)
        value = self.critic(state, evidence[None], torch.ones_like(logits, dtype=torch.bool))
        self.pending = {"state": state[0].cpu(), "text": features["text"][0].cpu(),
                        "belief": evidence.cpu(), "base_logits": logits[0].cpu(),
                        "behavior_logits": (result[0] / self.temperature).cpu(),
                        "value": float(value[0]), "weight": 1., "run_exit": False}
        return result


def collect(args):
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    identity = ["Reactor Lab", "Normal", args.seed]
    if identity in saved["validation_instances"]:
        raise ValueError("Training rollout overlaps validation")
    agent = CollectingPolicy(args.checkpoint, args.feature_cache, device=args.device,
                             encoder_path=args.encoder_path, local_probes=True, temperature=args.temperature)
    agent.critic = ReturnPredictor(saved["config"]["state_dim"]).to(args.device).eval()
    if "critic" in saved:
        agent.critic.load_state_dict(saved["critic"], strict=True)
    torch.manual_seed(args.seed)
    env = DiscoveryWorldEnv(args.output, args.max_steps)
    observation = env.reset(*identity)
    rng_snapshot = env.save_initial_rng(args.output / "initial_rng.json.gz")
    config = {**env.configuration, "action_descriptions": env.actions, "teleport_locations": env.teleports,
              "policy": agent.policy_type, "training_phase": "on_policy_belief_actions",
              "checkpoint": str(args.checkpoint), "temperature": args.temperature,
              "action_cost": args.action_cost, "gamma": args.gamma,
              "initial_rng_snapshot": {"path": "initial_rng.json.gz", **rng_snapshot},
              "policy_gradient_actions": "learned_reader_only; numerical_primitive_has_reward_and_cost"}
    rows, started = [], perf_counter()
    with TrajectoryWriter(args.output, config) as writer:
        while not env.ended:
            action = agent(observation)
            transition = env.step(action)
            writer.append(transition)
            row = agent.pending
            selected = agent.decisions[-1]["selected"]
            controlled = agent.decisions[-1].get("action_source") != "explicit_native_scalar_control"
            row.update(target=selected, old_log_probability=float(row.pop("behavior_logits").log_softmax(-1)[selected]),
                       controlled=controlled, reward=float(transition["training_reward"]) - args.action_cost,
                       native_reward=float(transition["training_reward"]), sequence=transition["index"],
                       temperature=args.temperature)
            rows.append(row)
            agent.observe(transition)
            observation = env.policy_input()
    future = 0.
    for row in reversed(rows):
        future = row["reward"] + args.gamma * future
        row["return"] = future
        row["advantage"] = future - row["value"]
    summary = dict(identity=identity, steps=env.steps, completed=all(x["completed_successfully"] for x in env.metrics),
                   progress=sum(x["progress"] for x in env.metrics) / len(env.metrics),
                   learned_actions=sum(row["controlled"] for row in rows),
                   reward=sum(row["reward"] for row in rows), seconds=perf_counter() - started,
                   checkpoint=str(args.checkpoint), action_cost=args.action_cost, gamma=args.gamma,
                   finite_horizon=args.max_steps, temperature=args.temperature)
    torch.save({"rows": rows, "summary": summary}, args.output / "learning.pt")
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.output / "decisions.jsonl").write_text("".join(json.dumps(row) + "\n" for row in agent.decisions))
    print(json.dumps(summary), flush=True)
    agent.features.close()


def reader_objective(logits, target, old_log_probability, advantage, controlled, clip):
    """Only actions executed by the reader contribute its policy objective."""
    distribution = torch.distributions.Categorical(logits=logits)
    if not controlled.any():
        zero = logits[torch.isfinite(logits)].sum() * 0
        return zero, zero
    ratio = (distribution.log_prob(target) - old_log_probability).exp()
    policy = -torch.minimum(ratio * advantage,
                            ratio.clamp(1 - clip, 1 + clip) * advantage)[controlled].mean()
    return policy, distribution.entropy()[controlled].mean()


@torch.no_grad()
def project_reader_step(actor, reference, rows, device, budget, batch_size=64):
    """Choose step length from behaviour-state KL, without evaluation tasks."""
    controlled = [row for row in rows if row["controlled"]]
    old = reference.state_dict()
    proposed = {key: value.clone() for key, value in actor.state_dict().items()}
    history, fraction = [], 1.
    while True:
        actor.load_state_dict({key: old[key].lerp(value, fraction) for key, value in proposed.items()})
        total = 0.
        for offset in range(0, len(controlled), batch_size):
            group = controlled[offset:offset + batch_size]
            state, text, belief, base, _, _ = batch(group, device)
            temperature = torch.tensor([row["temperature"] for row in group], device=device)[:, None]
            p = (reference(state, text, belief, base) / temperature).log_softmax(-1)
            q = (actor(state, text, belief, base) / temperature).log_softmax(-1)
            difference = torch.where(torch.isfinite(base), p - q, torch.zeros_like(p))
            total += float((p.exp() * difference).sum())
        divergence = total / len(controlled)
        history.append({"step_fraction": fraction, "mean_policy_kl": divergence})
        if divergence <= budget:
            return {"budget": budget, "selected_fraction": fraction, "measurements": history,
                    "states": len(controlled), "source": "current_training_rollouts_only"}
        fraction *= .5


def update(args):
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    collections = [torch.load(path, map_location="cpu", weights_only=True) for root in args.rollouts
                   for path in sorted(root.rglob("learning.pt"))]
    if not collections:
        raise ValueError("No completed training rollouts")
    if any(row["summary"]["checkpoint"] != str(args.checkpoint) for row in collections):
        raise ValueError("PPO update must use trajectories from its current behavior checkpoint")
    rows = [row for collection in collections for row in collection["rows"]]
    actor = BeliefActionHead(**saved["config"]).to(args.device)
    actor.load_state_dict(saved["model"], strict=True)
    critic = ReturnPredictor(saved["config"]["state_dim"]).to(args.device)
    if "critic" in saved:
        critic.load_state_dict(saved["critic"], strict=True)
    optimizer = torch.optim.AdamW([*actor.parameters(), *critic.parameters()], lr=args.lr, weight_decay=1e-4)
    advantages = torch.tensor([row["advantage"] for row in rows if row["controlled"]])
    mean, scale = float(advantages.mean()), float(advantages.std().clamp_min(1e-6))
    log = []
    for number in range(args.epochs):
        order = list(range(len(rows)))
        random.Random(args.seed + number).shuffle(order)
        totals = dict(policy_loss=0., value_loss=0., entropy=0., batches=0)
        for offset in range(0, len(rows), args.batch_size):
            selected = [rows[i] for i in order[offset:offset + args.batch_size]]
            state, text, belief, base, target, _ = batch(selected, args.device)
            mask = torch.tensor([row["controlled"] for row in selected], device=args.device, dtype=torch.bool)
            temperature = torch.tensor([row["temperature"] for row in selected], device=args.device)
            logits = actor(state, text, belief, base) / temperature[:, None]
            old = torch.tensor([row["old_log_probability"] for row in selected], device=args.device)
            advantage = torch.tensor([(row["advantage"] - mean) / scale for row in selected], device=args.device)
            policy, entropy = reader_objective(logits, target, old, advantage, mask, args.clip)
            target_return = torch.tensor([row["return"] for row in selected], device=args.device)
            value = critic(state, belief, torch.isfinite(base))
            value_loss = F.mse_loss(value, target_return)
            loss = policy + .5 * value_loss - .001 * entropy
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([*actor.parameters(), *critic.parameters()], 1.)
            optimizer.step()
            totals["policy_loss"] += float(policy.detach())
            totals["value_loss"] += float(value_loss.detach())
            totals["entropy"] += float(entropy.detach())
            totals["batches"] += 1
        log.append({"epoch": number + 1, **{key: value / totals["batches"] for key, value in totals.items() if key != "batches"}})
    args.output.mkdir(parents=True, exist_ok=True)
    identities = {tuple(row) for row in saved["train_instances"]}
    identities.update(tuple(row["summary"]["identity"]) for row in collections)
    projection = None
    if args.policy_step_kl is not None:
        reference = BeliefActionHead(**saved["config"]).to(args.device).eval().requires_grad_(False)
        reference.load_state_dict(saved["model"], strict=True)
        projection = project_reader_step(actor, reference, rows, args.device, args.policy_step_kl, args.batch_size)
    result = {**saved, "model": actor.state_dict(), "critic": critic.state_dict(),
              "train_instances": [list(row) for row in sorted(identities)],
              "on_policy_training": {"parent": str(args.checkpoint), "rollouts": [row["summary"] for row in collections],
                                     "epochs": log, "lr": args.lr, "clip": args.clip,
                                     "policy_step_projection": projection,
                                     "objective": "actual native progress minus per-action cost; finite episode returns"}}
    torch.save(result, args.output / "last.pt")
    (args.output / "training.json").write_text(json.dumps(result["on_policy_training"], indent=2) + "\n")
    print(json.dumps(result["on_policy_training"]), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collect", action="store_true")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rollouts", type=Path, nargs="+", default=[])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=710)
    parser.add_argument("--feature-cache", type=Path, default=Path("runs/discoveryworld/features"))
    parser.add_argument("--encoder-path", type=Path, default=Path("data/cache/siglip2"))
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--action-cost", type=float, default=0.001)
    parser.add_argument("--gamma", type=float, default=0.999)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--clip", type=float, default=0.1)
    parser.add_argument("--policy-step-kl", type=float, help="Bound empirical policy change on the collected training decisions")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args(argv)
    if (args.output / "last.pt").exists() or (args.output / "trajectory.jsonl").exists():
        parser.error("Choose a new run directory")
    if args.temperature <= 0 or args.seed < 5:
        parser.error("Use positive rollout temperature and a non-official training instance")
    if args.policy_step_kl is not None and args.policy_step_kl <= 0:
        parser.error("The policy-step KL budget must be positive")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    collect(args) if args.collect else update(args)


if __name__ == "__main__":
    main()
