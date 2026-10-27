"""On-policy recurrent actor-critic updates from new DiscoveryWorld interactions.

Warm-start the existing digital policy after imitation. Each action is sampled
from its current candidate distribution and executed once in the real simulator.
An update consumes that window once; it is not replay or PPO. Fixed refinement
counts are inherited from the policy. This trains shared policy parameters,
not instance-local adapters or a learned computation schedule.
"""
from __future__ import annotations

import argparse
import contextlib
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from cdm.digital import DigitalPolicy, PublicObjectMemory, public_candidates
from .adapter import DiscoveryWorldEnv, TrajectoryWriter
from .features import FeatureCache, FeatureConfig, batch_features, load_features
from .train import create_measurement_log, load_checkpoint, with_measurement_records


class StateCritic(nn.Module):
    """Training-only value head, stored separately from the deployable policy."""

    def __init__(self, width: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2 * width, width), nn.Tanh(), nn.Linear(width, 1))

    def forward(self, model: DigitalPolicy, state: dict[str, Tensor], mode: str = "cdm") -> Tensor:
        features = model.core.state_features(state["h"], state["z"])
        if mode == "gru":
            features = torch.cat((features[:, :model.config.width], torch.zeros_like(state["h"])), -1)
        return self.net(features).squeeze(-1)


def discounted_returns(rewards: list[float], terminated: list[bool], bootstrap: Tensor,
                       gamma: float) -> Tensor:
    """N-step returns: true termination cuts continuation, time limits do not."""
    value = bootstrap.detach().reshape(())
    returns = []
    for reward, terminal in zip(reversed(rewards), reversed(terminated)):
        value = value.new_tensor(reward) + gamma * value * (not terminal)
        returns.append(value)
    return torch.stack(list(reversed(returns)))


def _features(observation: dict, memory: PublicObjectMemory, cache: FeatureCache, device, measurement_log=None):
    observation = with_measurement_records(memory.observe(observation), measurement_log,
                                           object_relations=cache.config.object_relations)
    candidates = public_candidates(observation)
    path = cache.encode(observation, candidates)
    obs, actions = batch_features([load_features(str(path))], device)
    return candidates, obs, actions


def run_episode(model: DigitalPolicy, critic: StateCritic, optimizer, cache: FeatureCache,
                environment: DiscoveryWorldEnv, *, scenario: str, difficulty: str, seed: int,
                sequence_length: int = 16, gamma: float = .99, value_weight: float = .5,
                entropy_weight: float = .01, mode: str = "cdm", persistent: bool = True,
                restart_workspace: bool = False, provenance: dict | None = None) -> dict:
    """Collect and train on fresh windows, carrying h/z across their boundaries.

    Bootstrap evaluates the next public observation without advancing the
    carried state. After the update, that observation is assimilated once with
    gradients in the next window. Ended time-limit episodes also bootstrap.
    """
    device = next(model.parameters()).device
    model.train()
    critic.train()
    observation = environment.reset(scenario, difficulty, seed)
    state = model.initial(1, device)
    memory = PublicObjectMemory()
    measurement_log = create_measurement_log(observation, cache.config)
    kwargs = {"mode": mode, "persistent": persistent, "restart_workspace": restart_workspace}
    configuration = {**environment.configuration, "policy": "on_policy_recurrent_actor_critic",
                     "action_descriptions": environment.actions, "teleport_locations": environment.teleports,
                     "initial_task_metrics": environment.metrics, "features": asdict(cache.config),
                     "refinements": model.config.refinements, **kwargs,
                     "supervision": "returned_training_reward", "provenance": provenance or {}}
    windows = []
    episode_reward = 0.0
    started = perf_counter()
    with TrajectoryWriter(environment.output, configuration) as writer:
        while not environment.ended:
            optimizer.zero_grad(set_to_none=True)
            log_probabilities, entropies, values, rewards, terminals = [], [], [], [], []
            for _ in range(sequence_length):
                candidates, obs, actions = _features(observation, memory, cache, device, measurement_log)
                logits, state, _ = model(obs, actions, state, **kwargs)
                distribution = torch.distributions.Categorical(logits=logits[0])
                selected = distribution.sample()
                log_probability = distribution.log_prob(selected)
                value = critic(model, state, mode)[0]
                transition = environment.step(dict(candidates[int(selected)].action))
                reward = float(transition["training_reward"])
                log_probabilities.append(log_probability)
                entropies.append(distribution.entropy())
                values.append(value)
                rewards.append(reward)
                terminals.append(bool(transition["terminated"]))
                episode_reward += reward
                writer.append({**transition, "behavior_log_probability": float(log_probability.detach()),
                               "critic_value": float(value.detach()), "policy_update": len(windows)})
                observation = environment.policy_input()
                if measurement_log is not None:
                    measurement_log.observe(transition["action"], transition["action_result"], observation,
                                            sequence=transition["index"])
                if environment.ended:
                    break
            # Read the next observation under the pre-update policy for a TD
            # target, but do not consume that read into the continuing h/z.
            bootstrap = torch.zeros((), device=device)
            if not terminals[-1]:
                with torch.no_grad():
                    _, obs, actions = _features(observation, memory, cache, device, measurement_log)
                    _, next_state, _ = model(obs, actions, state, **kwargs)
                    bootstrap = critic(model, next_state, mode)[0]
            returns = discounted_returns(rewards, terminals, bootstrap, gamma)
            value_predictions = torch.stack(values)
            advantages = returns - value_predictions.detach()
            policy_loss = -(torch.stack(log_probabilities) * advantages).mean()
            value_loss = F.mse_loss(value_predictions, returns)
            entropy = torch.stack(entropies).mean()
            loss = policy_loss + value_weight * value_loss - entropy_weight * entropy
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                [*model.parameters(), *critic.parameters()], 1.0)
            optimizer.step()
            # Truncated BPTT retains the computed state values. Every subsequent
            # action is sampled with the updated parameters, without old-policy replay.
            state = {key: value.detach() for key, value in state.items()}
            windows.append({"update": len(windows), "steps": len(rewards), "reward": sum(rewards),
                            "policy_loss": float(policy_loss.detach()), "value_loss": float(value_loss.detach()),
                            "entropy": float(entropy.detach()), "gradient_norm": float(gradient_norm),
                            "bootstrap": float(bootstrap), "terminated": terminals[-1],
                            "truncated": bool(transition["truncated"])})
    summary = {"scenario": scenario, "difficulty": difficulty, "seed": seed,
               "steps": environment.steps, "updates": len(windows), "training_reward": episode_reward,
               "completed_successfully": all(item["completed_successfully"] for item in environment.metrics),
               "task_metrics": environment.metrics, "elapsed_seconds": perf_counter() - started,
               "protocol": "official_step_based", "training_phase": "on_policy_actor_critic",
               "refinements": model.config.refinements, **kwargs}
    (environment.output / "updates.jsonl").write_text("".join(json.dumps(row) + "\n" for row in windows))
    (environment.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def save_checkpoint(model: DigitalPolicy, critic: StateCritic, optimizer, source: dict,
                    output: Path, episodes: list[dict], args: dict, source_hash: str):
    """Keep strict policy loading unchanged; the critic/optimizer are separate."""
    identities = {tuple(identity) for identity in source.get("train_instances", [])}
    identities.update((item["scenario"], item["difficulty"], item["seed"]) for item in episodes)
    checkpoint = {**source, "model": model.state_dict(), "training_phase": "on_policy_actor_critic",
                  "digital_config": asdict(model.config), "core_config": asdict(model.core.config),
                  "warmstart_metrics": source.get("metrics"),
                  "metrics": {"on_policy_training": episodes[-1] if episodes else None,
                              "heldout_evaluation": None},
                  "train_instances": [list(identity) for identity in sorted(identities)],
                  "finetune": {"source_sha256": source_hash, "args": args, "episodes": episodes,
                               "fresh_transitions": sum(item["steps"] for item in episodes)},
                  "training_args": {**source["training_args"], "mode": args["mode"],
                                    "restart_state": args["restart_state"],
                                    "restart_workspace": args["restart_workspace"]}}
    torch.save(checkpoint, output / "last.pt")
    torch.save({"critic": critic.state_dict(), "optimizer": optimizer.state_dict(),
                "width": model.config.width, "episodes": len(episodes)}, output / "critic.pt")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--feature-cache", required=True, type=Path)
    parser.add_argument("--encoder-path", type=Path)
    parser.add_argument("--scenario", default="Reactor Lab")
    parser.add_argument("--difficulty", default="Normal")
    parser.add_argument("--seeds", nargs="+", type=int, required=True, help="Training seeds; official 0--4 are reserved")
    parser.add_argument("--episodes", type=int, default=1, help="Cycle through supplied training seeds")
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--sequence-length", type=int, default=16)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--feature-device")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--gamma", type=float, default=.99)
    parser.add_argument("--value-weight", type=float, default=.5)
    parser.add_argument("--entropy-weight", type=float, default=.01)
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--mode", choices=("cdm", "gru"))
    parser.add_argument("--restart-state", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--restart-workspace", action=argparse.BooleanOptionalAction, default=None)
    args = parser.parse_args(argv)
    if min(args.episodes, args.max_steps, args.sequence_length, args.threads) <= 0:
        parser.error("Episodes, steps, sequence length and threads must be positive")
    if any(seed < 5 for seed in args.seeds):
        parser.error("Training seeds must be >= 5; official evaluation seeds 0--4 are reserved")
    if not 0 <= args.gamma <= 1 or args.lr <= 0 or min(args.value_weight, args.entropy_weight) < 0:
        parser.error("Use gamma in [0,1], positive learning rate and nonnegative loss weights")
    if (args.output / "training.jsonl").exists() or (args.output / "last.pt").exists():
        parser.error("Choose a new fine-tuning output directory")
    return args


def main(argv=None):
    args = parse_args(argv)
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.random_seed)
    model, source = load_checkpoint(args.checkpoint, args.device)
    training = source["training_args"]
    args.mode = args.mode or training["mode"]
    args.restart_state = training.get("restart_state", False) if args.restart_state is None else args.restart_state
    args.restart_workspace = training.get("restart_workspace", False) if args.restart_workspace is None else args.restart_workspace
    reserved = {tuple(identity) for identity in source.get("validation_instances", [])}
    if any((args.scenario, args.difficulty, seed) in reserved for seed in args.seeds):
        raise ValueError("Fine-tuning seeds overlap the checkpoint's held-out validation instances")
    critic = StateCritic(model.config.width).to(args.device)
    optimizer = torch.optim.AdamW([*model.parameters(), *critic.parameters()], lr=args.lr)
    # Legacy checkpoints predate explicit clock metadata. Keep their original
    # input meaning; new checkpoints specify native world_steps/schema 2.
    feature_config = FeatureConfig(**{"schema": 1, "time_basis": "wall_clock", **source["features"]})
    cache = FeatureCache(args.feature_cache, feature_config,
                         args.feature_device or args.device, args.encoder_path)
    source = {**source, "features": asdict(feature_config)}
    args.output.mkdir(parents=True, exist_ok=True)
    source_hash = sha256(args.checkpoint.read_bytes()).hexdigest()
    arguments = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    metadata = {"args": arguments, "source_sha256": source_hash,
                "features": asdict(feature_config),
                "training_phase": "on_policy_actor_critic", "policy_updates_per_window": 1,
                "refinements": model.config.refinements, "critic": "separate_training_state_head",
                "truncation": "bootstrap_next_public_observation", "termination": "zero_bootstrap"}
    (args.output / "config.json").write_text(json.dumps(metadata, indent=2) + "\n")
    episodes = []
    try:
        with (args.output / "training.jsonl").open("x") as log:
            for index in range(args.episodes):
                seed = args.seeds[index % len(args.seeds)]
                output = args.output / "episodes" / f"episode{index:04d}_seed{seed}"
                environment = DiscoveryWorldEnv(output, args.max_steps)
                with (output / "simulator.log").open("w") as simulator_log, contextlib.redirect_stdout(simulator_log):
                    result = run_episode(model, critic, optimizer, cache, environment,
                                         scenario=args.scenario, difficulty=args.difficulty, seed=seed,
                                         sequence_length=args.sequence_length, gamma=args.gamma,
                                         value_weight=args.value_weight, entropy_weight=args.entropy_weight,
                                         mode=args.mode, persistent=not args.restart_state,
                                         restart_workspace=args.restart_workspace,
                                         provenance={"source_sha256": source_hash, "episode": index})
                episodes.append(result)
                save_checkpoint(model, critic, optimizer, source, args.output, episodes, arguments, source_hash)
                log.write(json.dumps(result) + "\n")
                log.flush()
                print(json.dumps(result), flush=True)
    finally:
        cache.close()


if __name__ == "__main__":
    main()
