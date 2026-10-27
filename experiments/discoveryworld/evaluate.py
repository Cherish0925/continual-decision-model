"""Execute a trained dynamic-action policy in the actual DiscoveryWorld simulator."""
from __future__ import annotations

import argparse
import atexit
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from hashlib import sha256
import json
import multiprocessing
import os
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from cdm.digital import PublicObjectMemory, public_candidates
from .adapter import DiscoveryWorldEnv, TrajectoryWriter
from .features import FeatureCache, FeatureConfig, batch_features, load_features
from .train import load_checkpoint


class LearnedPolicy:
    """One continuing learned state per episode; no teacher or action heuristics."""
    def __init__(self, checkpoint: str | Path, feature_cache: str | Path, *,
                 device: str = "cpu", feature_device: str | None = None,
                 encoder_path: str | Path | None = None, restart_state: bool | None = None, restart_workspace: bool | None = None,
                 mode: str | None = None, refinements: int | None = None,
                 temperature: float = 0.0):
        self.model, self.checkpoint = load_checkpoint(checkpoint, device)
        self.model.eval()
        self.device = torch.device(device)
        self.features = FeatureCache(feature_cache, FeatureConfig(**{"schema": 1, "time_basis": "wall_clock", **self.checkpoint["features"]}),
                                     feature_device or device, encoder_path)
        self.mode = mode or self.checkpoint["training_args"]["mode"]
        self.persistent = not (self.checkpoint["training_args"].get("restart_state", False) if restart_state is None else restart_state)
        self.restart_workspace = self.checkpoint["training_args"].get("restart_workspace", False) if restart_workspace is None else restart_workspace
        self.refinements, self.temperature = refinements, temperature
        self.reset()

    def reset(self):
        self.state = self.model.initial(1, self.device)
        self.memory = PublicObjectMemory()
        self.measurements = None
        if self.features.config.measurement_memory:
            from .measurements import MeasurementLog
            self.measurements = MeasurementLog()
        self.decisions = []

    def observe(self, transition: dict):
        """Receive the action actually executed, including teacher interventions."""
        if self.measurements is not None:
            self.measurements.append_transition(transition)

    def action_logits(self, logits, observation, candidates, features):
        """Optional learned readout over the state just updated by this action."""
        return logits

    @torch.no_grad()
    def __call__(self, observation: dict) -> dict:
        start = perf_counter()
        if self.measurements is not None:
            self.measurements.observe_initial(observation)
            observation = {**observation, "measurement_records": self.measurements.summaries()}
            if self.features.config.object_relations:
                observation["measurement_dialog_owner"] = self.measurements.dialog_owner
        observation = self.memory.observe(observation)
        candidates = public_candidates(observation)
        path = self.features.encode(observation, candidates)
        obs, actions = batch_features([load_features(str(path))], self.device)
        features_done = perf_counter()
        logits, self.state, _ = self.model(obs, actions, self.state,
                                                   persistent=self.persistent, mode=self.mode,
                                                   refinements=self.refinements, restart_workspace=self.restart_workspace,
                                                   predict_outcomes=False)
        logits = self.action_logits(logits, observation, candidates, actions)
        if self.temperature > 0:
            selected = int(torch.multinomial((logits[0] / self.temperature).softmax(-1), 1))
        else:
            selected = int(logits[0].argmax())
        # .item()/int above synchronize the selected action before timing it.
        done = perf_counter()
        self.decisions.append({"candidates": len(candidates), "selected": selected,
                               "feature_seconds": features_done - start,
                               "policy_seconds": done - features_done,
                               "decision_seconds": done - start})
        return dict(candidates[selected].action)


def create(config: dict):
    """Factory for experiments.discoveryworld.collect --policy ...evaluate:create."""
    return LearnedPolicy(**config)


def evaluate_episode(agent: LearnedPolicy, output: Path, scenario: str, difficulty: str,
                     seed: int, max_steps: int, *, random_seed: int | None = None,
                     shared_random_seed: int | None = None, initial_rng_root: Path | None = None) -> dict:
    if (output / "trajectory.jsonl").exists():
        raise ValueError(f"Refusing to overwrite previous evaluation: {output}")
    if random_seed is not None:
        # Policy sampling is independent of process scheduling and previous episodes.
        torch.manual_seed(random_seed)
    rng_mode = "per_episode_rng" if random_seed is not None else "serial_shared_rng" if shared_random_seed is not None else "caller_rng"
    recorded_seed = random_seed if random_seed is not None else shared_random_seed
    agent.reset()
    env = DiscoveryWorldEnv(output, max_steps)
    observation = env.reset(scenario, difficulty, seed)
    if initial_rng_root is not None:
        env.restore_initial_rng(Path(initial_rng_root) / f"seed{seed}" / "initial_rng.json.gz")
    rng_snapshot = env.save_initial_rng(output / "initial_rng.json.gz")
    configuration = {**env.configuration, "policy": getattr(agent, "policy_type", "learned_dynamic_action_policy"),
                     "action_descriptions": env.actions, "teleport_locations": env.teleports,
                     "initial_task_metrics": env.metrics, "features": asdict(agent.features.config),
                     "mode": agent.mode, "persistent": agent.persistent, "restart_workspace": agent.restart_workspace,
                     "refinements": getattr(agent, "refinement_schedule",
                                            agent.refinements if agent.refinements is not None else agent.model.config.refinements),
                     "training_epoch": agent.checkpoint["epoch"], "random_seed": recorded_seed, "rng_mode": rng_mode,
                     "temperature": agent.temperature, "device": str(agent.device),
                     "feature_device": str(agent.features.device), "worker_pid": os.getpid()}
    configuration.update(initial_rng_snapshot={"path": "initial_rng.json.gz", **rng_snapshot},
                         initial_rng_source=str(initial_rng_root) if initial_rng_root else "fresh_native_reset")
    if hasattr(agent, "evaluation_metadata"):
        configuration["policy_details"] = agent.evaluation_metadata
    start = perf_counter()
    with TrajectoryWriter(output, configuration) as writer:
        while not env.ended:
            action = agent(observation)
            transition = env.step(action)
            writer.append(transition)
            agent.observe(transition)
            observation = env.policy_input()
    identity = [scenario, difficulty, seed]
    split = "training_instance" if identity in agent.checkpoint["train_instances"] else \
        "validation_instance" if identity in agent.checkpoint["validation_instances"] else "new_instance"
    latencies = np.asarray([row["decision_seconds"] for row in agent.decisions])
    summary = {"scenario": scenario, "difficulty": difficulty, "seed": seed,
               "split": split, "steps": env.steps, "task_metrics": env.metrics,
               "random_seed": recorded_seed, "rng_mode": rng_mode, "temperature": agent.temperature,
               "device": str(agent.device), "feature_device": str(agent.features.device), "worker_pid": os.getpid(),
               "completed": all(row["completed"] for row in env.metrics),
               "completed_successfully": all(row["completed_successfully"] for row in env.metrics),
               "progress": float(np.mean([row["progress"] for row in env.metrics])),
               "truncated": env.steps >= max_steps and not all(row["completed"] for row in env.metrics),
               "elapsed_seconds": perf_counter() - start,
               "decision_latency_mean_seconds": float(latencies.mean()) if len(latencies) else None,
               "decision_latency_p95_seconds": float(np.quantile(latencies, .95)) if len(latencies) else None,
               "protocol": "official_step_based", "discovery_knowledge_score": None}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (output / "decisions.jsonl").write_text("".join(json.dumps(row) + "\n" for row in agent.decisions))
    return summary



_WORKER_AGENT = None


def create_evaluation_policy(policy_config):
    config = dict(policy_config)
    kind = config.pop("policy_kind", "learned")
    if kind == "learned":
        return LearnedPolicy(**config)
    from .belief_actions import BeliefPolicy
    from .adaptive_compute import AdaptivePolicy
    from .mixture_policy import MixturePolicy
    from .predictive_actions import PredictivePolicy
    interventions = {key: config.pop(key, None) for key in ("mode", "restart_state", "restart_workspace")}
    policy = {"belief": BeliefPolicy, "mixture": MixturePolicy, "adaptive": AdaptivePolicy,
              "predictive": PredictivePolicy}[kind](**config)
    if interventions["mode"] is not None:
        policy.mode = interventions["mode"]
    if interventions["restart_state"] is not None:
        policy.persistent = not interventions["restart_state"]
    if interventions["restart_workspace"] is not None:
        policy.restart_workspace = interventions["restart_workspace"]
    return policy


def _initialize_worker(policy_config: dict, threads: int, assignments):
    """Spawned processes own their models, CUDA contexts and simulator state."""
    global _WORKER_AGENT
    device = assignments.get()
    torch.set_num_threads(threads)
    _WORKER_AGENT = create_evaluation_policy({**policy_config, "device": device})
    atexit.register(_WORKER_AGENT.features.close)


def _worker_episode(task: dict) -> dict:
    return evaluate_episode(_WORKER_AGENT, **task)

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--policy-kind", choices=("learned", "belief", "mixture", "adaptive", "predictive"), default="learned")
    parser.add_argument("--use-forecasts", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--local-probes", action="store_true")
    parser.add_argument("--mixture-probes", action="store_true", help="Use inactive feedback across all numerical hypotheses")
    parser.add_argument("--initial-rng-root", type=Path, help="Replay the original native noise streams from a previous matched evaluation")
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--encoder-path", type=Path)
    parser.add_argument("--scenario", default="Reactor Lab")
    parser.add_argument("--difficulty", default="Normal")
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--devices", nargs="+", help="Worker devices, assigned round-robin; defaults to --device")
    parser.add_argument("--workers", type=int, default=1, help="Independent spawned evaluator processes; 1 keeps serial execution")
    parser.add_argument("--feature-device")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--restart-state", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--restart-workspace", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--mode", choices=("cdm", "gru"), help="Defaults to checkpoint's trained mode")
    parser.add_argument("--refinements", type=int)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--random-seed", type=int, default=0,
                        help="Serial: one shared RNG seed. Parallel: each episode independently uses this value plus its input-list index")
    args = parser.parse_args(argv)
    if min(args.workers, args.threads, args.max_steps) <= 0:
        parser.error("Workers, threads and max_steps must be positive")
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("Episode seeds must be unique within one output directory")
    if (args.output / "summary.json").exists() or any(
        (args.output / f"seed{seed}" / "trajectory.jsonl").exists() for seed in args.seeds
    ):
        parser.error("Choose a new evaluation output directory")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.random_seed)
    devices = args.devices or [args.device]
    workers = min(args.workers, len(args.seeds))
    policy_config = {"checkpoint": args.checkpoint, "feature_cache": args.feature_cache,
                     "feature_device": args.feature_device, "encoder_path": args.encoder_path,
                     "restart_state": args.restart_state, "restart_workspace": args.restart_workspace,
                     "mode": args.mode, "refinements": args.refinements, "temperature": args.temperature}
    policy_config["policy_kind"] = args.policy_kind
    if args.policy_kind == "predictive":
        policy_config["use_forecasts"] = args.use_forecasts
    if args.policy_kind != "learned":
        policy_config["local_probes"] = args.local_probes
        policy_config["mixture_probes"] = args.mixture_probes
    tasks = [{"output": args.output / f"seed{seed}", "scenario": args.scenario,
              "difficulty": args.difficulty, "seed": seed, "max_steps": args.max_steps,
              "initial_rng_root": args.initial_rng_root,
              **({"shared_random_seed": args.random_seed} if workers == 1 else {"random_seed": args.random_seed + index})}
             for index, seed in enumerate(args.seeds)]
    if workers == 1:
        agent = create_evaluation_policy({**policy_config, "device": devices[0]})
        try:
            results = [evaluate_episode(agent, **task) for task in tasks]
        finally:
            agent.features.close()
    else:
        context = multiprocessing.get_context("spawn")
        assignments = context.Queue()
        for worker in range(workers):
            assignments.put(devices[worker % len(devices)])
        try:
            with ProcessPoolExecutor(max_workers=workers, mp_context=context,
                                     initializer=_initialize_worker,
                                     initargs=(policy_config, args.threads, assignments)) as pool:
                results = list(pool.map(_worker_episode, tasks))
        finally:
            assignments.close()
            assignments.join_thread()
    first = json.loads((tasks[0]["output"] / "run.json").read_text())
    summary = {"checkpoint": str(args.checkpoint.resolve()),
               "checkpoint_sha256": sha256(args.checkpoint.read_bytes()).hexdigest(),
               "mode": first["mode"], "persistent": first["persistent"], "restart_workspace": first["restart_workspace"],
               "features": first["features"], "episodes": results, "workers": workers, "devices": devices,
               "temperature": args.temperature, "random_seed": args.random_seed,
               "rng_mode": "serial_shared_rng" if workers == 1 else "per_episode_rng",
               "episode_random_seed_policy": "one initial torch.manual_seed shared across serial episodes" if workers == 1 else
                   "base random_seed + input seed-list index, independently reset before each episode",
               "successful_episodes": sum(row["completed_successfully"] for row in results),
               "episode_count": len(results), "mean_progress": float(np.mean([row["progress"] for row in results])),
               "discovery_knowledge_score": None, "protocol": "official_step_based"}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
