"""Collect official DiscoveryWorld transitions from a local policy or action file."""
from __future__ import annotations

import argparse
from importlib import import_module
import json
from pathlib import Path

from .adapter import DiscoveryWorldEnv, TrajectoryWriter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", default="Reactor Lab")
    parser.add_argument("--difficulty", default="Normal")
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--observation-mode", choices=("native", "png"), default="native",
                        help="png preserves native image inputs without per-step frame-file writes")
    parser.add_argument("--output", type=Path, required=True)
    policy = parser.add_mutually_exclusive_group(required=True)
    policy.add_argument("--policy", help="module:factory; factory(config) returns an observation-to-action callable")
    policy.add_argument("--actions", type=Path, help="Replay native action dictionaries, one JSON object per line")
    parser.add_argument("--policy-config", type=Path)
    args = parser.parse_args()
    if (args.output / "trajectory.jsonl").exists():
        parser.error("Choose a new output directory for each trajectory")

    env = DiscoveryWorldEnv(args.output, args.max_steps, args.observation_mode)
    observation = env.reset(args.scenario, args.difficulty, args.seed)
    config = {} if args.policy_config is None else json.loads(args.policy_config.read_text())
    if args.policy:
        module, factory = args.policy.split(":", 1)
        agent = getattr(import_module(module), factory)(config)
        actions = None
    else:
        actions = iter(json.loads(line) for line in args.actions.read_text().splitlines() if line.strip())
        agent = None

    metadata = {
        **env.configuration, "policy": args.policy,
        "policy_config": config, "action_file": None if args.actions is None else str(args.actions),
        "action_descriptions": env.actions, "teleport_locations": env.teleports,
        "initial_task_metrics": env.metrics,
    }
    with TrajectoryWriter(args.output, metadata) as writer:
        while not env.ended:
            if agent is None:
                action = next(actions, None)
                if action is None:
                    break
            else:
                action = agent(observation)
            transition = env.step(action)
            writer.append(transition)
            observation = env.policy_input()
    summary = {"transitions": env.steps, "task_metrics": env.metrics,
               "ended": env.ended, "protocol": "official_step_based"}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
