"""Collect state-corrected DAgger labels during real DiscoveryWorld interaction.

Both policies receive every actual observation. Expert and learner suggestions
are obtained before choosing the executed action. Only that executed action is
paired with the simulator's returned outcome; expert_action is a separate BC
label. This collector neither edits candidate actions nor substitutes outcomes.
"""
from __future__ import annotations

import argparse
import contextlib
from copy import deepcopy
from dataclasses import asdict
from hashlib import file_digest
import inspect
import json
from pathlib import Path
import random
import subprocess
from time import perf_counter

import torch

from .adapter import DiscoveryWorldEnv, TrajectoryWriter
from .evaluate import LearnedPolicy
from .features import canonical_action


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def _sources(expert_factory) -> dict:
    root = Path(__file__).resolve().parents[2]
    paths = [Path(__file__).resolve(), *[Path(__file__).with_name(name).resolve()
             for name in ("adapter.py", "evaluate.py", "features.py", "reference.py")], root / "cdm" / "digital.py"]
    expert_source = inspect.getsourcefile(expert_factory)
    if expert_source:
        paths.append(Path(expert_source).resolve())
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, text=True,
                              capture_output=True, check=True).stdout.strip()
    return {"revision": revision, "sha256": {str(path): _sha256(path) for path in paths}}


def _controller(step: int, teacher_prefix_steps: int, student_prefix_steps: int,
                teacher_probability: float, randomizer: random.Random) -> tuple[str, str]:
    if step < teacher_prefix_steps:
        return "expert", "teacher_prefix"
    if step < teacher_prefix_steps + student_prefix_steps:
        return "learner", "student_prefix"
    return ("expert" if randomizer.random() < teacher_probability else "learner"), "mixture"


def run_episode(agent, output: str | Path, *, scenario: str = "Reactor Lab",
                difficulty: str = "Normal", seed: int = 110, max_steps: int = 150,
                teacher_probability: float = .5, teacher_prefix_steps: int = 0,
                student_prefix_steps: int = 0, mixture_seed: int = 0,
                expert_factory=None, provenance: dict | None = None) -> dict:
    """Collect one complete actual prefix and mixed-control continuation.

    The optional teacher prefix is followed by the student prefix, then the
    Bernoulli mixture. Both policies still make suggestions during either
    prefix. Expert state changes only through observe(actual_action, ...).
    RecoveryError returns a blocked summary and preserves completed transitions;
    unexpected errors preserve the same evidence and are re-raised.
    """
    from .recovery import RecoveryError, RecoveryExpert

    if not 0 <= teacher_probability <= 1:
        raise ValueError("teacher_probability must be in [0, 1]")
    if min(teacher_prefix_steps, student_prefix_steps) < 0 or max_steps <= 0:
        raise ValueError("Prefix lengths must be nonnegative and max_steps positive")
    output = Path(output)
    if (output / "trajectory.jsonl").exists() or (output / "summary.json").exists():
        raise ValueError(f"Refusing to overwrite an existing collection: {output}")
    output.mkdir(parents=True, exist_ok=True)
    expert_factory = expert_factory or RecoveryExpert
    sources = _sources(expert_factory)
    agent.reset()
    env = DiscoveryWorldEnv(output, max_steps)
    observation = env.reset(scenario, difficulty, seed)
    randomizer = random.Random(mixture_seed)
    configuration = {**env.configuration, "policy": "dagger_recovery_collection",
                     "action_descriptions": env.actions, "teleport_locations": env.teleports,
                     "initial_task_metrics": env.metrics, "source": sources,
                     "provenance": provenance or {},
                     "expert": f"{expert_factory.__module__}:{expert_factory.__qualname__}",
                     "teacher_probability": teacher_probability, "teacher_prefix_steps": teacher_prefix_steps,
                     "student_prefix_steps": student_prefix_steps, "mixture_seed": mixture_seed,
                     "schedule": "teacher prefix, student prefix, Bernoulli mixture",
                     "imitation_label": "expert_action", "outcome_action": "action",
                     "candidate_filtering": False}
    if hasattr(agent, "checkpoint"):
        configuration.update(features=asdict(agent.features.config), mode=agent.mode,
                             persistent=agent.persistent, restart_workspace=agent.restart_workspace,
                             refinements=agent.refinements if agent.refinements is not None else agent.model.config.refinements)
    counts, agreements = {"expert": 0, "learner": 0}, 0
    expert = error = None
    pending = None
    stage = "expert_initialization"
    started = perf_counter()
    with TrajectoryWriter(output, configuration) as writer:
        try:
            expert = expert_factory(deepcopy(observation))
            while not env.ended:
                stage = "expert_recommend"
                expert_action = deepcopy(expert.recommend(deepcopy(observation)))
                stage = "learner_recommend"
                learner_action = deepcopy(agent(deepcopy(observation)))
                source, phase = _controller(env.steps, teacher_prefix_steps, student_prefix_steps,
                                            teacher_probability, randomizer)
                executed = expert_action if source == "expert" else learner_action
                pending = {"index": env.steps, "expert_action": expert_action, "learner_action": learner_action,
                           "action": executed, "controller_source": source, "controller_phase": phase}
                stage = "environment_step"
                transition = env.step(deepcopy(executed))
                # Record before expert assimilation: a blocked expert must not
                # erase the action/result that exposed the recovery limitation.
                stage = "trajectory_write"
                writer.append({**transition, "expert_action": expert_action, "learner_action": learner_action,
                               "controller_source": source, "controller_phase": phase})
                if hasattr(agent, "observe"):
                    agent.observe(transition)
                counts[source] += 1
                agreements += canonical_action(expert_action) == canonical_action(learner_action)
                observation = env.policy_input()
                stage = "expert_observe"
                expert.observe(deepcopy(executed), deepcopy(observation), deepcopy(transition["action_result"]))
                pending = None
        except Exception as caught:
            error = caught
    expert_summary = expert.summary() if expert is not None else {}
    completed = all(item["completed"] for item in env.metrics)
    successful = all(item["completed_successfully"] for item in env.metrics)
    status = "blocked" if isinstance(error, RecoveryError) else "error" if error else \
        "completed" if successful else "task_ended" if completed else "truncated"
    summary = {"scenario": scenario, "difficulty": difficulty, "seed": seed,
               "steps": env.steps, "status": status, "blocked": isinstance(error, RecoveryError),
               "completed": completed, "completed_successfully": successful,
               "task_metrics": env.metrics, "progress": sum(item["progress"] for item in env.metrics) / len(env.metrics),
               "truncated": env.steps >= max_steps and not completed,
               "controller_counts": counts, "learner_expert_agreement": agreements / env.steps if env.steps else None,
               "expert": expert_summary, "elapsed_seconds": perf_counter() - started,
               "protocol": "official_step_based", "collection": "dagger_recovery",
               "error": None if error is None else {"type": type(error).__name__, "message": str(error), "stage": stage}}
    (output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    if error is not None:
        (output / "failure.json").write_text(json.dumps(
            {"error": summary["error"], "pending": pending}, indent=2, ensure_ascii=False) + "\n")
    if hasattr(agent, "decisions"):
        (output / "decisions.jsonl").write_text("".join(json.dumps(row) + "\n" for row in agent.decisions))
    if error is not None and not isinstance(error, RecoveryError):
        raise error
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--encoder-path", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scenario", default="Reactor Lab")
    parser.add_argument("--difficulty", default="Normal")
    parser.add_argument("--seed", type=int, default=110)
    parser.add_argument("--max-steps", type=int, default=150)
    parser.add_argument("--teacher-probability", type=float, default=.5)
    parser.add_argument("--teacher-prefix-steps", type=int, default=0)
    parser.add_argument("--student-prefix-steps", type=int, default=0)
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--feature-device")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--restart-state", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--restart-workspace", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--mode", choices=("cdm", "gru"))
    parser.add_argument("--refinements", type=int)
    parser.add_argument("--temperature", type=float, default=0.)
    args = parser.parse_args(argv)
    if args.threads <= 0 or args.max_steps <= 0 or min(args.teacher_prefix_steps, args.student_prefix_steps) < 0:
        parser.error("Use positive threads/max_steps and nonnegative prefix lengths")
    if not 0 <= args.teacher_probability <= 1:
        parser.error("--teacher-probability must be in [0,1]")
    if (args.output / "trajectory.jsonl").exists() or (args.output / "summary.json").exists():
        parser.error("Choose a new output directory")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.random_seed)
    args.output.mkdir(parents=True, exist_ok=True)
    provenance = {"checkpoint": str(args.checkpoint.resolve()), "checkpoint_sha256": _sha256(args.checkpoint),
                  "device": args.device, "feature_device": args.feature_device or args.device,
                  "temperature": args.temperature, "random_seed": args.random_seed}
    agent = LearnedPolicy(args.checkpoint, args.feature_cache, device=args.device,
                          feature_device=args.feature_device, encoder_path=args.encoder_path,
                          restart_state=args.restart_state, restart_workspace=args.restart_workspace,
                          mode=args.mode, refinements=args.refinements, temperature=args.temperature)
    try:
        with (args.output / "simulator.log").open("x") as log, contextlib.redirect_stdout(log):
            result = run_episode(agent, args.output, scenario=args.scenario, difficulty=args.difficulty,
                                 seed=args.seed, max_steps=args.max_steps, teacher_probability=args.teacher_probability,
                                 teacher_prefix_steps=args.teacher_prefix_steps, student_prefix_steps=args.student_prefix_steps,
                                 mixture_seed=args.random_seed, provenance=provenance)
        brief = {key: value for key, value in result.items() if key != "expert"}
        brief["expert"] = {key: value for key, value in result["expert"].items()
                           if key not in ("measurements", "trials")}
        print(json.dumps(brief, indent=2))
    finally:
        agent.features.close()


if __name__ == "__main__":
    main()
