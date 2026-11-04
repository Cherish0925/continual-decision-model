"""Run one native-action policy across the eight published DiscoveryWorld themes."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from contextlib import redirect_stdout
from hashlib import sha256
from importlib import import_module, metadata
import json
import multiprocessing
from pathlib import Path
import platform
import subprocess
from time import perf_counter
import traceback

from .adapter import DiscoveryWorldEnv, TrajectoryWriter, UPSTREAM_REVISION
from .structured_observation import StructuredDiscoveryWorldEnv


THEMES = ("Combinatorial Chemistry", "Archaeology Dating", "Plant Nutrients",
          "Reactor Lab", "Lost in Translation", "Space Sick", "Proteomics",
          "It's (not) Rocket Science!")
LIMITS = {"Easy": 100, "Normal": 1000, "Challenge": 1000}
SKILLS = ("Small Skills: Dialog Test", "Small Skills: Pick and Place Test",
          "Small Skills: Pick and Give Test", "Small Skills: Instrument Measurement Test",
          "Small Skills: Doors Test", "Small Skills: Doors with Keys Test",
          "Small Skills: Navigation in a House Test", "Small Skills: Search Test",
          "Small Skills: Discovery Feed Test", "Small Skills: Moving Agents Test")


def official_cases(seeds=range(5)):
    return [dict(id=f"{index:02d}-{difficulty.lower()}-{seed}", theme=theme,
                 difficulty=difficulty, seed=seed, max_steps=limit)
            for index, theme in enumerate(THEMES, 1)
            for difficulty, limit in LIMITS.items() for seed in seeds]


_POLICIES = {}


def initialize_worker(threads):
    import torch
    torch.set_num_threads(threads)


def reusable_policy(cfg):
    """Load weights once per worker; each case still starts with empty state."""
    key = (cfg["policy"], json.dumps(cfg["config"], sort_keys=True))
    if key not in _POLICIES:
        module, factory = cfg["policy"].split(":", 1)
        created = getattr(import_module(module), factory)(cfg["config"])
        if not hasattr(created, "reset"):
            return created
        _POLICIES[key] = created
    agent = _POLICIES[key]
    if hasattr(agent, "reset"):
        agent.reset()
    return agent


def simulator_build():
    import discoveryworld.ScenarioMaker as scenarios
    for theme in THEMES:
        entry = scenarios.SCENARIO_INFOS[theme]
        if set(entry["difficulty"]) != set(LIMITS) or len(entry["variations"]) != 5:
            raise ValueError(f"Installed scenario roster differs for {theme}")
    source = Path(scenarios.__file__).resolve().parents[1]
    revision = None
    if (source / ".git").exists():
        revision = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    else:
        try:
            raw = metadata.distribution("discoveryworld").read_text("direct_url.json")
            revision = json.loads(raw or "{}").get("vcs_info", {}).get("commit_id")
        except metadata.PackageNotFoundError:
            pass
    rocket = Path(scenarios.__file__).parent / "scenarios/not_rocket_science.py"
    return {"revision": revision, "rocket_source_sha256": sha256(rocket.read_bytes()).hexdigest(),
            "python": platform.python_version()}


def write_json(path, data):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def run_case(job):
    case, cfg = job
    path = Path(cfg["output"]) / case["id"]
    path.mkdir()
    row = dict(case)
    try:
        # The simulator prints asset and world-history messages on every episode/tick.
        with (path / "environment.log").open("w") as log, redirect_stdout(log):
            limit = 1 if cfg["mode"] == "probe" else case["max_steps"]
            if cfg["observation"] == "structured":
                env = StructuredDiscoveryWorldEnv(path, limit)
            else:
                env = DiscoveryWorldEnv(path, limit, observation_mode="png")
            obs = env.reset(case["theme"], case["difficulty"], case["seed"])
            if not env.metrics:
                raise ValueError("No task scorecard after reset")
            start = obs["ui"]["world_steps"]
            if cfg["mode"] == "probe":
                env.save_initial_rng(path / "initial_rng.json.gz")
                action = "DISCOVERY_FEED_GET_UPDATES"
                step = env.step({"action": action})
                end = step["next_observation"]["ui"]["world_steps"]
                if not step["action_result"].get("success") or end != start + 1:
                    raise RuntimeError(f"Native action/tick failed: {step['action_result']}, {start} -> {end}")
                row.update(status="environment_ready", action=action, initial_tick=start, final_tick=end)
            else:
                if cfg["rng_from"]:
                    env.restore_initial_rng(Path(cfg["rng_from"]) / case["id"] / "initial_rng.json.gz")
                env.save_initial_rng(path / "initial_rng.json.gz")
                agent = reusable_policy(cfg)
                decisions, execution = [], []
                started = perf_counter()
                configuration = dict(env.configuration, action_descriptions=env.actions,
                                     teleport_locations=env.teleports)
                with (path / "actions.jsonl").open("w") as actions, TrajectoryWriter(path, configuration) as writer:
                    while not env.ended:
                        before = perf_counter()
                        action = agent(obs)
                        selected = perf_counter()
                        step = env.step(action)
                        acted = perf_counter()
                        writer.append(step)
                        decisions.append(selected-before)
                        execution.append(acted-selected)
                        public = {k: step[k] for k in ("index", "observation", "action", "action_result", "next_observation")}
                        if hasattr(agent, "observe"):
                            agent.observe(public)
                        actions.write(json.dumps({"step": env.steps, "action": step["action"],
                                                   "result": step["action_result"],
                                                   "decision_seconds": decisions[-1],
                                                   "environment_seconds": execution[-1]}) + "\n")
                        obs = env.policy_input()
                scores = env.metrics
                row.update(status="evaluated", steps=env.steps, success=all(x["completed_successfully"] for x in scores),
                           terminated=all(x["completed"] for x in scores),
                           progress=sum(x["progress"] for x in scores) / len(scores),
                           task_metrics=scores, knowledge_score=None,
                           rollout_seconds=perf_counter()-started,
                           decision_seconds=sum(decisions), environment_seconds=sum(execution))
                if callable(getattr(agent, "get_knowledge", None)):
                    write_json(path / "submitted_knowledge.json", agent.get_knowledge())
                elif getattr(agent, "model_authored_notes", None) is not None:
                    write_json(path / "submitted_knowledge.json", agent.model_authored_notes)
    except (Exception, SystemExit) as error:
        row.update(status="error", error=f"{type(error).__name__}: {error}")
        (path / "error.txt").write_text(traceback.format_exc())
    write_json(path / "case.json", row)
    return row


def summarize(rows, mode):
    def group(items):
        status = "environment_ready" if mode == "probe" else "evaluated"
        valid = [x for x in items if x["status"] == status]
        out = dict(cases=len(items), recorded=len(valid), errors=len(items) - len(valid))
        if mode == "run" and items and len(valid) == len(items):
            out.update(successes=sum(x["success"] for x in valid),
                       mean_progress=sum(x["progress"] for x in valid) / len(valid), knowledge_score=None)
        return out
    return dict(mode=mode, overall=group(rows), by_theme={
        theme: {d: group([x for x in rows if (x["theme"], x["difficulty"]) == (theme, d)])
                for d in LIMITS} for theme in THEMES},
        by_difficulty={d: group([x for x in rows if x["difficulty"] == d]) for d in LIMITS})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("list", "probe", "run"))
    p.add_argument("--output", type=Path)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--seeds", type=int, nargs="+", default=list(range(5)),
                   help="Official evaluation uses 0--4; use separate seeds for development")
    p.add_argument("--difficulties", nargs="+", choices=tuple(LIMITS), default=list(LIMITS),
                   help="Development may use a subset; the official suite includes all three")
    p.add_argument("--observation", choices=("structured", "png"), default="structured")
    p.add_argument("--policy", help="module:factory; weights load once per worker, state resets per case")
    p.add_argument("--policy-config", type=Path, help="JSON passed to the factory")
    p.add_argument("--policy-id", help="Checkpoint revision or run name recorded with the results")
    p.add_argument("--rng-from", type=Path, help="Reuse an earlier scored run's initial simulator RNG snapshots")
    args = p.parse_args()
    cases = [case for case in official_cases(args.seeds) if case['difficulty'] in args.difficulties]
    if args.mode == "list":
        print(json.dumps(cases, indent=2))
        return
    if not args.output or min(args.workers, args.threads) < 1 or (args.mode == "run") != bool(args.policy):
        p.error("Specify a fresh --output and positive --workers; --policy is required only for run")
    if args.mode == "probe" and (args.policy_config or args.policy_id or args.rng_from):
        p.error("A probe has no policy configuration or paired run")
    if args.policy and (args.policy.count(":") != 1 or not all(args.policy.split(":"))):
        p.error("--policy must be module:factory")
    build = simulator_build()
    if build["revision"] is not None and build["revision"] != UPSTREAM_REVISION:
        p.error(f"Simulator revision {build['revision']} differs from {UPSTREAM_REVISION}")
    config = json.loads(args.policy_config.read_text()) if args.policy_config else {}
    if not isinstance(config, dict):
        p.error("--policy-config must be a JSON object")
    path = args.output.resolve()
    path.mkdir(parents=True, exist_ok=False)
    cfg = dict(mode=args.mode, output=str(path), observation=args.observation, policy=args.policy,
               config=config, rng_from=str(args.rng_from.resolve()) if args.rng_from else None)
    meta = dict(mode=args.mode, simulator=build, target_revision=UPSTREAM_REVISION,
                official_instances=args.seeds == list(range(5)) and set(args.difficulties) == set(LIMITS),
                observation=args.observation, policy=args.policy, policy_id=args.policy_id,
                policy_config_sha256=sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
                rng_from=cfg["rng_from"], cases=cases)
    write_json(path / "run.json", meta)
    jobs = [(case, cfg) for case in cases]
    if args.workers == 1:
        initialize_worker(args.threads)
        rows = [run_case(job) for job in jobs]
    else:
        with ProcessPoolExecutor(args.workers, mp_context=multiprocessing.get_context("spawn"),
                                 initializer=initialize_worker, initargs=(args.threads,)) as pool:
            rows = list(pool.map(run_case, jobs))
    write_json(path / "cases.json", rows)
    summary = summarize(rows, args.mode)
    write_json(path / "summary.json", summary)
    print(json.dumps(summary["overall"]))
    if summary["overall"]["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
