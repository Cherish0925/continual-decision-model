"""DiscoveryWorld observations, native actions, and recorded transitions.

This adapter uses the official step-based protocol: one submitted action,
one world tick, then a fresh observation. Task scores are training/evaluation
outputs and are not passed to the acting policy.
"""
from __future__ import annotations

import base64
from copy import deepcopy
import gzip
from hashlib import sha256
import json
from pathlib import Path
import random
from time import perf_counter
from uuid import uuid4

UPSTREAM_REVISION = "fd591323920be0d3786ef350955de1945aa571e5"
RNG_SNAPSHOT_SCHEMA = "discoveryworld_initial_python_rng_v1"


class DiscoveryWorldEnv:
    def __init__(self, output: str | Path, max_steps: int = 1000, *, observation_mode: str = "png"):
        self.output = Path(output).resolve()
        self.output.mkdir(parents=True, exist_ok=True)
        if max_steps <= 0:
            raise ValueError("max_steps must be positive")
        self.max_steps = max_steps
        if observation_mode not in ("native", "png"):
            raise ValueError("Use native or PNG observations with the trajectory adapter")
        self.observation_mode = observation_mode
        self.api = None

    def reset(self, scenario: str, difficulty: str, seed: int) -> dict:
        from discoveryworld.DiscoveryWorldAPI import DiscoveryWorldAPI
        # A fresh API also clears upstream per-agent UI and action bookkeeping.
        self.api = DiscoveryWorldAPI(threadID=0)
        self.episode_id = uuid4().hex
        self.api.FRAME_DIR = str(self.output / "frames" / self.episode_id)
        self.started = perf_counter()
        self.steps = 0
        if not self.api.loadScenario(scenario, difficulty, randomSeed=seed, numUserAgents=1):
            raise ValueError(f"Could not load {scenario!r}, {difficulty!r}, seed {seed}")
        self.configuration = {
            "episode_id": self.episode_id, "scenario": scenario,
            "difficulty": difficulty, "seed": seed, "max_steps": self.max_steps,
            "protocol": "official_step_based", "api_version": self.api.VERSION,
            "requested_upstream_revision": UPSTREAM_REVISION,
            "observation_adapter": "active_dialog_render_sync_v1",
            "observation_mode": self.observation_mode,
        }
        self.actions = self.api.listKnownActions(limited=False)
        self.teleports = self.api.listTeleportLocationsDict()
        self.current = self._observe()
        self.metrics = self._metrics()
        self.ended = all(m["completed"] for m in self.metrics)
        return self.policy_input()

    def _initial_rng_generators(self) -> dict:
        """Evaluator-only RNG handles. Never include these in policy_input.

        Native Object creates an unseeded random.Random, so scenario seed alone
        cannot replay instrument noise. Contents and parts also own such RNGs.
        """
        if self.api is None or self.steps != 0:
            raise RuntimeError("Initial RNG snapshots require reset and zero executed actions")
        world = self.api.world
        objects = {}
        for outer in world.getAllWorldObjects():
            for obj in outer.getAllContainedObjectsAndParts():
                objects[str(obj.uuid)] = obj.rng
        return {"world": world.rng, "api": self.api.r, "uuid_generator": world.uuidGenerator.random,
                "python_global": random, "objects": objects}

    def save_initial_rng(self, path: str | Path) -> dict:
        """Capture native Python RNG states without reseeding, drawing or ticking.

        This is an evaluator artifact, not an observation or a complete world
        checkpoint. Only the post-reset/pre-action boundary is supported.
        """
        generators = self._initial_rng_generators()
        snapshot = {"schema": RNG_SNAPSHOT_SCHEMA,
                    "environment": {key: self.configuration[key] for key in
                                    ("scenario", "difficulty", "seed", "api_version", "requested_upstream_revision")},
                    "world_steps": self.current["ui"]["world_steps"],
                    "rng": {name: rng.getstate() for name, rng in generators.items() if name != "objects"},
                    "object_rng": {key: rng.getstate() for key, rng in sorted(generators["objects"].items())}}
        payload = gzip.compress(json.dumps(snapshot, separators=(",", ":")).encode(), compresslevel=6, mtime=0)
        with Path(path).open("xb") as stream:
            stream.write(payload)
        return {"schema": RNG_SNAPSHOT_SCHEMA, "sha256": sha256(payload).hexdigest(),
                "object_rngs": len(snapshot["object_rng"]), "world_steps": snapshot["world_steps"]}

    def restore_initial_rng(self, path: str | Path, *, expected_sha256: str | None = None) -> dict:
        """Restore a matching initial evaluator snapshot, leaving world state intact."""
        generators = self._initial_rng_generators()
        payload = Path(path).read_bytes()
        digest = sha256(payload).hexdigest()
        if expected_sha256 is not None and digest != expected_sha256:
            raise ValueError("Initial RNG snapshot hash mismatch")
        snapshot = json.loads(gzip.decompress(payload))
        identity = {key: self.configuration[key] for key in
                    ("scenario", "difficulty", "seed", "api_version", "requested_upstream_revision")}
        if snapshot.get("schema") != RNG_SNAPSHOT_SCHEMA or snapshot.get("environment") != identity:
            raise ValueError("Initial RNG snapshot environment/schema mismatch")
        if snapshot.get("world_steps") != self.current["ui"]["world_steps"]:
            raise ValueError("Initial RNG snapshot world-step mismatch")
        if set(snapshot["object_rng"]) != set(generators["objects"]):
            raise ValueError("Initial RNG snapshot object UUID set mismatch")
        if set(snapshot["rng"]) != set(generators) - {"objects"}:
            raise ValueError("Initial RNG snapshot generator set mismatch")

        def tuples(value):
            return tuple(tuples(item) for item in value) if isinstance(value, list) else value

        states = [(generators[name], tuples(state)) for name, state in snapshot["rng"].items()]
        states += [(generators["objects"][key], tuples(state)) for key, state in snapshot["object_rng"].items()]
        # Validate every state before mutating any live generator.
        probe = random.Random(0)
        for _, state in states:
            probe.setstate(state)
        for rng, state in states:
            rng.setstate(state)
        return {"schema": RNG_SNAPSHOT_SCHEMA, "sha256": digest,
                "object_rngs": len(snapshot["object_rng"]), "world_steps": snapshot["world_steps"]}

    def _observe(self) -> dict:
        capture = perf_counter() - self.started
        # The pinned JSON USE handler can enter an NPC device's dialogue,
        # including after swapping arguments, without updating dialogToDisplay.
        # TALK and keyboard USE already render this exact current-dialog view.
        # Synchronize that UI cache before both image and JSON rendering. This
        # reads only the current conversation, never task rules or unseen objects;
        # it submits no action and advances neither the dialogue nor the clock.
        if self.api.isAgentInDialog(agentIdx=0):
            ui = self.api.ui[0]
            text, options = ui.currentAgent.getAgentInDialogWith().dialogTree.getCurrentDialog()
            ui.dialogToDisplay = {"dialogText": text, "dialogOptions": options}
        # Leave a completed dialogue's final message visible for its native
        # final observation; empty options already permit normal actions.
        if self.observation_mode == "native":
            raw = self.api.getAgentObservation(agentIdx=0)
        else:
            from .observation import get_observation
            raw = get_observation(self.api, agentIdx=0)
        arrival = perf_counter() - self.started
        if raw["errors"]:
            raise RuntimeError("; ".join(raw["errors"]))
        return {
            "ui": deepcopy(raw["ui"]), "vision": deepcopy(raw["vision"]),
            "capture_seconds": capture, "arrival_seconds": arrival,
        }

    def _metrics(self) -> list[dict]:
        return [
            {"task": s["taskName"], "progress": float(s["scoreNormalized"]),
             "completed": bool(s["completed"]),
             "completed_successfully": bool(s["completedSuccessfully"])}
            for s in self.api.getTaskScorecard()
        ]

    def policy_input(self) -> dict:
        """Native observed UI/images and documented action/location information."""
        if self.api is None:
            raise RuntimeError("Call reset before requesting a policy input")
        return {**deepcopy(self.current), "actions": deepcopy(self.actions),
                "teleport_locations": deepcopy(self.teleports)}

    def step(self, action: dict) -> dict:
        if self.api is None or self.ended:
            raise RuntimeError("A live episode is required")
        action = deepcopy(action)
        before = self.current
        before_metrics = self.metrics
        submitted = perf_counter() - self.started
        result = self.api.performAgentAction(agentIdx=0, actionJSON=action)
        applied = perf_counter() - self.started
        tick = self.api.tick()
        if not tick["success"]:
            raise RuntimeError("; ".join(tick["errors"]))
        advanced = perf_counter() - self.started
        # Upstream areTasksComplete/getStepCounter use observation-cached fields.
        # Refresh before inspecting completion or returning the next policy input.
        self.current = self._observe()
        self.metrics = self._metrics()
        self.steps += 1
        terminated = all(m["completed"] for m in self.metrics)
        truncated = self.steps >= self.max_steps and not terminated
        self.ended = terminated or truncated
        reward = (
            sum(m["progress"] for m in self.metrics) / len(self.metrics)
            - sum(m["progress"] for m in before_metrics) / len(before_metrics)
        )
        return {
            "episode_id": self.episode_id, "index": self.steps - 1,
            "observation": deepcopy(before), "action": action,
            "action_result": deepcopy(result), "next_observation": deepcopy(self.current),
            "training_reward": reward, "task_metrics": deepcopy(self.metrics),
            "terminated": terminated, "truncated": truncated,
            "timing": {
                "action_submitted_seconds": submitted,
                "action_applied_seconds": applied,
                "world_advanced_seconds": advanced,
                "decision_seconds": submitted - before["arrival_seconds"],
                "observation_to_action_seconds": applied - before["capture_seconds"],
                "environment_action_seconds": applied - submitted,
                "environment_tick_seconds": advanced - applied,
            },
        }


class TrajectoryWriter:
    """Append complete transitions; store PNG bytes once per distinct image."""

    def __init__(self, output: str | Path, configuration: dict):
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        (self.output / "images").mkdir(exist_ok=True)
        # Each collection run has its own directory, preserving earlier runs.
        self.stream = (self.output / "trajectory.jsonl").open("x", encoding="utf-8")
        (self.output / "run.json").write_text(
            json.dumps(configuration, indent=2) + "\n", encoding="utf-8")

    def _observation(self, observation: dict) -> dict:
        record = deepcopy(observation)
        for key, data_url in record["vision"].items():
            header, encoded = data_url.split(",", 1)
            if header != "data:image/png;base64":
                raise ValueError(f"Unexpected image representation: {header}")
            payload = base64.b64decode(encoded, validate=True)
            name = sha256(payload).hexdigest() + ".png"
            path = self.output / "images" / name
            if not path.exists():
                path.write_bytes(payload)
            record["vision"][key] = {"path": f"images/{name}", "mime_type": "image/png"}
        return record

    def append(self, transition: dict):
        record = deepcopy(transition)
        record["observation"] = self._observation(record["observation"])
        record["next_observation"] = self._observation(record["next_observation"])
        self.stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        self.stream.flush()

    def close(self):
        self.stream.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
