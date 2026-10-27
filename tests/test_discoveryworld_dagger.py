"""Real simulator tests for causal expert labels and retained recovery prefixes."""
import contextlib
import io
import json
from importlib.util import find_spec
from pathlib import Path
from tempfile import TemporaryDirectory
from types import ModuleType
import unittest
from unittest.mock import patch

from experiments.discoveryworld.dagger import run_episode


class StubRecoveryError(RuntimeError):
    pass


class RecordingExpert:
    latest = None

    def __init__(self, observation):
        type(self).latest = self
        self.queries = []
        self.received = []
        self.status = {"blocked": False, "completed": False}

    def recommend(self, observation):
        self.queries.append(observation["ui"]["world_steps"])
        return {"action": "ROTATE_DIRECTION", "arg1": "east"}

    def observe(self, action, observation, result):
        self.received.append((dict(action), observation["ui"]["world_steps"], dict(result)))

    def summary(self):
        return {**self.status, "observed_results": len(self.received)}


class BlockingExpert(RecordingExpert):
    def observe(self, action, observation, result):
        super().observe(action, observation, result)
        if len(self.received) == 2:
            self.status["blocked"] = True
            raise StubRecoveryError("No supported recovery from the received public state")


class RecordingLearner:
    def reset(self):
        self.queries = []

    def __call__(self, observation):
        self.queries.append(observation["ui"]["world_steps"])
        return {"action": "ROTATE_DIRECTION", "arg1": "south"}


@unittest.skipUnless(find_spec("discoveryworld"), "Install the pinned DiscoveryWorld simulator")
class DaggerCollectionTests(unittest.TestCase):
    def recovery_module(self):
        module = ModuleType("experiments.discoveryworld.recovery")
        module.RecoveryError, module.RecoveryExpert = StubRecoveryError, RecordingExpert
        return patch.dict("sys.modules", {module.__name__: module})

    def test_both_policies_observe_actual_prefix_and_results_follow_executed_action(self):
        with TemporaryDirectory() as folder, self.recovery_module(), contextlib.redirect_stdout(io.StringIO()):
            output = Path(folder)
            agent = RecordingLearner()
            result = run_episode(agent, output, max_steps=4, teacher_prefix_steps=1,
                                 student_prefix_steps=2, teacher_probability=1., expert_factory=RecordingExpert)
            rows = [json.loads(line) for line in (output / "trajectory.jsonl").read_text().splitlines()]
            self.assertEqual(agent.queries, [1, 2, 3, 4])
            self.assertEqual(RecordingExpert.latest.queries, agent.queries)
            self.assertEqual([row["controller_source"] for row in rows], ["expert", "learner", "learner", "expert"])
            self.assertEqual([row["controller_phase"] for row in rows],
                             ["teacher_prefix", "student_prefix", "student_prefix", "mixture"])
            self.assertEqual([row["action"]["arg1"] for row in rows], ["east", "south", "south", "east"])
            for row, received in zip(rows, RecordingExpert.latest.received):
                self.assertEqual(row["expert_action"]["arg1"], "east")
                self.assertEqual(row["learner_action"]["arg1"], "south")
                self.assertEqual(row["action_result"], received[2])
                self.assertEqual(row["action"], received[0])
                self.assertEqual(row["next_observation"]["ui"]["agentLocation"]["faceDirection"], row["action"]["arg1"])
                for ref in row["next_observation"]["vision"].values():
                    self.assertTrue((output / ref["path"]).read_bytes().startswith(b"\x89PNG\r\n\x1a\n"))
            self.assertTrue(result["truncated"])
            self.assertFalse(result["completed_successfully"])
            self.assertEqual(result["controller_counts"], {"expert": 2, "learner": 2})
            self.assertEqual(result["learner_expert_agreement"], 0.)
            configuration = json.loads((output / "run.json").read_text())
            self.assertTrue(configuration["source"]["sha256"])
            self.assertFalse(configuration["candidate_filtering"])

    def test_blocked_expert_preserves_last_executed_result_and_prefix(self):
        with TemporaryDirectory() as folder, self.recovery_module(), contextlib.redirect_stdout(io.StringIO()):
            output = Path(folder)
            result = run_episode(RecordingLearner(), output, max_steps=4, student_prefix_steps=4,
                                 expert_factory=BlockingExpert)
            rows = [json.loads(line) for line in (output / "trajectory.jsonl").read_text().splitlines()]
            self.assertEqual(len(rows), 2)
            self.assertEqual(result["steps"], 2)
            self.assertTrue(result["blocked"])
            self.assertFalse(result["completed"])
            self.assertFalse(rows[-1]["terminated"])
            self.assertTrue(rows[-1]["action_result"]["success"])
            self.assertEqual(result["error"]["stage"], "expert_observe")
            self.assertEqual(json.loads((output / "failure.json").read_text())["pending"]["action"], rows[-1]["action"])


if __name__ == "__main__":
    unittest.main()
