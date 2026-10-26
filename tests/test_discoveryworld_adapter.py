"""Integration with the actual pinned simulator, without model weights or GPUs."""
import contextlib
import io
import json
from importlib.util import find_spec
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from experiments.discoveryworld.adapter import DiscoveryWorldEnv, TrajectoryWriter


@unittest.skipUnless(find_spec("discoveryworld"), "Install the pinned DiscoveryWorld simulator")
class DiscoveryWorldAdapterTests(unittest.TestCase):
    def setUp(self):
        self.folder = TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.env = DiscoveryWorldEnv(self.folder.name, max_steps=2)
        with contextlib.redirect_stdout(io.StringIO()):
            self.observation = self.env.reset("Reactor Lab", "Normal", 100)

    def test_native_action_advances_world_and_records_only_public_inputs(self):
        first_step = self.observation["ui"]["world_steps"]
        self.assertIn("USE", self.observation["actions"])
        self.assertTrue(self.observation["vision"]["base64_no_grid"].startswith("data:image/png;base64,"))
        result = self.env.step({"action": "ROTATE_DIRECTION", "arg1": "north"})
        self.assertTrue(result["action_result"]["success"])
        self.assertEqual(result["next_observation"]["ui"]["world_steps"], first_step + 1)
        self.assertEqual(self.env.policy_input()["ui"]["world_steps"], first_step + 1)
        self.assertNotIn("criticalHypotheses", json.dumps(result))
        self.assertNotIn("scoreCard", json.dumps(self.env.policy_input()))
        self.assertNotIn("training_reward", self.env.policy_input())
        self.assertGreaterEqual(result["timing"]["observation_to_action_seconds"], 0)
        self.assertFalse(result["terminated"])
        result = self.env.step({"action": "ROTATE_DIRECTION", "arg1": "south"})
        self.assertTrue(result["truncated"])
        self.assertFalse(result["terminated"])

    def test_reset_replaces_previous_world_and_trajectory_images_are_readable(self):
        first_episode = self.env.episode_id
        result = self.env.step({"action": "ROTATE_DIRECTION", "arg1": "north"})
        with TrajectoryWriter(self.folder.name, self.env.configuration) as writer:
            writer.append(result)
        recorded = json.loads((Path(self.folder.name) / "trajectory.jsonl").read_text())
        for name in ("observation", "next_observation"):
            for ref in recorded[name]["vision"].values():
                payload = (Path(self.folder.name) / ref["path"]).read_bytes()
                self.assertTrue(payload.startswith(b"\x89PNG\r\n\x1a\n"))
        with contextlib.redirect_stdout(io.StringIO()):
            fresh = self.env.reset("Reactor Lab", "Normal", 101)
        self.assertNotEqual(first_episode, self.env.episode_id)
        self.assertEqual(len(self.env.api.ui), 1)
        self.assertEqual(fresh["ui"]["world_steps"], 1)


if __name__ == "__main__":
    unittest.main()
