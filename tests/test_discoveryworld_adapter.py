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

    def test_use_npc_device_renders_dialog_in_both_argument_orders(self):
        env = DiscoveryWorldEnv(Path(self.folder.name) / "dialog", max_steps=8)
        with contextlib.redirect_stdout(io.StringIO()):
            observation = env.reset("Reactor Lab", "Normal", 310)
            first_step = observation["ui"]["world_steps"]
            env.step({"action": "TELEPORT_TO_LOCATION", "arg1": "reactor lab"})
            observation = env.policy_input()
            visible = [obj for group in observation["ui"]["nearbyObjects"]["objects"].values() for obj in group]
            reactor = next(obj["uuid"] for obj in visible if obj["name"] == "crystal reactor (activated)")
            env.step({"action": "TELEPORT_TO_OBJECT", "arg1": reactor})
            crystal = next(obj["uuid"] for obj in env.policy_input()["ui"]["accessibleEnvironmentObjects"]
                           if obj["name"].startswith("quantum crystal") and f"[uuid: {reactor}]" in obj["description"])
            for first, second in ((reactor, crystal), (crystal, reactor)):
                result = env.step({"action": "USE", "arg1": first, "arg2": second})
                self.assertTrue(result["action_result"]["success"])
                self.assertTrue(env.api.isAgentInDialog(agentIdx=0))
                dialog = result["next_observation"]["ui"]["dialog_box"]
                self.assertTrue(dialog["is_in_dialog"])
                self.assertIn("current resonance", dialog["dialogIn"])
                self.assertTrue(dialog["dialogOptions"])
                exit_index = next(int(index) for index, label in dialog["dialogOptions"].items() if label == "Exit")
                result = env.step({"chosen_dialog_option_int": exit_index})
                self.assertTrue(result["action_result"]["success"])
                self.assertFalse(env.api.isAgentInDialog(agentIdx=0))
            result = env.step({"action": "ROTATE_DIRECTION", "arg1": "north"})
        self.assertTrue(result["action_result"]["success"])
        self.assertEqual(env.steps, 7)
        self.assertEqual(result["next_observation"]["ui"]["world_steps"], first_step + 7)
        self.assertNotIn("criticalHypotheses", json.dumps(env.policy_input()))


if __name__ == "__main__":
    unittest.main()
