"""Actual noisy native measurements reproduced from evaluator-only RNG state."""
import contextlib
import gzip
import io
from importlib.util import find_spec
import json
from pathlib import Path
import random
from tempfile import TemporaryDirectory
import unittest

from experiments.discoveryworld.adapter import DiscoveryWorldEnv, RNG_SNAPSHOT_SCHEMA
from experiments.discoveryworld.prefix_evaluate import evaluate_prefix
from experiments.discoveryworld.reference import collect_episode


class FeedbackPolicy:
    """Protocol test receiver, not a trained-model capability claim."""
    def reset(self, **kwargs):
        self.received = []
        self.calls = 0

    def observe(self, transition):
        self.received.append(transition)

    def __call__(self, observation):
        self.calls += 1
        return {"action": "ROTATE_DIRECTION", "arg1": "north"}


@unittest.skipUnless(find_spec("discoveryworld"), "Install the pinned DiscoveryWorld simulator")
class NativeRNGTests(unittest.TestCase):
    def test_noisy_reference_prefix_replays_exactly_and_then_takes_over(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            result = collect_episode(source, seed=100, max_steps=28, capture_initial_rng=True)
            self.assertIsNone(result["error"])
            metadata = json.loads((source / "run.json").read_text())
            snapshot_info = metadata["initial_rng_snapshot"]
            self.assertEqual(snapshot_info["schema"], RNG_SNAPSHOT_SCHEMA)
            snapshot = json.loads(gzip.decompress((source / snapshot_info["path"]).read_bytes()))
            self.assertGreater(len(snapshot["object_rng"]), 100)
            self.assertEqual(set(snapshot), {"schema", "environment", "world_steps", "rng", "object_rng"})
            expected = [json.loads(line) for line in (source / "trajectory.jsonl").read_text().splitlines()]
            readings = [row["next_observation"]["ui"]["lastActionMessage"] for row in expected
                        if "radiation meter reports a level" in row["next_observation"]["ui"].get("lastActionMessage", "")]
            self.assertGreaterEqual(len(readings), 4)
            policy = FeedbackPolicy()
            output = root / "replayed"
            with contextlib.redirect_stdout(io.StringIO()):
                summary = evaluate_prefix(policy, source, output, prefix_steps=28, takeover_steps=1,
                                          policy_type="frozen_semantic_native_generation")
            self.assertTrue(summary["rng_replay"]["initial_rng_restored"])
            self.assertTrue(summary["prefix_replay_matched"])
            self.assertEqual(summary["prefix_mismatches"], [])
            self.assertEqual(summary["takeover_steps"], 1)
            self.assertEqual(policy.calls, 1)
            self.assertEqual(len(policy.received), 29)
            self.assertTrue(all(set(row) == {"index", "observation", "action", "action_result", "next_observation"}
                                for row in policy.received))
            self.assertNotIn("object_rng", json.dumps(policy.received))
            actual = [json.loads(line) for line in (output / "trajectory.jsonl").read_text().splitlines()]
            for original, replayed in zip(expected, actual):
                self.assertEqual(replayed["action_result"], original["action_result"])
                self.assertEqual(replayed["next_observation"]["ui"], original["next_observation"]["ui"])
            self.assertEqual((output / "initial_rng.json.gz").read_bytes(),
                             (source / "initial_rng.json.gz").read_bytes())

    def test_snapshot_does_not_draw_and_only_restores_matching_initial_world(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            env = DiscoveryWorldEnv(root / "world", max_steps=2)
            with self.assertRaisesRegex(RuntimeError, "zero executed actions"):
                env.save_initial_rng(root / "too_early.gz")
            with contextlib.redirect_stdout(io.StringIO()):
                env.reset("Reactor Lab", "Normal", 100)
            before = {name: rng.getstate() for name, rng in env._initial_rng_generators().items() if name != "objects"}
            public = env.policy_input()
            path = root / "initial_rng.json.gz"
            info = env.save_initial_rng(path)
            self.assertEqual(before, {name: rng.getstate() for name, rng in env._initial_rng_generators().items() if name != "objects"})
            self.assertEqual(public, env.policy_input())
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                env.restore_initial_rng(path, expected_sha256="wrong")
            self.assertEqual(env.api.world.rng.getstate(), before["world"])
            env.api.world.rng.random()
            env.api.r.random()
            random.random()
            env.restore_initial_rng(path, expected_sha256=info["sha256"])
            self.assertEqual(before, {name: rng.getstate() for name, rng in env._initial_rng_generators().items() if name != "objects"})
            with contextlib.redirect_stdout(io.StringIO()):
                env.reset("Reactor Lab", "Normal", 101)
            with self.assertRaisesRegex(ValueError, "environment/schema mismatch"):
                env.restore_initial_rng(path)
            env.step({"action": "ROTATE_DIRECTION", "arg1": "north"})
            with self.assertRaisesRegex(RuntimeError, "zero executed actions"):
                env.restore_initial_rng(path)


if __name__ == "__main__":
    unittest.main()
