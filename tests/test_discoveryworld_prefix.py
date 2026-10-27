"""Prefix boundaries, native replay failures and causal policy priming."""
import contextlib
from copy import deepcopy
import io
from importlib.util import find_spec
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from experiments.discoveryworld.adapter import DiscoveryWorldEnv, TrajectoryWriter
from experiments.discoveryworld.prefix_evaluate import evaluate_prefix, load_prefix, _differences
from experiments.discoveryworld.reference import observed_objects


class PrefixReaderTests(unittest.TestCase):
    def test_native_dialogue_keys_match_serialized_keys_without_changing_readings(self):
        actual = {"dialog_box": {"dialogIn": "2171.12 Hertz", "dialogOptions": {1: "Increase", 9: "Exit"}}}
        saved = json.loads(json.dumps(actual))
        self.assertEqual(_differences(saved, actual, "ui"), [])
        actual["dialog_box"]["dialogIn"] = "2171.13 Hertz"
        self.assertEqual(len(_differences(saved, actual, "ui")), 1)

    def test_only_requested_prefix_is_read_and_annotations_are_not_retained(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "run.json").write_text(json.dumps({"scenario": "Reactor Lab", "difficulty": "Normal",
                                                      "seed": 100, "policy": "test_source", "hypothesis_family": "unused"}))
            row = {"index": 0, "action": {"action": "WAIT"}, "action_result": {"success": True},
                   "observation": {"ui": {}}, "next_observation": {"ui": {}},
                   "expert_phase": "must not be retained", "hypothesis": "must not be retained"}
            path = root / "trajectory.jsonl"
            path.write_text(json.dumps(row) + "\nDO NOT READ OR PARSE THE SUFFIX\n")
            # No summary or future answer file exists or is needed.
            prefix = load_prefix(root, 1)
            self.assertEqual(prefix["rows"][0]["action"], row["action"])
            self.assertNotIn("hypothesis", json.dumps(prefix))
            self.assertNotIn("expert_phase", json.dumps(prefix))
            self.assertEqual(load_prefix(path, 0)["rows"], [])
            with self.assertRaises(ValueError):
                load_prefix(root, 2)


class ObservingPolicy:
    """Protocol spy only; never a reported learned/semantic model result."""
    def reset(self, **kwargs):
        self.observed, self.calls = [], []

    def __call__(self, observation):
        self.calls.append({"received": len(self.observed), "world_step": observation["ui"]["world_steps"]})
        return {"action": "ROTATE_DIRECTION", "arg1": "east"}

    def observe(self, transition):
        self.observed.append(deepcopy(transition))


@unittest.skipUnless(find_spec("discoveryworld"), "Install the pinned DiscoveryWorld simulator")
class NativePrefixTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        cls.source = cls.root / "source"
        cls.source.mkdir()
        env = DiscoveryWorldEnv(cls.source, max_steps=2)
        with contextlib.redirect_stdout(io.StringIO()):
            env.reset("Reactor Lab", "Normal", 100)
            with TrajectoryWriter(cls.source, {**env.configuration, "policy": "native_protocol_test_source"}) as writer:
                first = env.step({"action": "TELEPORT_TO_LOCATION", "arg1": "science lab (instruments)"})
                writer.append(first)
                instrument = next(obj for obj in observed_objects(env.policy_input()).values()
                                  if obj["name"] == "radiation meter")
                second = env.step({"action": "TELEPORT_TO_OBJECT", "arg1": instrument["uuid"]})
                writer.append(second)
        cls.actions = [first["action"], second["action"]]

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def run_takeover(self, policy, name, policy_type, *, source=None):
        output = self.root / name
        with contextlib.redirect_stdout(io.StringIO()):
            summary = evaluate_prefix(policy, source or self.source, output, prefix_steps=2,
                                      takeover_steps=1, policy_type=policy_type)
        rows = [json.loads(line) for line in (output / "trajectory.jsonl").read_text().splitlines()]
        return summary, rows, output

    def test_native_prefix_and_feedback_with_both_priming_modes(self):
        for policy_type, calls, received in (("learned_dynamic_action_policy", 3, [0, 1, 2]),
                                             ("learned_calibration_with_native_scalar_control", 3, [0, 1, 2]),
                                             ("frozen_semantic_native_generation", 1, [2])):
            with self.subTest(policy_type=policy_type):
                policy = ObservingPolicy()
                summary, rows, output = self.run_takeover(policy, policy_type, policy_type)
                self.assertTrue(summary["prefix_replay_matched"])
                self.assertFalse(summary["rng_replay"]["initial_rng_restored"])
                self.assertIn("unguaranteed", summary["rng_replay"]["status"])
                self.assertEqual(summary["takeover_steps"], 1)
                self.assertEqual(len(policy.calls), calls)
                self.assertEqual([call["received"] for call in policy.calls], received)
                self.assertEqual(len(policy.observed), 3)
                self.assertTrue(all("task_metrics" not in row and "training_reward" not in row for row in policy.observed))
                self.assertEqual([row["action"] for row in rows[:2]], self.actions)
                self.assertEqual([row["control"] for row in rows], ["external_prefix"] * 2 + ["policy_takeover"])
                self.assertEqual(rows[-1]["takeover_index"], 0)
                self.assertEqual(len(summary["prefix_progress_by_action"]), 3)
                self.assertEqual(len(summary["takeover_progress_by_action"]), 2)
                self.assertFalse(summary["is_end_to_end_evaluation"])
                metadata = json.loads((output / "run.json").read_text())
                self.assertEqual(metadata["policy"], "prefix_takeover_evaluation")
                self.assertEqual(metadata["takeover_policy"], policy_type)
                proposal_calls = [json.loads(line) for line in (output / "policy_calls.jsonl").read_text().splitlines()]
                self.assertEqual(sum(call["proposal_executed"] for call in proposal_calls), 1)

    def test_unreplayable_object_reference_is_recorded_and_prevents_takeover(self):
        source = self.root / "bad_source"
        source.mkdir()
        (source / "run.json").write_text((self.source / "run.json").read_text())
        rows = [json.loads(line) for line in (self.source / "trajectory.jsonl").read_text().splitlines()]
        rows[-1]["action"]["arg1"] = 999999999
        (source / "trajectory.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
        policy = ObservingPolicy()
        summary, actual, _ = self.run_takeover(policy, "mismatch", "frozen_semantic_native_generation", source=source)
        self.assertFalse(summary["prefix_replay_matched"])
        self.assertEqual(summary["takeover_steps"], 0)
        self.assertEqual(summary["prefix_steps_executed"], 2)
        self.assertEqual(summary["prefix_mismatches"][0]["stage"], "after_action")
        self.assertFalse(actual[-1]["action_result"]["success"])
        self.assertEqual(len(policy.observed), 2)
        self.assertEqual(policy.calls, [])

    def test_first_action_intervention_does_not_execute_or_label_the_proposal(self):
        class PrimingPolicy(ObservingPolicy):
            def prime(self, observation):
                self.calls.append({"prime": True})
                return {"action": "ROTATE_DIRECTION", "arg1": "west"}

        policy = PrimingPolicy()
        output = self.root / "action_intervention"
        forced = {"action": "ROTATE_DIRECTION", "arg1": "north"}
        with contextlib.redirect_stdout(io.StringIO()):
            summary = evaluate_prefix(policy, self.source, output, prefix_steps=2,
                                      takeover_steps=2, policy_type="learned_dynamic_action_policy",
                                      first_action=forced)
        rows = [json.loads(line) for line in (output / "trajectory.jsonl").read_text().splitlines()]
        calls = [json.loads(line) for line in (output / "policy_calls.jsonl").read_text().splitlines()]
        self.assertIsNone(summary["error"])
        self.assertTrue(summary["prefix_replay_matched"])
        self.assertEqual(rows[2]["action"], forced)
        self.assertEqual(policy.observed[2]["action"], forced)
        self.assertEqual(rows[3]["action"]["arg1"], "east")
        self.assertEqual(sum(bool(row.get("prime")) for row in policy.calls), 3)
        self.assertFalse(calls[2]["proposal_executed"])
        self.assertEqual(calls[2]["executed_action"], forced)
        self.assertTrue(calls[3]["proposal_executed"])

    def test_compute_intervention_is_one_decision_and_does_not_add_world_ticks(self):
        class ComputePolicy(ObservingPolicy):
            refinements = 1

            def __call__(self, observation):
                action = super().__call__(observation)
                self.calls[-1]["refinements"] = self.refinements
                return action

        policy = ComputePolicy()
        with contextlib.redirect_stdout(io.StringIO()):
            summary = evaluate_prefix(policy, self.source, self.root / "compute_intervention",
                                      prefix_steps=2, takeover_steps=3,
                                      policy_type="learned_dynamic_action_policy", first_refinements=4)
        self.assertTrue(summary["prefix_replay_matched"])
        self.assertEqual([r["refinements"] for r in policy.calls], [1, 1, 4, 1, 1])
        self.assertEqual(summary["takeover_steps"], 3)
        self.assertEqual(len(policy.observed), 5)
        self.assertEqual(policy.refinements, 1)

    def test_actual_transition_is_saved_even_if_policy_feedback_raises(self):
        class BrokenObserver(ObservingPolicy):
            def observe(self, transition):
                raise RuntimeError("test receiver failed")

        summary, actual, _ = self.run_takeover(BrokenObserver(), "observer_error", "frozen_semantic_native_generation")
        self.assertEqual(summary["steps"], 1)
        self.assertEqual(len(actual), 1)
        self.assertEqual(summary["prefix_steps_executed"], 1)
        self.assertEqual(summary["error"]["message"], "test receiver failed")
        self.assertEqual(summary["takeover_steps"], 0)


if __name__ == "__main__":
    unittest.main()
