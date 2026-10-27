"""Behaviour accounting with real native messages and explicit synthetic edges."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from experiments.discoveryworld.analyze_rollouts import analyze_episode, analyze_models, markdown_table


class RolloutAnalysisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixture = Path(__file__).parent / "fixtures" / "discoveryworld_measurements.json"
        cls.samples = {case["label"]: case["transition"] for case in json.loads(fixture.read_text())["cases"]}

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def episode(self, labels, *, seed=310):
        """Composite fixture, not presented as a newly executed native episode."""
        rows = [deepcopy(self.samples[label]) for label in labels]
        return self.write_episode(rows, seed=seed)

    def write_episode(self, rows, *, seed=310):
        directory = self.root / f"seed{seed}"
        directory.mkdir(exist_ok=True)
        for index, row in enumerate(rows):
            row["index"] = index
            row["task_metrics"] = [{"progress": .08 + (index + 1) * .01}]
        meta = {"policy": "learned_dynamic_action_policy", "seed": seed, "scenario": "Reactor Lab",
                "difficulty": "Normal", "initial_task_metrics": [{"progress": .08}]}
        summary = {"steps": len(rows), "progress": .08 + len(rows) * .01,
                   "completed_successfully": False, "completed": False, "truncated": True}
        (directory / "run.json").write_text(json.dumps(meta))
        (directory / "summary.json").write_text(json.dumps(summary))
        (directory / "trajectory.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
        return directory

    def test_real_readings_dialog_binding_and_exact_official_curve(self):
        directory = self.episode(["radiation_a1", "radiation_a2", "radiation_a3", "radiation_b",
                                  "reactor1", "exit1", "reactor3", "adjust3a", "adjust3b", "failed_action"])
        report = analyze_episode(directory)
        self.assertEqual(report["radiation_samples"], 4)
        self.assertEqual(report["measurement_actions"], 4)
        self.assertEqual(report["max_same_action_run"], 3)
        self.assertEqual(len(report["tools_acquired"]), 5)
        self.assertEqual(len(report["measured_targets"]), 2)
        self.assertEqual(report["exit_after_sampling"], 1)
        self.assertEqual(report["sampling_exits_with_new_evidence"], 1)
        self.assertEqual(report["calibration_dialogues"], 1)
        self.assertEqual(report["calibration_events"][0]["reactor"], 50759)
        self.assertEqual(report["calibration_events"][0]["frequency_hz"], 1657.6)
        self.assertEqual(report["tuning_actions"], 2)
        self.assertEqual(report["tuning_events"][0]["returned_settings_hz"], [4000.])
        self.assertEqual(report["new_activations"], 0)
        self.assertEqual(report["progress_by_action"], [.08] + [.08 + i * .01 for i in range(1, 11)])
        self.assertFalse(report["success"])

    def test_truncation_failed_exit_and_condition_change_are_not_successful_sampling_exits(self):
        directory = self.episode(["radiation_a1", "radiation_a2"])
        self.assertEqual(analyze_episode(directory)["exit_after_sampling"], 0)
        directory = self.episode(["radiation_a1", "radiation_a2", "failed_action"])
        report = analyze_episode(directory)
        self.assertEqual(report["measurement_actions"], 2)
        self.assertEqual(report["exit_after_sampling"], 0)
        rows = [deepcopy(self.samples[key]) for key in ("radiation_a1", "radiation_a2", "radiation_a3")]
        rows[-1]["next_observation"]["ui"]["agentLocation"]["x"] += 1
        report = analyze_episode(self.write_episode(rows))
        self.assertEqual(report["max_same_action_run"], 3)
        self.assertEqual(report["exit_after_sampling"], 0)

    def test_successful_exit_can_have_no_new_evidence(self):
        directory = self.episode(["density", "radiation_a1", "radiation_a2", "density"])
        report = analyze_episode(directory)
        self.assertEqual(report["exit_after_sampling"], 1)
        self.assertEqual(report["sampling_exits_with_new_evidence"], 0)

    def test_activation_needs_visible_transition_and_excludes_known_restoration(self):
        # Synthetic state transitions use actual public object descriptions.
        rows = [deepcopy(self.samples["reactor1"])]
        novel = deepcopy(self.samples["adjust3a"])
        for item in novel["next_observation"]["ui"]["accessibleEnvironmentObjects"]:
            if item["uuid"] == 10581:
                item["name"] = "crystal reactor (activated)"
        rows.append(novel)
        restored = deepcopy(self.samples["reactor1"])
        for item in restored["observation"]["ui"]["accessibleEnvironmentObjects"]:
            if item["uuid"] == 50759:
                item["name"] = "crystal reactor (uncalibrated)"
        rows.append(restored)
        report = analyze_episode(self.write_episode(rows))
        self.assertEqual(report["activation_transitions"], 2)
        self.assertEqual(report["new_activations"], 1)
        self.assertEqual(report["activation_events"][0]["newly_activated_crystals"], [63031])
        self.assertEqual(report["activation_events"][1]["newly_activated_crystals"], [])
        # A first sighting of a working reactor after navigation is not causal evidence.
        unseen = deepcopy(self.samples["reactor1"])
        unseen["observation"] = {"ui": {}}
        self.assertEqual(analyze_episode(self.write_episode([unseen]))["activation_transitions"], 0)

    def test_reference_rejected_and_summary_must_match_actual_trace(self):
        directory = self.episode(["radiation_a1"])
        path = directory / "summary.json"
        summary = json.loads(path.read_text())
        summary["progress"] = 1.
        path.write_text(json.dumps(summary))
        with self.assertRaisesRegex(ValueError, "progress mismatch"):
            analyze_episode(directory)
        path = directory / "run.json"
        meta = json.loads(path.read_text())
        for policy in ("scripted_public_reactor_reference_v1", "public_history_recovery_expert", "unknown_policy"):
            with self.subTest(policy=policy):
                meta["policy"] = policy
                path.write_text(json.dumps(meta))
                with self.assertRaisesRegex(ValueError, "independent model-policy"):
                    analyze_episode(directory)

    def test_frozen_semantic_generation_is_accepted_with_distinct_policy_identity(self):
        directory = self.episode(["density"])
        path = directory / "run.json"
        meta = json.loads(path.read_text())
        meta["policy"] = "frozen_semantic_native_generation"
        path.write_text(json.dumps(meta))
        report = analyze_models({"frozen semantic": self.root})
        model = report["models"]["frozen semantic"]
        self.assertEqual(model["policy_types"], ["frozen_semantic_native_generation"])
        self.assertEqual(model["episodes"][0]["policy"], "frozen_semantic_native_generation")
        self.assertEqual(model["episodes"][0]["measurement_actions"], 1)
        self.assertIn("| frozen semantic | frozen_semantic_native_generation | 1 |", markdown_table(report))

    def test_model_json_and_table_keep_per_episode_curves(self):
        self.episode(["radiation_a1", "radiation_a2"], seed=311)
        self.episode(["density"], seed=310)
        report = analyze_models({"test": self.root})
        model = report["models"]["test"]
        self.assertEqual(model["seeds"], [310, 311])
        self.assertEqual([len(curve) for curve in model["progress_by_action"]], [2, 3])
        self.assertEqual(report["max_actions"], 2)
        self.assertEqual(model["successful_episodes"], 0)
        self.assertEqual(model["policy_types"], ["learned_dynamic_action_policy"])
        self.assertTrue(all(episode["policy"] == "learned_dynamic_action_policy" for episode in model["episodes"]))
        self.assertIn("| test | learned_dynamic_action_policy | 2 |", markdown_table(report))
        self.assertIn("do not establish sufficient", markdown_table(report))


if __name__ == "__main__":
    unittest.main()
