"""Public native excerpts test execution plumbing, not model task ability."""
from copy import deepcopy
from dataclasses import asdict
import gzip
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import torch

from cdm.core import ContinualCore, CoreConfig
from cdm.digital import DigitalConfig, DigitalPolicy
from experiments.discoveryworld.features import FeatureConfig
from experiments.discoveryworld.quantitative_policy import QuantitativePolicy, scalar_dialog_action


def fixtures():
    folder = Path(__file__).parent / "fixtures"
    cases = json.loads((folder / "discoveryworld_measurements.json").read_text())["cases"]
    rows = {case["label"]: case["transition"] for case in cases}
    activation = json.loads((folder / "discoveryworld_calibration_activation.json").read_text())
    rows["activated"] = activation["transition"]
    rows["exit_after_activation"] = activation["following_transition"]
    # Raw trajectory observations omit the adapter's separately documented
    # action schema; policy_input supplies it in actual native execution.
    schema = {"USE": {"args": ["arg1", "arg2"]}, "TALK": {"args": ["arg1"]},
              "ROTATE_DIRECTION": {"args": ["arg1"]}}
    for row in rows.values():
        for key in ("observation", "next_observation"):
            row[key]["actions"] = deepcopy(schema)
            row[key]["teleport_locations"] = {}
    return rows


class PredictorSpy(torch.nn.Module):
    """Deterministic protocol fixture, never a trained-prediction result."""
    def __init__(self, source):
        super().__init__()
        self.source = source
        self.predictions, self.updates = [], []

    def context(self, state):
        return self.source.core.state_features(state["h"], state["z"])

    def predict(self, queries, contexts, adapters):
        self.predictions.append((deepcopy(queries), contexts.clone()))
        return {"mean_hz": contexts.new_full((len(queries),), 1650.),
                "std_hz": contexts.new_full((len(queries),), 1.)}

    def update(self, queries, contexts, values, adapters, *, differentiable=False):
        self.updates.append((deepcopy(queries), contexts.clone(), values.clone()))
        return {key: value.detach() + .25 for key, value in adapters.items()}, values.new_tensor([1.])


class QuantitativePolicyTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(13)
        torch.set_num_threads(1)
        self.rows = fixtures()
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)

    def make_policy(self, retain=True):
        core = ContinualCore(CoreConfig(event_dim=16, record_dim=8, actions=1, width=16,
                                       hidden=16, heads=2, slots=2, action_width=16), torch.zeros(1, 1))
        source = DigitalPolicy(DigitalConfig(width=16, text_dim=256, image_dim=256, outcome_dim=14), core)
        predictor = PredictorSpy(source)
        metadata = {"source_mode": "cdm", "source": {
            "features": asdict(FeatureConfig(mode="lexical", measurement_memory=True)),
            "training_args": {"mode": "cdm"}, "epoch": 0}}
        policy = QuantitativePolicy.__new__(QuantitativePolicy)
        policy._configure(predictor, metadata, Path(self.temporary.name) / str(retain), retain_updates=retain)
        self.addCleanup(policy.features.close)
        return policy

    def receive(self, policy, name):
        row = deepcopy(self.rows[name])
        policy(row["observation"])
        policy.observe(row)
        return row

    def test_actual_native_units_options_and_activation_exit(self):
        dialog = self.rows["reactor3"]["next_observation"]["ui"]["dialog_box"]
        self.assertEqual(scalar_dialog_action(dialog, 5000., 1650.), {"chosen_dialog_option_int": 5})
        self.assertEqual(scalar_dialog_action(dialog, 1650., 1651.49), {"chosen_dialog_option_int": 4})
        self.assertEqual(scalar_dialog_action(dialog, 1651., 1651.49), {"chosen_dialog_option_int": 9})
        self.assertEqual(scalar_dialog_action(dialog, 10000., 10050.), {"chosen_dialog_option_int": 9})
        self.assertEqual(scalar_dialog_action(dialog, 5000., 1650., activated=True), {"chosen_dialog_option_int": 9})
        renamed = deepcopy(dialog)
        renamed["dialogOptions"] = {int(key) + 20: value for key, value in dialog["dialogOptions"].items()}
        self.assertEqual(scalar_dialog_action(renamed, 5000., 1650.), {"chosen_dialog_option_int": 25})

    def test_unknown_dialog_prediction_has_no_later_calibration_label(self):
        policy = self.make_policy()
        self.assertIs(policy.model, policy.predictor.source)
        self.assertIs(policy.measurements, policy.calibration.measurements)
        for name in ("density", "temperature", "gap", "spectrum", "radiation_a1", "radiation_a2", "radiation_a3", "reactor3"):
            self.receive(policy, name)
        self.assertEqual(policy.predictor.updates, [])
        self.assertEqual(policy.calibration.labels, ())
        observation = self.rows["reactor3"]["next_observation"]
        prior_h = policy.state["h"].clone()
        selected = policy(observation)
        self.assertEqual(selected, {"chosen_dialog_option_int": 5})
        self.assertFalse(torch.equal(prior_h, policy.state["h"]))
        query = policy.predictor.predictions[0][0][0]
        self.assertEqual(query.object_handle, 63031)
        self.assertTrue(all(record.sequence <= 204 for record in query.records))
        self.assertTrue(all(record.kind == "instrument" for record in query.records))
        self.assertNotIn("resonance_frequency_hz", query.features)
        policy.observe(deepcopy(self.rows["adjust3a"]))
        self.assertEqual(policy(self.rows["adjust3a"]["next_observation"]), {"chosen_dialog_option_int": 5})
        self.assertEqual(len(policy.predictor.predictions), 1)  # Fixed requested setpoint for this dialogue.
        self.assertEqual(policy.predictor.updates, [])

    def test_returned_activation_alone_admits_support_and_controls_retention(self):
        policies = []
        for retain in (True, False):
            torch.manual_seed(31)
            policy = self.make_policy(retain)
            initial = {key: value.clone() for key, value in policy.adapters.items()}
            for name in ("density", "temperature", "gap", "spectrum", "radiation_a1", "reactor3"):
                self.receive(policy, name)
            self.assertEqual(policy.update_events, [])
            # This excerpt is the actual native feedback at index 215. The
            # source's later answer is not available to any earlier call.
            row = deepcopy(self.rows["activated"])
            policy(row["observation"])
            context_before_feedback = policy.predictor.context(policy.state).clone()
            policy.observe(row)
            self.assertEqual(len(policy.predictor.updates), 1)
            query, contexts, target = policy.predictor.updates[0]
            self.assertEqual(float(target[0]), 1650.)
            torch.testing.assert_close(contexts, context_before_feedback, rtol=0, atol=0)
            self.assertEqual(query[0].object_handle, 63031)
            self.assertEqual(policy.update_events[0]["sequence"], 215)
            for key in initial:
                torch.testing.assert_close(policy.adapters[key], initial[key] + (.25 if retain else 0), rtol=0, atol=0)
                self.assertIs(policy.state[key], policy.adapters[key])
            self.assertEqual(policy(row["next_observation"]), {"chosen_dialog_option_int": 9})
            policy.observe(deepcopy(self.rows["exit_after_activation"]))
            self.assertEqual(len(policy.predictor.updates), 1)  # Old support is not a new observation.
            policies.append(policy)
        for key in ("h", "z"):
            torch.testing.assert_close(policies[0].state[key], policies[1].state[key], rtol=0, atol=0)
        self.assertEqual(len(policies[0].decisions), len(policies[1].decisions))

    def test_real_predictor_checkpoint_runs_deployed_update_on_native_feedback(self):
        from experiments.discoveryworld.calibration_predictor import CalibrationPredictor, save_calibration_checkpoint
        prototype = self.make_policy()
        model = CalibrationPredictor(prototype.model)
        metadata = {**prototype.checkpoint, "core_config": asdict(model.core.config),
                    "digital_config": asdict(model.source.config)}
        checkpoint = Path(self.temporary.name) / "calibration.pt"
        saved = {key: value.clone() for key, value in prototype.adapters.items()}
        saved["B"] += .015
        save_calibration_checkpoint(model, metadata, checkpoint, initial_adapters=saved)
        policy = QuantitativePolicy(checkpoint, Path(self.temporary.name) / "loaded")
        self.addCleanup(policy.features.close)
        self.assertIs(policy.model.core, policy.predictor.core)
        torch.testing.assert_close(policy.adapters["B"], saved["B"], rtol=0, atol=0)
        before = policy.adapters["A"].clone()
        for name in ("density", "temperature", "gap", "spectrum", "radiation_a1", "reactor3", "activated"):
            self.receive(policy, name)
        self.assertEqual(len(policy.update_events), 1)
        self.assertTrue(torch.isfinite(policy.adapters["A"]).all())
        self.assertFalse(torch.equal(before, policy.adapters["A"]))
        self.assertFalse(policy.adapters["A"].requires_grad)
        self.assertTrue(all(parameter.grad is None for parameter in policy.model.parameters()))

    def test_fresh_instances_restore_saved_adapters_and_preserve_caller_rng(self):
        prototype = self.make_policy()
        saved = {"A": prototype.adapters["A"].clone(),
                 "B": torch.full_like(prototype.adapters["B"], .123)}
        metadata = {"source_mode": "cdm", "source": prototype.checkpoint,
                    "training": {"adapter_initialization_seed": 17}, "initial_adapters": saved}
        initialized = []
        for global_seed in (11, 928):
            torch.manual_seed(global_seed)
            caller_rng = torch.random.get_rng_state().clone()
            policy = QuantitativePolicy.__new__(QuantitativePolicy)
            policy._configure(PredictorSpy(deepcopy(prototype.model)), metadata,
                              Path(self.temporary.name) / f"saved-{global_seed}")
            self.addCleanup(policy.features.close)
            self.assertTrue(torch.equal(caller_rng, torch.random.get_rng_state()))
            for key in ("A", "B"):
                torch.testing.assert_close(policy.state[key], saved[key], rtol=0, atol=0)
                self.assertIs(policy.state[key], policy.adapters[key])
            initialized.append({key: value.clone() for key, value in policy.state.items()})
            policy.adapters["B"].add_(3.)
            policy.reset()
            self.assertTrue(torch.equal(caller_rng, torch.random.get_rng_state()))
            torch.testing.assert_close(policy.adapters["B"], saved["B"], rtol=0, atol=0)
            self.assertEqual(policy.evaluation_metadata["adapter_initialization"], "checkpoint_tensors")
            self.assertEqual(policy.evaluation_metadata["adapter_initialization_seed"], 17)
        for key in initialized[0]:
            torch.testing.assert_close(initialized[0][key], initialized[1][key], rtol=0, atol=0)

    def test_legacy_seed_fallback_is_explicit_and_does_not_consume_rng(self):
        prototype = self.make_policy()
        for seed in (None, 17):
            metadata = {"source_mode": "cdm", "source": prototype.checkpoint}
            if seed is not None:
                metadata["training"] = {"adapter_initialization_seed": seed}
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(0 if seed is None else seed)
                expected = prototype.model.initial(1, "cpu")
            torch.manual_seed(329)
            before = torch.random.get_rng_state().clone()
            policy = QuantitativePolicy.__new__(QuantitativePolicy)
            policy._configure(PredictorSpy(deepcopy(prototype.model)), metadata,
                              Path(self.temporary.name) / f"legacy-{seed}")
            self.addCleanup(policy.features.close)
            self.assertTrue(torch.equal(before, torch.random.get_rng_state()))
            for key in expected:
                torch.testing.assert_close(policy.state[key], expected[key], rtol=0, atol=0)
            self.assertEqual(policy.evaluation_metadata["adapter_initialization"], "seed_fallback")
            self.assertEqual(policy.evaluation_metadata["adapter_initialization_seed"], 0 if seed is None else seed)

    def test_later_anchor_readings_update_without_recounting_old_labels(self):
        path = Path(__file__).parent / "fixtures/discoveryworld_calibration.json.gz"
        rows = json.loads(gzip.decompress(path.read_bytes()))["transitions"]
        policy = self.make_policy()
        schema = self.rows["density"]["observation"]["actions"]
        for row in rows:
            if row["index"] > 204:
                break
            policy({**row["observation"], "actions": schema, "teleport_locations": {}})
            policy.observe(row)
        self.assertEqual([event["sequence"] for event in policy.update_events], [114, 115, 116, 161, 162, 163])
        self.assertEqual([event["support_count"] for event in policy.update_events], [1, 1, 1, 2, 2, 2])
        first = policy.predictor.updates[0][0][0].features["radiation_usv_h"][0]
        later = policy.predictor.updates[2][0][0].features["radiation_usv_h"][0]
        self.assertEqual((first.count, later.count), (1, 3))
        self.assertEqual(len(policy.calibration.labels), 2)


if __name__ == "__main__":
    unittest.main()
