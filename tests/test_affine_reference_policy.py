"""Frozen reference parity, support replacement and native-policy construction."""
from dataclasses import asdict
import gzip
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import torch

from cdm.core import ContinualCore, CoreConfig
from cdm.digital import DigitalConfig, DigitalPolicy
from experiments.discoveryworld.affine_reference_policy import (
    AffineReferencePolicy, AffineTaskPredictor, FrozenAffineReference, main)
from experiments.discoveryworld.features import FeatureConfig
from experiments.discoveryworld.numeric_reference import QUANTITIES, affine_fit, mixture


def configuration():
    prior = [.6] + [.05] * 8
    return {"kind": "frozen_affine_reference_v1", "quantities": list(QUANTITIES),
            "priors": {"2": prior, "3": prior}, "temperatures_hz": {"2": 1., "3": 1.},
            "global_prior_mean_hz": 3674.635333333334, "global_prior_std_hz": 1703.815729734423}


def source_policy():
    core = ContinualCore(CoreConfig(event_dim=16, record_dim=8, actions=1, width=8,
                                   hidden=8, slots=2, heads=2, action_width=8), torch.zeros(1, 1))
    return DigitalPolicy(DigitalConfig(width=8, text_dim=256, image_dim=256, outcome_dim=14), core)


class AffineReferenceTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        # Training instance 130's public support properties and working Hz.
        self.values = np.array([
            [36.23, 20.75, 45.5228125, 47.87, 23.02, 18.31, 30.54, 42.77, 36.97],
            [49.81, 31.90, 31.5365625, 42.89, 60.66, 65.23, 65.85, 17.77, 61.30],
            [16.95, 16.20, 13.4734375, 19.01, 36.38, 16.21, 36.21, 63.38, 34.69]])
        self.targets = np.array([4046.07, 5526.29, 1943.])
        self.config = configuration()
        self.reference = FrozenAffineReference(self.config)

    def queries(self):
        return [{"features": {name: [{"mean": value, "first_sequence": 0, "last_sequence": 1}]
                              for name, value in zip(QUANTITIES, row)}} for row in self.values]

    def test_matches_existing_reference_and_executes_map_not_mixture_average(self):
        for count in (2, 3):
            data = {"affine": affine_fit(self.values[:count], self.targets[:count], self.values[-1])}
            expected_mean, _, expected_weights = mixture(data, 1., np.array(self.config["priors"][str(count)]))
            state = self.reference.fit_posterior(self.values[:count], self.targets[:count])
            prediction = self.reference.predict(self.values[-1:], state)
            np.testing.assert_allclose(prediction["weights"][0], expected_weights, rtol=1e-10, atol=1e-12)
            np.testing.assert_allclose(prediction["expert_predictions_hz"][0], data["affine"][0], atol=1e-9)
            self.assertAlmostEqual(float(prediction["mixture_mean_hz"][0]), expected_mean, places=8)
            expected_map = data["affine"][0][expected_weights.argmax()]
            self.assertAlmostEqual(float(prediction["map_expert_mean_hz"][0]), expected_map, places=8)
            if count == 2:
                self.assertGreater(abs(expected_map - expected_mean), 10.)

    def test_complete_support_replaces_fit_and_discard_keeps_global_prior_and_AB(self):
        predictor = AffineTaskPredictor(source_policy(), self.reference)
        state = predictor.source.initial(1, "cpu")
        adapters = {key: state[key] for key in ("A", "B")}
        original = {key: value.clone() for key, value in adapters.items()}
        queries = self.queries()
        with patch.object(self.reference, "fit_posterior", wraps=self.reference.fit_posterior) as fit:
            retained, _ = predictor.update(queries[:2], None, self.targets[:2], adapters)
            discarded, _ = predictor.update(queries[:2], None, self.targets[:2], adapters)
        self.assertEqual(fit.call_count, 2)
        retained_prediction = predictor.predict(queries[-1:], None, retained)
        prior_prediction = predictor.predict(queries[-1:], None, adapters)
        self.assertEqual(float(prior_prediction["mean_hz"][0]), self.config["global_prior_mean_hz"])
        self.assertNotEqual(float(retained_prediction["mean_hz"][0]), float(prior_prediction["mean_hz"][0]))
        revised, _ = predictor.update(queries, None, self.targets, retained)
        repeated, _ = predictor.update(queries, None, self.targets, revised)
        self.assertEqual(len(revised["affine_reference_state"].targets), 3)
        torch.testing.assert_close(revised["affine_reference_state"].weights, repeated["affine_reference_state"].weights)
        for key in ("A", "B"):
            self.assertIs(revised[key], adapters[key])
            torch.testing.assert_close(revised[key], original[key], rtol=0, atol=0)
        self.assertFalse(any(parameter.requires_grad for parameter in predictor.parameters()))
        with self.assertRaises(ValueError):
            predictor.update(queries, None, self.targets, adapters, differentiable=True)

    def test_policy_loads_only_one_source_and_records_reference_identity(self):
        from experiments.discoveryworld.train import load_checkpoint
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = source_policy()
            source_path = root / "source.pt"
            torch.save({"model": source.state_dict(), "core_config": asdict(source.core.config),
                        "digital_config": asdict(source.config), "training_args": {"mode": "cdm"},
                        "features": asdict(FeatureConfig(mode="lexical", measurement_memory=True)), "epoch": 0}, source_path)
            config_path = root / "reference.json"
            config_path.write_text(json.dumps({**self.config, "source_checkpoint": str(source_path)}))
            with patch("experiments.discoveryworld.train.load_checkpoint", wraps=load_checkpoint) as load:
                policy = AffineReferencePolicy(config_path, root / "cache")
            self.addCleanup(policy.features.close)
            self.assertEqual(load.call_count, 1)
            self.assertIs(policy.model, policy.predictor.source)
            self.assertEqual(policy.policy_type, "affine_reference_with_native_scalar_control")
            self.assertFalse(policy.evaluation_metadata["source_core_adapter_update"])
            self.assertEqual(policy.evaluation_metadata["action_point_prediction"], "largest_weight_expert_prediction")

    def test_configuration_fit_rejects_available_nontraining_source(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "run.json").write_text(json.dumps({"split": "validation"}))
            archive = root / "public_inputs.json.gz"
            archive.write_bytes(gzip.compress(json.dumps([{"source": str(root / "trajectory.jsonl"), "samples": []}]).encode()))
            with self.assertRaisesRegex(ValueError, "Non-training source"):
                main(["--public-inputs", str(archive), "--source-checkpoint", "unused",
                      "--output", str(root / "output.json")])
            self.assertFalse((root / "output.json").exists())


if __name__ == "__main__":
    unittest.main()
