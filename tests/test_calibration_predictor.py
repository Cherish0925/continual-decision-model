from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import torch

from cdm.core import ContinualCore, CoreConfig
from cdm.digital import DigitalPolicy, DigitalConfig
from experiments.discoveryworld.calibration_predictor import (
    CalibrationPredictor, QueryEncoder, query_rows, save_calibration_checkpoint, load_calibration_checkpoint)


def query(handle=1, value=12.):
    summary = dict(conditions={"inventory": [{"uuid": handle, "name": "quantum crystal"}],
                               "accessible": [], "nearby": {}, "agent_location": {}},
                   count=32, minimum=value-.1, maximum=value+.1, mean=value,
                   latest=value, last_change=.02, first_sequence=4, last_sequence=35)
    return SimpleNamespace(object_handle=handle, features={"radiation_usv_h": [summary]})


class CalibrationPredictorTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(3)
        core = ContinualCore(CoreConfig(width=8, hidden=8, heads=2, slots=2, rank=2,
                                         event_dim=3, record_dim=2, actions=1, step_rule="evidence"),
                              torch.zeros(1, 1))
        self.model = CalibrationPredictor(DigitalPolicy(DigitalConfig(width=8, text_dim=8, image_dim=8), core))
        self.context = torch.randn(1, 16)
        self.adapters = {key: value for key, value in core.initial(1, "cpu").items() if key in ("A", "B")}

    def test_handles_only_bind_and_update_has_future_gradient(self):
        for first, second in zip(query_rows(query(1)), query_rows(query(99991))):
            self.assertTrue(torch.equal(first, second))
        frozen = self.model.core.gru.weight_hh_l0.detach().clone()
        before = self.model.predict([query()], self.context, self.adapters)["mean_hz"]
        updated, _ = self.model.update([query()], self.context.expand(1, -1), [5000.], self.adapters,
                                       differentiable=True)
        mean, logvar = self.model.raw_prediction([query(value=15.)], self.context, updated)
        self.model.likelihood(mean, logvar, torch.tensor([[5100.]])).mean().backward()
        self.assertGreater(float((self.model.predict([query()], self.context, updated)["mean_hz"] - before).abs()), 0.)
        for parameter in (self.model.query_encoder.numeric[0].weight, self.model.core.pred_in.weight,
                          self.model.core.log_step, self.model.core.log_kappa):
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(float(parameter.grad.abs().sum()), 0.)
        self.assertIsNone(self.model.core.gru.weight_hh_l0.grad)
        self.assertTrue(torch.equal(frozen, self.model.core.gru.weight_hh_l0))

    def test_checkpoint_retains_source_and_exact_prediction(self):
        metadata = {"core_config": asdict(self.model.core.config),
                    "digital_config": asdict(self.model.source.config),
                    "features": {"mode": "lexical"}, "training_args": {"mode": "cdm"}}
        with TemporaryDirectory() as folder:
            path = Path(folder) / "model.pt"
            save_calibration_checkpoint(self.model, metadata, path, initial_adapters=self.adapters)
            loaded, saved = load_calibration_checkpoint(path)
            self.assertEqual(saved["source"]["features"], metadata["features"])
            self.assertIs(loaded.core, loaded.source.core)
            for key, value in self.adapters.items():
                self.assertTrue(torch.equal(value, saved["initial_adapters"][key]))
            for key, value in self.model.predict([query()], self.context, self.adapters).items():
                self.assertTrue(torch.equal(value, loaded.predict([query()], self.context, self.adapters)[key]))

    def test_object_only_masks_numeric_context_without_resetting_source_state(self):
        state = self.model.source.initial(1, "cpu")
        state["h"].fill_(3.)
        state["z"].add_(7.)
        untouched = {key: value.clone() for key, value in state.items()}
        current_inputs = self.model.inputs([query()], self.context)
        self.model.context_mode = "object_only"
        first = self.model.predict([query()], self.context, self.adapters)
        second = self.model.predict([query()], self.context * 13 + 5, self.adapters)
        for key in first:
            self.assertTrue(torch.equal(first[key], second[key]))
        object_inputs = self.model.inputs([query()], self.model.context(state))
        self.assertEqual(float(object_inputs[..., :16].detach().abs().sum()), 0.)
        self.assertTrue(torch.equal(object_inputs[..., 16:], current_inputs[..., 16:]))
        for key, value in state.items():
            self.assertTrue(torch.equal(value, untouched[key]))
        metadata = {"core_config": asdict(self.model.core.config),
                    "digital_config": asdict(self.model.source.config),
                    "features": {"mode": "lexical"}, "training_args": {"mode": "cdm"}}
        with TemporaryDirectory() as folder:
            path = Path(folder) / "model.pt"
            save_calibration_checkpoint(self.model, metadata, path)
            loaded, checkpoint = load_calibration_checkpoint(path)
            self.assertEqual(loaded.context_mode, "object_only")
            self.assertTrue(torch.equal(first["mean_hz"], loaded.predict([query()], self.context, self.adapters)["mean_hz"]))
            checkpoint.pop("context_mode")
            torch.save(checkpoint, path)
            legacy, _ = load_calibration_checkpoint(path)
            self.assertEqual(legacy.context_mode, "current")

    def test_typed_interaction_preserves_parameters_and_checkpoint_semantics(self):
        typed = CalibrationPredictor(self.model.source, query_encoding="typed_interaction")
        typed.query_encoder.load_state_dict(self.model.query_encoder.state_dict(), strict=True)
        self.assertEqual({k: v.shape for k, v in typed.state_dict().items()},
                         {k: v.shape for k, v in self.model.state_dict().items()})
        for key, value in typed.state_dict().items():
            self.assertTrue(torch.equal(value, self.model.state_dict()[key]))
        small, large = query(value=10.), query(value=20.)
        response = typed.query_encoder([large]) - typed.query_encoder([small])
        small.features = {"density_g_cm3": small.features["radiation_usv_h"]}
        large.features = {"density_g_cm3": large.features["radiation_usv_h"]}
        another_type = typed.query_encoder([large]) - typed.query_encoder([small])
        self.assertGreater(float((response - another_type).detach().abs().sum()), 1e-4)
        self.assertFalse(torch.equal(typed.query_encoder([small]), self.model.query_encoder([small])))
        typed.query_encoder([small]).square().sum().backward()
        self.assertIsNotNone(typed.query_encoder.quantity.weight.grad)
        metadata = {"core_config": asdict(typed.core.config), "digital_config": asdict(typed.source.config),
                    "features": {"mode": "lexical"}, "training_args": {"mode": "cdm"}}
        with TemporaryDirectory() as folder:
            path = Path(folder) / "typed.pt"
            save_calibration_checkpoint(typed, metadata, path)
            loaded, checkpoint = load_calibration_checkpoint(path)
            self.assertEqual(loaded.query_encoding, "typed_interaction")
            self.assertTrue(torch.equal(typed.predict([small], self.context, self.adapters)["mean_hz"],
                                        loaded.predict([small], self.context, self.adapters)["mean_hz"]))
            checkpoint.pop("query_encoding")
            torch.save(checkpoint, path)
            legacy, _ = load_calibration_checkpoint(path)
            self.assertEqual(legacy.query_encoding, "late_binding")
            self.assertTrue(torch.equal(legacy.query_encoder([small]), self.model.query_encoder([small])))
        with self.assertRaisesRegex(ValueError, "query_encoding"):
            QueryEncoder(8, "unknown")


if __name__ == "__main__":
    unittest.main()
