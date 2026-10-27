from copy import deepcopy
import gzip
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch

from cdm.core import ContinualCore, CoreConfig
from cdm.digital import DigitalPolicy, DigitalConfig
from experiments.discoveryworld.calibration_predictor import CalibrationPredictor
from experiments.discoveryworld.train_calibration import build_program, run_program, initialize_quantity_readout


def recorded_program():
    fixture = Path(__file__).parent / "fixtures/discoveryworld_calibration.json.gz"
    rows = json.loads(gzip.decompress(fixture.read_bytes()))["transitions"]
    episode = SimpleNamespace(transitions=rows, identity=("Reactor Lab", "Normal", 310), path=fixture)
    return episode, build_program(episode)


class CalibrationTrainingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_real_future_label_cannot_enter_its_own_query_or_support(self):
        episode, original = recorded_program()
        altered = deepcopy(episode)
        for row in altered.transitions:
            if row["index"] == 213:
                ui = row["next_observation"]["ui"]
                for key in ("lastActionMessage", "extended_action_message"):
                    ui[key] = ui.get(key, "").replace("6592", "6501")
                ui["dialog_box"]["dialogIn"] = ui["dialog_box"]["dialogIn"].replace("6592", "6501")
        changed = build_program(altered)
        query = [event for event in original["events"] if event["kind"] == "query"]
        self.assertEqual([(row["sequence"], row["support_count"]) for row in query], [(205, 2), (219, 3)])
        first, other = query[0], next(row for row in changed["events"] if row["kind"] == "query")
        self.assertEqual((first["target"], other["target"]), (6592., 6501.))
        self.assertEqual(original["queries"][first["query"]], changed["queries"][other["query"]])
        self.assertEqual([event for event in original["events"] if event["sequence"] < 205],
                         [event for event in changed["events"] if event["sequence"] < 205])
        for event in original["events"]:
            if event["kind"] == "update":
                self.assertTrue(all(index <= event["sequence"] for index in event["label_sequences"]))
                self.assertTrue(all(original["queries"][index]["available_after_sequence"] <= event["sequence"]
                                    for index in event["queries"]))

    def test_equal_compute_discard_and_actual_future_gradient(self):
        torch.manual_seed(4)
        core = ContinualCore(CoreConfig(width=8, hidden=8, heads=2, slots=2, rank=2,
                                         event_dim=3, record_dim=2, actions=1, step_rule="evidence"), torch.zeros(1, 1))
        model = CalibrationPredictor(DigitalPolicy(DigitalConfig(width=8, text_dim=8, image_dim=8), core))
        _, program = recorded_program()
        program["initial_adapters"] = {key: value for key, value in core.initial(1, "cpu").items() if key in ("A", "B")}
        program["contexts"] = {event["sequence"]: torch.randn(1, 16) for event in program["events"]}
        retained_loss, retained = run_program(model, program, differentiable=True)
        retained_loss.backward()
        self.assertGreater(float(model.query_encoder.numeric[0].weight.grad.abs().sum()), 0.)
        self.assertGreater(float(core.log_step.grad.abs()), 0.)
        with torch.no_grad():
            _, discarded = run_program(model, program, retain_updates=False)
            no_updates = {**program, "events": [event for event in program["events"] if event["kind"] == "query"]}
            _, baseline = run_program(model, no_updates)
        self.assertEqual(retained["updates_computed"], discarded["updates_computed"])
        self.assertGreater(retained["updates_computed"], 2)
        for first, second in zip(discarded["predictions"], baseline["predictions"]):
            self.assertEqual(first["mean_hz"], second["mean_hz"])
        self.assertNotEqual(retained["predictions"][0]["mean_hz"], discarded["predictions"][0]["mean_hz"])

    def test_train_only_readout_prior_retains_hidden_core_and_variance_gradient(self):
        torch.manual_seed(9)
        core = ContinualCore(CoreConfig(width=8, hidden=8, heads=2, slots=2, rank=2,
                                         event_dim=3, record_dim=2, actions=1), torch.zeros(1, 1))
        model = CalibrationPredictor(DigitalPolicy(DigitalConfig(width=8, text_dim=8, image_dim=8), core))
        _, program = recorded_program()
        hidden, recurrent = core.pred_in.weight.detach().clone(), core.gru.weight_hh_l0.detach().clone()
        prior = initialize_quantity_readout(model, [program], seed=31)
        self.assertEqual(prior["unique_training_crystals"], 4)
        self.assertAlmostEqual(prior["offset_hz"], (2171.12 + 2049.22 + 6592. + 7119.) / 4)
        self.assertTrue(torch.equal(hidden, core.pred_in.weight))
        self.assertTrue(torch.equal(recurrent, core.gru.weight_hh_l0))
        query = next(row for row in program["events"] if row["kind"] == "query")
        q = SimpleNamespace(**program["queries"][query["query"]])
        state = core.initial(1, "cpu")
        mean, logvar = model.raw_prediction([q], torch.zeros(1, 16), state)
        native_mean, native_logvar = model.likelihood.distribution(mean, logvar)
        self.assertGreater(float(native_logvar.detach()), -8.)
        self.assertLess(float(native_logvar.detach()), 20.)
        model.likelihood(mean, logvar, native_mean.detach()).mean().backward()
        self.assertGreater(abs(float(core.pred_out[-1].bias.grad[1])), .1)


if __name__ == "__main__":
    unittest.main()
