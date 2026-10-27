import unittest

import torch

from cdm.core import ContinualCore, CoreConfig
from cdm.digital import DigitalConfig, DigitalPolicy
from experiments.discoveryworld.calibration import FEATURE_QUANTITIES
from experiments.discoveryworld.calibration_mixture import AffineMixture
from experiments.discoveryworld.mixture_policy import MixtureTaskPredictor


def query(values):
    # Complete named observations with a fixed public chronology.
    return {"features": {name: [{"mean": float(value), "first_sequence": 1,
                                "last_sequence": 2}]
                         for name, value in zip(FEATURE_QUANTITIES, values)}}


class MixturePolicyTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(3)
        core = ContinualCore(CoreConfig(width=8, hidden=8, heads=2, slots=2,
                                        rank=2, event_dim=3, record_dim=2, actions=1),
                             torch.zeros(1, 1))
        source = DigitalPolicy(DigitalConfig(width=8, text_dim=8, image_dim=8), core)
        mixture = AffineMixture({"x_mean": [0.] * 9, "x_scale": [1.] * 9,
                                "y_mean_hz": 3000., "y_scale_hz": 1500.})
        self.predictor = MixtureTaskPredictor(source, mixture)
        self.adapters = {k: v for k, v in source.initial(1, "cpu").items() if k in ("A", "B")}

    def test_support_update_is_instance_owned_and_improves_known_settings(self):
        qs = [query([x] * 9) for x in (-1., 1.)]
        context = torch.randn(2, 16)
        initial = self.predictor.predict(qs, context, self.adapters)["mean_hz"]
        updated, _ = self.predictor.update(qs, context, [2000., 4000.], self.adapters)
        learned = self.predictor.predict(qs, context, updated)["mean_hz"]
        self.assertLess(float((learned - torch.tensor([2000., 4000.])).abs().max().detach()), .01)
        self.assertTrue(torch.equal(self.predictor.predict(qs, context, self.adapters)["mean_hz"], initial))
        for name in ("A", "B"):
            self.assertTrue(torch.equal(updated[name], self.adapters[name]))
        changed_context = self.predictor.predict(qs, context * 100., updated)["mean_hz"]
        self.assertTrue(torch.equal(learned, changed_context))

    def test_native_setting_uses_one_supported_hypothesis_not_mixture_mean(self):
        mixture = self.predictor.mixture
        with torch.no_grad():
            mixture.prior_mean[:, 0] = torch.arange(9, dtype=torch.float64)
            mixture.mixture_logits[0] = 1.
        result = self.predictor.predict([query([1.] * 9)], torch.zeros(1, 16), self.adapters)
        distribution = self.predictor.last_prediction
        self.assertTrue(torch.equal(result["mean_hz"], distribution["expert_mean_hz"][:, 0]))
        self.assertGreater(float((result["mean_hz"] - distribution["mixture_mean_hz"]).abs().detach()), 100.)


if __name__ == "__main__":
    unittest.main()
