from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import torch

from experiments.discoveryworld.calibration_mixture import AffineMixture, save_mixture_checkpoint, load_mixture_checkpoint


class AffineMixtureTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.normalization = {"x_mean": [0.] * 9, "x_scale": [1.] * 9, "y_mean_hz": 0., "y_scale_hz": 10.}
        self.model = AffineMixture(self.normalization, noise_floor_hz=.01)
        with torch.no_grad():
            self.model.log_noise_extra_hz.fill_(-8.)
        self.x = torch.tensor([[0.] * 9, [1.] * 9, [2.] + [4.] * 8], dtype=torch.float64)
        self.y = torch.tensor([10., 12., 14.], dtype=torch.float64)

    def test_two_points_retain_hypotheses_then_third_observation_separates(self):
        before = {key: value.clone() for key, value in self.model.state_dict().items()}
        two = self.model.fit_posterior(self.x[:2], self.y[:2])
        fit = self.model.predict(self.x[:2], two)
        self.assertLess(float((fit["expert_mean_hz"] - self.y[:2, None]).detach().abs().max()), .001)
        torch.testing.assert_close(fit["expert_posterior"], torch.full((2, 9), 1 / 9, dtype=torch.float64))
        third = self.model.fit_posterior(self.x, self.y)
        pred = self.model.predict(torch.tensor([[3.] + [9.] * 8]), third)
        self.assertEqual(int(pred["map_expert_index"]), 0)
        self.assertGreater(float(pred["expert_posterior"][0, 0].detach()), .999)
        self.assertAlmostEqual(float(pred["map_expert_mean_hz"].detach()), 16., places=3)
        for key, value in self.model.state_dict().items():
            self.assertTrue(torch.equal(value, before[key]))

    def test_native_noise_covariance_gradients_and_checkpoint(self):
        self.model = AffineMixture(self.normalization, noise_floor_hz=2.)
        posterior = self.model.fit_posterior(self.x[:2], self.y[:2])
        nll = self.model.predictive_nll(self.x[2:], self.y[2:], posterior).mean() + self.model.prior_penalty(examples=10)
        nll.backward()
        for parameter in self.model.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(float(parameter.grad.abs().sum()), 0.)
        empty = self.model.fit_posterior(torch.empty(0, 9), torch.empty(0))
        query = torch.zeros(1, 9)
        prediction = self.model.predict(query, empty)
        # Intercept prior variance is 10^2 Hz^2; native observation noise is
        # floor^2 + extra^2 = 8 Hz^2, never added in standardized units.
        torch.testing.assert_close(prediction["expert_std_hz"].square(), torch.full((1, 9), 108., dtype=torch.float64))
        with TemporaryDirectory() as folder:
            path = Path(folder) / "model.pt"
            save_mixture_checkpoint(self.model, path, source_checkpoint="source.pt", source_sha256="abc")
            loaded, metadata = load_mixture_checkpoint(path)
            self.assertEqual(metadata["source_checkpoint"], "source.pt")
            for key, value in self.model.predict(query, posterior).items():
                self.assertTrue(torch.equal(value, loaded.predict(query, loaded.fit_posterior(self.x[:2], self.y[:2]))[key]))

    def test_shared_five_parameters_match_initial_prior_and_keep_uniform_weights(self):
        typed = AffineMixture(self.normalization)
        shared = AffineMixture(self.normalization, prior_mode="shared")
        self.assertEqual(sum(p.numel() for p in shared.parameters()), 5)
        self.assertEqual(sum(p.numel() for p in typed.parameters()), 54)
        typed_posterior = typed.fit_posterior(self.x[:2], self.y[:2])
        shared_posterior = shared.fit_posterior(self.x[:2], self.y[:2])
        for key, value in typed.predict(self.x, typed_posterior).items():
            self.assertTrue(torch.equal(value, shared.predict(self.x, shared_posterior)[key]))
        optimizer = torch.optim.Adam(shared.parameters(), lr=.001)
        shared.predictive_nll(self.x[2:], self.y[2:], shared_posterior).mean().backward()
        for parameter in shared.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertGreater(float(parameter.grad.abs().sum()), 0.)
        optimizer.step()
        self.assertTrue(torch.equal(shared.mixture_logits, torch.zeros_like(shared.mixture_logits)))
        with TemporaryDirectory() as folder:
            path = Path(folder) / "shared.pt"
            save_mixture_checkpoint(shared, path, source_checkpoint="source.pt", source_sha256="abc")
            loaded, saved = load_mixture_checkpoint(path)
            self.assertEqual(loaded.prior_mode, "shared")
            self.assertEqual(sum(p.numel() for p in loaded.parameters()), 5)
            for key, value in shared.predict(self.x, shared.fit_posterior(self.x[:2], self.y[:2])).items():
                self.assertTrue(torch.equal(value, loaded.predict(self.x, loaded.fit_posterior(self.x[:2], self.y[:2]))[key]))
            save_mixture_checkpoint(typed, path, source_checkpoint="source.pt", source_sha256="abc")
            saved = torch.load(path, weights_only=True)
            saved.pop("prior_mode")
            torch.save(saved, path)
            old, _ = load_mixture_checkpoint(path)
            self.assertEqual(old.prior_mode, "typed")


if __name__ == "__main__":
    unittest.main()
