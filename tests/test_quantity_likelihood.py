"""Precision and deployed-update semantics for scalar tasks in native units."""
import math
import unittest

import torch

from cdm.core import (ContinualCore, CoreConfig, GaussianQuantityLikelihood,
                      adapter_update, reading_nll)


class QuantityLikelihoodTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(19)
        torch.set_num_threads(1)
        self.core = ContinualCore(CoreConfig(event_dim=3, record_dim=2, actions=3,
                                             width=8, hidden=8, heads=2, slots=2,
                                             rank=2, sensor_noise=True, step_rule="evidence"),
                                  torch.tensor([-1., 0., 1.]))
        self.state = self.core.initial(1, "cpu")
        self.x = torch.randn(1, 3, 24)

    def test_default_prediction_and_process_likelihood_stay_unchanged(self):
        # The existing all-action path is independent of the new raw method.
        features = torch.randn(1, 16)
        x = torch.cat((features[:, None].expand(-1, 3, -1),
                       self.core.action_features()[None]), -1)
        expected = self.core.predict_actions(features, self.state["A"], self.state["B"])
        actual = self.core.predict(x, self.state["A"], self.state["B"])
        for got, old in zip(actual, expected):
            torch.testing.assert_close(got, old)
        with torch.no_grad():
            self.core.pred_out[-1].weight.zero_()
            self.core.pred_out[-1].bias.copy_(torch.tensor([3., -18.]))
        mean, logvar = self.core.predict(self.x)
        self.assertTrue(torch.equal(mean, torch.full_like(mean, 3.)))
        self.assertTrue(torch.equal(logvar, torch.full_like(logvar, -8.)))
        self.assertTrue(torch.equal(self.core.predict_raw(self.x)[1], torch.full_like(logvar, -18.)))
        precise = torch.tensor([[False, True, False]])
        noise = torch.where(precise, self.core.log_noise[1], self.core.log_noise[0])
        variance = logvar.exp() + (2 * noise).exp()
        expected_nll = .5 * (variance.log() + 1 / variance)
        torch.testing.assert_close(reading_nll(self.core, mean, logvar, mean + 1, precise),
                                   expected_nll, rtol=0, atol=0)

    def test_one_native_unit_error_keeps_its_scale_and_gradient(self):
        # A scalar near 5000 represented near 5 still needs sub-unit precision.
        with torch.no_grad():
            self.core.pred_out[-1].weight.zero_()
            self.core.pred_out[-1].bias.copy_(torch.tensor([5., 2 * math.log(.5 / 1000)]))
            self.core.log_noise.fill_(math.log(1e6))
        likelihood = GaussianQuantityLikelihood(scale=1000., logvar_bounds=(-8., 20.))
        mean, logvar = self.core.predict_raw(self.x)
        exact, off_by_one = torch.full_like(mean, 5000.), torch.full_like(mean, 5001.)
        penalty = likelihood(mean, logvar, off_by_one) - likelihood(mean, logvar, exact)
        torch.testing.assert_close(penalty, torch.full_like(penalty, 2.), rtol=1e-6, atol=1e-6)
        value, native_logvar = likelihood.distribution(mean, logvar)
        torch.testing.assert_close(value, exact, rtol=0, atol=0)
        torch.testing.assert_close(native_logvar.exp(), torch.full_like(mean, .25), rtol=1e-6, atol=1e-6)
        likelihood(mean, logvar, off_by_one).sum().backward()
        self.assertGreater(abs(float(self.core.pred_out[-1].bias.grad[1])), .1)
        self.assertIsNone(self.core.log_noise.grad)

    def test_physical_variance_bounds_and_measurement_noise(self):
        likelihood = GaussianQuantityLikelihood(scale=1000., offset=10., noise_std=.5,
                                                  logvar_bounds=(math.log(.25), math.log(4.)))
        mean = torch.tensor([[2., 3.]], dtype=torch.bfloat16)
        logvar = torch.tensor([[-100., 100.]], dtype=torch.bfloat16)
        value, variance = likelihood.distribution(mean, logvar)
        self.assertEqual(value.dtype, torch.float32)
        torch.testing.assert_close(value, torch.tensor([[2010., 3010.]]))
        torch.testing.assert_close(variance.exp(), torch.tensor([[.5, 4.25]]))

    def test_future_loss_differentiates_the_actual_instance_update(self):
        likelihood = GaussianQuantityLikelihood(scale=10., offset=-2., noise_std=.1)
        A, B = self.state["A"], self.state["B"]
        with torch.no_grad():
            self.core.pred_out[-1].bias[1] = -6.
            targets = likelihood.distribution(*self.core.predict_raw(self.x, A, B))[0] + 1.
        next_A, next_B, support_loss = adapter_update(self.core, A, B, self.x, targets,
                                                     None, None, .05, True, likelihood=likelihood)
        query = self.x[:, :1] + .05
        before = self.core.predict_raw(query, A, B)[0]
        after, logvar = self.core.predict_raw(query, next_A, next_B)
        self.assertGreater(float((after - before).detach().abs().max()), 1e-6)
        self.assertTrue(support_loss.requires_grad)
        likelihood(after, logvar, targets[:, :1]).mean().backward()
        for parameter in (self.core.pred_in.weight, self.core.pred_out[-1].weight,
                          self.core.log_step, self.core.log_kappa):
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(float(parameter.grad.abs().sum()), 0.)
        self.assertIsNone(self.core.log_noise.grad)

    def test_deployed_updates_return_no_graph_for_both_likelihoods(self):
        for likelihood in (None, GaussianQuantityLikelihood(scale=10., noise_std=.2)):
            with self.subTest(likelihood=likelihood), torch.no_grad():
                updated = adapter_update(self.core, self.state["A"], self.state["B"], self.x,
                                         torch.ones(1, 3), None, torch.tensor([[1., 1., 0.]]),
                                         .05, likelihood=likelihood)
                for value in updated:
                    self.assertFalse(value.requires_grad)
                    self.assertIsNone(value.grad_fn)
                self.assertGreater(float(updated[0].abs().sum()), 0.)


if __name__ == "__main__":
    unittest.main()
