"""Support fitting and future-loss gradients of the generic residual update."""
import unittest
from unittest.mock import patch

import torch

from cdm.online_regression import ridge_predict, ridge_update, ridge_variance


class OnlineRegressionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(13)
        torch.set_num_threads(1)
        # Actual then-available public properties/working returns, train130,
        # update 216. Order: density, temperature, radiation, gap, channels 1–5.
        # This tests known-support fitting, without claiming future-law inference.
        self.properties = torch.tensor([
            [36.23, 20.75, 45.5228125, 47.87, 23.02, 18.31, 30.54, 42.77, 36.97],
            [49.81, 31.90, 31.5365625, 42.89, 60.66, 65.23, 65.85, 17.77, 61.30],
            [16.95, 16.20, 13.4734375, 19.01, 36.38, 16.21, 36.21, 63.38, 34.69],
        ], dtype=torch.float64)
        self.targets = torch.tensor([[4046.07, 5526.29, 1943.]], dtype=torch.float64)
        self.features = torch.cat((torch.ones(3, 1, dtype=torch.float64), self.properties / 50), -1)[None]
        self.prior = torch.full_like(self.targets, 3674.635333333334)

    def test_distinct_real_known_values_are_fitted_in_double_precision(self):
        state = ridge_update(self.features, self.targets, self.prior, 1e-8)
        prediction = ridge_predict(state, self.features, self.prior)
        self.assertEqual(state.cholesky.dtype, torch.float64)
        self.assertEqual(state.coefficients.dtype, torch.float64)
        torch.testing.assert_close(prediction, self.targets, rtol=0, atol=1e-3)
        self.assertGreater(float(prediction.max() - prediction.min()), 3500.)

    def test_replacement_does_not_recount_old_support(self):
        old = ridge_update(self.features[:, :2], self.targets[:, :2], self.prior[:, :2], .01)
        current = ridge_update(self.features, self.targets, self.prior, .01)
        repeated = ridge_update(self.features, self.targets, self.prior, .01)
        self.assertEqual(old.features.shape[1], 2)
        self.assertEqual(current.features.shape[1], 3)
        torch.testing.assert_close(current.coefficients, repeated.coefficients, rtol=0, atol=0)
        changed = self.targets.clone()
        changed[0, 0] += 10
        revised = ridge_update(self.features, changed, self.prior, .01)
        self.assertEqual(revised.features.shape[1], 3)
        self.assertFalse(torch.equal(current.coefficients, revised.coefficients))

    def test_later_query_loss_reaches_support_features_and_positive_ridge(self):
        encoder = torch.nn.Sequential(torch.nn.Linear(9, 6, dtype=torch.float64), torch.nn.Tanh())
        features = encoder(self.properties / 50).unsqueeze(0)
        features.retain_grad()
        log_ridge = torch.nn.Parameter(torch.tensor(-2., dtype=torch.float64))
        prior = torch.nn.Parameter(torch.tensor(3674., dtype=torch.float64))
        state = ridge_update(features[:, :2], self.targets[:, :2], prior.expand(1, 2), log_ridge.exp())
        future = ridge_predict(state, features[:, 2:], prior.expand(1, 1))
        loss = ((future - self.targets[:, 2:]) / 1000).square().mean()
        loss.backward()
        for gradient in (encoder[0].weight.grad, log_ridge.grad, prior.grad,
                         features.grad[:, :2], features.grad[:, 2:]):
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertGreater(float(gradient.abs().sum()), 1e-10)
        deployed = state.detach()
        self.assertFalse(deployed.coefficients.requires_grad)
        self.assertFalse(deployed.features.requires_grad)
        self.assertFalse(deployed.ridge.requires_grad)

    def test_empty_and_masked_support_preserve_prior_and_batch_independence(self):
        query = torch.tensor([[[1., 2.]], [[3., 4.]]])
        prior = torch.tensor([[1200.25], [5600.75]])
        empty = ridge_update(torch.empty(2, 0, 2), torch.empty(2, 0), torch.empty(2, 0), .1)
        torch.testing.assert_close(ridge_predict(empty, query, prior), prior, rtol=0, atol=0)
        torch.testing.assert_close(ridge_variance(empty, query), query.square().sum(-1))
        features = torch.tensor([[[1., 0.], [float("nan"), float("nan")]],
                                 [[float("nan"), float("nan")], [float("nan"), float("nan")]]])
        targets = torch.tensor([[1201., float("nan")], [float("nan"), float("nan")]])
        present = torch.tensor([[True, False], [False, False]])
        state = ridge_update(features, targets, torch.zeros_like(targets), torch.tensor([.1, .2]), present=present)
        alone = ridge_update(features[:1, :1], targets[:1, :1], torch.zeros(1, 1), .1)
        predicted = ridge_predict(state, query, prior)
        torch.testing.assert_close(predicted[:1], ridge_predict(alone, query[:1], prior[:1]))
        torch.testing.assert_close(predicted[1:], prior[1:], rtol=0, atol=0)
        torch.testing.assert_close(ridge_variance(state, query)[1:], query[1:].square().sum(-1))

    def test_conditional_variance_has_explicit_feature_prior_units(self):
        state = ridge_update(torch.tensor([[[1., 0.]]]), torch.tensor([[10.]]), torch.tensor([[0.]]), .25)
        variance = ridge_variance(state, torch.tensor([[[1., 0.], [0., 1.]]]))
        torch.testing.assert_close(variance, torch.tensor([[.2, 1.]]))

    def test_computed_discarded_control_performs_the_same_solve(self):
        empty = ridge_update(self.features[:, :0], self.targets[:, :0], self.prior[:, :0], .01)
        with patch("torch.linalg.cholesky", wraps=torch.linalg.cholesky) as solve:
            retained = ridge_update(self.features, self.targets, self.prior, .01)
            discarded_candidate = ridge_update(self.features, self.targets, self.prior, .01)
        self.assertEqual(solve.call_count, 2)
        torch.testing.assert_close(retained.coefficients, discarded_candidate.coefficients, rtol=0, atol=0)
        torch.testing.assert_close(ridge_predict(empty, self.features, self.prior), self.prior, rtol=0, atol=0)
        self.assertFalse(torch.equal(ridge_predict(retained, self.features, self.prior), self.prior))


if __name__ == "__main__":
    unittest.main()
