"""Negative physical observations: union semantics and conditional moments."""
import math

import numpy as np
import torch
from scipy.stats import norm, truncnorm

from cdm.censored_observation import (censored_gaussian_observation, condition_gaussian_mixture,
                                      interval_log_probability)


def test_empty_and_masked_exclusions_preserve_prediction():
    means = torch.tensor([[0., 4000.]], dtype=torch.float64)
    std = torch.tensor([[1., 2.]], dtype=torch.float64)
    for intervals, mask in ((torch.empty(1, 0, 2), None),
                            (torch.tensor([[[-1., 1.], [float("nan"), float("nan")]]]), torch.zeros(1, 2, dtype=torch.bool))):
        result = censored_gaussian_observation(means, std, intervals, mask)
        torch.testing.assert_close(result["mean"], means)
        torch.testing.assert_close(result["variance"], std.square())
        torch.testing.assert_close(result["log_survival"], torch.zeros_like(means), atol=1e-14, rtol=0)


def test_overlaps_duplicates_order_and_touching_intervals_are_one_observation():
    means = torch.tensor([[-1., 0., 1.]], dtype=torch.float64)
    std = torch.tensor([[.5, 1., 2.]], dtype=torch.float64)
    redundant = torch.tensor([[[.5, 1.], [-2., -.5], [-1., .5], [-2., -.5], [1., 2.]]])
    single = torch.tensor([[[-2., 2.]]])
    a = censored_gaussian_observation(means, std, redundant)
    b = censored_gaussian_observation(means, std, single)
    for key in ("log_survival", "mean", "variance"):
        torch.testing.assert_close(a[key], b[key], rtol=1e-12, atol=1e-12)


def test_symmetric_failure_has_exact_two_tail_probability_and_moments():
    for cutoff in (.1, 1., 4., 20., 40.):
        result = censored_gaussian_observation(torch.zeros(1, 1), torch.ones(1, 1),
                                               torch.tensor([[[-cutoff, cutoff]]], dtype=torch.float64))
        expected_log_mass = math.log(2) + norm.logsf(cutoff)
        expected_variance = 1 + cutoff * math.exp(norm.logpdf(cutoff) - norm.logsf(cutoff))
        np.testing.assert_allclose(result["log_survival"].item(), expected_log_mass, rtol=1e-12)
        np.testing.assert_allclose(result["mean"].item(), 0., atol=1e-12)
        np.testing.assert_allclose(result["variance"].item(), expected_variance, rtol=1e-10)


def test_one_sided_extreme_tail_retains_small_positive_variance():
    cutoffs = torch.tensor([[0., 8., 25., 100.]], dtype=torch.float64)
    # Each component has a different standardized cutoff at physical zero.
    result = censored_gaussian_observation(-cutoffs, torch.ones_like(cutoffs),
                                           torch.tensor([[[-torch.inf, 0.]]]))
    expected_mean, expected_var = truncnorm.stats(cutoffs.numpy()[0], np.inf, moments="mv")
    np.testing.assert_allclose(result["mean"].numpy()[0], expected_mean - cutoffs.numpy()[0], rtol=2e-7, atol=1e-10)
    np.testing.assert_allclose(result["variance"].numpy()[0], expected_var, rtol=2e-5, atol=1e-10)
    assert (result["variance"] > 0).all()
    assert torch.isfinite(result["log_survival"]).all()


def test_disjoint_failures_agree_with_direct_sample_conditioning():
    generator = torch.Generator().manual_seed(19)
    values = torch.randn(1_000_000, generator=generator, dtype=torch.float64) * 1.4 + .3
    intervals = torch.tensor([[[-2., -1.], [-.2, .7], [1., 1.4]]], dtype=torch.float64)
    keep = ~((values[:, None] > intervals[0, :, 0]) & (values[:, None] < intervals[0, :, 1])).any(-1)
    result = censored_gaussian_observation(torch.tensor([[.3]]), torch.tensor([[1.4]]), intervals)
    np.testing.assert_allclose(result["log_survival"].exp().item(), keep.double().mean().item(), atol=.002)
    np.testing.assert_allclose(result["mean"].item(), values[keep].mean().item(), atol=.008)
    np.testing.assert_allclose(result["variance"].item(), values[keep].var().item(), atol=.02)


def test_gradients_of_physical_prediction_and_interval_likelihood():
    intervals = torch.tensor([[[-.5, .5], [.4, 1.5]]], dtype=torch.float64)
    means = torch.tensor([[0., 2.]], dtype=torch.float64, requires_grad=True)
    std = torch.tensor([[1., .5]], dtype=torch.float64, requires_grad=True)
    def call(mu, sigma):
        result = censored_gaussian_observation(mu, sigma, intervals)
        return result["log_survival"], result["mean"], result["variance"]
    assert torch.autograd.gradcheck(call, (means, std), eps=1e-6, atol=1e-5, rtol=1e-4)


def test_failed_setting_reweights_hypotheses_without_hidden_target():
    prior = {"means": torch.tensor([[100., 110.]], dtype=torch.float64),
             "std": torch.tensor([[.5, .5]], dtype=torch.float64),
             "weights": torch.tensor([[.8, .2]], dtype=torch.float64)}
    result = condition_gaussian_mixture(prior, torch.tensor([[[98., 102.]]]))
    assert result["weights"][0, 1] > .999
    # A symmetric excluded interval leaves a component's mean at its old
    # setting; moments do not assert that mean is a permissible next action.
    torch.testing.assert_close(result["means"][0, 0], prior["means"][0, 0])


def test_impossible_observation_is_explicitly_undefined():
    result = censored_gaussian_observation(torch.zeros(1, 2), torch.ones(1, 2),
                                           torch.tensor([[[-torch.inf, torch.inf]]]))
    assert (~result["possible"]).all()
    assert torch.isneginf(result["log_survival"]).all()
    assert torch.isnan(result["mean"]).all() and torch.isnan(result["variance"]).all()


def test_later_success_probability_uses_exact_intersection_with_exclusions():
    prediction = {"means": torch.tensor([[0., 2.]], dtype=torch.float64),
                  "std": torch.tensor([[1., .5]], dtype=torch.float64),
                  "weights": torch.tensor([[.4, .6]], dtype=torch.float64)}
    exclude = torch.tensor([[[-1., .5], [0., 1.]]], dtype=torch.float64)
    success = torch.tensor([[.5, 2.5]], dtype=torch.float64)
    actual = interval_log_probability(prediction, success, exclude).exp().item()
    means, scales, weights = [prediction[k].numpy()[0] for k in ("means", "std", "weights")]
    # The exclusion union is [-1,1], so the surviving success interval is [1,2.5].
    numerator = (weights * (norm.cdf(2.5, means, scales) - norm.cdf(1., means, scales))).sum()
    denominator = (weights * (norm.cdf(-1., means, scales) + norm.sf(1., means, scales))).sum()
    np.testing.assert_allclose(actual, numerator / denominator, rtol=1e-12)
    assert torch.isneginf(interval_log_probability(prediction, torch.tensor([[-.5, .5]]), exclude)).all()
    torch.testing.assert_close(interval_log_probability(prediction, torch.tensor([[-torch.inf, torch.inf]]), exclude),
                               torch.zeros(1, dtype=torch.float64), atol=1e-14, rtol=0)
