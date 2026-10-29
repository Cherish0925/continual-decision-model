import math
import pytest
import torch

from cdm.censored_observation import setting_event_observation


def event(failures, domain):
    return torch.tensor([[*failures, [-math.inf, domain[0]], [domain[1], math.inf]]], dtype=torch.float64)


def test_real_open_failed_intervals_identify_one_integer_target():
    exclusions = event([[2388., 2392.], [2392., 2396.]], [2391., 2395.])
    means = torch.tensor([[2390., 2395.]], dtype=torch.float64, requires_grad=True)
    scales = torch.tensor([[2., 3.]], dtype=torch.float64, requires_grad=True)
    result = setting_event_observation(means, scales, exclusions, torch.ones(1, 4, dtype=torch.bool),
                                       torch.tensor([[2391., 2395.]]))
    assert result['score_is_density'].item()
    torch.testing.assert_close(result['mean'], torch.full_like(means, 2392.))
    torch.testing.assert_close(result['variance'], torch.zeros_like(means))
    expected = torch.distributions.Normal(means, scales).log_prob(torch.tensor(2392.))
    torch.testing.assert_close(result['log_score'], expected)
    first = torch.autograd.grad(-result['log_score'].sum(), (means, scales), create_graph=True)
    second = torch.autograd.grad(sum(x.square().sum() for x in first), (means, scales))
    assert all(torch.isfinite(x).all() for x in (*first, *second))


def test_success_boundary_is_not_a_surviving_failed_endpoint():
    exclusions = event([[2388., 2392.], [2392., 2396.]], [2392., 2395.])
    result = setting_event_observation(torch.tensor([[2392.]]), torch.tensor([[1.]]), exclusions,
                                       torch.ones(1, 4, dtype=torch.bool), torch.tensor([[2392., 2395.]]))
    assert not result['possible'].any()
    assert not result['score_is_density'].any()


def test_point_is_zero_mass_when_positive_length_support_remains():
    exclusions = event([[0., 1.], [1., 2.]], [.5, 3.])
    result = setting_event_observation(torch.tensor([[1.]]), torch.tensor([[1.]]), exclusions,
                                       torch.ones(1, 4, dtype=torch.bool), torch.tensor([[.5, 3.]]))
    assert not result['score_is_density'].any()
    expected = torch.special.ndtr(torch.tensor(2., dtype=torch.float64)) - torch.special.ndtr(torch.tensor(1., dtype=torch.float64))
    torch.testing.assert_close(result['log_score'].exp(), expected.reshape(1, 1))


def test_multiple_residual_points_require_a_discrete_measure():
    exclusions = event([[0., 1.], [1., 2.], [2., 3.]], [.5, 2.5])
    with pytest.raises(ValueError, match='Multiple isolated'):
        setting_event_observation(torch.tensor([[1.]]), torch.tensor([[1.]]), exclusions,
                                  torch.ones(1, 5, dtype=torch.bool), torch.tensor([[.5, 2.5]]))
