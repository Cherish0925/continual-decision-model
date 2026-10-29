import torch

from cdm.decision_belief import failure_conditioned_weights


def test_failed_high_confidence_setting_changes_relation_odds():
    means = torch.tensor([[0., 8.]], dtype=torch.float64, requires_grad=True)
    std = torch.ones_like(means)
    prior = torch.tensor([[.8, .2]], dtype=torch.float64)
    interval = torch.tensor([[[-2., 2.]]], dtype=torch.float64)
    mask = torch.ones(1, 1, dtype=torch.bool)
    result = failure_conditioned_weights(means, std, prior, interval, mask)
    expected_survival = torch.special.ndtr((interval[:, 0, 0, None] - means) / std)
    expected_survival += torch.special.ndtr((means - interval[:, 0, 1, None]) / std)
    expected = prior * expected_survival
    expected /= expected.sum(-1, keepdim=True)
    torch.testing.assert_close(result['weights'], expected)
    assert result['weights'][0, 1] > .8
    result['weights'][0, 1].backward()
    assert means.grad.isfinite().all()


def test_repeated_or_split_exclusions_are_one_public_event():
    means = torch.tensor([[1., 5.]], dtype=torch.float64)
    std, prior = torch.ones_like(means), torch.tensor([[.6, .4]], dtype=torch.float64)
    union = torch.tensor([[[-2., 2.]]], dtype=torch.float64)
    pieces = torch.tensor([[[-2., 1.], [0., 2.], [-2., 1.]]], dtype=torch.float64)
    first = failure_conditioned_weights(means, std, prior, union, torch.ones(1, 1, dtype=torch.bool))
    second = failure_conditioned_weights(means, std, prior, pieces, torch.ones(1, 3, dtype=torch.bool))
    torch.testing.assert_close(first['weights'], second['weights'])
    third = failure_conditioned_weights(means, std, prior, union, torch.ones(1, 1, dtype=torch.bool))
    torch.testing.assert_close(first['weights'], third['weights'], rtol=0, atol=0)
    empty = failure_conditioned_weights(means, std, prior, union, torch.zeros(1, 1, dtype=torch.bool))
    assert torch.equal(empty['weights'], prior)


def test_zero_prior_mass_has_finite_training_gradient():
    means = torch.tensor([[0., 3., 5.]], dtype=torch.float64, requires_grad=True)
    std = torch.ones_like(means, requires_grad=True)
    weights = torch.tensor([[0., .3, .7]], dtype=torch.float64, requires_grad=True)
    result = failure_conditioned_weights(means, std, weights,
        torch.tensor([[[2., 4.]]], dtype=torch.float64), torch.tensor([[True]]))
    assert result["weights"][0, 0] == 0.
    result["weights"][0, 1].backward()
    for value in (means, std, weights):
        assert torch.isfinite(value.grad).all()
