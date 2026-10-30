import torch
from torch.nn import functional as F

from cdm.adaptive_response import AdaptiveResponse, ResponseAdaptationConfig
from cdm.conditional_response import observable_log_probabilities


def test_censored_update_matches_likelihood_gradient_and_trains_future_use():
    torch.manual_seed(11)
    model = AdaptiveResponse(5, 4, ResponseAdaptationConfig(memory_size=4))
    state = model.initial(3, 'cpu')
    x = model.features(torch.randn(3, 5))
    base = torch.randn(3, 4)
    before, after = torch.tensor([3, 2, 1]), torch.tensor([1, 2, 1])
    state = model.record(state, x, base, before, after)
    weight = state['response_fast'].clone().requires_grad_()
    logits = base + torch.bmm(x[:, None], weight).squeeze(1)
    loss = F.nll_loss(observable_log_probabilities(logits, before), after, reduction='sum')
    expected = torch.autograd.grad(loss, weight)[0]
    revised = model.update(state, 'fast', torch.ones(3, dtype=torch.bool))
    actual = (state['response_fast'] - revised['response_fast']) / model.fast_rate.exp()
    torch.testing.assert_close(actual, expected)
    later = model.predict(revised, base[:, None], x[:, None])[:, 0]
    future_loss = F.nll_loss(observable_log_probabilities(later, before), after)
    future_loss.backward()
    assert model.fast_rate.grad.abs() > 0


def test_discarded_update_preserves_prediction_at_the_same_computation():
    model = AdaptiveResponse(2, 4)
    state = model.initial(1, 'cpu')
    x = model.features(torch.tensor([[1., 0.]]))
    base = torch.zeros(1, 4)
    state = model.record(state, x, base, torch.tensor([3]), torch.tensor([0]))
    kept = model.update(state, 'fast', torch.tensor([True]))
    discarded = model.update(state, 'fast', torch.tensor([True]), retain=False)
    assert kept['response_fast'].abs().sum() > 0
    assert torch.equal(discarded['response_fast'], state['response_fast'])
    old = model.predict(state, base[:, None], x[:, None]).softmax(-1)[0, 0, 0]
    new = model.predict(kept, base[:, None], x[:, None]).softmax(-1)[0, 0, 0]
    assert new > old
