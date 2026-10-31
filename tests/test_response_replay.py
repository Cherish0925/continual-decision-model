import torch
from torch.nn import functional as F

from cdm.response_replay import ResponseReplay, ResponseReplayConfig, response_gradient, recent_support
from cdm.conditional_response import observable_log_probabilities


def test_censored_gradient_matches_observed_likelihood():
    torch.manual_seed(31)
    logits = torch.randn(2, 3, 4, requires_grad=True)
    before = torch.tensor([[3, 2, 1], [3, 2, 1]])
    after = torch.tensor([[3, 0, 1], [2, 2, 0]])
    loss = -observable_log_probabilities(logits, before).gather(-1, after[..., None]).sum()
    expected, = torch.autograd.grad(loss, logits)
    torch.testing.assert_close(response_gradient(logits, before, after), expected)


def test_padding_and_ingredient_names_do_not_change_relation_update():
    torch.manual_seed(32)
    model = ResponseReplay(config=ResponseReplayConfig(mode='relation'))
    base, support = torch.randn(2, 3, 4), torch.randn(2, 5, 4)
    q, s = torch.rand(2, 3, 8), torch.rand(2, 5, 8)
    qx, sx = F.normalize(torch.rand(2, 3, 7), dim=-1), F.normalize(torch.rand(2, 5, 7), dim=-1)
    before = torch.full((2, 5), 3); after = torch.tensor([[0, 1, 2, 3, 2], [2, 3, 3, 0, 0]])
    mask = torch.tensor([[1, 1, 0, 0, 0], [1, 1, 1, 0, 0]], dtype=torch.bool)
    result = model(base, qx, q, support, sx, s, before, after, mask)
    permutation = torch.randperm(8)
    torch.testing.assert_close(result, model(base, qx, q[..., permutation], support, sx,
                                             s[..., permutation], before, after, mask))
    support[~mask] = 100.; s[~mask] = 200.
    torch.testing.assert_close(result, model(base, qx, q, support, sx, s, before, after, mask))


def test_future_loss_trains_update_and_empty_history_is_identity():
    torch.manual_seed(33)
    model = ResponseReplay(config=ResponseReplayConfig(mode='refreshed'))
    base, support = torch.randn(2, 1, 4), torch.randn(2, 3, 4)
    x = F.normalize(torch.rand(2, 3, 6), dim=-1); qx = x[:, :1]
    q, s = torch.rand(2, 1, 8), torch.rand(2, 3, 8)
    before = torch.full((2, 3), 3); after = torch.tensor([[0, 1, 2], [1, 2, 3]])
    mask = torch.ones(2, 3, dtype=torch.bool)
    result = model(base, qx, q, support, x, s, before, after, mask)
    F.cross_entropy(result[:, 0], torch.tensor([0, 1])).backward()
    assert torch.isfinite(model.log_rates.grad).all() and model.log_rates.grad.abs().sum() > 0
    torch.testing.assert_close(base, model(base, qx, q, support, x, s, before, after, ~mask))


def test_history_capacity_is_applied_per_stream_not_to_padded_batch():
    sequence = torch.arange(80).repeat(2, 1)
    mask = torch.arange(80)[None] < torch.tensor([80, 3])[:, None]
    result = recent_support({'inputs': sequence[..., None], 'valid': mask}, 64)
    torch.testing.assert_close(result['inputs'][0, :, 0], torch.arange(16, 80))
    torch.testing.assert_close(result['inputs'][1, :3, 0], torch.arange(3))
    assert result['valid'].sum(1).tolist() == [64, 3]
