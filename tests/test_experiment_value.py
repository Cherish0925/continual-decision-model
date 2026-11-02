import torch

from cdm.experiment_value import ExperimentValue


def inputs():
    torch.manual_seed(7)
    return (torch.randn(2, 7, 30), torch.tensor([[1, 1, 1, 1, 1, 0, 0]] * 2).bool(),
            torch.randn(2, 7, 4, 4).log_softmax(-1), torch.randn(2, 4).softmax(-1),
            torch.randn(2, 7, 8))


def test_initial_adapter_preserves_actions_and_receives_learning_signal():
    model = ExperimentValue(8)
    args = inputs()
    out = model(*args)
    assert torch.equal(out, torch.zeros_like(out))
    out[:, 0].sum().backward()
    assert model.output[-1].weight.grad.abs().sum() > 0


def test_action_and_hypothesis_order_do_not_change_the_decision():
    model = ExperimentValue(8)
    args = inputs()
    torch.nn.init.normal_(model.output[-1].weight)
    q, mask, p, w, features = args
    h = torch.tensor([3, 1, 0, 2])
    a = torch.tensor([6, 1, 4, 3, 0, 2, 5])
    expected = model(*args)
    torch.testing.assert_close(model(q, mask, p[:, :, h], w[:, h], features), expected)
    torch.testing.assert_close(model(q[:, a], mask[:, a], p[:, a], w, features[:, a]), expected[:, a])


def test_joint_hypotheses_retain_information_lost_by_action_marginals():
    model = ExperimentValue(8)
    q, mask, p, w, features = inputs()
    torch.nn.init.normal_(model.output[-1].weight)
    q, mask, features = q[:1], mask[:1], features[:1]
    w = torch.full((1, 2), .5)
    p = torch.randn(1, 7, 2, 4).log_softmax(-1)
    switched = p.clone()
    switched[:, 0] = switched[:, 0, [1, 0]]
    torch.testing.assert_close(p.exp().mean(2), switched.exp().mean(2))
    assert not torch.allclose(model(q, mask, p, w, features), model(q, mask, switched, w, features))
