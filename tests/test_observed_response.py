import torch
from cdm.observed_response import condition_on_observed_responses


def test_observations_constrain_only_matching_queries_and_respect_censoring():
    inputs = torch.tensor([[[1., 0.], [0., 1.]]])
    queries = torch.tensor([[[1., 0.], [0., 1.], [.5, .5]]])
    logits = torch.tensor([[[1., 2., 3., 4.]] * 3])
    out = condition_on_observed_responses(logits, queries, inputs,
        torch.tensor([[3, 2]]), torch.tensor([[0, 2]]), torch.ones(1, 2, dtype=torch.bool))
    probability = out.exp()
    assert probability[0, 0, 0] > .999
    assert probability[0, 1, :2].sum() < 1e-6
    torch.testing.assert_close(out[0, 2], logits.log_softmax(-1)[0, 2])
    torch.testing.assert_close(probability.sum(-1), torch.ones(1, 3))
    assert out.isfinite().all()
