import torch

from cdm.adaptive_response import ResponseAdaptationConfig
from cdm.experimental_memory import ExperimentalMemoryConfig
from experiments.continual_chemistry.active_training import proposed_experiments
from experiments.continual_chemistry.reaction import ReactionExperimentPolicy
from test_conditional_response import model, observed


def test_action_credit_reaches_reader_without_changing_forecasts_or_training_experts():
    torch.set_num_threads(1)
    base = model()
    m = ReactionExperimentPolicy(base.base, base.continual_config, base.response.config,
        ResponseAdaptationConfig(), memory_config=ExperimentalMemoryConfig(slots=2))
    torch.nn.init.normal_(m.response.output[-1].weight, std=.2)
    obs, _ = observed()
    state = m.receive(m.initial(1, 'cpu'), obs)
    state = m.next_apparatus(state, torch.tensor([True]), torch.tensor([False]))
    # New task, with its old ledger available only through the archive.
    obs = {**obs, 'evidence': obs['evidence'][:, :1], 'evidence_mask': obs['evidence_mask'][:, :1]}
    state = m.receive(state, obs)
    baseline = proposed_experiments(m, state, obs)
    m.memory_action_credit = True
    revised = proposed_experiments(m, state, obs)
    torch.testing.assert_close(revised, baseline, rtol=0, atol=0)
    valid = obs['action_mask'][0].nonzero().flatten()
    loss = -revised.log_softmax(-1)[0, valid[0]]
    loss.backward()
    assert m.experimental_memory.new_log_prior.grad.abs() > 0
    assert m.experimental_memory.compatibility[-1].weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in m.response.parameters())
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters())
