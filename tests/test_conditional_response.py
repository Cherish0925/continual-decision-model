import tempfile
from pathlib import Path

import torch
from torch.nn import functional as F

from cdm.active_experiment import ActiveExperimentConfig, ActiveExperimentPolicy
from cdm.conditional_response import ConditionalResponseConfig, observable_log_probabilities
from cdm.continual_experiment import ContinualExperimentConfig
from cdm.core import ContinualCore, CoreConfig
from experiments.continual_chemistry.observations import observe
from experiments.continual_chemistry.reaction import (ReactionExperimentPolicy,
    initialize_response_prior, load_reaction_checkpoint, primitive_observation_mask,
    permute_observation_components)


def model():
    torch.manual_seed(901)
    core = ContinualCore(CoreConfig(event_dim=3, record_dim=3, actions=2, width=24,
                                   slots=2, heads=2, hidden=24, action_width=16), torch.tensor([0., 1.]))
    base = ActiveExperimentPolicy(ActiveExperimentConfig(36, 20, 30, outcome_dim=4, action_width=16), core)
    return ReactionExperimentPolicy(base, ContinualExperimentConfig(memory_size=4, adapter_width=8),
                                    ConditionalResponseConfig(width=12, history_capacity=8))


def observed():
    return observe({'components': ['A', 'B', 'C'], 'rust_level': 2, 'history': [
        {'component_counts': [1, 0, 0], 'rust_before': 3, 'rust_after': 3, 'native_actions': 12},
        {'component_counts': [0, 1, 1], 'rust_before': 3, 'rust_after': 2, 'native_actions': 13}]})


def test_censored_likelihood_and_gradient():
    p = torch.tensor([.1, .2, .3, .4], dtype=torch.float64)
    logits = p.log().requires_grad_()
    out = observable_log_probabilities(logits, torch.tensor(1))
    torch.testing.assert_close(out.exp(), torch.tensor([.1, .9, 0., 0.], dtype=torch.float64))
    loss = -out[1]
    loss.backward()
    assert torch.isfinite(logits.grad).all()
    assert logits.grad[0] > 0 and (logits.grad[1:] < 0).all()


def test_frequency_initialization_and_dose_invariance():
    m = model()
    records = [{'component_counts': [1, 0, 0], 'rust_before': 2, 'rust_after': after}
               for after in [2, 2, 2, 1, 0]]
    initialize_response_prior(m, records)
    obs, choices = observed()
    state = m.receive(m.initial(1, 'cpu'), obs)
    out = m.readout(state, obs['candidates'], obs['action_mask'])
    # Prior counts are [2,2,4] after unit pseudocounts on valid observed classes.
    torch.testing.assert_close(out['outcomes'][0, 0].exp(), torch.tensor([.25, .25, .5, 0.]))
    # The *same* internal/history state sees only candidate proportions here.
    first, scaled = choices.index((1, 1, 0)), choices.index((2, 2, 0))
    torch.testing.assert_close(out['outcomes'][:, first], out['outcomes'][:, scaled])


def test_history_changes_response_and_core_has_a_learning_path():
    m = model()
    obs, choices = observed()
    candidate = obs['candidates'][:, choices.index((1, 1, 0)):choices.index((1, 1, 0)) + 1]
    optimizer = torch.optim.Adam(m.parameters(), lr=.02)
    # Output starts at the calibrated prior: first learn its connection weights.
    state = m.receive(m.initial(1, 'cpu'), obs)
    out = m.readout(state, candidate, torch.ones(1, 1, dtype=torch.bool))
    (-out['outcomes'][0, 0, 1]).backward(); optimizer.step(); optimizer.zero_grad()
    state = m.receive(m.initial(1, 'cpu'), obs)
    out = m.readout(state, candidate, torch.ones(1, 1, dtype=torch.bool))
    (-out['outcomes'][0, 0, 1]).backward()
    for module in [m.core.gru, m.core.block, m.response.component]:
        norm = sum(p.grad.square().sum() for p in module.parameters() if p.grad is not None)
        assert torch.isfinite(norm) and norm > 0
    with torch.no_grad():
        changed = {**state, 'reaction_history': state['reaction_history'].clone()}
        # Change only the correspondence between mixtures and their public
        # responses; the continuing state and outcome histogram stay fixed.
        changed['reaction_history'][:, :2, :8] = changed['reaction_history'][:, :2, :8].flip(1)
        other = m.readout(changed, candidate, torch.ones(1, 1, dtype=torch.bool))
        assert (out['latent_response_logits'] - other['latent_response_logits']).abs().max() > 1e-7
        m.core_response = False
        no_core = m.readout(state, candidate, torch.ones(1, 1, dtype=torch.bool))
        assert (out['latent_response_logits'] - no_core['latent_response_logits']).abs().max() > 1e-7


def test_checkpoint_and_received_history_remain_consistent():
    m = model(); obs, _ = observed()
    state = m.receive(m.initial(1, 'cpu'), obs)
    assert state['reaction_mask'].sum() == 2
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / 'model.pt'
        torch.save({'kind': 'chemistry_reaction_v1', **m.configuration(), 'model': m.state_dict()}, path)
        restored, _ = load_reaction_checkpoint(path)
        a = m.readout(state, obs['candidates'], obs['action_mask'])
        b = restored.readout(state, obs['candidates'], obs['action_mask'])
        torch.testing.assert_close(a['outcomes'], b['outcomes'])
    reset = m.reset_rows(state, torch.ones(1, dtype=torch.bool))
    assert not reset['reaction_mask'].any()


def test_ingredient_renaming_and_primitive_public_history():
    m = model(); obs, choices = observed()
    # Make the learned residual nonconstant to exercise the representation,
    # rather than accidentally testing only a constant initialization.
    with torch.no_grad():
        m.response.output[-1].weight.normal_(std=.03)
    state = m.receive(m.initial(1, 'cpu'), obs)
    base = m.readout(state, obs['candidates'], obs['action_mask'])
    permutation = torch.tensor([2, 0, 1])
    renamed = permute_observation_components(obs, permutation)
    renamed_state = {**state, 'reaction_history': state['reaction_history'].clone()}
    renamed_state['reaction_history'][:, :2] = renamed['evidence'][:, 1:]
    # Hold the persistent core state fixed: this checks the direct relational
    # path, not an unproven invariance of the inherited recurrent network.
    changed = m.readout(renamed_state, renamed['candidates'], renamed['action_mask'])
    torch.testing.assert_close(base['outcomes'], changed['outcomes'])
    masked = primitive_observation_mask(obs)
    assert not masked['action_mask'][0, choices.index((2, 2, 0))]
    assert not masked['action_mask'][0, choices.index((0, 1, 1))]
    assert masked['action_mask'][0, choices.index((1, 1, 0))]
