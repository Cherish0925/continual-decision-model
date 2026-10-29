import copy
import itertools

import torch

from experiments.continual_chemistry.foresight import (
    choose_experiment, forecast_state, two_experiment_values,
)
from experiments.continual_chemistry.observations import observe
from experiments.continual_chemistry.reaction import primitive_observation_mask
from test_conditional_response import model


def test_two_experiment_rate_matches_enumerated_conditional_plans():
    first_p = torch.tensor([.1, .2], dtype=torch.float64)
    first_c = torch.ones(2, dtype=torch.float64)
    parents = torch.tensor([0, 0, 1, 1])
    branch_p = torch.tensor([.3, .6, .4, .4], dtype=torch.float64)
    next_p = torch.tensor([[.9, .1], [.8, .2], [.21, .05], [.1, .15]], dtype=torch.float64)
    next_c = torch.tensor([[1., 2.], [1., 3.], [1., 2.], [2., 1.]], dtype=torch.float64)
    mask = torch.ones_like(next_p, dtype=torch.bool)
    finish, cost, followup = two_experiment_values(first_p, first_c, parents, branch_p,
                                                  next_p, next_c, mask)
    expected = []
    for first in range(2):
        branches = (parents == first).nonzero().flatten().tolist()
        for decisions in itertools.product([-1, 0, 1], repeat=len(branches)):
            p, c = float(first_p[first]), float(first_c[first])
            for branch, action in zip(branches, decisions):
                if action >= 0:
                    p += float(branch_p[branch] * next_p[branch, action])
                    c += float(branch_p[branch] * next_c[branch, action])
            expected.append((p / c, first))
    optimum, selected = max(expected)
    torch.testing.assert_close((finish / cost).max(), torch.tensor(optimum, dtype=torch.float64))
    assert int((first_p / first_c).argmax()) == 1
    assert int((finish / cost).argmax()) == selected == 0
    assert followup.shape == parents.shape


def test_forecasts_change_only_isolated_hypothetical_state():
    torch.set_num_threads(1)
    m = model().eval()
    public = {'components': ['A', 'B', 'C'], 'rust_level': 2, 'history': [
        {'component_counts': [1, 0, 0], 'rust_before': 3, 'rust_after': 3, 'native_actions': 12},
        {'component_counts': [0, 1, 1], 'rust_before': 3, 'rust_after': 2, 'native_actions': 13}]}
    obs, choices = observe(public)
    obs = primitive_observation_mask(obs)
    with torch.no_grad():
        state = m.receive(m.initial(1, 'cpu'), obs)
        out = m.readout(state, obs['candidates'], obs['action_mask'])
        saved = {key: value.clone() for key, value in state.items()}
        original = copy.deepcopy(public)
        action, trace = choose_experiment(m, state, public, obs, choices, out, elapsed_actions=25)
        assert obs['action_mask'][0, action]
        assert trace['imagined_branches'] > 0
        assert 0 <= trace['predicted_completion'] <= 1
        assert public == original
        for key in state:
            torch.testing.assert_close(state[key], saved[key], rtol=0, atol=0)
        future_public = copy.deepcopy(public)
        future_public['history'].append({'component_counts': [1, 1, 0],
            'rust_before': 2, 'rust_after': 1, 'native_actions': 14})
        future_public['rust_level'] = 1
        future_observation, _ = observe(future_public, elapsed_actions=39)
        future = forecast_state(m, state, future_observation)
        assert not torch.equal(future['h'], state['h'])
        assert not torch.equal(future['z'], state['z'])
        assert int(future['reaction_mask'].sum()) == 3
        received = m.receive(state, future_observation)
        for key in ('h', 'z', 'reaction_history', 'reaction_mask', 'fast'):
            torch.testing.assert_close(future[key], received[key], rtol=0, atol=0)
        # Imagined returns never become received observations or replay targets.
        for key in ('observation_count', 'memory_cursor', 'memory_target', 'pending'):
            if key in state:
                torch.testing.assert_close(future[key], state[key], rtol=0, atol=0)


def test_no_feasible_followup_keeps_one_step_completion():
    p = torch.tensor([.2, .1])
    c = torch.tensor([2., 2.])
    parent = torch.tensor([0, 1])
    future = torch.tensor([[.9], [.9]])
    finish, cost, next_action = two_experiment_values(p, c, parent, 1-p,
        future, torch.ones_like(future), torch.zeros_like(future, dtype=torch.bool))
    torch.testing.assert_close(finish, p)
    torch.testing.assert_close(cost, c)
    assert (next_action == -1).all()
