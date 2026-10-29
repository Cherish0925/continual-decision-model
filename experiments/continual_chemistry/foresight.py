"""Two-experiment forecasts from a fixed learned public-response model.

Forecasts are private computation branches. They update hypothetical h/z and a
hypothetical ledger, but never write into received observations, feedback memory
or the actual continuing state. The same model evaluates the next experiments.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from time import perf_counter

import torch

from .observations import observe
from .reaction import primitive_observation_mask
from .warmup_reaction import batch_observations


@dataclass(frozen=True)
class ForesightConfig:
    topk: int = 6
    maximum_iterations: int = 24
    terminal_reserve: float = 5.  # Public executor uses at most approach+open+3 moves.


def predicted_cost(out):
    return out['predictions'][..., 1].clamp(0, math.log(1001)).expm1().clamp_min(1.)


def two_experiment_values(first_probability, first_cost, parent, branch_probability,
                          next_probability, next_cost, next_mask, iterations=24):
    """Maximize predicted completion probability / expected experiment operations.

    For each possible first action and returned state, the second action may be
    chosen separately. Stopping that branch costs zero and adds no completion.
    Fractional optimization uses the finite-action Bellman expression p-lambda*c;
    a fixed point gives the best modeled two-experiment rate on this shortlist.
    Door operations are excluded from the ratio but reserved in feasibility and
    are charged by the native executor. A common terminal cost is a monotone
    transform of this ratio and does not change its maximizer.
    """
    rate = (first_probability / first_cost).max()
    best_next = torch.full((len(parent),), -1, dtype=torch.long)
    finish, cost = first_probability.clone(), first_cost.clone()
    for _ in range(iterations):
        advantage = (next_probability - rate * next_cost).masked_fill(~next_mask, -torch.inf)
        advantage = torch.cat((advantage, advantage.new_zeros(len(parent), 1)), -1)
        chosen = advantage.argmax(-1)
        usable = chosen < next_probability.shape[-1]
        safe = chosen.clamp_max(next_probability.shape[-1] - 1)
        p = next_probability.gather(1, safe[:, None]).squeeze(1) * usable
        c = next_cost.gather(1, safe[:, None]).squeeze(1) * usable
        finish = first_probability.index_add(0, parent, branch_probability * p)
        cost = first_cost.index_add(0, parent, branch_probability * c)
        new_rate, selected = (finish / cost).max(0)
        best_next = torch.where(usable, chosen, -1)
        if float((new_rate - rate).abs()) < 1e-8:
            break
        rate = new_rate
    return finish, cost, best_next


def forecast_state(model, actual, observation):
    """Apply the learned observation transition to isolated predicted branches.

    No receive()/feedback() call is made. Counters and real replay records remain
    unchanged. h/z here describe an imagined future, not another actual reading.
    """
    batch = len(observation['event'])
    imagined = {key: value.expand(batch, *value.shape[1:]).clone() for key, value in actual.items()}
    advanced = model.base.advance(observation['event'], observation['evidence'],
                                  observation['evidence_mask'], imagined)
    imagined.update(advanced)
    history = observation['evidence'][:, 1:]
    mask = observation['evidence_mask'][:, 1:]
    if history.shape[1] > imagined['reaction_history'].shape[1]:
        raise ValueError('Forecast history exceeds the saved model capacity')
    imagined['reaction_history'].zero_(); imagined['reaction_mask'].zero_()
    imagined['reaction_history'][:, :history.shape[1]] = history
    imagined['reaction_mask'][:, :mask.shape[1]] = mask
    imagined['fast'] = imagined['fast'] * model.continual_config.fast_decay
    return imagined


@torch.no_grad()
def choose_experiment(model, state, public, observation, choices, out, *, elapsed_actions,
                      max_actions=1000, mode='two_step', config=ForesightConfig()):
    started = perf_counter()
    probability = out['outcomes'][0].softmax(-1)
    cost = predicted_cost(out)[0]
    valid = observation['action_mask'][0].clone()
    remaining = max_actions - elapsed_actions
    feasible = valid & (cost + config.terminal_reserve <= remaining)
    completion_score = (probability[:, 0] / cost).masked_fill(~(feasible if feasible.any() else valid), -torch.inf)
    direct = int(completion_score.argmax())
    if mode == 'completion':
        return direct, {'mode': mode, 'predicted_completion': float(probability[direct, 0]),
                        'predicted_experiment_operations': float(cost[direct]),
                        'imagined_branches': 0, 'seconds': perf_counter() - started}
    if mode != 'two_step':
        raise ValueError('Planning mode must be completion or two_step')
    if not feasible.any():
        return direct, {'mode': mode, 'imagined_branches': 0,
                        'reason': 'no predicted experiment fits remaining budget', 'seconds': perf_counter() - started}
    # Retain direct-value and uncertain-outcome candidates. Entropy is only a
    # shortlist heuristic; the final score uses the forecast's actual consequence.
    entropy = -(probability * probability.clamp_min(1e-30).log()).sum(-1)
    ranks = [(probability[:, 0] / cost).masked_fill(~feasible, -torch.inf).argsort(descending=True),
             (entropy / cost).masked_fill(~feasible, -torch.inf).argsort(descending=True)]
    selected = []
    for rank, quota in zip(ranks, [(config.topk + 1) // 2, config.topk]):
        for index in rank.tolist():
            if feasible[index] and index not in selected:
                selected.append(index)
            if len(selected) >= min(quota, int(feasible.sum())):
                break
    selected = selected[:config.topk]
    observations, parent, weights = [], [], []
    current = int(public['rust_level'])
    for position, action in enumerate(selected):
        for returned in range(1, current + 1):
            weight = float(probability[action, returned])
            if weight == 0:
                continue
            record = {'component_counts': list(choices[action]), 'rust_before': current,
                      'rust_after': returned, 'native_actions': float(cost[action])}
            imagined = {'components': public['components'], 'rust_level': returned,
                        'history': [*public['history'], record]}
            obs, _ = observe(imagined, elapsed_actions=elapsed_actions + float(cost[action]), max_actions=max_actions)
            observations.append(primitive_observation_mask(obs))
            parent.append(position); weights.append(weight)
    ids = torch.tensor(selected)
    if observations:
        obs = batch_observations(observations)
        imagined = forecast_state(model, state, obs)
        future = model.readout(imagined, obs['candidates'], obs['action_mask'])
        next_p = future['outcomes'].softmax(-1)[..., 0]
        next_c = predicted_cost(future)
        parents = torch.tensor(parent)
        next_mask = obs['action_mask'] & (cost[ids[parents], None] + next_c + config.terminal_reserve <= remaining)
        finish, expected_cost, next_action = two_experiment_values(probability[ids, 0], cost[ids], parents,
            torch.tensor(weights), next_p, next_c, next_mask, config.maximum_iterations)
        chosen_position = int((finish / expected_cost).argmax())
        chosen = selected[chosen_position]
        followups = [{'returned_rust': int((o['candidates'][0, 0, 25] * 3).round()),
                      'probability': weights[i], 'action': int(next_action[i])}
                     for i, o in enumerate(observations) if parent[i] == chosen_position]
    else:
        chosen_position = int((probability[ids, 0] / cost[ids]).argmax())
        chosen = selected[chosen_position]
        finish, expected_cost = probability[ids, 0], cost[ids]
        followups = []
    return chosen, {'mode': mode, 'shortlist': selected, 'imagined_branches': len(observations),
                    'predicted_completion': float(finish[chosen_position]),
                    'predicted_experiment_operations': float(expected_cost[chosen_position]),
                    'first_return_probability': probability[chosen].tolist(), 'conditional_followups': followups,
                    'seconds': perf_counter() - started,
                    'cost_scope': 'predicted native experiment actions; terminal actions reserved and charged at execution'}
