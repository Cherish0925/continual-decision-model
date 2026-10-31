"""Task-return and response learning on experiments selected by the live model."""
from __future__ import annotations

from dataclasses import replace
import torch
from torch.nn import functional as F

from cdm.adaptive_response import ResponseAdaptationConfig
from cdm.continual_experiment import Activity, ContinualExperimentPolicy, select_rows
from .reaction import ReactionExperimentPolicy
from .objectives import scaled_feedback_loss


def enable_response_learning(parent, activity_mode='learned', max_internal_steps=2):
    config = replace(parent.continual_config, activity_mode=activity_mode,
                     fixed_refinements=0, max_internal_steps=max_internal_steps)
    model = ReactionExperimentPolicy(parent.base, config, parent.response.config,
                                    ResponseAdaptationConfig()).to(next(parent.parameters()).device)
    incompatible = model.load_state_dict(parent.state_dict(), strict=False)
    if incompatible.unexpected_keys or any(not k.startswith(('response_adaptation.', 'experiment_residual.'))
                                          for k in incompatible.missing_keys):
        raise ValueError('The inherited chemical policy changed outside the new response update')
    return model


def start_apparatus(model, state, task_boundary, lifetime_boundary):
    return model.next_apparatus(state, task_boundary, lifetime_boundary)


def choose_activity(model, state, obs, *, sample, retain_grad=True):
    """Choose internal work without re-encoding an external observation."""
    done = torch.zeros(len(state['h']), dtype=torch.bool, device=state['h'].device)
    logps, entropies = [], []
    costs = torch.zeros_like(done, dtype=torch.float32)
    value = None
    for _ in range(model.continual_config.max_internal_steps + 1):
        out = ContinualExperimentPolicy.readout(model, state, obs['candidates'][:, :1],
                                                obs['action_mask'].any(-1, keepdim=True))
        if value is None:
            value = out['value']
        distribution = torch.distributions.Categorical(logits=out['activity_logits'])
        kind = distribution.sample() if sample else out['activity_logits'].argmax(-1)
        logps.append(torch.where(done, 0., distribution.log_prob(kind)))
        entropies.append(torch.where(done, 0., distribution.entropy()))
        choice = torch.where(done, int(Activity.WAIT), kind)
        before = state
        state = model.internal(state, choice, obs)
        state = select_rows(before, state, ~done)
        internal = ~done & (choice > Activity.ACT) & (choice < Activity.WAIT)
        costs += internal
        done = done | (choice == Activity.ACT) | (choice == Activity.WAIT)
        if done.all():
            break
    return state, torch.stack(logps).sum(0), torch.stack(entropies).sum(0), costs, value


def proposed_experiments(model, state, obs):
    # Action selection may inspect every candidate, but only the executed one
    # receives an outcome target. The response loss below supplies its gradient.
    with torch.no_grad():
        prediction = model.readout(state, obs['candidates'], obs['action_mask'])
    base = ContinualExperimentPolicy.readout(model, state, obs['candidates'], obs['action_mask'])
    out = {**prediction, 'features': base['features']}
    return model.experiment_logits(out, obs)


def active_chemical_window(model, state, env, count, statistics, *, compute_cost=.002,
                           discount=.995, entropy_weight=.01):
    losses, logps, rewards, values, entropies, continuing = [], [], [], [], [], []
    nlls, counts = [], []
    starts = env.dispatched
    for _ in range(count):
        boundary, lifetime = (env.new_task, env.new_lifetime) if env.dispatched == 0 else env.reset_finished()
        state = start_apparatus(model, state, boundary, lifetime)
        obs = env.observation()
        state = model.receive(state, obs)
        state, activity_lp, activity_entropy, computations, value = choose_activity(model, state, obs, sample=True)
        action_logits = proposed_experiments(model, state, obs)
        distribution = torch.distributions.Categorical(logits=action_logits)
        action = distribution.sample()
        row = torch.arange(len(action), device=action.device)
        query = obs['candidates'][row, action][:, None]
        out = model.readout(state, query, torch.ones(len(action), 1, dtype=torch.bool, device=action.device))
        zero = torch.zeros_like(action)
        state = model.begin_action(state, out['features'], zero)
        result = env.step(action)
        valid = result['returned'] & ~result['leave']
        nll = F.nll_loss(out['outcomes'][:, 0], result['after'], reduction='none')
        nll = (nll * valid).sum() / valid.sum().clamp_min(1)
        # Cost includes attempted operations even if the budget ends mid-test;
        # an unfinished response receives no chemical target or replay entry.
        received = result['returned']
        if received.any():
            cost_loss = scaled_feedback_loss(out['predictions'][received, 0], result['targets'][received], statistics)
        else:
            cost_loss = out['predictions'].sum() * 0
        losses.append(nll + cost_loss); nlls.append(nll.detach())
        state = model.feedback(state, result['targets'], received)
        recorded = model.record_response(state, out, zero, result['before'], result['after'])
        state = select_rows(state, recorded, valid)
        logps.append(distribution.log_prob(action) + activity_lp)
        values.append(value)
        rewards.append(result['reward'] - compute_cost * computations)
        entropies.append(distribution.entropy() + activity_entropy)
        counts.append(computations.detach())
        end_lifetime = result['terminal'] & torch.tensor(
            [i % env.lifetime_length == 0 for i in env.task_index], device=action.device)
        continuing.append(~end_lifetime)
    # Bootstrap the next received observation, including task boundaries, using
    # a copied environment state so the continuing rollout is not advanced twice.
    from copy import deepcopy
    with torch.no_grad():
        future = deepcopy(env)
        boundary, lifetime = future.reset_finished()
        next_state = start_apparatus(model, state, boundary, lifetime)
        next_obs = future.observation()
        next_state = model.receive(next_state, next_obs)
        tail = ContinualExperimentPolicy.readout(model, next_state, next_obs['candidates'][:, :1],
                                                 next_obs['action_mask'].any(-1, keepdim=True))['value']
    returns, target = [], tail
    for reward, carry in zip(reversed(rewards), reversed(continuing)):
        target = reward + discount * target * carry
        returns.append(target)
    returns = torch.stack(list(reversed(returns))).detach()
    values = torch.stack(values)
    advantage = returns - values.detach()
    policy = -(torch.stack(logps) * advantage).mean()
    critic = F.smooth_l1_loss(values, returns)
    entropy = torch.stack(entropies).mean()
    loss = torch.stack(losses).mean() + .1 * policy + .1 * critic - entropy_weight * entropy
    return loss, state, {'response_nll': float(torch.stack(nlls).mean()),
        'policy_loss': float(policy.detach()), 'value_loss': float(critic.detach()),
        'mean_internal_activities': float(torch.stack(counts).mean()),
        'requests': env.dispatched - starts, 'returned': env.returned,
        'native_actions': env.total_actions, 'episodes_ended': env.ended}
