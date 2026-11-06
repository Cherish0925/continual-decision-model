"""Sequence learning for factorized requests, with one state update per event."""
import json
import torch
import torch.nn.functional as F

from cdm.digital import PublicActionHistory, PublicObjectMemory
from .features import batch_features, load_features
from .request_actions import request_stages, request_candidates
from .train import (PreparedEpisode, PreparedStep, public_observation, create_measurement_log,
                    with_measurement_records, next_numeric_target, outcome_target, action_run_weights)


def prepare_requests(episodes, cache, *, allow_failed_actions, imitate_successful_episodes):
    prepared = []
    for episode in episodes:
        if episode.metadata.get('action_interface') != 'object_requests_v1':
            raise ValueError('Object request training requires shared-executor trajectories')
        memory = PublicObjectMemory()
        history = PublicActionHistory(cache.config.action_history)
        measurements = create_measurement_log(public_observation(episode, episode.transitions[0]), cache.config)
        weights, exits = action_run_weights(episode.transitions)
        metrics = episode.transitions[-1].get('task_metrics', [])
        successful = bool(metrics) and all(m['completed_successfully'] for m in metrics)
        steps = []
        for i,row in enumerate(episode.transitions):
            raw = public_observation(episode, row)
            memory.objects.update((o['uuid'],o) for o in raw.get('remembered_objects',[]))
            observation = history.augment(with_measurement_records(memory.observe(raw), measurements,
                                          object_relations=cache.config.object_relations))
            stages = request_stages(observation, row['action'])
            paths = [(str(cache.encode(observation, choices, episode.path.parent)), index)
                     for choices,index in stages]
            labeled = row.get('imitate_action', True) and (
                not allow_failed_actions or row.get('executor',{}).get('fulfilled',False))
            if allow_failed_actions and row.get('terminated') and not all(
                    m.get('completed_successfully',False) for m in row.get('task_metrics',[])):
                labeled = False
            labeled = labeled and (successful or not imitate_successful_episodes)
            path,index = paths[-1]
            steps.append(PreparedStep(path,index if labeled else -100,index,outcome_target(row),
                weights[i],exits[i],next_numeric_target(row['next_observation'],memory),paths))
            history.append_transition(row,observation)
            if measurements is not None:
                measurements.observe(row['action'],row['action_result'],row['next_observation'],sequence=i)
        prepared.append(PreparedEpisode(episode,steps))
        print(json.dumps(dict(prepared=str(episode.path),requests=len(steps),features=cache.config.mode)),flush=True)
    return prepared


def score_requests(model, steps, obs, final_candidates, state, *, mode, persistent,
                   restart_workspace, refinements=None):
    device = obs['numeric'].device
    event,state = model.assimilate(obs,state,persistent=persistent,restart_workspace=restart_workspace)
    if mode == 'cdm':
        state = model.refine_state(event,state,model.config.refinements if refinements is None else refinements)
    stage_counts = [len(s.request_stages) for s in steps]
    lengths = torch.tensor(stage_counts,device=device)
    loss = torch.zeros(len(steps),device=device)
    correct = torch.ones(len(steps),dtype=torch.bool,device=device)
    for stage in range(max(stage_counts)):
        entries = [s.request_stages[min(stage,len(s.request_stages)-1)] for s in steps]
        _,actions = batch_features([load_features(path) for path,index in entries],device)
        targets = torch.tensor([index for path,index in entries],device=device)
        logits,_ = model.readout(event,actions,state,mode=mode,predict_outcomes=False)
        alive = stage < lengths
        loss = loss + F.cross_entropy(logits,targets,reduction='none')*alive/lengths
        correct = correct & ((logits.argmax(-1)==targets) | ~alive)
    selected = torch.tensor([s.executed_target for s in steps],device=device)
    logits,outcomes = model.readout(event,final_candidates,state,mode=mode,outcome_indices=selected)
    return logits,state,outcomes,loss,correct


def sample_requests(model, observations, choices, obs, actions, state, cache, *, mode,
                    persistent, restart_workspace, predict_outcomes):
    """Sample a joint request; sum conditional log probabilities for policy RL."""
    device = obs['numeric'].device
    event,state = model.assimilate(obs,state,persistent=persistent,restart_workspace=restart_workspace)
    if mode == 'cdm':
        state = model.refine_state(event,state,model.config.refinements)
    count = len(observations)
    logp = torch.zeros(count,device=device)
    entropy = torch.zeros(count,device=device)
    for stage in range(3):
        logits,_ = model.readout(event,actions,state,mode=mode,predict_outcomes=False)
        distribution = torch.distributions.Categorical(logits=logits)
        selected = distribution.sample()
        logp = logp + distribution.log_prob(selected)
        entropy = entropy + distribution.entropy()
        chosen = [rows[int(index)] for rows,index in zip(choices,selected)]
        requests = [dict(candidate.action) for candidate in chosen]
        following = [request_candidates(o,r) for o,r in zip(observations,requests)]
        if stage == 2 or not any(following):
            break
        # Finished rows have one already selected alternative, contributing
        # log probability and entropy zero in the remaining argument stages.
        choices = [rows or [candidate] for rows,candidate in zip(following,chosen)]
        features = [load_features(str(cache.encode(o,c))) for o,c in zip(observations,choices)]
        _,actions = batch_features(features,device)
    _,outcomes = model.readout(event,actions,state,mode=mode,
        predict_outcomes=predict_outcomes,outcome_indices=selected)
    state,forecast = model.commit_action(obs,actions,state,selected,mode=mode)
    return requests,state,logp,entropy,outcomes,forecast


def request_return(before, after, *, success, done, native_actions, gamma, action_cost):
    """Finite-budget semi-Markov reward with potential shaping.

    The task objective is successful completion minus actual native costs.
    Terminal potential is zero, including an exhausted action budget, so a
    partially solved failure cannot retain an artificial terminal bonus.
    The raw simulator increment is retained separately in traces.
    """
    discount = gamma ** native_actions
    shaped = float(success) + discount * (0. if done else after) - before - action_cost*native_actions
    return shaped,discount
