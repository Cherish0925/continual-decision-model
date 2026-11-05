"""Factorized learned requests using the existing persistent digital core."""
from time import perf_counter

import torch

from .features import batch_features, load_features
from .request_actions import request_candidates


@torch.no_grad()
def choose_request(policy, observation):
    started = perf_counter()
    policy.memory.objects.update((o['uuid'],o) for o in observation.get('remembered_objects',[]))
    if policy.measurements is not None:
        policy.measurements.observe_initial(observation)
        observation = {**observation,'measurement_records':policy.measurements.summaries()}
        if policy.features.config.object_relations:
            observation['measurement_dialog_owner'] = policy.measurements.dialog_owner
    observation = policy.memory.observe(observation)
    policy.previous_observation = observation
    observation = policy.action_history.augment(observation)
    prefix = {}
    events = []
    event = None
    feature_seconds = 0.
    for _ in range(3):
        candidates = request_candidates(observation,prefix)
        if not candidates:
            break
        feature_start = perf_counter()
        path = policy.features.encode(observation,candidates)
        obs,actions = batch_features([load_features(str(path))],policy.device)
        feature_seconds += perf_counter()-feature_start
        if event is None:
            event,policy.state = policy.model.assimilate(obs,policy.state,
                persistent=policy.persistent,restart_workspace=policy.restart_workspace)
            if policy.mode == 'cdm':
                depth = policy.model.config.refinements if policy.refinements is None else policy.refinements
                policy.state = policy.model.refine_state(event,policy.state,depth)
        logits,_ = policy.model.readout(event,actions,policy.state,mode=policy.mode,predict_outcomes=False)
        if policy.temperature > 0:
            selected = int(torch.multinomial((logits[0]/policy.temperature).softmax(-1),1,generator=policy.generator))
        else:
            selected = int(logits[0].argmax())
        prefix = dict(candidates[selected].action)
        policy.previous_features,policy.previous_candidates = (obs,actions),candidates
        events.append(dict(candidates=len(candidates),selected=selected))
    elapsed = perf_counter()-started
    policy.decisions.append(dict(stages=events,feature_seconds=feature_seconds,
        policy_seconds=elapsed-feature_seconds,decision_seconds=elapsed))
    return prefix
