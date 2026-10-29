"""Compare inherited and wrapped policies on the same received native evidence.

No simulator is run. All inputs and feedback come from public execution logs.
Model counterfactuals replay the recorded actions; they are not task scores.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path

import torch

from cdm.active_experiment import load_active_checkpoint
from cdm.continual_experiment import Activity, ContinualExperimentPolicy
from cdm.continual_physics import ContinualPhysicsPolicy, load_continual_physics_checkpoint
from experiments.discoveryworld.active_native import NativeActiveEpisode
from experiments.discoveryworld.calibration import FEATURE_QUANTITIES
from experiments.discoveryworld.calibration_mixture import load_mixture_checkpoint
from experiments.discoveryworld.evaluate_active_native import _event_from_result
from experiments.discoveryworld.evaluate_active_stream import observation, feedback_targets
from experiments.discoveryworld.train_active_core import changed_hypotheses
from experiments.continual_physics.evaluate_native import native_observation
from experiments.continual_physics.runtime import physical_feedback, setting_outcomes


def episode_from_record(record):
    """The native interface's initial ledger, using only its public bootstrap."""
    episode = NativeActiveEpisode(None)
    rows = record['bootstrap']['public_object_order']
    episode.objects = [r['public_handle'] for r in rows]
    episode.reactors = [r['reactor_handle'] for r in rows]
    episode.known[0, :2] = True
    episode.known_hz[0, :2] = torch.tensor(record['bootstrap']['known_support_hz'], dtype=torch.float64)
    episode.current_hz[0, :2] = episode.known_hz[0, :2]
    episode.current_hz[0, 2:] = 5000.
    # These are the control range/step explicitly returned by the task's dialog.
    episode.controls = [dict(lower_hz=0., upper_hz=10000., minimum_step_hz=1.) for _ in rows]
    return episode


def apply_public_record(episode, decision):
    previous_values, previous_counts = episode.values.clone(), episode.counts.clone()
    for row in decision['result'].get('measurements', []):
        if row['kind'] != 'instrument' or row['quantity'] not in FEATURE_QUANTITIES:
            continue
        if row['target_handle'] not in episode.objects:
            continue
        obj = episode.objects.index(row['target_handle'])
        q = FEATURE_QUANTITIES.index(row['quantity'])
        n = int(episode.counts[0, obj, q])
        value = float(row['value'])
        episode.values[0, obj, q] = (min(float(episode.values[0, obj, q]), value)
            if q == 2 and n else (episode.values[0, obj, q] * n + value) / (n + 1))
        episode.counts[0, obj, q] += 1
    action, query = decision['action_index'], decision['query_index']
    if action >= 20:
        outcomes = setting_outcomes(decision)
        if outcomes:
            episode.current_hz[0, query] = outcomes[-1][0]
        if decision['success']:
            activated = [value for value, success in outcomes if success]
            if not activated:
                raise ValueError('A successful public record must identify its setting')
            episode.known[0, query] = True
            episode.known_hz[0, query] = activated[-1]
            episode.completed_targets.add(query)
            if query == 2:
                episode.query[0] = 3
                episode.failed.zero_()
        else:
            episode.failed[0, action - 20] = True
    else:
        episode.failed &= ~changed_hypotheses(previous_values, previous_counts,
            episode.values, episode.counts, episode.known, episode.query)
    episode.decisions.append(decision)


def difference(a, b):
    mask = a.isfinite() & b.isfinite()
    return {'maximum': float((a[mask] - b[mask]).abs().max()) if mask.any() else 0.,
            'mask_disagreement': int((a.isfinite() != b.isfinite()).sum())}


@torch.no_grad()
def compare(args):
    torch.set_num_threads(1)
    parent, _ = load_active_checkpoint(args.parent, 'cpu')
    model, _ = load_continual_physics_checkpoint(args.initial, 'cpu')
    parent.eval(); model.eval()
    candidate = ContinualPhysicsPolicy(deepcopy(model.base), deepcopy(model.belief),
        model.continual_config, replace(model.physics_config, inherit_parent_interface=True))
    candidate.load_state_dict(model.state_dict(), strict=False)
    candidate.activity[-1].bias[Activity.WAIT] = 1.
    candidate.eval()
    mixture, _ = load_mixture_checkpoint(args.mixture, 'cpu')
    result = {'parent': str(args.parent), 'initial': str(args.initial),
              'base_tensors_equal': all(torch.equal(v, model.base.state_dict()[k]) for k, v in parent.state_dict().items()),
              'normalization_difference': {k: difference(getattr(mixture, k), getattr(model.belief, k))
                   for k in ('x_mean', 'x_scale', 'y_mean', 'y_scale')}, 'episodes': []}
    artifact = None
    for path in args.records:
        record = json.loads(path.read_text())
        episode = episode_from_record(record)
        parent_state = parent.initial(1, 'cpu')
        states = {mode: model.initial(1, 'cpu') for mode in ('wrapped_wait', 'wrapped_delay2')}
        states['inherited_candidate'] = candidate.initial(1, 'cpu')
        event = torch.zeros(1, 36); event[0, 34] = 1.
        trace = []
        for step, decision in enumerate(record['decisions']):
            tensors = episode.tensors('cpu')
            beliefs, _ = model.belief.observe_predict(tensors['values'], tensors['seen'], tensors['known_hz'],
                tensors['known'], tensors['query'], model.belief_memory(states['wrapped_wait']))
            old_obs, _ = observation(episode, event, mixture, step, 160, 'cpu',
                belief_source=lambda _: beliefs, physical_context=True)
            parent_state = parent.advance(event, old_obs['evidence'], old_obs['evidence_mask'], parent_state, 1)
            parent_out = parent.readout(parent_state, old_obs['candidates'], old_obs['action_mask'])
            row = {'step': step, 'recorded_action': decision['action_index'],
                'parent_action': int(parent_out['logits'].argmax()), 'failed_relations': episode.failed[0].nonzero().flatten().tolist()}
            for mode in states:
                active_model = candidate if mode == 'inherited_candidate' else model
                obs, _ = native_observation(active_model, states[mode], episode, event, step, 160, 'cpu')
                state = active_model.receive(states[mode], obs)
                out = active_model.readout(state, obs['candidates'], obs['action_mask'])
                # This restores only the original parent's tested-law feature/mask.
                old_candidates = obs['candidates'].clone()
                old_candidates[:, :, 28] = old_obs['candidates'][:, :, 28]
                old_candidates[:, :, 21] = old_obs['candidates'][:, :, 21]
                restored = active_model.readout(state, old_candidates, old_obs['action_mask'])
                inherited = ContinualExperimentPolicy.readout(active_model, state, old_obs['candidates'], old_obs['action_mask'])
                row[mode] = {'action': int(out['logits'].argmax()),
                    'restore_failed_action': int(restored['logits'].argmax()),
                    'inherited_input_action': int(inherited['logits'].argmax()),
                    'parent_state_h': float((state['h']-parent_state['h']).norm()),
                    'parent_state_z': float((state['z']-parent_state['z']).norm()),
                    'parent_logits': difference(parent_out['logits'], inherited['logits']),
                    'actual_parent_logits': difference(parent_out['logits'], out['logits']),
                    'candidate_change': difference(obs['candidates'], out['physical_candidates']),
                    'evidence_change': difference(old_obs['evidence'], obs['evidence']),
                    'action_mask_difference': int((old_obs['action_mask'] != obs['action_mask']).sum()),
                    'activities': [], 'top_actions': out['logits'][0].topk(4).indices.tolist(),
                    'top_logits': out['logits'][0].topk(4).values.tolist()}
                action = torch.tensor([decision['action_index']])
                state = active_model.begin_action(state, out['features'], action, physical=out)
                if mode in ('wrapped_delay2', 'inherited_candidate'):
                    for _ in range(2):
                        waiting = active_model.readout(state, obs['candidates'], obs['action_mask'])
                        activity = waiting['activity_logits'].argmax(-1)
                        row[mode]['activities'].append(Activity(int(activity[0])).name)
                        state = active_model.internal(state, activity, obs)
                feedback = physical_feedback(decision, 'cpu')
                state = active_model.feedback(state, feedback_targets(decision, 'cpu'),
                    physical_feedback=feedback if feedback['mask'].any() else None)
                states[mode] = state
                if artifact is None and decision['action_index'] >= 20:
                    artifact = {'observation': obs, 'action': action, 'feedback': feedback_targets(decision, 'cpu'),
                        'physical_feedback': feedback, 'source': str(path), 'decision': step,
                        'note': 'Only public measurements, actually tested settings and returned success/cost.'}
            trace.append(row)
            # Restrict causal analysis to the parent's first discrete divergence.
            if row['wrapped_delay2']['action'] != row['parent_action']:
                break
            apply_public_record(episode, decision)
            event = _event_from_result(decision, episode, mixture, step=step+1, max_steps=160, device='cpu')
        result['episodes'].append({'source': str(path), 'seed': record['seed'], 'trace': trace})
    args.output.mkdir(parents=True, exist_ok=True)
    saved = {**candidate.configuration(), 'model': candidate.state_dict(),
             'training': {'mode': 'inherited_interface_candidate', 'source_initial': str(args.initial)}, 'update': 0}
    torch.save(saved, args.output/'inherited_candidate.pt')
    loaded, _ = load_continual_physics_checkpoint(args.output/'inherited_candidate.pt')
    result['candidate_roundtrip_equal'] = all(torch.equal(v, loaded.state_dict()[k]) for k,v in candidate.state_dict().items())
    result['candidate_inherits_interface'] = loaded.physics_config.inherit_parent_interface
    result['candidate_pending_wait_bias'] = float(loaded.activity[-1].bias[Activity.WAIT])
    (args.output/'prefix_comparison.json').write_text(json.dumps(result, indent=2)+'\n')
    if artifact is not None:
        torch.save(artifact, args.output/'public_physics_sample.pt')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent', type=Path, required=True)
    p.add_argument('--initial', type=Path, required=True)
    p.add_argument('--mixture', type=Path, default=Path('runs/discoveryworld/calibration/mixture/shared_train_only_seed31/last.pt'))
    p.add_argument('--records', type=Path, nargs='+', required=True)
    p.add_argument('--output', type=Path, required=True)
    args=p.parse_args(); result=compare(args)
    for episode in result['episodes']:
        row=episode['trace'][-1]
        print(episode['seed'], row['step'], row['parent_action'], row['wrapped_delay2']['action'],
              row['wrapped_wait']['action'], row['wrapped_wait']['restore_failed_action'],
              row['inherited_candidate']['action'], row['inherited_candidate']['actual_parent_logits'])


if __name__=='__main__': main()
