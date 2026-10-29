"""Measure conditional response learning on already recorded public histories.

These are same-prefix prediction interventions, not claims about native policy
performance. Core and direct-history paths are altered separately at readout.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from .dataset import load_episode
from .observations import feedback
from .reaction import load_reaction_checkpoint
from .train_reaction import selected_prediction


def average_precision(rows):
    """Tie-aware empirical precision-recall area for improvement probabilities."""
    positives = sum(row['improvement'] for row in rows)
    if not positives:
        return None
    ordered = sorted(rows, key=lambda row: -row['improvement_probability'])
    tp = count = ap = 0
    index = 0
    while index < len(ordered):
        stop = index + 1
        while stop < len(ordered) and ordered[stop]['improvement_probability'] == ordered[index]['improvement_probability']:
            stop += 1
        gained = sum(row['improvement'] for row in ordered[index:stop])
        tp += gained; count += stop - index
        ap += (tp / count) * (gained / positives)
        index = stop
    return ap


def summarize(rows):
    return {'transitions': len(rows), 'rust_nll': sum(r['rust_nll'] for r in rows) / len(rows),
            'improvement_frequency': sum(r['improvement'] for r in rows) / len(rows),
            'improvement_probability': sum(r['improvement_probability'] for r in rows) / len(rows),
            'improvement_brier': sum((r['improvement_probability'] - r['improvement'])**2 for r in rows) / len(rows),
            'improvement_average_precision': average_precision(rows),
            'cost_mae': sum(r['cost_error'] for r in rows) / len(rows),
            'calibration': [dict(lower=i / 10, upper=(i + 1) / 10, count=len(group),
                                prediction=sum(r['improvement_probability'] for r in group) / len(group),
                                frequency=sum(r['improvement'] for r in group) / len(group))
                            for i in range(10)
                            if (group := [r for r in rows if i / 10 <= r['improvement_probability'] < (i + 1) / 10])]}


def evaluate(checkpoint, data_root, output, *, limit_episodes=None):
    torch.set_num_threads(2)
    model, saved = load_reaction_checkpoint(checkpoint)
    model.eval()
    manifest = json.loads((data_root / 'manifest.json').read_text())
    episodes = [r for r in manifest['episodes'] if r['split'] == 'validation']
    if limit_episodes:
        episodes = episodes[:limit_episodes]
    modes = ['full', 'direct_history_off', 'response_core_off', 'history_pairing_reversed', 'frequency_prior']
    all_rows = {name: [] for name in modes}
    episode_rows = []
    with torch.no_grad():
        for episode in episodes:
            state = model.initial(1, 'cpu')
            grouped = {name: [] for name in modes}
            for obs, target, record in load_episode(data_root / episode['directory'] / 'experiments.jsonl'):
                state = model.receive(state, obs)
                actual = selected_prediction(model, state, obs, target)
                for mode in modes:
                    model.direct_history = mode != 'direct_history_off'
                    model.core_response = mode != 'response_core_off'
                    intervention = state
                    if mode == 'history_pairing_reversed':
                        history = state['reaction_history'].clone()
                        length = int(state['reaction_mask'][0].sum())
                        # Preserve each marginal and the continuing trajectory;
                        # alter only which mixture produced each returned result.
                        history[:, :length, :8] = history[:, :length, :8].flip(1)
                        intervention = {**state, 'reaction_history': history}
                    out = actual if mode == 'full' else selected_prediction(model, intervention, obs, target)
                    if mode == 'frequency_prior':
                        from cdm.conditional_response import observable_log_probabilities
                        n = len(record['component_counts']); before = record['rust_before']
                        # The model prior is trainable; compare with the initial
                        # train-only frequencies reconstructed from metadata.
                        prior = torch.tensor(saved['training']['initial_response_prior'])[n, before]
                        logp = observable_log_probabilities(prior, torch.tensor(before))
                    else:
                        logp = out['outcomes'][0, 0]
                    p = logp.exp()
                    row = {'rust_nll': -float(logp[record['rust_after']]),
                           'improvement': int(record['rust_after'] < record['rust_before']),
                           'improvement_probability': float(p[:record['rust_before']].sum()),
                           'cost_error': abs(math.expm1(float(out['predictions'][0, 0, 1].clamp(0, math.log(1001)))) - record['native_actions'])}
                    grouped[mode].append(row); all_rows[mode].append(row)
                model.direct_history = model.core_response = True
                state = model.begin_action(state, actual['features'], target['action'] * 0)
                state = model.feedback(state, feedback(record))
            episode_rows.append({'directory': episode['directory'],
                                 'evaluation_regime': episode.get('evaluation_regime', 'validation'),
                                 'conditions': {key: summarize(rows) for key, rows in grouped.items()}})
    result = {'checkpoint': checkpoint.name, 'training_updates': saved['training']['updates'],
              'scope': 'fixed public prefixes; readout interventions; no native acting result',
              'conditions': {key: summarize(rows) for key, rows in all_rows.items()}, 'episodes': episode_rows}
    for condition in result['conditions']:
        result['conditions'][condition]['episode_mean_nll'] = sum(e['conditions'][condition]['rust_nll'] for e in episode_rows) / len(episode_rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({key: value for key, value in result.items() if key != 'episodes'}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--data-root', type=Path, default=Path('runs/continual_physics/chemistry_data'))
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--limit-episodes', type=int)
    args = parser.parse_args()
    evaluate(args.checkpoint, args.data_root, args.output, limit_episodes=args.limit_episodes)


if __name__ == '__main__':
    main()
