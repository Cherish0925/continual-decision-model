"""Localize changed predictions between two trained stages on public prefixes.

Parameter combinations measure functional compatibility, not an independently
trained factorial comparison. They do not isolate the effects of a training
schedule or augmentation policy.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .dataset import load_episode
from .observations import feedback
from .reaction import load_reaction_checkpoint
from .train_reaction import selected_prediction


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--warmup', required=True, type=Path)
    p.add_argument('--joint', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--data-root', type=Path, default=Path('runs/continual_physics/chemistry_data'))
    args = p.parse_args(); torch.set_num_threads(2)
    warm, _ = load_reaction_checkpoint(args.warmup); joint, _ = load_reaction_checkpoint(args.joint)
    warm.eval(); joint.eval()
    own = {'core': joint.base.core, 'event': joint.base.event, 'evidence': joint.base.evidence, 'response': joint.response}
    manifest = json.loads((args.data_root / 'manifest.json').read_text())
    sequences = [(entry, load_episode(args.data_root / entry['directory'] / 'experiments.jsonl'))
                 for entry in manifest['episodes'] if entry['split'] == 'validation']
    results = {}
    for condition in ['joint_response_warm_encoder', 'warm_response_joint_encoder', 'joint_with_warm_core']:
        joint.base.core, joint.base.event, joint.base.evidence, joint.response = (own[key] for key in ['core', 'event', 'evidence', 'response'])
        if condition == 'joint_response_warm_encoder':
            joint.base.core = warm.base.core; joint.base.event = warm.base.event; joint.base.evidence = warm.base.evidence
        elif condition == 'warm_response_joint_encoder':
            joint.response = warm.response
        else:
            joint.base.core = warm.base.core
        records = []
        with torch.no_grad():
            for entry, sequence in sequences:
                state = joint.initial(1, 'cpu'); nll = 0.
                for obs, target, row in sequence:
                    state = joint.receive(state, obs)
                    out = selected_prediction(joint, state, obs, target)
                    nll -= float(out['outcomes'][0, 0, int(target['rust_after'][0])])
                    state = joint.begin_action(state, out['features'], target['action'] * 0)
                    state = joint.feedback(state, feedback(row))
                records.append({'directory': entry['directory'], 'regime': entry['evaluation_regime'],
                                'queries': len(sequence), 'nll': nll / len(sequence)})
        results[condition] = {regime: sum(r['nll'] for r in records if regime == 'all' or r['regime'] == regime) /
                             sum(regime == 'all' or r['regime'] == regime for r in records)
                             for regime in ['all', 'untrained_recipe', 'seen_recipe_new_instance']}
    result = {'scope': 'same old development prefixes; parameter substitution, not trained ablation or confirmation tuning',
              'warmup': args.warmup.name, 'joint': args.joint.name, 'episode_mean_nll': results}
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
