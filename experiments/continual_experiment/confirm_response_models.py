"""Confirm the selector-disabled continuation on a new, fixed task-stream cohort."""
import argparse
import json
from pathlib import Path
import random

import torch

from experiments.continual_chemistry.response_families import TRAIN_FAMILIES, sample_lifetime
from .evaluate_memory import execute
from .shared_tasks import load_shared_checkpoint, save_shared_checkpoint
from .summarize_response_families import summarize


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--previous', type=Path, required=True)
    p.add_argument('--workers', type=int, default=12)
    args = p.parse_args()
    torch.set_num_threads(1)
    target = args.root / 'confirmation'
    target.mkdir(exist_ok=False)
    seed, streams = 903117, []
    groups = ['new_parameters'] * 24 + ['wider_parameters'] * 12 + ['heldout_combination'] * 12
    for i, group in enumerate(groups):
        families = ('interaction_inhibition',) if group == 'heldout_combination' else TRAIN_FAMILIES
        streams.append(dict(name=f'{group}_{i:03d}', group=group,
            tasks=sample_lifetime(random.Random(seed + 104729 * i), families,
                                  wider=group == 'wider_parameters')))
    manifest = dict(seed=seed, streams=streams,
        split='Drawn after primary-cohort selector removal; fixed before confirmation; no training or checkpoint selection afterward')
    manifest_path = args.root / 'cohorts/confirmation.json'
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    jobs, pairs = [], []
    for continuation in (911, 912):
        source = args.root / 'train' / f'hypotheses{continuation}'
        model, saved = load_shared_checkpoint(source / 'last.pt')
        model.tasks['chemistry'].experiment_value = None
        exported = source / 'core_only.pt'
        save_shared_checkpoint(exported, model, training={**saved['training'],
            'execution_variant': 'Remove new selector after the primary comparison; all remaining weights unchanged',
            'source_checkpoint': str(source / 'last.pt')})
        restored, _ = load_shared_checkpoint(exported)
        assert restored.tasks['chemistry'].experiment_value is None
        assert all(torch.equal(v, restored.state_dict()[k]) for k, v in model.state_dict().items())
        for condition, directory, endpoint in (
            ('core', source, 'core_only.pt'),
            ('inherited', args.root / 'train' / f'inherited{continuation}', 'last.pt'),
            ('parent', args.previous / 'train' / f'memory{continuation}', 'initial.pt')):
            name = f'{condition}{continuation}'
            command = ['-m', 'experiments.continual_experiment.evaluate_response_families',
                '--checkpoint', str(directory / endpoint), '--manifest', str(manifest_path),
                '--output', str(target / name), '--workers', str(args.workers), '--batch', '1',
                '--record-predictions', '--memory-mode', 'evidence']
            jobs.append(dict(name=name, directory=str(directory), endpoint=endpoint, arguments=command))
        pairs.extend((f'core{continuation}', f'{other}{continuation}') for other in ('parent', 'inherited'))
        pairs.append((f'inherited{continuation}', f'parent{continuation}'))
    command = ['-m', 'experiments.continual_experiment.evaluate_family_reference',
        '--manifest', str(manifest_path), '--output', str(target / 'belief'), '--selection', 'information',
        '--workers', str(args.workers)]
    jobs.append(dict(name='belief', directory=str(args.previous / 'train/memory911'),
                     endpoint='initial.pt', arguments=command))
    plan = dict(jobs=jobs, paired_conditions=pairs, training_updates=256, manifest=str(manifest_path),
        decision='Test removal of the new selector after primary-cohort results; compare with independently continued inherited actors',
        selection='All fixed final endpoints; no extra training, coefficient search or repeated confirmation cohorts',
        scope='Digital extension; new streams and continuous parameters; discrete targets can recur from training')
    (target / 'plan.json').write_text(json.dumps(plan, indent=2) + '\n')
    execute(plan, target, args.root / 'train', parallel_conditions=7)
    summarize(target, manifest, pairs)


if __name__ == '__main__':
    main()
