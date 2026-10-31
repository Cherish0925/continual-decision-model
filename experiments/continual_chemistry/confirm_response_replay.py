"""Native decisions using fitted response updates on a fresh instance cohort."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing
import os
from pathlib import Path

import torch

from cdm.response_replay import ResponseReplay, ResponseReplayConfig
from experiments.continual_experiment.shared_tasks import load_shared_checkpoint, save_shared_checkpoint
from .collect import recipe_group
from .native import run
from .summarize import paired


def episode(job):
    name, parent, update, row, output = job
    torch.set_num_threads(1)
    model, _ = load_shared_checkpoint(parent)
    chemical = model.tasks['chemistry']
    if update is not None:
        data = torch.load(update, map_location='cpu', weights_only=True)
        config = dict(data['config'])
        if name.startswith('relation_no_update'):
            config['mode'] = 'calibration'
        chemical.response_replay = ResponseReplay(config=ResponseReplayConfig(**config))
        chemical.response_replay.load_state_dict(data['model'])
    options = dict(policy='model', checkpoint=str(parent), seed=row['seed'], difficulty=row['difficulty'],
        output=str(output / name / row['directory']), max_actions=1000, max_experiments=209, threads=1,
        primitive_candidates=True, initial_rng_root=str(output / 'initial_rng'),
        discard_response_updates=name.startswith('zero'))
    result = run(options, model=chemical)
    return {**result, 'condition': name, 'directory': row['directory'],
            'recipe_group': recipe_group(Path(options['output']) / 'native.log')}


def summarize(rows, protocol):
    summaries, comparisons = [], {}
    names = sorted({row['condition'] for row in rows})
    for name in names:
        group = [row for row in rows if row['condition'] == name]
        summaries.append(dict(condition=name, instances=len(group),
            completed=sum(r['completed_successfully'] for r in group),
            errors=sum(bool(r['error']) for r in group),
            **{f'mean_{key}': sum(r[key] for r in group) / len(group)
               for key in ('native_actions', 'experiments', 'inference_seconds', 'execution_seconds', 'seconds')}))
        for reference in ('original' + name[-3:], 'zero' + name[-3:], 'calibration' + name[-3:]):
            other = [r for r in rows if r['condition'] == reference]
            if name != reference and len(group) == len(other) == len(protocol['episodes']):
                comparisons[f'{name}:{reference}'] = paired(group, other)
    return dict(protocol=protocol, conditions=summaries, comparisons=comparisons, episodes=rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--workers', type=int, default=16)
    p.add_argument('--per-difficulty', type=int, default=20)
    p.add_argument('--seed-base', type=int, default=241000)
    p.add_argument('--conditions', nargs='+', default=['original', 'zero', 'calibration', 'stored', 'refreshed', 'relation'])
    p.add_argument('--parents', nargs='+', type=int, default=[911, 912])
    args = p.parse_args()
    if os.environ.get('PYTHONHASHSEED') != '0':
        raise ValueError('Run with PYTHONHASHSEED=0')
    rows = [dict(difficulty=d, seed=args.seed_base + 7*i, directory=f'{d.lower()}/seed{args.seed_base + 7*i}')
            for d in ('Normal', 'Challenge') for i in range(args.per_difficulty)]
    protocol = dict(episodes=rows, conditions=args.conditions, parents=args.parents,
        selection='new seeds fixed before response-update results; all four fitted methods included',
        compute='actual inference time includes response re-encoding and four inner updates; not equal compute',
        training='original core, action head and cost activity policy remain frozen',
        state='reset at each apparatus; this experiment does not evaluate long-term consolidation',
        max_native_actions=1000, bootstrap_unit='named recipe parameter group')
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
    jobs, results = [], []
    for seed in args.parents:
        parent = Path(f'runs/continual_physics/active_feedback_joint_20261030/learned_seed{seed}/last.pt')
        for condition in args.conditions:
            fitted = 'relation' if condition == 'relation_no_update' else condition
            update = None if condition in ('original', 'zero') else args.study / f'{fitted}{seed}' / 'update.pt'
            if update is not None and not update.exists():
                raise FileNotFoundError(update)
            for row in rows:
                saved = args.output / (condition + str(seed)) / row['directory'] / 'comparison.json'
                if saved.exists():
                    results.append(json.loads(saved.read_text()))
                else:
                    jobs.append((condition + str(seed), parent, update, row, args.output))
    total = len(jobs) + len(results)
    with ProcessPoolExecutor(args.workers, mp_context=multiprocessing.get_context('spawn')) as pool:
        futures = [pool.submit(episode, job) for job in jobs]
        for future in as_completed(futures):
            result = future.result(); results.append(result)
            (args.output / result['condition'] / result['directory'] / 'comparison.json').write_text(json.dumps(result, indent=2) + '\n')
            if len(results) % 20 == 0 or len(results) == total:
                (args.output / 'results.json').write_text(json.dumps(summarize(results, protocol), indent=2) + '\n')
                print(json.dumps(dict(completed=len(results), total=total)), flush=True)
    (args.output / 'results.json').write_text(json.dumps(summarize(results, protocol), indent=2) + '\n')
    (args.output / 'completed.json').write_text(json.dumps(dict(executions=len(results))) + '\n')


if __name__ == '__main__':
    main()
