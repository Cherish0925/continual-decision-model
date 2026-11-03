"""Joint prediction/action learning with four continuations and new task regimes."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import threading
from time import perf_counter, sleep

from experiments.continual_chemistry.response_families import EXPANDED_FAMILIES, sample_lifetime
from .launch_active_tasks import jobs_for


def cohort(seed):
    streams = []
    groups = [('new_parameters', 24, EXPANDED_FAMILIES, {}),
              ('wider_parameters', 12, EXPANDED_FAMILIES, {'wider': True}),
              ('heldout_combination', 12, ('interaction_inhibition',), {}),
              ('heldout_threshold', 12, ('threshold_balance',), {}),
              ('new_target_ratios', 12, EXPANDED_FAMILIES, {'max_parts': 8, 'minimum_target_parts': 7}),
              ('five_ingredients', 12, EXPANDED_FAMILIES, {'components': (5,)})]
    for group, size, families, kwargs in groups:
        for _ in range(size):
            index = len(streams)
            streams.append(dict(name=f'{group}_{index:03d}', group=group, max_parts=kwargs.get('max_parts', 6),
                tasks=sample_lifetime(random.Random(seed + 104729 * index), families, **kwargs)))
    for index, source in enumerate(streams[:12]):
        streams.append({**deepcopy(source), 'name': f'renamed_{index:03d}', 'group': 'renamed_inputs',
                        'rename_components': True, 'source_stream': source['name'],
                        'public_permutation': random.Random(seed + index + 93113).sample(range(8), 8)})
    return dict(seed=seed, streams=streams,
        scope='1008 task instances in 84 independent streams, plus 144 consistently renamed replays; digital extension',
        split='New test streams; target ratios with 7 or 8 primitive parts and 5-ingredient inputs are absent from both training stages')


def training_jobs(root, updates, batch):
    jobs = []
    for condition in ('joint', 'prediction_only', 'inherited'):
        for index, seed in enumerate((201, 202, 203, 204)):
            parent = 911 + index % 2
            command = jobs_for(root / 'train', updates, batch)[0]['arguments']
            def put(key, value):
                if key in command:
                    command[command.index(key) + 1] = str(value)
                else:
                    command.extend([key, str(value)])
            name = f'{condition}{seed}'
            for key, value in {'--output': root / 'train' / name, '--seed': seed + 1500,
                '--device': f'cuda:{index % 2}', '--external-per-stream': 8, '--chemical-steps': 8,
                '--shared-initial': f'runs/continual_physics/experimental_memory_20261101/train/memory{parent}/initial.pt',
                '--experimental-memory': 'evidence', '--memory-slots': 12,
                '--chemical-families': 'expanded'}.items():
                put(key, value)
            command.append('--rename-components')
            if condition != 'inherited':
                put('--relation-predictor', root / 'predictors' / f'seed{seed}' / 'last.pt')
                put('--relation-learning', condition)
            jobs.append(dict(name=name, condition=condition, seed=seed, parent=parent,
                             gpu=index % 2, arguments=command))
    return jobs


def execute(jobs, directory, parallel):
    directory.mkdir(parents=True, exist_ok=True)
    devices = {}
    for job in jobs:
        arguments = job['arguments']
        if '--device' in arguments:
            device = arguments[arguments.index('--device') + 1]
            if device.startswith('cuda:'):
                devices.setdefault(device, threading.Lock())
    def run(job):
        started = perf_counter()
        if job.get('ready'):
            while not Path(job['ready']).exists():
                completed = Path(job['ready']).parent.parent / 'completed.json'
                if completed.exists():
                    failed = next((r for r in json.loads(completed.read_text())
                                   if r['name'] == job['name'] and r['exit_code']), None)
                    if failed:
                        return dict(name=job['name'], exit_code=failed['exit_code'], dependency_failed=True)
                sleep(10)
        endpoint = Path(job['arguments'][job['arguments'].index('--output') + 1])
        marker = job.get('completion', 'finished.json' if job.get('ready') else 'summary.json')
        if (endpoint / marker).exists() or (endpoint / 'finished.json').exists():
            return dict(name=job['name'], exit_code=0, reused=True)
        arguments = job['arguments']
        device = arguments[arguments.index('--device') + 1] if '--device' in arguments else None
        from contextlib import nullcontext
        with devices.get(device, nullcontext()):
            with (directory / (job['name'] + '.log')).open('w') as log:
                result = subprocess.run([sys.executable, *arguments], stdout=log, stderr=subprocess.STDOUT,
                    env={**os.environ, 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
                         'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True'})
        row = dict(name=job['name'], exit_code=result.returncode, seconds=perf_counter() - started)
        print(json.dumps(row), flush=True)
        return row
    done = []
    with ThreadPoolExecutor(parallel) as pool:
        for future in as_completed([pool.submit(run, job) for job in jobs]):
            done.append(future.result())
            temporary = directory / 'completed.next.json'
            temporary.write_text(json.dumps(done, indent=2) + '\n')
            temporary.replace(directory / 'completed.json')
    if any(row['exit_code'] for row in done):
        raise SystemExit(1)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--stage', choices=('plan', 'predictors', 'references', 'train', 'test', 'native'), default='plan')
    p.add_argument('--updates', type=int, default=256)
    p.add_argument('--batch', type=int, default=12)
    p.add_argument('--parallel', type=int, default=4)
    p.add_argument('--evaluation-device', choices=('cpu', 'cuda'), default='cpu')
    p.add_argument('--workers', type=int, default=8)
    args = p.parse_args()
    if args.stage == 'plan':
        jobs = training_jobs(args.root, args.updates, args.batch)
        plan = dict(jobs=jobs, updates=args.updates, batch=args.batch,
            repetitions='Four continuation seeds from two shared-core parents, four independently initialized response predictors',
            comparison='Joint action/response gradients; same architecture with response-only gradients; inherited actor continuation',
            selection='Fixed final update, all four repetitions; no test selection',
            objective='Selected observed likelihood, completion and all executed operation costs, with physical task retention',
            learning='Shared core, task interfaces and learned response functions; public ingredient identities randomized per lifetime',
            endpoint='A complete joint continuation and held-out evaluation, not a completed autonomous real-time CDM')
        (args.root / 'plan.json').write_text(json.dumps(plan, indent=2) + '\n')
        (args.root / 'test_cohort.json').write_text(json.dumps(cohort(583194), indent=2) + '\n')
        return
    plan = json.loads((args.root / 'plan.json').read_text())
    if args.stage == 'predictors':
        jobs = [dict(name=f'seed{seed}', arguments=['-m',
            'experiments.continual_experiment.train_relation_belief',
            '--data', 'runs/continual_physics/function_belief_20261103/data',
            '--output', str(args.root / 'predictors' / f'seed{seed}'),
            '--kind', 'functions', '--decoder', 'modulated', '--updates', '24000',
            '--seed', str(seed), '--device', f'cuda:{index % 2}'])
            for index, seed in enumerate((201, 202, 203, 204))]
        execute(jobs, args.root / 'predictors', args.parallel)
    elif args.stage == 'references':
        jobs = [dict(name=f'information{seed}', arguments=['-m',
            'experiments.continual_experiment.evaluate_relation_belief',
            '--checkpoint', str(args.root / 'predictors' / f'seed{seed}' / 'last.pt'),
            '--manifest', str(args.root / 'test_cohort.json'),
            '--output', str(args.root / 'references' / f'information{seed}'),
            '--selection', 'information', '--capacity', '12', '--workers', str(args.workers)])
            for seed in (201, 202, 203, 204)]
        execute(jobs, args.root / 'references', args.parallel)
    elif args.stage == 'train':
        execute(plan['jobs'], args.root / 'train', args.parallel)
    elif args.stage == 'native':
        jobs = []
        for source in plan['jobs']:
            path = args.root / 'train' / source['name']
            jobs.append(dict(name=source['name'], ready=str(path / 'finished.json'), arguments=['-m',
                'experiments.continual_experiment.evaluate_shared_tasks', '--checkpoint', str(path / 'last.pt'),
                '--manifest', 'runs/continual_physics/experimental_memory_20261101/cohort/manifest.json',
                '--initial-rng-root', 'runs/continual_physics/experimental_memory_20261101/native/initial_rng',
                '--output', str(args.root / 'native' / source['name']), '--state-sharing', 'separated',
                '--memory-mode', 'evidence', '--memory-slots', '12', '--workers', '8']))
        (args.root / 'native').mkdir(exist_ok=True)
        (args.root / 'native' / 'plan.json').write_text(json.dumps(dict(jobs=jobs,
            scope='Original native cohort and saved starting states; retention comparison, not a new independent cohort'), indent=2)+'\n')
        execute(jobs, args.root / 'native', args.parallel)
    else:
        jobs = []
        for index, source in enumerate(plan['jobs']):
            path = args.root / 'train' / source['name']
            jobs.append(dict(name=source['name'], ready=str(path / 'finished.json'),
                completion='summary.json', arguments=['-m',
                'experiments.continual_experiment.evaluate_response_families',
                '--checkpoint', str(path / 'last.pt'), '--manifest', str(args.root / 'test_cohort.json'),
                '--output', str(args.root / 'test' / source['name']),
                '--device', f'cuda:{index % 2}' if args.evaluation_device == 'cuda' else 'cpu',
                '--workers', str(args.workers if args.evaluation_device == 'cpu' else 1),
                '--batch', '4', '--record-predictions']))
        for index, parent in enumerate((911, 912)):
            name = f'parent{parent}'
            jobs.append(dict(name=name, arguments=['-m',
                'experiments.continual_experiment.evaluate_response_families',
                '--checkpoint', f'runs/continual_physics/experimental_memory_20261101/train/memory{parent}/initial.pt',
                '--manifest', str(args.root / 'test_cohort.json'),
                '--output', str(args.root / 'test' / name),
                '--device', f'cuda:{index}' if args.evaluation_device == 'cuda' else 'cpu',
                '--workers', str(args.workers if args.evaluation_device == 'cpu' else 1), '--batch', '4',
                '--record-predictions']))
        (args.root / 'test').mkdir(exist_ok=True)
        (args.root / 'test' / 'plan.json').write_text(json.dumps(dict(jobs=jobs), indent=2) + '\n')
        execute(jobs, args.root / 'test', args.parallel)


if __name__ == '__main__':
    main()
