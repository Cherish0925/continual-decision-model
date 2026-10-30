"""Evaluate every declared shared-core endpoint on fixed interleaved task streams."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from experiments.physical_belief.summarize_native import paired


def summarize(root, plan):
    expected = json.loads(Path(plan['manifest']).read_text())['streams']
    expected_keys = {
        'physics': {(s['name'], i, seed) for s in expected for i, seed in enumerate(s['seeds'])},
        'chemistry': {(s['name'], i, row['seed']) for s in expected for i, row in enumerate(s['chemistry'])}}
    rows, conditions, flat = {}, {}, []
    for job in plan['jobs']:
        name = job['name']
        streams = json.loads((root / name / 'summary.json').read_text())
        rows[name], conditions[name] = {}, {}
        for task in ('physics', 'chemistry'):
            values = [r for stream in streams for r in stream[task]]
            if len(values) != len(expected_keys[task]) or {
                (r['stream'], r['episode_index'], r['seed']) for r in values} != expected_keys[task]:
                raise ValueError(f'Incomplete {name}/{task}')
            if any(r['error'] for r in values):
                raise ValueError(f'Execution error in {name}/{task}')
            rows[name][task] = values
            conditions[name][task] = {'instances': len(values),
                'completed': sum(r['completed_successfully'] for r in values),
                'mean_native_actions': sum(r['native_actions'] for r in values) / len(values),
                'mean_seconds': sum(r['seconds'] for r in values) / len(values)}
            flat.extend({'condition': name, 'task': task, **{k: r[k] for k in
                ('stream', 'episode_index', 'seed', 'completed_successfully', 'native_actions', 'seconds')}} for r in values)
    comparisons = {}
    for seed in (831, 832):
        full = f'continuing_seed{seed}'
        for control in (f'separated_seed{seed}', f'initial_separated_seed{seed}', f'continuing_separated_runtime_seed{seed}'):
            comparisons[full + ':' + control] = {task: paired(rows[full][task], rows[control][task])
                                                for task in ('physics', 'chemistry')}
    result = {'protocol': plan, 'conditions': conditions, 'comparisons': comparisons}
    (root / 'comparisons.json').write_text(json.dumps(result, indent=2) + '\n')
    with (root / 'instances.csv').open('w') as file:
        writer = csv.DictWriter(file, fieldnames=list(flat[0])); writer.writeheader(); writer.writerows(flat)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--training-root', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--manifest', type=Path, default=Path('experiments/continual_experiment/shared_confirmation.json'))
    p.add_argument('--workers', type=int, default=6)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    jobs = []
    for seed in (831, 832):
        for label, source, endpoint, state in (
            ('initial_separated', 'continuing', 'initial.pt', 'separated'),
            ('continuing', 'continuing', 'last.pt', 'continuing'),
            ('separated', 'separated', 'last.pt', 'separated'),
            ('continuing_separated_runtime', 'continuing', 'last.pt', 'separated')):
            directory = args.training_root / f'{source}_seed{seed}'
            name = f'{label}_seed{seed}'
            command = ['-m', 'experiments.continual_experiment.evaluate_shared_tasks',
                '--checkpoint', str(directory / endpoint), '--manifest', str(args.manifest),
                '--output', str(args.output / name), '--initial-rng-root', str(args.output / 'initial_rng'),
                '--state-sharing', state, '--workers', str(args.workers)]
            jobs.append({'name': name, 'training_directory': str(directory), 'endpoint': endpoint,
                         'arguments': command})
    plan = {'jobs': jobs, 'manifest': str(args.manifest), 'training_updates': 200,
            'selection': 'all final endpoints and paired initial/same-weight comparisons',
            'tasks': 'six native Reactor Lab tasks interleaved with six native chemistry tasks in each stream',
            'scope': 'new instances of known simulator relation and reaction families',
            'budget': '400 native operations per physical task, 1000 per chemical task',
            'timing': 'sequential tasks; physical delivery is delayed two internal ticks; world clocks do not advance asynchronously',
            'bootstrap_unit': 'whole ordered six-physical/six-chemical task stream',
            'training_seed_scope': 'two continuations of the same physical parent and two chemical interface warmups'}
    (args.output / 'plan.json').write_text(json.dumps(plan, indent=2) + '\n')
    pending, running, completed = list(jobs), [], []
    while pending or running:
        for job in list(pending):
            if len(running) >= 2:
                break
            directory = Path(job['training_directory'])
            if job['endpoint'] == 'last.pt':
                status = args.training_root / 'completed.json'
                statuses = json.loads(status.read_text()) if status.exists() else []
                failure = next((r for r in statuses if r['name'] == directory.name and r['exit_code']), None)
                if failure:
                    completed.append({'name': job['name'], 'exit_code': failure['exit_code'], 'reason': 'training_failed'})
                    pending.remove(job)
                    continue
                marker = directory / 'finished.json'
                if not marker.exists():
                    continue
                if json.loads(marker.read_text())['updates'] != 200:
                    raise ValueError('Expected fixed final update 200')
            elif not (directory / job['endpoint']).exists():
                continue
            log = (args.output / (job['name'] + '.log')).open('w')
            process = subprocess.Popen([sys.executable, *job['arguments']], stdout=log, stderr=subprocess.STDOUT,
                env={**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'PYTHONHASHSEED': '0',
                     'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1'})
            running.append((job, process, log)); pending.remove(job)
        for entry in list(running):
            job, process, log = entry
            code = process.poll()
            if code is None:
                continue
            log.close(); running.remove(entry)
            row = {'name': job['name'], 'exit_code': code}; completed.append(row)
            print(json.dumps(row), flush=True)
        (args.output / 'completed.json').write_text(json.dumps(completed, indent=2) + '\n')
        if pending or running:
            time.sleep(5)
    if any(r['exit_code'] for r in completed):
        raise SystemExit(1)
    summarize(args.output, plan)


if __name__ == '__main__':
    main()
