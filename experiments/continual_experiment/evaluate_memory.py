"""Complete native comparisons for learned experimental memory."""
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from experiments.physical_belief.summarize_native import paired


def summarize(root, plan):
    manifest = json.loads(Path(plan['manifest']).read_text())
    expected = {(s['name'], i, t['seed']) for s in manifest['streams']
                for i, t in enumerate(s['chemistry'])}
    records, conditions, flat = {}, {}, []
    for job in plan['jobs']:
        name = job['name']
        streams = json.loads((root / name / 'summary.json').read_text())
        records[name], conditions[name] = {}, {}
        for domain in ('chemistry', 'physics'):
            rows = [r for stream in streams for r in stream[domain]]
            if not rows:
                continue
            if domain == 'chemistry' and {(r['stream'], r['episode_index'], r['seed']) for r in rows} != expected:
                raise ValueError(f'Incomplete chemistry cohort: {name}')
            if any(r['error'] for r in rows):
                raise ValueError(f'Execution error: {name}/{domain}')
            records[name][domain] = rows
            groups = {'all': rows}
            if domain == 'chemistry':
                groups.update({exposure: [r for r in rows if r['exposure'] == exposure]
                               for exposure in ('new', 'return')})
                groups['long_return'] = [r for r in rows if (r['intervening_tasks'] or 0) >= 4]
                groups['untrained_composition'] = [r for r in rows if r['training_regime'] == 'untrained_named_composition']
            conditions[name][domain] = {key: {
                'instances': len(group), 'completed': sum(r['completed_successfully'] for r in group),
                **{'mean_' + metric: float(np.mean([r[metric] for r in group]))
                   for metric in ('native_actions', 'seconds', 'inference_seconds') if metric in group[0]}}
                for key, group in groups.items() if group}
            flat.extend({'condition': name, 'task': domain, **{key: r.get(key) for key in (
                'stream', 'episode_index', 'seed', 'difficulty', 'exposure', 'intervening_tasks',
                'training_regime', 'completed_successfully', 'native_actions', 'seconds', 'inference_seconds')}}
                for r in rows)
    comparisons = {}
    for seed in (911, 912):
        left = f'memory{seed}'
        for right in (f'no_memory{seed}', f'disabled{seed}', f'evidence{seed}', f'initial{seed}'):
            comparisons[left + ':' + right] = {}
            for domain in records[left].keys() & records[right].keys():
                a, b = records[left][domain], records[right][domain]
                result = {'all': paired(a, b)}
                if domain == 'chemistry':
                    for label, predicate in (
                        ('new', lambda r: r['exposure'] == 'new'),
                        ('return', lambda r: r['exposure'] == 'return'),
                        ('long_return', lambda r: (r['intervening_tasks'] or 0) >= 4),
                        ('untrained_composition', lambda r: r['training_regime'] == 'untrained_named_composition')):
                        result[label] = paired([r for r in a if predicate(r)], [r for r in b if predicate(r)])
                comparisons[left + ':' + right][domain] = result
    result = dict(protocol=plan, conditions=conditions, comparisons=comparisons)
    (root / 'comparisons.json').write_text(json.dumps(result, indent=2) + '\n')
    with (root / 'instances.csv').open('w') as file:
        writer = csv.DictWriter(file, fieldnames=list(flat[0])); writer.writeheader(); writer.writerows(flat)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--training-root', type=Path, required=True)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--workers', type=int, default=10)
    p.add_argument('--parallel-conditions', type=int, default=3)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    jobs = []
    for seed in (911, 912):
        for condition, source, endpoint, mode, chemistry_only in (
            ('initial', 'memory', 'initial.pt', 'disabled', False),
            ('memory', 'memory', 'last.pt', 'learned', False),
            ('no_memory', 'no_memory', 'last.pt', 'disabled', False),
            ('disabled', 'memory', 'last.pt', 'disabled', True),
            ('evidence', 'memory', 'last.pt', 'evidence', True)):
            name, directory = f'{condition}{seed}', args.training_root / f'{source}{seed}'
            command = ['-m', 'experiments.continual_experiment.evaluate_shared_tasks',
                '--checkpoint', str(directory / endpoint), '--manifest', str(args.manifest),
                '--output', str(args.output / name), '--initial-rng-root', str(args.output / 'initial_rng'),
                '--state-sharing', 'separated', '--memory-mode', mode, '--workers', str(args.workers)]
            if chemistry_only:
                command += ['--chemistry-only']
            jobs.append(dict(name=name, directory=str(directory), endpoint=endpoint, arguments=command))
    plan = dict(jobs=jobs, manifest=str(args.manifest), training_updates=200,
        comparison='both trained endpoints; same-weight no archive use; fixed evidence retrieval; original parent',
        scope='new native instances, known simulator families, structured public observations',
        tasks='12 chemical tasks interleaved with 12 physical tasks; separate states and shared core weights',
        timing='sequential environment tasks; elapsed time is measured, not an asynchronous world',
        controls='same-weight chemistry interventions skip independent, unchanged physics executions',
        uncertainty='paired bootstrap of entire 12-task streams, 10000 resamples',
        endpoint='all fixed final training endpoints; no outcome-based selection')
    with (args.output / 'plan.json').open('x') as file:
        file.write(json.dumps(plan, indent=2) + '\n')
    pending, running, completed = list(jobs), [], []
    while pending or running:
        statuses = args.training_root / 'completed.json'
        statuses = json.loads(statuses.read_text()) if statuses.exists() else []
        for job in list(pending):
            if len(running) >= args.parallel_conditions:
                break
            directory = Path(job['directory'])
            failed = next((r for r in statuses if r['name'] == directory.name and r['exit_code']), None)
            if failed:
                completed.append(dict(name=job['name'], exit_code=failed['exit_code'], reason='training_failed'))
                pending.remove(job); continue
            if job['endpoint'] == 'last.pt':
                marker = directory / 'finished.json'
                if not marker.exists():
                    continue
                if json.loads(marker.read_text())['updates'] != plan['training_updates']:
                    raise ValueError('Training endpoint differs from declared update count')
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
            row = dict(name=job['name'], exit_code=code); completed.append(row)
            print(json.dumps(row), flush=True)
        (args.output / 'completed.json').write_text(json.dumps(completed, indent=2) + '\n')
        if pending or running:
            time.sleep(5)
    if any(r['exit_code'] for r in completed):
        raise SystemExit(1)
    summarize(args.output, plan)


if __name__ == '__main__':
    main()
