"""Train one native policy per condition, with independent GPU queues."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def wait_for_collections(roots):
    """Start training only after every declared collection attempt has ended."""
    previous = None
    while True:
        progress = []
        for root in roots:
            metadata_path, summary_path = root / 'collection.json', root / 'summary.json'
            metadata = json.loads(metadata_path.read_text())
            try:
                rows = json.loads(summary_path.read_text()) if summary_path.exists() else []
            except json.JSONDecodeError:
                # Collectors started before atomic summary writes may still be
                # replacing this file. Wait for their next complete snapshot.
                rows = []
            expected = {case['id'] for case in metadata['cases']}
            finished = {row['id'] for row in rows}
            if metadata.get('status') == 'error':
                raise RuntimeError(f"Collection failed: {root}: {metadata.get('error')}")
            progress.append(dict(path=str(root), completed=len(finished & expected),
                                 total=len(expected), complete=finished == expected))
        if progress != previous:
            print(json.dumps(dict(collections=progress)), flush=True)
            previous = progress
        if all(row['complete'] for row in progress):
            return
        time.sleep(30)


def collection_data(roots):
    """Keep nonempty traces, recording supervision coverage before fitting."""
    files, coverage = [], []
    for root in roots:
        for path in sorted(root.rglob('trajectory.jsonl')):
            if path.stat().st_size:
                files.append(path)
        for path in sorted(root.glob('*/summary.json')):
            row = json.loads(path.read_text())
            coverage.append({key: row[key] for key in
                ('theme', 'difficulty', 'seed', 'status', 'steps', 'success', 'progress')})
    if not files:
        raise ValueError('Collection produced no executed trajectories')
    return files, coverage


def commands(args):
    protocol = json.loads(args.protocol.read_text())
    jobs = []
    for initial in protocol['initializations']:
        for condition in protocol['conditions']:
            name = f"{condition['name']}{initial['seed']}"
            checkpoint = (args.output / 'initializations' / f"random{initial['seed']}.pt"
                          if condition.get('initialization') == 'random' else args.source_root / initial['source'])
            command = [sys.executable, '-m', 'experiments.discoveryworld.train',
                       '--train-trajectories', *map(str, args.train_trajectories),
                       '--validation-trajectories', *map(str, args.validation_trajectories),
                       '--initial-checkpoint', str(checkpoint),
                       '--output', str(args.output / name),
                       '--feature-cache', str(args.feature_cache),
                       '--encoder-path', str(args.encoder_path),
                       '--feature-device', 'cuda:0', '--device', 'cuda:0', '--threads', '2',
                       '--seed', str(initial['seed']), '--mode', condition['mode']]
            for key, value in {**protocol['training'], **condition.get('training', {})}.items():
                flag = '--' + key.replace('_', '-')
                if isinstance(value, bool):
                    if value:
                        command.append(flag)
                else:
                    command.extend((flag, str(value)))
            if condition.get('holdout', False):
                command.extend(('--exclude-themes', *protocol['theme_holdout']))
            jobs.append(dict(name=name, command=command))
    return jobs


def prepare_random_cores(args, protocol):
    """Random comparison cores retain the inherited architecture and action grid."""
    if not any(row.get('initialization') == 'random' for row in protocol['conditions']):
        return
    import torch
    from cdm.core import ContinualCore, CoreConfig
    directory = args.output / 'initializations'
    directory.mkdir(exist_ok=True)
    for initial in protocol['initializations']:
        source = torch.load(args.source_root / initial['source'], map_location='cpu', weights_only=True)
        if source.get('kind') != 'shared_experiment_core_v1':
            raise ValueError('Matched random initialization requires a shared scientific core')
        torch.manual_seed(initial['seed'])
        core = ContinualCore(CoreConfig(**source['core_config']), source['core']['action_values'])
        torch.save(dict(kind='shared_experiment_core_v1', core_config=source['core_config'], core=core.state_dict(),
                        initialization='random core with the inherited architecture', seed=initial['seed']),
                   directory / f"random{initial['seed']}.pt")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--protocol', type=Path, default=Path(__file__).with_name('native_learning.json'))
    p.add_argument('--train-trajectories', nargs='+', type=Path, required=True)
    p.add_argument('--validation-trajectories', nargs='+', type=Path, required=True)
    p.add_argument('--source-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--feature-cache', type=Path, required=True)
    p.add_argument('--encoder-path', type=Path, required=True)
    p.add_argument('--gpus', nargs='+', default=['0', '1'])
    p.add_argument('--plan-only', action='store_true')
    p.add_argument('--wait-for-collections', action='store_true',
                   help='Wait for all declared teacher episodes, then use their nonempty traces')
    p.add_argument('--evaluate', action='store_true',
                   help='Execute all 120 official cases at each declared final training endpoint')
    p.add_argument('--evaluation-workers', type=int, default=2)
    args = p.parse_args()
    if not args.plan_only:
        args.output.mkdir(parents=True, exist_ok=True)
    if args.wait_for_collections and not args.plan_only:
        wait_for_collections([*args.train_trajectories, *args.validation_trajectories])
        args.train_trajectories, train_coverage = collection_data(args.train_trajectories)
        args.validation_trajectories, validation_coverage = collection_data(args.validation_trajectories)
        (args.output / 'data.json').write_text(json.dumps(dict(training=train_coverage,
            development=validation_coverage), indent=2) + '\n')
    jobs = commands(args)
    if args.plan_only:
        print(json.dumps(jobs, indent=2))
        return
    protocol = json.loads(args.protocol.read_text())
    prepare_random_cores(args, protocol)
    (args.output / 'plan.json').write_text(json.dumps(jobs, indent=2) + '\n')

    def queue(index):
        env = {**os.environ, 'CUDA_VISIBLE_DEVICES': args.gpus[index], 'OMP_NUM_THREADS': '2'}
        results = []
        for job in jobs[index::len(args.gpus)]:
            with (args.output / (job['name'] + '.log')).open('x') as log:
                process = subprocess.run(job['command'], env=env, stdout=log, stderr=subprocess.STDOUT)
            row = dict(name=job['name'], exit_code=process.returncode)
            results.append(row)
            print(json.dumps(row), flush=True)
            if process.returncode:
                break
            if args.evaluate:
                policy = args.output / (job['name'] + '-policy.json')
                policy.write_text(json.dumps(dict(checkpoint=str(args.output / job['name'] / 'last.pt'),
                    feature_cache=str(args.feature_cache), device='cuda:0',
                    encoder_path=str(args.encoder_path)), indent=2) + '\n')
                with (args.output / (job['name'] + '-evaluation.log')).open('x') as log:
                    evaluation_command = [sys.executable, '-m', 'experiments.discoveryworld.full_suite',
                        'run', '--policy', 'experiments.discoveryworld.evaluate:create',
                        '--policy-config', str(policy), '--policy-id', job['name'],
                        '--workers', str(args.evaluation_workers), '--threads', '1',
                        '--output', str(args.output / (job['name'] + '-evaluation'))]
                    for key, value in protocol.get('evaluation', {}).items():
                        evaluation_command += ['--'+key.replace('_','-')]
                        evaluation_command += list(map(str, value)) if isinstance(value, list) else [str(value)]
                    evaluation = subprocess.run(evaluation_command,
                        env=env, stdout=log, stderr=subprocess.STDOUT)
                row['evaluation_exit_code'] = evaluation.returncode
                if evaluation.returncode:
                    break
        return results

    with ThreadPoolExecutor(len(args.gpus)) as pool:
        results = [row for queue_rows in pool.map(queue, range(len(args.gpus))) for row in queue_rows]
    (args.output / 'completed.json').write_text(json.dumps(results, indent=2) + '\n')
    if len(results) != len(jobs) or any(row['exit_code'] or row.get('evaluation_exit_code', 0) for row in results):
        raise SystemExit('Training queue stopped; see the named run log.')


if __name__ == '__main__':
    main()
