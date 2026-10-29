"""Execute a fixed native comparison as its final training checkpoints arrive."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


def make_jobs(root, training_roots, workers):
    mixture = 'runs/discoveryworld/calibration/mixture/shared_train_only_seed31/last.pt'
    parents = {221: 'runs/discoveryworld/active_experiments/rehearsal/counterfactual_seed183/update1200.pt',
               222: 'runs/discoveryworld/active_experiments/rehearsal_repeat/counterfactual_seed184/update1200.pt'}
    manifest = 'experiments/physical_belief/native_confirmation.json'
    common = ['--manifest', manifest, '--mixture-checkpoint', mixture,
              '--initial-rng-root', str(root / 'initial_rng'), '--workers', str(workers), '--threads', '1']
    jobs = []

    def append(name, checkpoint, *, belief=None, belief_mode='carry', ready=None, extra=None):
        cmd = [sys.executable, '-m', 'experiments.physical_belief.evaluate_native',
               '--output', str(root / name), '--checkpoint', checkpoint, *common]
        if belief:
            cmd += ['--belief-checkpoint', belief, '--belief-mode', belief_mode]
        if extra:
            cmd += extra
        jobs.append({'name': name, 'command': cmd, 'ready': str(ready) if ready else None})

    for seed, checkpoint in parents.items():
        append(f'baseline_rehearsal_{seed-38}', checkpoint)
    jobs.append({'name': 'baseline_explicit', 'ready': None,
                 'command': [sys.executable, '-m', 'experiments.discoveryworld.evaluate_relation_memory',
                             '--output', str(root / 'baseline_explicit'), '--state-mode', 'carry', *common]})
    # This earlier-development parent comparison uses the same 30 instances as
    # its seed-183 counterpart. It cannot select which trained arms are run.
    append('development_parent184', parents[222], extra=[
        '--manifest', 'experiments/physical_belief/native_development.json',
        '--initial-rng-root', 'runs/physical_belief/native_development/initial_rng',
        '--output', 'runs/physical_belief/native_development/baseline_rehearsal184'])
    for seed, training_root in training_roots.items():
        for mode in ('ordinary', 'reset', 'carry'):
            trained = training_root / f'{mode}_seed{seed}'
            append(f'{mode}_seed{seed}', str(trained / 'policy.pt'),
                   belief=None if mode == 'ordinary' else str(trained / 'belief.pt'),
                   belief_mode='reset' if mode == 'reset' else 'carry', ready=trained / 'finished.json')
            if mode == 'carry':
                for intervention in ('reset', 'discard'):
                    append(f'carry_seed{seed}_memory_{intervention}', str(trained / 'policy.pt'),
                           belief=str(trained / 'belief.pt'), belief_mode=intervention,
                           ready=trained / 'finished.json')
    for job in jobs:
        manifest_index = max(index for index, value in enumerate(job["command"]) if value == "--manifest")
        job["manifest"] = job["command"][manifest_index + 1]
        cohort = json.loads(Path(job["manifest"]).read_text())
        job["expected_instances"] = sum(len(stream["seeds"]) for stream in cohort["streams"])
    return jobs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('runs/physical_belief/native_confirmation'))
    parser.add_argument('--training-root221', type=Path, default=Path('runs/physical_belief/joint_20261029'))
    parser.add_argument('--training-root222', type=Path, default=Path('runs/physical_belief/joint_repeat_20261029'))
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--parallel', type=int, default=8)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    jobs = make_jobs(args.output, {221: args.training_root221, 222: args.training_root222}, args.workers)
    plan = {'jobs': jobs, 'selection': 'All six fixed-final-budget trainings, independent of development score',
            'native_instances_per_confirmation': 120, 'total_native_executions': sum(job['expected_instances'] for job in jobs), 'cohort': '20 ordered A-B-A streams, two fresh tasks per segment',
            'matched_rng': str(args.output / 'initial_rng'), 'threads_per_worker': 1,
            'maximum_workers': args.parallel * args.workers,
            'same_weight_interventions': ['carry', 'reset physical memory at task boundaries',
                                          'compute and discard physical memory updates'],
            'neural_state': 'reset at native task boundaries in all neural-policy conditions'}
    (args.output / 'evaluation_plan.json').write_text(json.dumps(plan, indent=2) + '\n')
    running, complete = [], []
    pending = list(jobs)
    while pending or running:
        for job in list(pending):
            if len(running) >= args.parallel:
                break
            if job['ready'] and not Path(job['ready']).exists():
                continue
            log = (args.output / f'{job["name"]}.log').open('w')
            process = subprocess.Popen(job['command'], stdout=log, stderr=subprocess.STDOUT)
            running.append((job, process, log, time.monotonic()))
            pending.remove(job)
            print(json.dumps({'event': 'started', 'name': job['name'], 'pid': process.pid}), flush=True)
        for entry in list(running):
            job, process, log, started = entry
            code = process.poll()
            if code is None:
                continue
            log.close()
            row = {'name': job['name'], 'exit_code': code, 'seconds': time.monotonic()-started}
            complete.append(row)
            running.remove(entry)
            (args.output / 'evaluation_completion.json').write_text(json.dumps(complete, indent=2) + '\n')
            print(json.dumps({'event': 'finished', **row}), flush=True)
        if pending or running:
            time.sleep(2)
    if any(row['exit_code'] for row in complete):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
