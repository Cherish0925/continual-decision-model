"""Run paired continuing/separate-state trainings on the two available GPUs."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--updates', type=int, default=200)
    p.add_argument('--batch', type=int, default=32)
    p.add_argument('--external-per-stream', type=int, default=16)
    p.add_argument('--chemical-steps', type=int, default=4)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    jobs = []
    for gpu, seed, warmup in ((0, 831, 753), (1, 832, 754)):
        for mode in ('continuing', 'separated'):
            name = f'{mode}_seed{seed}'
            command = ['-m', 'experiments.continual_experiment.train_shared_tasks',
                '--data', 'runs/discoveryworld/active_experiments/data_large',
                '--checkpoint', 'runs/physical_belief/joint_20261029/carry_seed221/policy.pt',
                '--belief-checkpoint', 'runs/physical_belief/joint_20261029/carry_seed221/belief.pt',
                '--mixture-checkpoint', 'runs/discoveryworld/calibration/mixture/shared_train_only_seed31/last.pt',
                '--environment-truth', 'runs/continual_physics/environment_train/environment.npz',
                '--execution-geometry', 'runs/continual_physics/environment_train/execution_geometry.npz',
                '--chemistry-checkpoint', f'runs/continual_physics/chemistry_reaction/warmup{warmup}/warmup.pt',
                '--chemical-data', 'runs/continual_physics/chemistry_exploration',
                '--output', str(args.output / name), '--device', f'cuda:{gpu}',
                '--seed', str(seed), '--state-sharing', mode, '--updates', str(args.updates),
                '--batch', str(args.batch), '--external-per-stream', str(args.external_per_stream),
                '--chemical-steps', str(args.chemical_steps)]
            jobs.append({'name': name, 'gpu': gpu, 'arguments': command})
    plan = {'jobs': jobs, 'updates': args.updates,
        'physical_requests_per_training': args.updates * args.batch * args.external_per_stream,
        'chemical_returns_per_training': args.updates * args.batch * args.chemical_steps,
        'shared_parent': 'physical carry_seed221; chemical warmups retain this identical core',
        'comparison': 'same weights, optimizer groups, task budgets and seed within each pair; h/z continuation differs',
        'selection': 'fixed final update; no intermediate evaluation selection'}
    (args.output / 'plan.json').write_text(json.dumps(plan, indent=2) + '\n')

    def run(job):
        start = perf_counter()
        with (args.output / (job['name'] + '.log')).open('w') as log:
            result = subprocess.run([sys.executable, *job['arguments']], stdout=log, stderr=subprocess.STDOUT)
        return {'name': job['name'], 'exit_code': result.returncode, 'seconds': perf_counter() - start}

    completed = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for future in as_completed([pool.submit(run, job) for job in jobs]):
            row = future.result(); completed.append(row)
            (args.output / 'completed.json').write_text(json.dumps(completed, indent=2) + '\n')
            print(json.dumps(row), flush=True)
    if any(r['exit_code'] for r in completed):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
