"""Paired joint training on policy-selected physical and chemical experiments."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter


def jobs_for(root, updates, batch):
    jobs = []
    for gpu, seed in ((0, 911), (1, 912)):
        for activity in ('learned', 'fixed'):
            name = f'{activity}_seed{seed}'
            command = ['-m', 'experiments.continual_experiment.train_shared_tasks',
                '--data', 'runs/discoveryworld/active_experiments/data_large',
                '--checkpoint', 'runs/physical_belief/joint_20261029/carry_seed221/policy.pt',
                '--belief-checkpoint', 'runs/physical_belief/joint_20261029/carry_seed221/belief.pt',
                '--mixture-checkpoint', 'runs/discoveryworld/calibration/mixture/shared_train_only_seed31/last.pt',
                '--environment-truth', 'runs/continual_physics/environment_train/environment.npz',
                '--execution-geometry', 'runs/continual_physics/environment_train/execution_geometry.npz',
                '--chemistry-checkpoint', 'runs/continual_physics/chemistry_reaction/warmup753/warmup.pt',
                '--chemical-data', 'runs/continual_physics/chemistry_exploration',
                '--output', str(root / name), '--device', f'cuda:{gpu}', '--seed', str(seed),
                '--state-sharing', 'separated', '--updates', str(updates), '--batch', str(batch),
                '--external-per-stream', '16', '--chemical-steps', '8', '--chemical-mode', 'active',
                '--chemical-activity', activity]
            jobs.append(dict(name=name, gpu=gpu, arguments=command))
    return jobs


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--updates', type=int, default=200)
    p.add_argument('--batch', type=int, default=32)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    jobs = jobs_for(args.output, args.updates, args.batch)
    plan = dict(jobs=jobs, updates=args.updates, batch=args.batch,
        physical_requests_per_run=args.updates * args.batch * 16,
        chemical_requests_per_run=args.updates * args.batch * 8,
        initialization='same physical parent and chemical warmup753; two continuation seeds',
        comparison='learned chemical internal work versus fixed single observation update; all other losses and budgets match',
        state='shared core parameters, per-domain h/z, chemical A A B B A A lifetimes',
        objective='received responses, physical/chemical native costs and subsequent task return',
        scope='known simulator families; structured public observations; no asynchronous-world claim',
        endpoint='fixed final update, all repetitions reported')
    (args.output / 'plan.json').write_text(json.dumps(plan, indent=2) + '\n')

    def run(job):
        start = perf_counter()
        with (args.output / (job['name'] + '.log')).open('w') as log:
            result = subprocess.run([sys.executable, *job['arguments']], stdout=log, stderr=subprocess.STDOUT)
        return dict(name=job['name'], exit_code=result.returncode, seconds=perf_counter() - start)

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
