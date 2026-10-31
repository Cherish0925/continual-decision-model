"""Continue both task interfaces with response learning and a matched control."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter

from .launch_active_tasks import jobs_for


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--study', required=True, type=Path)
    p.add_argument('--updates', type=int, default=200)
    p.add_argument('--batch', type=int, default=32)
    p.add_argument('--chemical-steps', type=int, default=16)
    p.add_argument('--smoke', action='store_true')
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    jobs = []
    for original in jobs_for(args.output, args.updates, args.batch):
        seed = int(original['name'][-3:])
        control = original['name'].startswith('fixed')
        name = ('calibration' if control else 'relation') + str(seed)
        command = original['arguments']
        command[command.index('--output') + 1] = str(args.output / name)
        command[command.index('--chemical-activity') + 1] = 'learned'
        command[command.index('--chemical-steps') + 1] = str(args.chemical_steps)
        command[command.index('--seed') + 1] = str(seed + 200)
        command += ['--shared-initial', f'runs/continual_physics/active_feedback_joint_20261030/learned_seed{seed}/last.pt',
                    '--response-replay-checkpoint', str(args.study / f'relation{seed}' / 'update.pt')]
        if control:
            command += ['--response-replay-control']
        jobs.append(dict(name=name, arguments=command))
    if args.smoke:
        jobs = [jobs[0], jobs[-1]]
    (args.output / 'plan.json').write_text(json.dumps(dict(jobs=jobs, updates=args.updates, batch=args.batch,
        chemical_steps=args.chemical_steps, endpoint='fixed final update; both repeats and controls',
        initial='inherited complete physical/chemical model and fitted response-update parameters',
        comparison='joint core and action training, with versus without relation-conditioned response fitting',
        control='identical initial temperature, bias, core, policy, cost learning and external-request budget',
        scope='known physical/chemical families; structured inputs; response fit has fixed compute; no new long-term memory'), indent=2) + '\n')

    def run(job):
        start = perf_counter()
        with (args.output / (job['name'] + '.log')).open('w') as log:
            result = subprocess.run([sys.executable, *job['arguments']], stdout=log, stderr=subprocess.STDOUT)
        return dict(name=job['name'], exit_code=result.returncode, seconds=perf_counter() - start)
    rows = []
    with ThreadPoolExecutor(len(jobs)) as pool:
        for job in as_completed([pool.submit(run, job) for job in jobs]):
            row = job.result(); rows.append(row)
            print(json.dumps(row), flush=True)
            (args.output / 'completed.json').write_text(json.dumps(rows, indent=2) + '\n')
    if any(row['exit_code'] for row in rows):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
