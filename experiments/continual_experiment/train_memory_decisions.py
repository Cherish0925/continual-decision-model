"""Compare prediction-only and action-return learning of memory applicability."""
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
    p.add_argument('--initial-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--updates', type=int, default=100)
    p.add_argument('--batch', type=int, default=32)
    p.add_argument('--smoke', action='store_true')
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    jobs = []
    for original in jobs_for(args.output, args.updates, args.batch):
        seed = int(original['name'][-3:])
        control = original['name'].startswith('fixed')
        name = ('prediction' if control else 'decision') + str(seed)
        command = original['arguments']
        command[command.index('--output') + 1] = str(args.output / name)
        command[command.index('--chemical-activity') + 1] = 'learned'
        command[command.index('--chemical-steps') + 1] = '16'
        command[command.index('--seed') + 1] = str(seed + 400)
        command += ['--shared-initial', str(args.initial_root / f'memory{seed}' / 'last.pt'),
                    '--chemical-schedule', 'mixed', '--memory-lr', '.001']
        if not control:
            command += ['--memory-action-credit']
        jobs.append(dict(name=name, arguments=command))
    if args.smoke:
        jobs = jobs[:1]
    plan = dict(jobs=jobs, updates=args.updates, batch=args.batch,
        motivation='Memory helps returning relations but increases new-relation costs in the completed comparison',
        comparison='Same parent, reader rate, joint training and request budget; action credit into applicability differs',
        prediction='Experts retain proper likelihood training; detached expert forecasts feed the actor reader path',
        endpoint='Fixed final window, both continuations retained; new native seeds before model evaluation')
    (args.output / 'plan.json').write_text(json.dumps(plan, indent=2) + '\n')

    def run(job):
        start = perf_counter()
        with (args.output / (job['name'] + '.log')).open('w') as log:
            result = subprocess.run([sys.executable, *job['arguments']], stdout=log, stderr=subprocess.STDOUT)
        return dict(name=job['name'], exit_code=result.returncode, seconds=perf_counter() - start)
    done = []
    with ThreadPoolExecutor(len(jobs)) as pool:
        for future in as_completed([pool.submit(run, job) for job in jobs]):
            row = future.result(); done.append(row)
            (args.output / 'completed.json').write_text(json.dumps(done, indent=2) + '\n')
            print(json.dumps(row), flush=True)
    if any(row['exit_code'] for row in done):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
