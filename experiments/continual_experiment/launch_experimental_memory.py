"""Joint training of experimental memory and a matched no-memory reference."""
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
    p.add_argument('--initial-root', required=True, type=Path)
    p.add_argument('--updates', type=int, default=200)
    p.add_argument('--batch', type=int, default=32)
    p.add_argument('--smoke', action='store_true')
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    jobs = []
    for original in jobs_for(args.output, args.updates, args.batch):
        seed = int(original['name'][-3:])
        control = original['name'].startswith('fixed')
        name = ('no_memory' if control else 'memory') + str(seed)
        command = original['arguments']
        command[command.index('--output') + 1] = str(args.output / name)
        command[command.index('--chemical-activity') + 1] = 'learned'
        command[command.index('--chemical-steps') + 1] = '16'
        command[command.index('--seed') + 1] = str(seed + 300)
        command += ['--shared-initial', str(args.initial_root / f'relation{seed}' / 'last.pt'),
                    '--experimental-memory', 'disabled' if control else 'learned',
                    '--chemical-schedule', 'mixed']
        jobs.append(dict(name=name, arguments=command))
    if args.smoke:
        jobs = [jobs[0], jobs[-1]]
    plan = dict(jobs=jobs, updates=args.updates, batch=args.batch,
        endpoint='fixed final update; no checkpoint choice from native results',
        comparison='shared-core memory learning versus no archive use, identical starting weights and request budgets',
        memory='actual returned ledgers; six slots; compatibility learned from prequential evidence and future outcomes',
        train_sequence='12 tasks; variable novelty and return lags, sampled from training split only',
        scope='known simulator response family; no hidden recipe identity in model inputs; memory writes automatic')
    (args.output / 'plan.json').write_text(json.dumps(plan, indent=2) + '\n')

    def run(job):
        start = perf_counter()
        with (args.output / (job['name'] + '.log')).open('w') as log:
            r = subprocess.run([sys.executable, *job['arguments']], stdout=log, stderr=subprocess.STDOUT)
        return dict(name=job['name'], exit_code=r.returncode, seconds=perf_counter() - start)
    rows = []
    with ThreadPoolExecutor(len(jobs)) as pool:
        for future in as_completed([pool.submit(run, job) for job in jobs]):
            rows.append(future.result())
            (args.output / 'completed.json').write_text(json.dumps(rows, indent=2) + '\n')
            print(json.dumps(rows[-1]), flush=True)
    if any(r['exit_code'] for r in rows):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
