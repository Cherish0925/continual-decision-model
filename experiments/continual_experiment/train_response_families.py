"""Joint response-family training with matched action representations."""
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
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--previous', type=Path, required=True)
    p.add_argument('--updates', type=int, default=256)
    p.add_argument('--batch', type=int, default=24)
    p.add_argument('--conditions', nargs='+', default=['inherited', 'marginal', 'hypotheses'])
    p.add_argument('--concurrency-per-gpu', type=int, default=1)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    jobs = []
    for gpu, seed in ((0, 911), (1, 912)):
        template = next(j for j in jobs_for(args.output, args.updates, args.batch)
                        if j['name'] == f'learned_seed{seed}')['arguments']
        for condition in args.conditions:
            name = f'{condition}{seed}'
            command = template.copy()
            for key, value in {'--output': args.output / name, '--seed': seed + 500,
                               '--chemical-steps': 16, '--external-per-stream': 8}.items():
                command[command.index(key) + 1] = str(value)
            command += ['--shared-initial', str(args.previous / 'train' / f'memory{seed}' / 'initial.pt'),
                '--experimental-memory', 'evidence', '--memory-slots', '12', '--chemical-families', 'diverse']
            if condition != 'inherited':
                command += ['--experiment-value', condition]
            jobs.append(dict(name=name, gpu=gpu, arguments=command))
    plan = dict(jobs=jobs, updates=args.updates, batch=args.batch,
        chemical_requests_per_model=args.updates * args.batch * 16,
        physical_requests_per_model=args.updates * args.batch * 8,
        endpoint='Fixed final update; no final-test selection',
        initial='Frozen original response parent; fixed twelve-ledger memory; within-task h/z retained',
        comparison='Inherited actor; same-size marginal selector; per-hypothesis selector',
        learned='Shared GRU/workspace, response predictor, actor/critic and the new action head',
        training='Five deterministic response families; new, changed-familiar and exact-return contexts',
        boundary='Digital extension, not an official DiscoveryWorld task; no task identities observed',
        unresolved='Within-fresh latent uncertainty, independent-time operation and consolidation remain separate work')
    (args.output / 'plan.json').write_text(json.dumps(plan, indent=2) + '\n')

    def run(job):
        started = perf_counter()
        with (args.output / (job['name'] + '.log')).open('w') as log:
            result = subprocess.run([sys.executable, *job['arguments']], stdout=log, stderr=subprocess.STDOUT)
        return dict(name=job['name'], exit_code=result.returncode, seconds=perf_counter() - started)

    completed = []
    # Each GPU owns an independent queue. Other research processes are not touched.
    def gpu_queue(gpu):
        with ThreadPoolExecutor(args.concurrency_per_gpu) as pool:
            return [f.result() for f in as_completed([pool.submit(run, j) for j in jobs if j['gpu'] == gpu])]

    with ThreadPoolExecutor(2) as pool:
        for future in as_completed([pool.submit(gpu_queue, gpu) for gpu in (0, 1)]):
            completed.extend(future.result())
            (args.output / 'completed.json').write_text(json.dumps(completed, indent=2) + '\n')
            print(json.dumps(completed), flush=True)
    if any(row['exit_code'] for row in completed):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
