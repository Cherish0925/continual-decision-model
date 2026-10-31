"""Direct recipe retrieval and recurrent-state controls for experimental memory."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--training-root', type=Path, required=True)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--workers', type=int, default=8)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    jobs = []
    for seed in (911, 912):
        for label, source, mode, flag in (
            ('direct', 'no_memory', 'disabled', '--direct-retrieval'),
            ('reset_neural', 'memory', 'learned', '--reset-chemical-neural-state')):
            directory = args.training_root / f'{source}{seed}'
            name = f'{label}{seed}'
            command = ['-m', 'experiments.continual_experiment.evaluate_shared_tasks',
                '--checkpoint', str(directory / 'last.pt'), '--manifest', str(args.manifest),
                '--output', str(args.output / name), '--initial-rng-root', str(args.output / 'initial_rng'),
                '--state-sharing', 'separated', '--memory-mode', mode,
                '--chemistry-only', '--workers', str(args.workers), flag]
            jobs.append(dict(name=name, directory=str(directory), arguments=command))
    plan = dict(jobs=jobs, manifest=str(args.manifest),
        declaration='Before the trained memory candidates were evaluated',
        direct='Try recent successful recipes still legal in this task, then use the trained no-memory policy',
        reset_neural='Keep raw experimental archives, reset h/z at each chemical task boundary')
    with (args.output / 'memory_controls_plan.json').open('x') as file:
        file.write(json.dumps(plan, indent=2) + '\n')

    def run(job):
        directory = Path(job['directory'])
        while not (directory / 'finished.json').exists():
            status = args.training_root / 'completed.json'
            statuses = json.loads(status.read_text()) if status.exists() else []
            failure = next((r for r in statuses if r['name'] == directory.name and r['exit_code']), None)
            if failure:
                return dict(name=job['name'], exit_code=failure['exit_code'], reason='training_failed')
            time.sleep(5)
        with (args.output / (job['name'] + '.log')).open('w') as log:
            r = subprocess.run([sys.executable, *job['arguments']], stdout=log, stderr=subprocess.STDOUT,
                env={**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'PYTHONHASHSEED': '0',
                     'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1'})
        return dict(name=job['name'], exit_code=r.returncode)
    completed = []
    with ThreadPoolExecutor(4) as pool:
        for future in as_completed([pool.submit(run, job) for job in jobs]):
            r = future.result(); completed.append(r)
            (args.output / 'memory_controls_completed.json').write_text(json.dumps(completed, indent=2) + '\n')
            print(json.dumps(r), flush=True)
    if any(r['exit_code'] for r in completed):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
