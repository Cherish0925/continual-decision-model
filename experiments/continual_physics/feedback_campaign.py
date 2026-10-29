"""Queue matched feedback trainings after the original GPU work, then evaluate."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .verify_feedback_start import arguments_from_settings


def read_json(path, default):
    try:
        return json.loads(path.read_text()) if path.exists() else default
    except json.JSONDecodeError:
        return default  # Existing launchers update their short completion files in place.


def dependencies(primary, reference, seed):
    return [(primary, f'{mode}_seed{seed}') for mode in
            ('full', 'discard_physical', 'task_reset', 'fixed_schedule')] + [
                (reference, f'recurrent_reference_seed{seed}')]


def dependency_state(items):
    waiting = []
    for root, name in items:
        completed = {row['name']: row for row in read_json(root / 'completed.json', [])}
        row = completed.get(name)
        if row is not None and row['exit_code']:
            raise RuntimeError(f'Dependency exited unsuccessfully: {name}')
        if row is None:
            waiting.append(name)
            continue
        finished = read_json(root / name / 'finished.json', None)
        if finished is None or not (root / name / 'last.pt').exists():
            raise RuntimeError(f'Dependency exited without final artifacts: {name}')
        if finished['updates'] != 200 or finished['external_requests'] != 204800:
            raise RuntimeError(f'Dependency did not finish its fixed budget: {name}')
    return waiting


def make_plan(args):
    jobs = []
    for gpu, seed in enumerate((231, 232)):
        settings = json.loads((args.primary / f'full_seed{seed}' / 'config.json').read_text())
        if (settings['updates'], settings['batch'], settings['external_per_stream']) != (200, 32, 32):
            raise ValueError('The candidate comparison must match the original fixed budget')
        for source in ('workspace', 'evidence'):
            name = f'{source}_seed{seed}'
            command = arguments_from_settings(settings, recurrent_feedback=source,
                initial_state=args.primary / f'full_seed{seed}' / 'initial.pt',
                output=args.output / name, device=f'cuda:{gpu}')
            jobs.append(dict(name=name, seed=seed, gpu=gpu, source=source,
                             arguments=['-m', 'experiments.continual_physics.train', *command],
                             dependencies=[dict(root=str(root), name=condition) for root, condition in
                                           dependencies(args.primary, args.reference, seed)]))
    evaluations = []
    for cohort in ('development', 'confirmation'):
        manifest = Path('experiments/continual_physics') / f'feedback_{cohort}.json'
        instances = json.loads(manifest.read_text())['instance_count']
        for seed in (231, 232):
            conditions = [(f'full_seed{seed}', args.primary / f'full_seed{seed}' / 'last.pt', False)]
            for source in ('workspace', 'evidence'):
                name = f'{source}_seed{seed}'
                conditions += [(name, args.output / name / 'last.pt', False),
                               (name + '_disabled', args.output / name / 'last.pt', True)]
            for name, checkpoint, disabled in conditions:
                output = args.output / 'native' / cohort / name
                command = ['-m', 'experiments.continual_physics.evaluate_native', '--checkpoint', str(checkpoint),
                    '--manifest', str(manifest), '--output', str(output),
                    '--initial-rng-root', str(args.output / 'native' / 'initial_rng'),
                    '--state-mode', 'carry', '--feedback-delay-ticks', '2',
                    '--max-actions', '400', '--max-decisions', '160', '--threads', '1',
                    '--device', 'cpu', '--workers', str(args.native_workers)]
                if disabled:
                    command.append('--disable-recurrent-feedback')
                evaluations.append(dict(name=f'{cohort}/{name}', arguments=command, cohort=cohort,
                                        instances=instances, checkpoint=str(checkpoint),
                                        output=str(output), feedback_enabled=(not name.startswith('full_') and not disabled)))
    return dict(question='Can workspace computation improve task transitions when it feeds the recurrent history?',
        evidence_control='Same-capacity direct public-ledger input; separates computation feedback from observation access',
        sources=['workspace', 'evidence'], trainings=jobs, evaluations=evaluations,
        external_requests_per_training=204800, total_new_external_requests=819200,
        additional_parameters_per_training=1048576,
        same_weight_control='Compute the learned projection, then discard its result; all other weights retained',
        gpu_start='All four original conditions and the recurrent reference on that GPU exited successfully at their final budget',
        failed_dependency='Exit that GPU candidate job; never substitute an earlier checkpoint',
        max_candidate_jobs_per_gpu=2, max_parallel_native_jobs=args.native_parallel,
        native_workers_per_job=args.native_workers, total_native_executions=1500,
        outcomes=['completion', 'actual native actions including failures', 'elapsed seconds',
                  'inference seconds', 'feedback processing seconds', 'internal activities',
                  'initial continuation', 'first switch', 'switch continuation', 'first return', 'return continuation'],
        selection='All fixed final checkpoints; no choice based on development or confirmation results',
        initialization='Original full initial.pt restores all shared weights; original parent, seed, objective and schedule retained')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--primary', type=Path, default=Path('runs/continual_physics/joint_20261029'))
    parser.add_argument('--reference', type=Path, default=Path('runs/continual_physics/recurrent_reference_20261029'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--native-parallel', type=int, default=2)
    parser.add_argument('--native-workers', type=int, default=6)
    parser.add_argument('--plan-only', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    plan = make_plan(args)
    plan_path = args.output / 'plan.json'
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise ValueError('Do not alter an already saved campaign plan')
    plan_path.write_text(json.dumps(plan, indent=2) + '\n')
    if args.plan_only:
        print(json.dumps({key: plan[key] for key in ('total_new_external_requests', 'total_native_executions')}))
        return
    verified = read_json(args.output / 'initial_equivalence.json', [])
    if {row['seed'] for row in verified} != {231, 232} or not all(
        row['shared_weights_exact'] and row['same_observation_outputs_exact'] and
        row['first_window_external_requests'] == 1024 for row in verified):
        raise ValueError('Complete both original-weight and first-window comparisons before queueing training')
    completion = args.output / 'completed.json'
    completed = read_json(completion, [])
    done = {row['name'] for row in completed if row['exit_code'] == 0}
    jobs = [job for job in plan['trainings'] if job['name'] not in done]
    (args.output / 'queue.json').write_text(json.dumps(dict(pid=os.getpid(), started=time.time(),
                                                        phase='waiting_or_training'), indent=2) + '\n')

    def run_training(job):
        items = [(Path(row['root']), row['name']) for row in job['dependencies']]
        try:
            waiting = dependency_state(items)
            print(json.dumps(dict(name=job['name'], waiting_for=waiting)), flush=True)
            while waiting:
                time.sleep(20)
                waiting = dependency_state(items)
        except RuntimeError as error:
            return dict(name=job['name'], exit_code=1, reason=str(error), stage='dependency')
        if (args.output / job['name']).exists():
            return dict(name=job['name'], exit_code=1, reason='Existing incomplete output requires explicit handling')
        start = time.monotonic()
        with (args.output / f"{job['name']}.log").open('w') as log:
            child = subprocess.Popen([sys.executable, *job['arguments']], stdout=log, stderr=subprocess.STDOUT)
            print(json.dumps(dict(started=job['name'], pid=child.pid, gpu=job['gpu'])), flush=True)
            code = child.wait()
        return dict(name=job['name'], exit_code=code, seconds=time.monotonic()-start)

    with ThreadPoolExecutor(max_workers=4) as executor:
        for future in as_completed([executor.submit(run_training, job) for job in jobs]):
            row = future.result(); completed.append(row)
            completion.write_text(json.dumps(completed, indent=2) + '\n')
            print(json.dumps(row), flush=True)
    if any(row['exit_code'] for row in completed):
        raise SystemExit('A candidate dependency or training failed; native campaign not started')
    evaluation_file = args.output / 'native_completed.json'
    evaluated = read_json(evaluation_file, [])
    done = {row['name'] for row in evaluated if row['exit_code'] == 0}

    def evaluate(job):
        output = Path(job['output'])
        if output.exists():
            return dict(name=job['name'], exit_code=1, reason='Existing incomplete native output requires explicit handling')
        output.parent.mkdir(parents=True, exist_ok=True)
        start = time.monotonic()
        with output.with_suffix('.log').open('w') as log:
            result = subprocess.run([sys.executable, *job['arguments']], stdout=log, stderr=subprocess.STDOUT,
                env={**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'OMP_NUM_THREADS': '1',
                     'MKL_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1'})
        return dict(name=job['name'], exit_code=result.returncode, seconds=time.monotonic()-start)

    for cohort in ('development', 'confirmation'):
        pending = [job for job in plan['evaluations'] if job['cohort'] == cohort and job['name'] not in done]
        with ThreadPoolExecutor(max_workers=args.native_parallel) as executor:
            for future in as_completed([executor.submit(evaluate, job) for job in pending]):
                row = future.result(); evaluated.append(row)
                evaluation_file.write_text(json.dumps(evaluated, indent=2) + '\n')
                print(json.dumps(row), flush=True)
    if any(row['exit_code'] for row in evaluated):
        raise SystemExit('A native execution failed; inspect the recorded condition')
    subprocess.run([sys.executable, '-m', 'experiments.continual_physics.summarize_feedback',
                    '--input', str(args.output)], check=True)
    (args.output / 'finished.json').write_text(json.dumps(dict(trainings=4, native_executions=1500,
                                                            completed=time.time()), indent=2)+'\n')


if __name__ == '__main__':
    main()
