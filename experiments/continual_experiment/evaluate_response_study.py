"""Evaluate fixed final response-family models and their native-task retention."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shutil

from .evaluate_memory import execute, summarize as summarize_native
from .summarize_response_families import summarize as summarize_digital


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--previous', type=Path, required=True)
    parser.add_argument('--updates', type=int, default=256)
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    train = args.root / 'train'
    digital = args.root / 'test'
    native = args.root / 'native'
    digital.mkdir(exist_ok=True)
    native.mkdir(exist_ok=False)
    cohorts = args.root / 'cohorts'
    cohorts.mkdir(exist_ok=True)
    for split in ('development', 'test'):
        target = cohorts / f'{split}.json'
        if not target.exists():
            shutil.copyfile(Path(__file__).parent / 'results' / f'response_families_{split}.json', target)
    digital_jobs, native_jobs, pairs, native_pairs = [], [], [], []
    for seed in (911, 912):
        name = f'parent{seed}'
        directory = args.previous / 'train' / f'memory{seed}'
        if not (digital / name / 'summary.json').exists():
            command = ['-m', 'experiments.continual_experiment.evaluate_response_families',
                '--checkpoint', str(directory / 'initial.pt'), '--manifest', str(cohorts / 'test.json'),
                '--output', str(digital / name), '--batch', '1', '--workers', str(args.workers),
                '--record-predictions']
            digital_jobs.append(dict(name=name, directory=str(directory), endpoint='initial.pt', arguments=command))
    for name in ('belief', 'oracle', 'enumeration'):
        if not (digital / name / 'summary.json').exists():
            command = ['-m', 'experiments.continual_experiment.evaluate_family_reference',
                '--manifest', str(cohorts / 'test.json'), '--output', str(digital / name),
                '--baseline', name, '--selection', 'information', '--workers', str(args.workers)]
            digital_jobs.append(dict(name=name, directory=str(args.previous / 'train/memory911'),
                                     endpoint='initial.pt', arguments=command))
    for seed in (911, 912):
        for condition in ('hypotheses', 'marginal', 'inherited', 'disabled'):
            name = f'{condition}{seed}'
            source = 'hypotheses' if condition == 'disabled' else condition
            directory = train / f'{source}{seed}'
            command = ['-m', 'experiments.continual_experiment.evaluate_response_families',
                '--checkpoint', str(directory / 'last.pt'),
                '--manifest', str(args.root / 'cohorts/test.json'),
                '--output', str(digital / name), '--batch', '1', '--workers', str(args.workers),
                '--record-predictions', '--memory-mode', 'disabled' if condition == 'disabled' else 'evidence']
            digital_jobs.append(dict(name=name, directory=str(directory), endpoint='last.pt', arguments=command))
            if condition == 'disabled':
                continue
            command = ['-m', 'experiments.continual_experiment.evaluate_shared_tasks',
                '--checkpoint', str(directory / 'last.pt'),
                '--manifest', str(args.previous / 'cohort/manifest.json'),
                '--output', str(native / name),
                '--initial-rng-root', str(args.previous / 'native/initial_rng'),
                '--state-sharing', 'separated', '--memory-mode', 'evidence', '--memory-slots', '12',
                '--workers', str(args.workers)]
            native_jobs.append(dict(name=name, directory=str(directory), endpoint='last.pt', arguments=command))
        pairs.extend((f'hypotheses{seed}', f'{other}{seed}')
                     for other in ('marginal', 'inherited', 'parent', 'disabled'))
        native_pairs.extend((f'hypotheses{seed}', f'{other}{seed}')
                            for other in ('marginal', 'inherited', 'parent'))

    # The chemistry comparison above added ledgers without changing the original
    # physics policy. Reuse those exact saved native outcomes in this reference.
    references = []
    for seed in (911, 912):
        original = json.loads((args.previous / 'native' / f'initial{seed}' / 'summary.json').read_text())
        memory = json.loads((args.root / 'parent_memory' / f'parent_memory{seed}' / 'summary.json').read_text())
        chemistry = {s['stream']: s['chemistry'] for s in memory}
        for stream in original:
            stream['chemistry'] = chemistry[stream['stream']]
        target = native / f'parent{seed}'
        target.mkdir()
        (target / 'summary.json').write_text(json.dumps(original, indent=2) + '\n')
        references.append(dict(name=f'parent{seed}', reused='original physics and frozen-parent-plus-memory chemistry'))

    common = dict(training_updates=args.updates, endpoint='fixed final training window; no test selection',
                  repetitions='Two continuations of existing parents, not independent pretraining')
    digital_plan = dict(**common, jobs=digital_jobs, paired_conditions=pairs,
        manifest=str(args.root / 'cohorts/test.json'),
        scope='Digital extension; frozen test cohort. Reuse complete parent/reference results when available.',
        reference_selection='Information rule selected on separate development streams; fixed for this evaluation')
    native_plan = dict(**common, jobs=native_jobs, references=references, paired_conditions=native_pairs,
        manifest=str(args.previous / 'cohort/manifest.json'),
        scope='Existing primary native cohort, shared saved initializations; retention and transfer comparison')
    (digital / 'plan.json').write_text(json.dumps(digital_plan, indent=2) + '\n')
    (native / 'plan.json').write_text(json.dumps(native_plan, indent=2) + '\n')

    def run_digital():
        execute(digital_plan, digital, train, parallel_conditions=3)
        summarize_digital(digital, json.loads(Path(digital_plan['manifest']).read_text()), pairs)

    def run_native():
        execute(native_plan, native, train, parallel_conditions=2)
        summarize_native(native, {**native_plan, 'jobs': native_jobs + references})

    with ThreadPoolExecutor(2) as pool:
        futures = [pool.submit(run_digital), pool.submit(run_native)]
        for future in futures:
            future.result()


if __name__ == '__main__':
    main()
