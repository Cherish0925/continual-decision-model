"""Compare archive capacities after identifying eviction in the primary cohort."""
import argparse
import json
from pathlib import Path

from experiments.physical_belief.summarize_native import paired
from .evaluate_memory import execute, summarize


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--fresh-cohort', type=Path,
                   help='Repeat both capacities on a different fixed cohort, instead of reusing primary references')
    p.add_argument('--initial-rng-root', type=Path,
                   help='Reuse the exact saved environment initialization of the reference conditions')
    p.add_argument('--workers', type=int, default=8)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    jobs, references, pairs = [], [], []
    manifest = args.fresh_cohort or args.input / 'cohort/manifest.json'
    initial_rng = args.initial_rng_root or args.input / 'native/initial_rng'
    for seed in (911, 912):
        name, ref = f'capacity12_{seed}', f'capacity6_{seed}'
        directory = args.input / 'train' / f'memory{seed}'
        command = ['-m', 'experiments.continual_experiment.evaluate_shared_tasks',
            '--checkpoint', str(directory / 'last.pt'),
            '--manifest', str(manifest),
            '--output', str(args.output / name),
            '--initial-rng-root', str(initial_rng),
            '--state-sharing', 'separated', '--memory-mode', 'evidence', '--memory-slots', '12',
            '--chemistry-only', '--workers', str(args.workers)]
        jobs.append(dict(name=name, directory=str(directory), endpoint='last.pt', arguments=command))
        if args.fresh_cohort:
            reference_command = command.copy()
            reference_command[reference_command.index('--memory-slots') + 1] = '6'
            reference_command[reference_command.index('--output') + 1] = str(args.output / ref)
            jobs.append(dict(name=ref, directory=str(directory), endpoint='last.pt', arguments=reference_command))
        else:
            source = args.input / 'native' / f'evidence{seed}' / 'summary.json'
            (args.output / ref).mkdir()
            (args.output / ref / 'summary.json').write_text(source.read_text())
            references.append(dict(name=ref, reused_summary=str(source)))
        pairs.append((name, ref))
    plan = dict(jobs=jobs, references=references, paired_conditions=pairs,
        training_updates=200, manifest=str(manifest), initial_rng_root=str(initial_rng),
        motivation='Six intervening tasks evict an earlier relationship from a six-task ring',
        scope=('Different fixed native cohort; same simulator families and frozen primary weights'
               if args.fresh_cohort else
               'Post-hoc comparison on the completed primary cohort, not an independent confirmation'),
        controls='Same weights and fixed evidence reader; 6 versus 12 task ledgers; same total old-context prior mass',
        changes='More retained ledgers also change the mixture of old forecasts; no label-based deduplication')
    (args.output / 'plan.json').write_text(json.dumps(plan, indent=2) + '\n')
    execute(plan, args.output, args.input / 'train', parallel_conditions=2)
    result = summarize(args.output, {**plan, 'jobs': jobs + references})
    lags = {}
    for left, right in pairs:
        a, b = ([r for s in json.loads((args.output / name / 'summary.json').read_text())
                 for r in s['chemistry']] for name in (left, right))
        lags[left + ':' + right] = {
            str(lag): paired([r for r in a if r.get('intervening_tasks') == lag],
                             [r for r in b if r.get('intervening_tasks') == lag])
            for lag in (2, 4, 6)}
    result['return_lags'] = lags
    (args.output / 'comparisons.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result['conditions'], indent=2))


if __name__ == '__main__':
    main()
