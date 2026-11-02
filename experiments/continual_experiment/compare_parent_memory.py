"""Compare the frozen response parents with and without twelve task ledgers."""
import argparse
import json
from pathlib import Path

from .evaluate_memory import execute, summarize


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--previous', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--workers', type=int, default=12)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    jobs, references, pairs = [], [], []
    for seed in (911, 912):
        name = f'parent_memory{seed}'
        directory = args.previous / 'train' / f'memory{seed}'
        command = ['-m', 'experiments.continual_experiment.evaluate_shared_tasks',
            '--checkpoint', str(directory / 'initial.pt'),
            '--manifest', str(args.previous / 'cohort/manifest.json'),
            '--output', str(args.output / name),
            '--initial-rng-root', str(args.previous / 'native/initial_rng'),
            '--state-sharing', 'separated', '--memory-mode', 'evidence',
            '--memory-slots', '12', '--chemistry-only', '--workers', str(args.workers)]
        jobs.append(dict(name=name, directory=str(directory), endpoint='initial.pt', arguments=command))
        for label, source in (
            (f'initial{seed}', args.previous / 'native' / f'initial{seed}' / 'summary.json'),
            (f'continued_memory{seed}', args.previous / 'capacity_native' / f'capacity12_{seed}' / 'summary.json')):
            target = args.output / label
            target.mkdir()
            (target / 'summary.json').write_text(source.read_text())
            references.append(dict(name=label, reused_summary=str(source)))
        pairs.extend([(name, f'initial{seed}'), (f'continued_memory{seed}', name)])
    plan = dict(jobs=jobs, references=references, paired_conditions=pairs,
        manifest=str(args.previous / 'cohort/manifest.json'), training_updates=0,
        comparison='Frozen parent with twelve ledgers; same predictor and within-task state; fixed evidence reader',
        cohort='Existing primary cohort, exact shared saved initialization; no new independent cohort',
        purpose='Separate adding experimental memory from continuation training')
    (args.output / 'plan.json').write_text(json.dumps(plan, indent=2) + '\n')
    execute(plan, args.output, args.previous / 'train', parallel_conditions=2)
    summarize(args.output, {**plan, 'jobs': jobs + references})


if __name__ == '__main__':
    main()
