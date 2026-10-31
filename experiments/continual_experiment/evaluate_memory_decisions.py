"""Native evaluation of action-return credit into memory applicability."""
import argparse
import json
from pathlib import Path

from .evaluate_memory import execute, summarize


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--training-root', type=Path, required=True)
    p.add_argument('--parent-root', type=Path, required=True)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--workers', type=int, default=8)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    jobs, pairs = [], []
    for seed in (911, 912):
        for label, root, source, mode, chemical_only, updates in (
            ('parent', args.parent_root, 'memory', 'learned', False, 200),
            ('reference', args.parent_root, 'no_memory', 'disabled', False, 200),
            ('decision', args.training_root, 'decision', 'learned', False, 100),
            ('prediction', args.training_root, 'prediction', 'learned', False, 100),
            ('disabled', args.training_root, 'decision', 'disabled', True, 100),
            ('evidence', args.training_root, 'decision', 'evidence', True, 100)):
            name, directory = f'{label}{seed}', root / f'{source}{seed}'
            command = ['-m', 'experiments.continual_experiment.evaluate_shared_tasks',
                '--checkpoint', str(directory / 'last.pt'), '--manifest', str(args.manifest),
                '--output', str(args.output / name), '--initial-rng-root', str(args.output / 'initial_rng'),
                '--state-sharing', 'separated', '--memory-mode', mode, '--workers', str(args.workers)]
            if chemical_only:
                command += ['--chemistry-only']
            jobs.append(dict(name=name, directory=str(directory), endpoint='last.pt',
                             required_updates=updates, arguments=command))
        pairs += [(f'decision{seed}', f'{name}{seed}') for name in
                  ('prediction', 'parent', 'reference', 'disabled', 'evidence')]
        pairs.append((f'prediction{seed}', f'parent{seed}'))
    plan = dict(jobs=jobs, training_updates=100, manifest=str(args.manifest), paired_conditions=pairs,
        question='Does action-return credit reduce the new-task penalty while retaining the return benefit?',
        cohort='Fresh native seed ranges; same previously studied simulator families',
        conditions='Two fixed continuations, prediction-only control, frozen parents and memory interventions',
        uncertainty='Paired whole-stream bootstrap; both training continuations reported separately')
    (args.output / 'plan.json').write_text(json.dumps(plan, indent=2) + '\n')
    execute(plan, args.output, args.training_root)
    summarize(args.output, plan)


if __name__ == '__main__':
    main()
