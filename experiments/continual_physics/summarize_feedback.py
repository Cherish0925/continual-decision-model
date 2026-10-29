"""Summarize the fixed feedback comparison by task-transition phase."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from experiments.continual_experiment.summarize_state_controls import read_rows, contrast
from experiments.continual_experiment.summarize_neural_states import PHASES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads((args.input / 'plan.json').read_text())
    summary = dict(question=plan['question'], cohorts={},
                   bootstrap='10,000 complete-stream resamples; training repetitions kept separate',
                   checkpoint_selection='none; every final fixed-budget checkpoint',
                   disabled_feedback='Projection computed then discarded; other weights unchanged')
    instance_rows = []
    for cohort in ('development', 'confirmation'):
        jobs = [job for job in plan['evaluations'] if job['cohort'] == cohort]
        methods = {job['name'].split('/', 1)[1]: read_rows(Path(job['output']) / 'summary.json') for job in jobs}
        keys = set(next(iter(methods.values())))
        if len(keys) != jobs[0]['instances'] or any(set(rows) != keys for rows in methods.values()):
            raise ValueError(f'Incomplete or unmatched evaluation: {cohort}')
        if any(row.get('error') for rows in methods.values() for row in rows.values()):
            raise ValueError('An execution error cannot count as task failure')
        groups = {}
        for name, rows in methods.items():
            groups[name] = {}
            for phase, predicate in PHASES.items():
                selected = [row for row in rows.values() if predicate(row)]
                groups[name][phase] = dict(instances=len(selected),
                    completed=sum(row['completed_successfully'] for row in selected),
                    mean_actions=float(np.mean([row['native_actions'] for row in selected])),
                    mean_seconds=float(np.mean([row['seconds'] for row in selected])),
                    mean_inference_seconds=float(np.mean([row['timing']['inference_seconds'] for row in selected])),
                    mean_feedback_seconds=float(np.mean([row['timing']['feedback_seconds'] for row in selected])),
                    internal_activities=np.asarray([row['internal_activity_counts'] for row in selected]).sum(0).tolist())
            for key in sorted(rows):
                row = rows[key]
                instance_rows.append(dict(cohort=cohort, condition=name, **{field: row[field] for field in
                    ('stream', 'seed', 'episode_index', 'segment_role', 'position_in_segment',
                     'completed_successfully', 'native_actions', 'seconds')}))
        comparisons = {}
        for seed in (231, 232):
            for source in ('workspace', 'evidence'):
                name = f'{source}_seed{seed}'
                for control in (f'full_seed{seed}', name + '_disabled'):
                    comparisons[f'{name}_minus_{control}'] = {
                        phase: contrast(methods, {name: 1, control: -1}, predicate,
                            np.random.default_rng(57210 + (seed - 231) * len(PHASES) + index))
                        for index, (phase, predicate) in enumerate(PHASES.items())}
            comparisons[f'workspace_minus_evidence_seed{seed}'] = {
                phase: contrast(methods, {f'workspace_seed{seed}': 1, f'evidence_seed{seed}': -1}, predicate,
                    np.random.default_rng(57210 + (seed - 231) * len(PHASES) + index))
                for index, (phase, predicate) in enumerate(PHASES.items())}
        summary['cohorts'][cohort] = dict(methods=groups, paired_comparisons=comparisons)
    (args.input / 'feedback_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    with (args.input / 'feedback_instances.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, list(instance_rows[0])); writer.writeheader(); writer.writerows(instance_rows)
    print(json.dumps(dict(native_executions=len(instance_rows), cohorts=list(summary['cohorts']))))


if __name__ == '__main__':
    main()
