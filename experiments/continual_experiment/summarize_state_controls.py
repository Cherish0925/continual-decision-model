"""Summarize matched 2x2 task-boundary state interventions."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


GROUPS = {
    'all': lambda r: True,
    'initial_later': lambda r: r['segment_role'] == 'initial' and r['position_in_segment'] > 0,
    'switch_first': lambda r: r['segment_role'] == 'switch' and r['position_in_segment'] == 0,
    'return_first': lambda r: r['segment_role'] == 'return' and r['position_in_segment'] == 0,
    'return_later': lambda r: r['segment_role'] == 'return' and r['position_in_segment'] > 0,
}


def read_rows(path):
    return {(r['stream'], r['seed']): r for stream in json.loads(path.read_text()) for r in stream['episodes']}


def contrast(methods, coefficients, predicate, rng, repetitions=10000):
    names = list(coefficients)
    keys = [key for key, row in methods[names[0]].items() if predicate(row)]
    streams = sorted({key[0] for key in keys})
    totals = np.zeros((len(streams), 3))
    for index, stream in enumerate(streams):
        selected = [key for key in keys if key[0] == stream]
        totals[index, 0] = len(selected)
        for name, coefficient in coefficients.items():
            totals[index, 1] += coefficient * sum(methods[name][key]['native_actions'] for key in selected)
            totals[index, 2] += coefficient * sum(methods[name][key]['completed_successfully'] for key in selected)
    samples = totals[rng.integers(len(streams), size=(repetitions, len(streams)))].sum(1)
    point = totals.sum(0)
    return {'instances': len(keys), 'streams': len(streams),
            'action_difference': point[1] / point[0],
            'action_interval_95': np.quantile(samples[:, 1] / samples[:, 0], [.025, .975]).tolist(),
            'completion_difference_pp': 100 * point[2] / point[0],
            'completion_interval_95_pp': (100 * np.quantile(samples[:, 2] / samples[:, 0], [.025, .975])).tolist()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--reference', type=Path, default=Path('runs/discoveryworld/continual_experiment/native'))
    parser.add_argument('--output', type=Path, default=Path('experiments/continual_experiment/results'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    methods, metadata = {}, {}
    for seed in (201, 202):
        sources = {f'full_seed{seed}': args.reference / f'full_seed{seed}',
                   f'reset_seed{seed}': args.reference / f'full_reset_seed{seed}',
                   **{f'{mode}_seed{seed}': args.input / f'{mode}_seed{seed}'
                      for mode in ('neural_only', 'learning_only')}}
        for name, directory in sources.items():
            methods[name] = read_rows(directory / 'summary.json')
            metadata[name] = {'source': str(directory), 'training_seed': seed,
                              'new_execution': name.startswith(('neural_only', 'learning_only'))}
    keys = set(next(iter(methods.values())))
    if len(keys) != 184 or any(set(rows) != keys for rows in methods.values()):
        raise ValueError('Every method must cover the same 184 original instances')
    if any(row.get('error') for rows in methods.values() for row in rows.values()):
        raise ValueError('A native execution error cannot be summarized as task failure')
    grouped = {}
    for name, rows in methods.items():
        grouped[name] = {}
        for group, predicate in GROUPS.items():
            selected = [r for r in rows.values() if predicate(r)]
            counts = np.asarray([r['internal_activity']['counts'] for r in selected])
            grouped[name][group] = {
                'instances': len(selected), 'completed': sum(r['completed_successfully'] for r in selected),
                'mean_actions': float(np.mean([r['native_actions'] for r in selected])),
                'mean_measurements': float(np.mean([r['measurement_requests'] for r in selected])),
                'mean_tests': float(np.mean([r['test_requests'] for r in selected])),
                'activity_counts': counts.sum(0).tolist()}
    # Reuse the actual complete-stream pairing for every contrast.
    rng, contrasts = np.random.default_rng(42581), {}
    for seed in (201, 202):
        comparisons = {
            'neural_without_learning': {f'neural_only_seed{seed}': 1, f'reset_seed{seed}': -1},
            'learning_without_neural': {f'learning_only_seed{seed}': 1, f'reset_seed{seed}': -1},
            'neural_with_learning': {f'full_seed{seed}': 1, f'learning_only_seed{seed}': -1},
            'learning_with_neural': {f'full_seed{seed}': 1, f'neural_only_seed{seed}': -1},
            'interaction': {f'full_seed{seed}': 1, f'neural_only_seed{seed}': -1,
                            f'learning_only_seed{seed}': -1, f'reset_seed{seed}': 1}}
        contrasts[str(seed)] = {label: {group: contrast(methods, coefficients, predicate, rng)
                                       for group, predicate in GROUPS.items()}
                                for label, coefficients in comparisons.items()}
    # First episodes precede all boundary treatments and must reproduce exactly.
    invariance = {}
    for name, rows in methods.items():
        reference = methods[f'full_seed{metadata[name]["training_seed"]}']
        mismatch = [key for key, row in rows.items() if row['episode_index'] == 0 and
                    any(row[field] != reference[key][field] for field in
                        ('native_actions', 'measurement_requests', 'test_requests', 'completed_successfully'))]
        invariance[name] = {'first_episodes': sum(r['episode_index'] == 0 for r in rows.values()),
                            'mismatches': mismatch}
    summary = {'source_release': 'v0.10.0', 'source': metadata, 'methods': grouped,
               'contrasts': contrasts, 'first_episode_reproduction': invariance,
               'bootstrap': {'unit': 'entire ordered task stream', 'repetitions': 10000, 'seed': 42581},
               'scope': 'Post-hoc state-mechanism intervention on the original cohort; no retraining or model selection'}
    (args.output / 'state_controls_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    fields = ('condition', 'stream', 'seed', 'episode_index', 'segment_role', 'position_in_segment',
              'completed_successfully', 'native_actions', 'measurement_requests', 'test_requests')
    with (args.output / 'state_controls_instances.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fields); writer.writeheader()
        for name, rows in methods.items():
            for key in sorted(rows):
                writer.writerow({'condition': name, **{field: rows[key][field] for field in fields[1:]}})
    lines = ['# What persists between scientific tasks', '',
        'This comparison separates the continuing neural state (`h`, `z`) from the outcome replay buffer and fast/slow prediction adapters in the released v0.10 model. The numerical physical posterior is recomputed from each instance; it is not part of the carried state.', '',
        'Both trained full models use their original ordered 184-instance cohort, initial simulator RNG, public observations, tools, and 400-action limit. Four new conditions execute 736 instances. Full carry and full reset reuse the original results. These are post-hoc mechanism controls, without training or checkpoint selection.', '',
        '| Repetition | Boundary state | Completed | Actions / instance | First return: completed | First return: actions |',
        '| --- | --- | ---: | ---: | ---: | ---: |']
    for seed in (201, 202):
        for mode, label in [('reset', 'Reset all'), ('neural_only', 'Carry h/z only'),
                            ('learning_only', 'Carry replay and adapters only'), ('full', 'Carry all')]:
            row = grouped[f'{mode}_seed{seed}']; all_row, returned = row['all'], row['return_first']
            lines.append(f'| {seed} | {label} | {all_row["completed"]}/{all_row["instances"]} | {all_row["mean_actions"]:.2f} | {returned["completed"]}/{returned["instances"]} | {returned["mean_actions"]:.2f} |')
    lines += ['', '## First-return contrasts', '',
        'Differences are added state minus the corresponding reset control. Negative action differences mean fewer native operations. Intervals resample complete streams; the two training repetitions are reported separately.', '',
        '| Repetition | Added state and existing state | Action difference [95% interval] | Completion difference, pp [95% interval] |',
        '| --- | --- | ---: | ---: |']
    labels = {'neural_without_learning': 'h/z, replay/adapters reset',
              'learning_without_neural': 'Replay/adapters, h/z reset',
              'neural_with_learning': 'h/z, replay/adapters carried',
              'learning_with_neural': 'Replay/adapters, h/z carried',
              'interaction': 'Interaction of the two carried groups'}
    for seed in (201, 202):
        for comparison, label in labels.items():
            row = contrasts[str(seed)][comparison]['return_first']; low, high = row['action_interval_95']; cl, ch = row['completion_interval_95_pp']
            lines.append(f'| {seed} | {label} | {row["action_difference"]:+.2f} [{low:+.2f}, {high:+.2f}] | {row["completion_difference_pp"]:+.1f} [{cl:+.1f}, {ch:+.1f}] |')
    lines += ['', 'The interaction is full − neural-only − learning-only + reset. A group intervention measures its total effect through subsequent decisions and observations; it does not identify an individual hidden variable or internal computation.', '',
        'Pending requests, terminal feedback, activity counts, and observation counts follow the original carry protocol. Only the stated neural tensors or replay/adaptation tensors are reset. Each task starts with a fresh public measurement ledger. The first episode in each stream precedes the intervention; its completion and action counts are checked against the original run.', '',
        '[Per-instance data](state_controls_instances.csv) · [Phase summaries and paired intervals](state_controls_summary.json)', '',
        'Reproduce with `python -m experiments.continual_experiment.state_controls --output runs/continual-experiment/state_controls`, then `python -m experiments.continual_experiment.summarize_state_controls --input runs/continual-experiment/state_controls`. The original release checkpoints and initial RNG records are required.', '']
    (args.output / 'state_controls.md').write_text('\n'.join(lines))
    print(json.dumps({'methods': grouped, 'first_episode_reproduction': invariance}, indent=2))


if __name__ == '__main__':
    main()
