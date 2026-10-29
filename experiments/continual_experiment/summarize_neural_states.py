"""Paired recurrent/workspace carry controls on the original v0.10 streams."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .summarize_state_controls import read_rows, contrast


PHASES = {
    'all': lambda r: True,
    'initial_later': lambda r: r['segment_role'] == 'initial' and r['position_in_segment'] > 0,
    'switch_first': lambda r: r['segment_role'] == 'switch' and r['position_in_segment'] == 0,
    'switch_later': lambda r: r['segment_role'] == 'switch' and r['position_in_segment'] > 0,
    'return_first': lambda r: r['segment_role'] == 'return' and r['position_in_segment'] == 0,
    'return_later': lambda r: r['segment_role'] == 'return' and r['position_in_segment'] > 0,
}
LABELS = {'reset': 'Reset', 'h_only': 'Carry h', 'z_only': 'Carry z', 'neural_only': 'Carry h and z'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--previous', type=Path,
                        default=Path('runs/continual-experiment/state_controls_v010/full'))
    parser.add_argument('--reference', type=Path,
                        default=Path('runs/discoveryworld/continual_experiment/native'))
    parser.add_argument('--output', type=Path, default=Path('experiments/continual_experiment/results'))
    args = parser.parse_args()
    methods, sources = {}, {}
    for seed in (201, 202):
        for mode in LABELS:
            source = ((args.reference / f'full_reset_seed{seed}') if mode == 'reset' else
                      (args.previous / f'neural_only_seed{seed}') if mode == 'neural_only' else
                      (args.input / f'{mode}_seed{seed}'))
            name = f'{mode}_seed{seed}'
            sources[name] = str(source)
            methods[name] = read_rows(source / 'summary.json')
    keys = set(next(iter(methods.values())))
    if len(keys) != 184 or any(set(rows) != keys for rows in methods.values()):
        raise ValueError('Each condition must cover the same original 184 instances')
    if any(row.get('error') for rows in methods.values() for row in rows.values()):
        raise ValueError('Execution errors cannot count as task failures')
    reproduction = {}
    for name, rows in methods.items():
        seed = name.rsplit('seed', 1)[1]
        reference = methods[f'reset_seed{seed}']
        first = [key for key in keys if rows[key]['episode_index'] == 0]
        fields = ('native_actions', 'measurement_requests', 'test_requests', 'completed_successfully')
        mismatches = [key for key in first if any(rows[key][f] != reference[key][f] for f in fields)]
        if mismatches:
            raise ValueError(f'Untreated first episodes do not reproduce: {name}: {mismatches}')
        reproduction[name] = {'first_episodes': len(first), 'mismatches': []}
    grouped = {}
    for name, rows in methods.items():
        grouped[name] = {}
        for phase, predicate in PHASES.items():
            selected = [row for row in rows.values() if predicate(row)]
            grouped[name][phase] = {
                'instances': len(selected),
                'completed': sum(row['completed_successfully'] for row in selected),
                'mean_actions': float(np.mean([row['native_actions'] for row in selected])),
                'mean_measurements': float(np.mean([row['measurement_requests'] for row in selected])),
                'mean_tests': float(np.mean([row['test_requests'] for row in selected])),
                'activity_counts': np.asarray([row['internal_activity']['counts'] for row in selected]).sum(0).tolist(),
            }
    comparisons = {'h_minus_reset': {'h_only': 1, 'reset': -1},
                   'z_minus_reset': {'z_only': 1, 'reset': -1},
                   'hz_minus_z': {'neural_only': 1, 'z_only': -1},
                   'hz_minus_h': {'neural_only': 1, 'h_only': -1},
                   'interaction': {'neural_only': 1, 'h_only': -1, 'z_only': -1, 'reset': 1}}
    agreement = {}
    for seed in (201, 202):
        agreement[str(seed)] = {}
        for left, right in (('h_only', 'neural_only'), ('z_only', 'reset')):
            a, b = methods[f'{left}_seed{seed}'], methods[f'{right}_seed{seed}']
            fields = ('native_actions', 'measurement_requests', 'test_requests', 'completed_successfully')
            changed = [key for key in sorted(keys) if any(a[key][field] != b[key][field] for field in fields)]
            agreement[str(seed)][f'{left}_versus_{right}'] = {'instances': len(keys), 'changed': changed}
    contrasts = {}
    for seed in (201, 202):
        contrasts[str(seed)] = {}
        for label, coefficients in comparisons.items():
            coefficients = {f'{name}_seed{seed}': value for name, value in coefficients.items()}
            contrasts[str(seed)][label] = {
                phase: contrast(methods, coefficients, predicate,
                                np.random.default_rng(42581 + (seed - 201) * len(PHASES) + index))
                for index, (phase, predicate) in enumerate(PHASES.items())}
    summary = {'source_release': 'v0.10.0', 'source': sources, 'methods': grouped,
               'contrasts': contrasts, 'first_episode_reproduction': reproduction,
               'per_instance_outcome_agreement': agreement,
               'bootstrap': {'unit': 'entire ordered stream', 'repetitions': 10000, 'seed': 42581},
               'scope': 'Post-hoc fixed-checkpoint mechanism controls; no new training or checkpoint selection'}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'neural_state_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    fields = ('condition', 'stream', 'seed', 'episode_index', 'segment_role', 'position_in_segment',
              'completed_successfully', 'native_actions', 'measurement_requests', 'test_requests')
    with (args.output / 'neural_state_instances.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fields); writer.writeheader()
        for name, rows in methods.items():
            for key in sorted(rows):
                writer.writerow({'condition': name, **{field: rows[key][field] for field in fields[1:]}})
    lines = ['# Recurrent state and workspace across task transitions', '',
        'This follow-up separates the GRU state h from the Transformer workspace z. Each single-state condition resets the other neural state, outcome replay and fast/slow adapters at task boundaries. The previous h-and-z-only and full-reset conditions provide the other two cells of the comparison. Model weights, ordered instances, initial simulator randomness, observations, tools and action budgets are unchanged.', '',
        'The new controls execute 736 native tasks using both released v0.10 full models. The original task cohort is reused to explain the previous state effect; these are not independent confirmation results for a new method.', '',
        '| Repetition | Carried state | All tasks: completed | All: actions | Initial later | First switch | Switch later | First return | Return later |',
        '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for seed in (201, 202):
        for mode, label in LABELS.items():
            row = grouped[f'{mode}_seed{seed}']; overall = row['all']
            values = ' | '.join(f'{row[phase]["mean_actions"]:.2f}' for phase in PHASES)
            lines.append(f'| {seed} | {label} | {overall["completed"]}/{overall["instances"]} | {values} |')
    lines += ['', 'Action means include failed tasks. Initial later and switch later denote subsequent fresh instances sharing the current relation. First return denotes the first fresh instance after an intervening relation; return later denotes subsequent instances in that returning segment.', '',
              'For both trained models, h-only and h-and-z carry match on every instance in completion, native action count, measurement count and test count. z-only and full reset also match instance by instance on all four outcomes. Thus the observed cross-task effects follow the retained recurrent state. This does not imply that the workspace is unused within a task or that its internal representations are identical.', '',
              'The trade-off persists across transitions. Carrying h reduces the mean cost of successive same-relation tasks, but increases first-switch and first-return costs. On first-return tasks, repetition 201 completes 19/20 with either h carry or reset; repetition 202 completes 16/20 with h carry and 18/20 with reset. Neither selective reset gives a free improvement that preserves all continuity benefits and removes all transition costs.', '',
              '## Paired transition effects', '',
              'Differences are candidate minus reset. Intervals resample complete ordered streams and retain all instances in each sampled stream. The two training repetitions share the cohort and are reported separately.', '',
              '| Repetition | Carried state | Initial later | First switch | First return | Return later |',
              '| --- | --- | ---: | ---: | ---: | ---: |']
    for seed in (201, 202):
        for label, title in [('h_minus_reset', 'h'), ('z_minus_reset', 'z')]:
            entries = []
            for phase in ('initial_later', 'switch_first', 'return_first', 'return_later'):
                row = contrasts[str(seed)][label][phase]
                lo, hi = row['action_interval_95']
                entries.append(f'{row["action_difference"]:+.2f} [{lo:+.2f}, {hi:+.2f}]')
            lines.append(f'| {seed} | {title} | ' + ' | '.join(entries) + ' |')
    lines += ['', '## Representation and inference', '',
        'The 1,024-dimensional GRU state receives public operation events, including task-local object indices, tool/quantity identifiers, returned measurements, execution feedback and query identity. The eight 1,024-dimensional workspace slots repeatedly attend to that GRU state and the current object–quantity evidence ledger. Their slots are not assigned to particular objects. Action and activity readouts concatenate the normalized GRU vector with the normalized mean workspace vector. Additional REFINE activity updates z alone; the GRU update has no direct input from z or the complete evidence ledger.', '',
        'The released model receives a new initial observation at a task boundary. The initial event has a distinct pattern, so a boundary is implicitly observable, although there is no dedicated boundary input or declared separation between instance-specific and transferable information. Object slots retain their roles: slots 0 and 1 are initial supports, and slots 2 and 3 are targets; the objects and their numerical values change. Reusing these role indices is not itself an implementation error. Obsolete instance context and excessive preference for the recent relation remain hypotheses; these interventions do not identify their encoded content.', '',
        '## Next training comparison', '',
        'A concrete candidate is to let the existing GRU update read the current public evidence and the preceding workspace result. Add zero-initialized projections of an evidence summary and the workspace summary to the received-event embedding before the GRU. Initialization exactly preserves the old policy; training can then use the existing GRU reset/update gates to retain useful prior information and revise current-instance content. Workspace computations can affect the recurrent state at the next received observation, without counting an internal computation as a second observation.', '',
        'Train this feedback connection on continuous streams with fresh objects and coefficients, and propagate future task rewards and received-outcome prediction losses across task boundaries. Any object relabeling must consistently change events, evidence, candidate actions, known/query masks and records; start with permutations within the existing support and target roles. Compare against an equally trained original recurrence on independent development streams, reporting same-relation continuation, first switch, first return, completion and total action cost together. Hidden relation identities and the known A–B–A evaluation schedule are not inputs. This is a proposed experiment, not a demonstrated repair or a novelty claim.', '',
        'All 20 untreated first episodes per condition reproduce the original completion, action, measurement and test counts. Replay and adapter state are reset in every selective neural condition; they can still form and be used within each task.', '',
        '[Per-instance outcomes](neural_state_instances.csv) · [Phase summaries and paired contrasts](neural_state_summary.json) · [Previous state-group control](state_controls.md)', '',
        'Reproduce the new runs with `python -m experiments.continual_experiment.state_controls --state-modes h_only z_only --output runs/continual-experiment/neural_state_controls`. Then run `python -m experiments.continual_experiment.summarize_neural_states --input runs/continual-experiment/neural_state_controls --previous runs/continual-experiment/state_controls_v010/full`. Both original checkpoints and their initial RNG records are required.', '']
    (args.output / 'neural_state_controls.md').write_text('\n'.join(lines))
    print(json.dumps({'methods': grouped, 'reproduction': reproduction}, indent=2))


if __name__ == '__main__':
    main()
