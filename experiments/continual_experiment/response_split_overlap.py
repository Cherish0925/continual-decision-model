"""Compare experimental response functions across the released v0.17.0 splits."""
import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import torch

from experiments.continual_chemistry.recipes import primitive_recipes
from experiments.continual_chemistry.response_families import (
    EXPANDED_FAMILIES, TRAIN_FAMILIES, response_levels, sample_law, sample_lifetime,
)


RESULTS = Path(__file__).parent / 'results'


def read_corpus(data):
    """Recover each sampled law, checking all four released measurement fields."""
    tables, targets, sequences = set(), set(), set()
    split_counts = {}
    for split, base_seed in [('train', 733091), ('validation', 833091)]:
        tensors = torch.load(data / (split + '.pt'), map_location='cpu', weights_only=True)
        counts = Counter(laws=0, measurements=0, same_menu_and_responses=0, same_ordered_queries=0)
        for n, dataset in sorted(tensors.items()):
            recipes = primitive_recipes(n)
            q = torch.tensor(recipes)
            base_cost = [6 + sum(r) + sum(x > 0 for x in r) for r in recipes]
            recorded = {k: v.tolist() for k, v in dataset.items()}
            for i, archived in enumerate(recorded['indices']):
                rng = random.Random(base_seed + 100003 * n + i)
                law = sample_law(rng, n, TRAIN_FAMILIES)
                intrinsic = response_levels(q, law).tolist()
                indices = rng.sample(range(len(q)), len(archived))
                before, after, costs, current = [], [], [], 3
                for j, action in enumerate(indices):
                    fresh = j % 6 == 0 or current == 0
                    if fresh:
                        current = 3
                    before.append(current)
                    current = min(current, intrinsic[action])
                    after.append(current)
                    costs.append(base_cost[action] + (14 if fresh else 2) + (4 if current == 0 else 0))
                if (indices, before, after, costs) != (archived, recorded['before'][i],
                                                      recorded['after'][i], recorded['cost'][i]):
                    raise ValueError(f'Measurement mismatch: {split}, {n} ingredients, row {i}')
                table, sequence = (n, bytes(intrinsic)), (n, tuple(indices))
                if split == 'train':
                    tables.add(table)
                    sequences.add(sequence)
                    targets.add((n, tuple(law['target'])))
                else:
                    counts['same_menu_and_responses'] += table in tables
                    counts['same_ordered_queries'] += sequence in sequences
                    if i < 128:  # train_relation_belief logs these 128 rows for each ingredient count.
                        counts['monitored_laws'] += 1
                        counts['monitored_same_menu_and_responses'] += table in tables
                counts['laws'] += 1
                counts['measurements'] += len(indices)
        split_counts[split] = dict(counts) if split == 'validation' else {
            'laws': counts['laws'], 'measurements': counts['measurements'], 'distinct_response_tables': len(tables)}
    return tables, targets, split_counts


def training_streams(root):
    """Replay the closed episodes of each continuation; include possible open tails separately."""
    plan = json.loads((root / 'plan.json').read_text())
    actual, possible, record = {}, {}, {}
    queries = {n: torch.tensor(primitive_recipes(n)) for n in (3, 4)}
    for job in plan['jobs']:
        name, args = job['name'], job['arguments']
        seed = int(args[args.index('--seed') + 1])
        if args[args.index('--chemical-families') + 1] != 'expanded':
            raise ValueError(f'Unexpected training distribution: {name}')
        lanes = defaultdict(list)
        for line in (root / 'train' / name / 'chemical_tasks.jsonl').read_text().splitlines():
            row = json.loads(line)
            lanes[row['lane']].append(row)
        actual[name], possible[name] = set(), set()
        if set(lanes) != set(range(plan['batch'])):
            raise ValueError(f'Missing training lane: {name}')
        for lane, rows in lanes.items():
            rng = random.Random(seed + 91827 + 104729 * lane)
            for i in range(len(rows) + 1):
                if i % 12 == 0:
                    stream = sample_lifetime(rng, EXPANDED_FAMILIES)
                task = stream[i % 12]
                law = task['law']
                n = law['components']
                table = (n, bytes(response_levels(queries[n], law).tolist()))
                if i < len(rows):
                    row = rows[i]
                    if (i, law['family'], task['exposure'], task['intervening_tasks']) != (
                            row['task_index'], row['family'], row['exposure'], row['intervening_tasks']):
                        raise ValueError(f'Training schedule mismatch: {name}, lane {lane}, task {i}')
                    actual[name].add(table)
                else:
                    # The last step need not end an episode. The following entry is an upper bound:
                    # it may already have supplied observations or may never have been started.
                    possible[name].add(table)
        possible[name].update(actual[name])
        record[name] = dict(condition=job['condition'], ended_training_tasks=sum(map(len, lanes.values())),
                            lanes=len(lanes))
        if job['condition'] != 'joint':  # The same predictor checkpoints initialize prediction_only.
            continue
        training_log = root / 'predictors' / f"seed{job['seed']}" / 'training.jsonl'
        validation = {}
        for line in training_log.read_text().splitlines():
            row = json.loads(line)
            if row['update'] in (500, 6000, 12000, 18000, 24000):
                views = row['validation']
                if set(views) != {f'{n}_{c}' for n in (3, 4) for c in (0, 4, 16)}:
                    raise ValueError('Unexpected predictor validation views')
                validation[row['update']] = round(sum(v[0] for v in views.values()) / len(views), 6)
        record[name]['predictor_validation_joint_nll'] = validation
    return actual, possible, record


def compare(data, cohort_path, rows_path, training_root):
    torch.set_num_threads(1)
    tables, targets, corpus = read_corpus(data)
    streams = json.loads(cohort_path.read_text())['streams']
    groups = defaultdict(Counter)
    bucket, test_tables = {}, {}
    ratio_subgroup = {}
    menus = {}
    for stream in streams:
        group = stream['group']
        if group == 'renamed_inputs':
            continue
        stats = groups[group]
        for i, task in enumerate(stream['tasks']):
            law = task['law']
            n, parts = law['components'], stream['max_parts']
            if (n, parts) not in menus:
                menus[n, parts] = torch.tensor(primitive_recipes(n, parts))
            q = menus[n, parts]
            levels = response_levels(q, law)
            same_menu = n in (3, 4) and parts == 6
            seen_table = same_menu and (n, bytes(levels.tolist())) in tables
            stats['tasks'] += 1
            stats['same_menu'] += same_menu
            stats['same_menu_and_responses'] += seen_table
            stats['seen_ordered_primary_target'] += (n, tuple(law['target'])) in targets
            short_solution = bool(((levels == 0) & (q.sum(-1) <= 6)).any())
            if not (levels == 0).any():
                raise ValueError('Test world has no successful recipe')
            stats['solvable_with_six_or_fewer_parts'] += short_solution
            stats['primary_target_uses_all_available_ingredients'] += all(x > 0 for x in law['target'])
            key = (stream['name'], i)
            if key in bucket:
                raise ValueError(f'Duplicate test position: {key}')
            bucket[key] = ('new_menu' if not same_menu else
                           'seen_response_table' if seen_table else 'unmatched_response_table')
            if same_menu:
                test_tables[key] = (n, bytes(levels.tolist()))
            if group == 'new_target_ratios':
                ratio_subgroup[key] = 'short_solution_exists' if short_solution else 'long_recipe_required'

    actual, possible, runs = training_streams(training_root)
    if Counter(v['condition'] for v in runs.values()) != {'joint': 4, 'inherited': 4, 'prediction_only': 4}:
        raise ValueError('Expected four continuations per training condition')
    for name, record in runs.items():
        prior = set() if record['condition'] == 'inherited' else tables
        record['test_matches_with_corpus_if_used_and_ended_tasks'] = sum(
            table in prior or table in actual[name] for table in test_tables.values())
        record['test_matches_including_possible_open_tasks'] = sum(
            table in prior or table in possible[name] for table in test_tables.values())
    joint_union = tables.union(*(possible[name] for name in runs if runs[name]['condition'] == 'joint'))
    union = tables.union(*possible.values())
    unmatched_all = {key for key, table in test_tables.items() if table not in union}
    by_model = defaultdict(lambda: defaultdict(Counter))
    same_unmatched = defaultdict(Counter)
    by_ratio = defaultdict(lambda: defaultdict(Counter))
    seen = defaultdict(set)
    with rows_path.open() as f:
        for row in csv.DictReader(f):
            if row['group'] == 'renamed_inputs':
                continue
            name, key = row['condition'], (row['stream'], int(row['task_index']))
            if key in seen[name]:
                raise ValueError(f'Duplicate task record: {name}, {key}')
            seen[name].add(key)
            entries = [by_model[name][bucket[key]]]
            if key in ratio_subgroup:
                entries.append(by_ratio[name][ratio_subgroup[key]])
            if key in unmatched_all:
                entries.append(same_unmatched[name])
            for agg in entries:
                agg['tasks'] += 1
                agg['completed'] += row['completed'] == 'True'
                agg['operations'] += int(row['native_actions'])
    if not seen or any(keys != set(bucket) for keys in seen.values()):
        raise ValueError('Per-model task records do not match the test cohort')
    for strata in [*by_model.values(), *by_ratio.values(), same_unmatched]:
        for agg in strata.values():
            agg['mean_operations'] = round(agg.pop('operations') / agg['tasks'], 3)
    return dict(reference='Strata and by_model use the released predictor corpus. continuation_training adds '
                          'the streams of all 12 evaluated continuations. Earlier parent training is not reconstructed.',
                matching='Same ingredient count and indexed candidate menu; identical intrinsic categorical '
                         'response on every candidate. No relabeling. Returns inside streams are included.',
                corpus=corpus, primary_streams=sum(s['group'] != 'renamed_inputs' for s in streams),
                primary_tasks=len(bucket), strata=dict(Counter(bucket.values())),
                groups=dict(groups), by_model=dict(by_model), new_target_ratios_by_model=dict(by_ratio),
                continuation_training=runs,
                same_menu_positions=len(test_tables), distinct_same_menu_tables=len(set(test_tables.values())),
                unmatched_against_corpus_and_four_joint_runs=sum(t not in joint_union for t in test_tables.values()),
                common_unmatched_distinct_tables=len({test_tables[k] for k in unmatched_all}),
                common_unmatched_streams=len({k[0] for k in unmatched_all}),
                common_unmatched_same_menu=dict(same_unmatched))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', required=True, type=Path, help='Directory containing released train.pt and validation.pt')
    p.add_argument('--cohort', type=Path, default=RESULTS / 'joint_belief_test_cohort.json')
    p.add_argument('--rows', type=Path, default=RESULTS / 'joint_belief_instances.csv')
    p.add_argument('--training-root', required=True, type=Path,
                   help='Released joint_belief_20261103 directory with plan, predictor logs and chemical task logs')
    p.add_argument('--output', required=True, type=Path)
    args = p.parse_args()
    out = compare(args.data, args.cohort, args.rows, args.training_root)
    args.output.write_text(json.dumps(out, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'corpus': out['corpus'], 'strata': out['strata']}))


if __name__ == '__main__':
    main()
