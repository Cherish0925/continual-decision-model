"""Compare released v0.17 policies on response functions modulo ingredient relabeling."""
import argparse
import csv
import json
from collections import Counter, defaultdict
from functools import lru_cache
from itertools import permutations
from pathlib import Path

import torch

from experiments.continual_chemistry.recipes import primitive_recipes
from experiments.continual_chemistry.response_families import response_levels
from experiments.continual_experiment.response_split_overlap import RESULTS, read_corpus, training_streams


@lru_cache(None)
def recipe_maps(n):
    """For each permutation of the ingredients, map every six-part-menu recipe."""
    recipes = [tuple(recipe) for recipe in primitive_recipes(n, 6)]
    index = {recipe: i for i, recipe in enumerate(recipes)}
    return tuple(tuple(index[tuple(r[j] for j in p)] for r in recipes)
                 for p in permutations(range(n)))


def compare(data, root, cohort, records):
    torch.set_num_threads(1)
    corpus, _, _ = read_corpus(data)
    _, possible, _ = training_streams(root)  # Includes possible open episodes in all 12 runs.
    training = corpus.union(*possible.values())

    @lru_cache(None)
    def classify(n, levels):
        orbit = {bytes(levels[j] for j in mapping) for mapping in recipe_maps(n)}
        label = ('indexed_match' if (n, levels) in training else
                 'permutation_only_match' if any((n, v) in training for v in orbit) else 'unmatched')
        return label, (n, min(orbit))

    menus, labels = {}, {}
    classes, streams = defaultdict(set), defaultdict(set)
    positions = Counter()
    for stream in json.loads(cohort.read_text())['streams']:
        if stream['group'] == 'renamed_inputs' or stream['max_parts'] != 6:
            continue
        encountered = set()
        for i, task in enumerate(stream['tasks']):
            n = task['law']['components']
            if n not in (3, 4):
                continue
            if n not in menus:
                menus[n] = torch.tensor(primitive_recipes(n, 6))
            table = bytes(response_levels(menus[n], task['law']).tolist())
            label, key = classify(n, table)
            strata = [label]
            if label == 'unmatched' and key not in encountered:
                strata.append('unmatched_first_in_stream')
            encountered.add(key)
            pos = (stream['name'], i)
            if pos in labels:
                raise ValueError(f'Duplicate position in test schedule: {pos}')
            labels[pos] = strata
            for s in strata:
                positions[s] += 1
                classes[s].add(key)
                streams[s].add(pos[0])

    totals = defaultdict(lambda: defaultdict(Counter))
    seen = defaultdict(set)
    with records.open() as f:
        for row in csv.DictReader(f):
            pos = (row['stream'], int(row['task_index']))
            if row['group'] == 'renamed_inputs' or pos not in labels:
                continue
            model = row['condition']
            if pos in seen[model]:
                raise ValueError(f'Duplicate instance for {model}: {pos}')
            seen[model].add(pos)
            for s in labels[pos]:
                totals[s][model].update(completed=int(row['completed'] == 'True'),
                                        operations=int(row['native_actions']))
    if not seen or any(keys != labels.keys() for keys in seen.values()):
        raise ValueError('Incomplete per-model test results')
    return {
        'scope': 'Primary three- and four-ingredient menus, at most six parts. Exact intrinsic '
                 'responses on all candidate recipes, allowing any one permutation of ingredients. '
                 'Training pool: predictor training data and all twelve continuation training '
                 'streams, including possible open episodes. Earlier parent training and validation '
                 'exposure are outside this pool. First in stream means first occurrence of that '
                 'response equivalence class in the evaluated stream, not statistically independent trials.',
        'same_menu_positions': len(labels),
        'strata': {s: {'positions': count, 'distinct_classes': len(classes[s]), 'streams': len(streams[s]),
                       'models': {m: {'completed': a['completed'],
                                      'mean_operations_including_failures': round(a['operations'] / count, 3)}
                                  for m, a in sorted(totals[s].items())}}
                   for s, count in sorted(positions.items())},
    }


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--training-root', type=Path, required=True)
    p.add_argument('--cohort', type=Path, default=RESULTS / 'joint_belief_test_cohort.json')
    p.add_argument('--rows', type=Path, default=RESULTS / 'joint_belief_instances.csv')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    result = compare(a.data, a.training_root, a.cohort, a.rows)
    a.output.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print({k: v['positions'] for k, v in result['strata'].items()})
