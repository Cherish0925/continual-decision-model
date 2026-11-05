"""Summarize completed native-action training and fixed-weight execution tests."""
import argparse
import json
from pathlib import Path


def read_lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def summarize(root):
    conditions = {}
    for descriptor in sorted(root.glob('*/condition.json')):
        folder = descriptor.parent
        condition = json.loads(descriptor.read_text())
        name = condition['route'] + '-' + condition['mode']
        if name in conditions:
            raise ValueError(f'Duplicate condition: {name}')
        updates = read_lines(folder / 'updates.jsonl')
        training = read_lines(folder / 'training.jsonl')
        evaluations = {}
        for split, expected in [('evaluation-seeded', 120), ('development-greedy', 8),
                                ('development-sampled', 8)]:
            path = folder / split
            rows = json.loads((path / 'cases.json').read_text())
            if len(rows) != expected or any(row['status'] != 'evaluated' for row in rows):
                raise ValueError(f'Incomplete evaluation: {name}/{split}')
            run = json.loads((path / 'run.json').read_text())
            action_count = invalid = repetitions = adjacent = 0
            for row in rows:
                actions = read_lines(path / row['id'] / 'actions.jsonl')
                if len(actions) != row['steps']:
                    raise ValueError(f'Incomplete actions: {name}/{split}/{row["id"]}')
                action_count += len(actions)
                invalid += sum(not event['result'].get('success', False) for event in actions)
                repetitions += sum(a['action'] == b['action'] for a, b in zip(actions, actions[1:]))
                adjacent += max(0, len(actions) - 1)
            evaluations[split] = {
                'cases': len(rows), 'successes': sum(row['success'] for row in rows),
                'mean_progress': sum(row['progress'] for row in rows) / len(rows),
                'actions': action_count, 'unsuccessful_native_actions': invalid,
                'consecutive_identical_actions': repetitions, 'adjacent_action_pairs': adjacent,
                'object_rng_seed': run['object_rng_seed'], 'case_results': rows,
            }
        conditions[name] = {
            **condition, 'training_episodes': len(training),
            'training_interactions': updates[-1]['interactions'],
            'optimizer_updates': updates[-1]['update'],
            'training_seconds': updates[-1]['elapsed_seconds'],
            'training_successes': sum(row['completed_successfully'] for row in training),
            'evaluations': evaluations,
        }
    if len(conditions) != 4:
        raise ValueError('Expected the four declared autonomous/basic-operation conditions')
    return {
        'training_seed': 301, 'comparison': 'Two initialization routes, each with CDM and GRU',
        'features': 'Fixed lexical/numeric features; no pretrained encoder',
        'official_execution': 'Greedy action selection, 120 cases, official action limits',
        'development_execution': {
            'environment_seed': 3001, 'themes': 8, 'difficulty': 'Easy',
            'sampled_temperature': 1.0, 'sampling_seed': 301,
            'purpose': 'Exploratory fixed-weight comparison after observing repeated greedy actions',
        },
        'scope': 'Instance splits; scientific-law disjointness is not established',
        'conditions': conditions,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.runs)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    for name, row in result['conditions'].items():
        print(name, {key: value['successes'] for key, value in row['evaluations'].items()})


if __name__ == '__main__':
    main()
