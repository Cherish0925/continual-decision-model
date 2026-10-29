"""Summarize the fixed-weight failure-confidence development intervention."""
import argparse
import csv
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path,
                        default=Path('runs/continual_physics/failure_weight_development'))
    parser.add_argument('--output', type=Path,
                        default=Path('experiments/continual_physics/results/failure_weights_development'))
    args = parser.parse_args()
    conditions = {}
    for name in ('reset232_original', 'reset232_conditioned'):
        rows = [json.loads(path.read_text()) for path in
                sorted((args.root / name).glob('*/episode*/summary.json'))]
        if len(rows) != 30 or any(row['error'] for row in rows):
            raise ValueError('This comparison requires all thirty completed executions per condition')
        conditions[name] = {row['seed']: row for row in rows}
    original, revised = conditions.values()
    if original.keys() != revised.keys():
        raise ValueError('The paired conditions must use the same instances')
    paired = [{
        'stream': original[seed]['stream'], 'seed': seed,
        'original_completed': original[seed]['completed_successfully'],
        'conditioned_completed': revised[seed]['completed_successfully'],
        'original_actions': original[seed]['native_actions'],
        'conditioned_actions': revised[seed]['native_actions'],
        'action_difference': revised[seed]['native_actions'] - original[seed]['native_actions'],
    } for seed in sorted(original)]
    summary = {
        'scope': 'Development intervention on identical fixed task-reset weights; no new training',
        'instances': 30, 'streams': len({row['stream'] for row in paired}),
        'independent_confirmation': False,
        'conditions': {name: {
            'completed': sum(row['completed_successfully'] for row in data.values()),
            'mean_native_actions': sum(row['native_actions'] for row in data.values()) / len(data),
            'mean_seconds': sum(row['seconds'] for row in data.values()) / len(data),
        } for name, data in conditions.items()},
        'mean_action_difference': sum(row['action_difference'] for row in paired) / len(paired),
        'gained_completions': sum(not row['original_completed'] and row['conditioned_completed'] for row in paired),
        'lost_completions': sum(row['original_completed'] and not row['conditioned_completed'] for row in paired),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    with (args.output / 'instances.csv').open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(paired[0]))
        writer.writeheader()
        writer.writerows(paired)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
