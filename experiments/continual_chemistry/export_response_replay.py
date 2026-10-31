"""Publish completed response-learning comparisons without runtime details."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .summarize import paired


def prediction_pair(left, right):
    assert [(r['episode'], r['position']) for r in left] == [(r['episode'], r['position']) for r in right]
    groups = sorted({r['recipe_group'] for r in left})
    differences = np.asarray([a['updated_nll'] - b['updated_nll'] for a, b in zip(left, right)])
    totals = np.asarray([sum(v for r, v in zip(left, differences) if r['recipe_group'] == g) for g in groups])
    counts = np.asarray([sum(r['recipe_group'] == g for r in left) for g in groups])
    selected = np.random.default_rng(1031).integers(len(groups), size=(10000, len(groups)))
    means = totals[selected].sum(1) / counts[selected].sum(1)
    return dict(mean_difference=float(differences.mean()), interval95=np.quantile(means, [.025, .975]).tolist(),
                bootstrap_unit='named recipe group', predictions=len(left), groups=len(groups))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, default=Path('experiments/continual_chemistry/results'))
    args = p.parse_args()
    result, predictions, instances = {}, [], []
    for phase, study, native in [('recorded', 'study', 'native_baseline'), ('selected', 'active_study', 'active_native')]:
        trained = {}
        for file in sorted((args.root / study).glob('*/development.json')):
            data = json.loads(file.read_text()); name = file.parent.name
            trained[name] = data
            for row in data['predictions']:
                predictions.append(dict(phase=phase, condition=name, **row))
        comparisons = {f'{mode}{seed}:calibration{seed}': prediction_pair(trained[f'{mode}{seed}']['predictions'],
                       trained[f'calibration{seed}']['predictions']) for seed in (911, 912) for mode in ('stored', 'refreshed', 'relation')}
        without = {}
        for seed in (911, 912):
            file = args.root / study / f'relation{seed}' / 'without_update.json'
            if file.exists():
                data = json.loads(file.read_text())
                without[str(seed)] = {k: v for k, v in data.items() if k != 'predictions'}
                comparisons[f'relation{seed}:same_weights_no_update'] = prediction_pair(
                    trained[f'relation{seed}']['predictions'], data['predictions'])
                predictions.extend(dict(phase=phase, condition=f'relation_no_update{seed}', **r) for r in data['predictions'])
        acting = json.loads((args.root / native / 'results.json').read_text())
        for seed in (911, 912):
            left = [r for r in acting['episodes'] if r['condition'] == f'relation{seed}']
            right = [r for r in acting['episodes'] if r['condition'] == f'relation_no_update{seed}']
            if right:
                acting['comparisons'][f'relation{seed}:relation_no_update{seed}'] = paired(left, right)
        columns = ('condition', 'directory', 'difficulty', 'seed', 'recipe_group', 'completed_successfully',
                   'native_actions', 'experiments', 'inference_seconds', 'execution_seconds', 'seconds', 'error')
        instances.extend(dict(phase=phase, **{k: row[k] for k in columns}) for row in acting['episodes'])
        result[phase] = dict(
            training={name: {k: v for k, v in data.items() if k not in ('predictions', 'parent')} for name, data in trained.items()},
            prediction_pairs=comparisons, same_weight_without_update=without,
            native_conditions=acting['conditions'], native_pairs=acting['comparisons'], protocol=acting['protocol'])
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'response_learning.json').write_text(json.dumps(result, indent=2) + '\n')
    for filename, rows in [('response_learning_predictions.csv', predictions), ('response_learning_instances.csv', instances)]:
        with (args.output / filename).open('w') as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]), lineterminator='\n')
            writer.writeheader(); writer.writerows(rows)
    print(json.dumps({phase: dict(trainings=len(data['training']), native=sum(r['instances'] for r in data['native_conditions']))
                      for phase, data in result.items()}))


if __name__ == '__main__':
    main()
