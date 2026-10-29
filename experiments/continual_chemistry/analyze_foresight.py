"""Describe executed planning decisions on the fixed development cohort."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary', type=Path, default=Path('runs/continual_physics/chemistry_planning/development/summary.json'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text())
    index, groups, readings, differences = {}, defaultdict(list), defaultdict(list), []
    for row in summary['episodes']:
        if row['condition'] in {'completion', 'two_step'}:
            index.setdefault((row['difficulty'], row['seed']), {})[row['condition']] = row
    for (difficulty, seed), conditions in sorted(index.items()):
        sequences = {}
        for mode, row in conditions.items():
            raw = args.summary.parent/mode/row['directory']/'experiments.jsonl'
            events = [json.loads(line) for line in raw.read_text().splitlines()]
            sequences[mode] = [event for event in events if 'component_counts' in event]
            for event in sequences[mode]:
                probability = [math.exp(value) for value in event['model_outcome_logits']]
                total = sum(probability)
                probability = [value/total for value in probability]
                readings[mode].append({'nll': -math.log(max(probability[event['rust_after']], 1e-30)),
                    'p_complete': probability[0], 'complete': event['rust_after'] == 0,
                    'cost_error': abs(math.expm1(event['model_progress_cost_prediction'][1])-event['native_actions'])})
        a, b = conditions['completion'], conditions['two_step']
        delta = b['native_actions']-a['native_actions']
        groups[a['recipe_group']].append(delta)
        split = next((i for i, (x, y) in enumerate(zip(sequences['completion'], sequences['two_step']))
                      if x['component_counts'] != y['component_counts']), None)
        differences.append({'difficulty': difficulty, 'seed': seed, 'recipe_group': a['recipe_group'],
            'first_different_experiment': split, 'native_action_difference': delta})
    predictions = {}
    for mode, values in readings.items():
        confident = [row for row in values if row['p_complete'] > .9]
        predictions[mode] = {'executed_experiments': len(values),
            'mean_return_nll': sum(row['nll'] for row in values)/len(values),
            'native_cost_mae': sum(row['cost_error'] for row in values)/len(values),
            'predicted_completion_above_0_9': {'count': len(confident),
                'completed': sum(row['complete'] for row in confident),
                'mean_predicted': sum(row['p_complete'] for row in confident)/len(confident) if confident else None}}
    result = {'source': str(args.summary), 'cohort': 'original 30 development instances; fixed joint753 and topK6',
        'interpretation': 'On-policy predictions use different selected histories; their score difference is not a matched-prefix causal comparison.',
        'instances': differences, 'paired_recipe_groups': [
            {'recipe_group': name, 'instances': len(values), 'mean_native_action_difference': sum(values)/len(values)}
            for name, values in sorted(groups.items())],
        'prediction_on_executed_choices': predictions}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(predictions))


if __name__ == '__main__': main()
