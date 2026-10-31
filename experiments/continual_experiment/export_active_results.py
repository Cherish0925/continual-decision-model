"""Publish completed active-feedback tables and observed activity summaries."""
import argparse
import json
import math
from pathlib import Path
import shutil


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', required=True, type=Path)
    p.add_argument('--output', type=Path, default=Path('experiments/continual_experiment/results'))
    p.add_argument('--prefix', default='active_feedback')
    args = p.parse_args()
    comparisons = json.loads((args.input / 'comparisons.json').read_text())
    activity = {}
    for condition in comparisons['conditions']:
        streams = json.loads((args.input / condition / 'summary.json').read_text())
        rows = [row for stream in streams for row in stream['chemistry']]
        counts = [sum(row['activity_counts'][i] for row in rows) for i in range(6)]
        n = high = clean = 0
        nll = confidence = 0.
        for file in (args.input / condition).glob('*/chemistry*/experiments.jsonl'):
            for line in file.open():
                row = json.loads(line)
                if row.get('activity') == 'leave':
                    continue
                n += 1
                nll -= row['model_outcome_logits'][row['rust_after']]
                probability = math.exp(row['model_outcome_logits'][0])
                if probability > .9:
                    high += 1
                    confidence += probability
                    clean += row['rust_after'] == 0
        activity[condition] = dict(instances=len(rows), returned_experiments=n,
            activities=dict(zip(('act', 'refine', 'retrieve', 'learn', 'consolidate', 'wait'), counts)),
            mean_response_update_norm=sum(row['response_adaptation_norm'] for row in rows) / len(rows),
            selected_return_nll=nll / n,
            confident_clean_predictions=high, confident_clean_successes=clean,
            mean_confident_clean_prediction=confidence / high if high else None)
    args.output.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.input / 'comparisons.json', args.output / f'{args.prefix}.json')
    (args.output / f'{args.prefix}_instances.csv').write_text((args.input / 'instances.csv').read_text())
    result = dict(scope='Actual deployed choices on each model\'s own trajectory; prediction losses are not same-prefix causal comparisons',
                  conditions=activity)
    if comparisons['protocol'].get('study') == 'response':
        result['response_update_schedule'] = ('Four fixed response-fit iterations at each prediction readout; '
            'LEARN and CONSOLIDATE counts refer to the separate progress/cost update, not response fitting.')
    (args.output / f'{args.prefix}_activity.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'conditions': len(activity), 'executions': sum(
        c[t]['instances'] for c in comparisons['conditions'].values() for t in ('physics', 'chemistry'))}))


if __name__ == '__main__':
    main()
