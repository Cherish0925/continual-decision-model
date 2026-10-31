"""Export the fixed-endpoint comparison of action-trained memory applicability."""
import argparse
import csv
import json
from pathlib import Path

from .analyze_memory_predictions import predictions, summarize as prediction_summary
from .evaluate_memory import summarize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path,
                        default=Path('experiments/continual_experiment/results'))
    args = parser.parse_args()
    native = args.input / 'decision_native'
    plan = json.loads((native / 'plan.json').read_text())
    result = summarize(native, plan)
    manifest = json.loads(Path(plan['manifest']).read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'memory_decisions.json').write_text(json.dumps(result, indent=2) + '\n')
    (args.output / 'memory_decisions_instances.csv').write_text((native / 'instances.csv').read_text())
    (args.output / 'memory_decisions_cohort.json').write_text(json.dumps(manifest, indent=2) + '\n')

    forecasts, flat, training = {}, [], []
    for seed in (911, 912):
        for kind in ('parent', 'decision', 'prediction'):
            name = f'{kind}{seed}'
            rows = predictions(native / name, manifest)
            forecasts[name] = prediction_summary(rows)
            flat.extend(dict(condition=name, **row) for row in rows)
            if kind == 'parent':
                continue
            root = args.input / 'decision_train' / name
            finished = json.loads((root / 'finished.json').read_text())
            last = json.loads((root / 'training.jsonl').read_text().splitlines()[-1])
            training.append(dict(condition=name, **finished,
                chemical_returns=last['chemical_activity_learning']['returned'],
                chemical_tasks_ended=last['chemical_activity_learning']['episodes_ended']))
    description = dict(
        method='Each policy: cached forecasts before the actual scored return; unchanged chosen history',
        uncertainty='Paired whole-stream bootstrap within each policy; different policies choose different experiments',
        fixed_reference='Current log prior 1, old total log prior 0, unit cumulative evidence coefficient',
        results=forecasts)
    (args.output / 'memory_decisions_predictions.json').write_text(json.dumps(description, indent=2) + '\n')
    (args.output / 'memory_decisions_training.json').write_text(json.dumps(training, indent=2) + '\n')
    with (args.output / 'memory_decisions_predictions.csv').open('w') as file:
        writer = csv.DictWriter(file, fieldnames=list(flat[0]))
        writer.writeheader(); writer.writerows(flat)
    print(json.dumps(result['conditions'], indent=2))


if __name__ == '__main__':
    main()
