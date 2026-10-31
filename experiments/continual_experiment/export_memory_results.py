"""Export the complete experimental-memory comparison and its paired figure."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from experiments.physical_belief.summarize_native import paired
from .evaluate_memory import summarize


def groups(rows):
    return {'all': rows, 'new': [r for r in rows if r['exposure'] == 'new'],
        'return': [r for r in rows if r['exposure'] == 'return'],
        'long_return': [r for r in rows if (r['intervening_tasks'] or 0) >= 4],
        'untrained_composition': [r for r in rows if r['training_regime'] == 'untrained_named_composition']}


def statistics(rows):
    return {label: dict(instances=len(values), completed=sum(r['completed_successfully'] for r in values),
        **{'mean_' + key: float(np.mean([r[key] for r in values]))
           for key in ('native_actions', 'inference_seconds', 'seconds')})
        for label, values in groups(rows).items() if values}


def comparison(left, right):
    a, b = groups(left), groups(right)
    return {key: paired(a[key], b[key]) for key in a if a[key]}


def plot(result, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 9, 'font.family': 'DejaVu Sans',
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.spines.left': False, 'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.1), sharey=True, sharex=True, layout='constrained')
    names = ['Learned memory, continuation 1', 'Learned memory, continuation 2', 'Bayesian memory']
    colors = ['#235789', '#3D8178', '#9B6C35']
    for ax, subset, title in zip(axes, ('new', 'return'), ('New relationships', 'Returning relationships')):
        comparisons = [result['comparisons'][f'memory{seed}:no_memory{seed}']['chemistry'][subset]
                       for seed in (911, 912)]
        comparisons.append(result['bayesian_comparison'][subset])
        ax.axvline(0, color='#777777', linewidth=.8)
        for y, (value, color) in enumerate(zip(comparisons, colors)):
            mean = value['mean_action_difference']
            lo, hi = value['mean_action_difference_95ci']
            ax.errorbar(mean, 2-y, xerr=[[mean-lo], [hi-mean]], fmt='o', markersize=5,
                        color=color, capsize=3, linewidth=1.4)
        ax.set_title(title, loc='left', fontweight='bold', pad=12)
        ax.set_yticks([2, 1, 0], names)
        ax.tick_params(axis='y', length=0)
        ax.grid(axis='x', color='#E6E8EB', linewidth=.6)
        ax.set_axisbelow(True)
        ax.set_ylim(-.6, 2.6)
    fig.supxlabel('Change in native operations (memory − matched no-memory reference)', fontsize=9)
    for extension in ('pdf', 'svg', 'png'):
        fig.savefig(output / f'experimental_memory.{extension}', dpi=180)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output', type=Path, default=Path('experiments/continual_experiment/results'))
    args = p.parse_args()
    native = args.input / 'native'
    plan = json.loads((native / 'plan.json').read_text())
    additional = json.loads((native / 'memory_controls_plan.json').read_text())
    plan['jobs'] += additional['jobs']
    plan['additional_controls'] = {k: v for k, v in additional.items() if k != 'jobs'}
    result = summarize(native, plan)
    cohort = json.loads(Path(plan['manifest']).read_text())
    expected_count = sum(len(s['chemistry']) for s in cohort['streams'])
    reference = {}
    for name in ('bayes_memory', 'bayes_reset'):
        streams = json.loads((args.input / 'references' / name / 'summary.json').read_text())
        rows = [r for s in streams for r in s['chemistry']]
        if len(rows) != expected_count or any(r['error'] for r in rows):
            raise ValueError('Incomplete Bayesian reference')
        reference[name] = rows
        result['conditions'][name] = {'chemistry': statistics(rows)}
    result['bayesian_comparison'] = comparison(reference['bayes_memory'], reference['bayes_reset'])
    for seed in (911, 912):
        rows = [r for s in json.loads((native / f'memory{seed}' / 'summary.json').read_text()) for r in s['chemistry']]
        result['comparisons'][f'memory{seed}:bayes_memory'] = {'chemistry': comparison(rows, reference['bayes_memory'])}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'experimental_memory.json').write_text(json.dumps(result, indent=2) + '\n')
    with (native / 'instances.csv').open() as file:
        reader = csv.DictReader(file); fieldnames = reader.fieldnames; flat = list(reader)
    for name, rows in reference.items():
        flat.extend({key: ({'condition': name, 'task': 'chemistry', **row}).get(key) for key in fieldnames} for row in rows)
    with (args.output / 'experimental_memory_instances.csv').open('w') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames); writer.writeheader(); writer.writerows(flat)
    manifest = Path(plan['manifest'])
    (args.output / 'experimental_memory_cohort.json').write_text(manifest.read_text())
    plot(result, args.output)
    print(json.dumps({name: {domain: values['all'] for domain, values in tasks.items()}
                      for name, tasks in result['conditions'].items()}, indent=2))


if __name__ == '__main__':
    main()
