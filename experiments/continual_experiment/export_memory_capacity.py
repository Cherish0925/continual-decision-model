"""Export archive-capacity comparisons and effects at each return interval."""
import argparse
import json
from pathlib import Path

from .export_memory_results import comparison, statistics


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output', type=Path,
                   default=Path('experiments/continual_experiment/results'))
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    results = {}
    for name in ('capacity_native', 'capacity_confirmation'):
        root = args.input / name
        result = json.loads((root / 'comparisons.json').read_text())
        references = ('initial', 'no_memory') if name == 'capacity_native' else ('reference',)
        reference_root = args.input / ('native' if name == 'capacity_native' else 'decision_native')
        result['reference_sources'] = {}
        for seed in (911, 912):
            left = f'capacity12_{seed}'
            rows = [r for s in json.loads((root / left / 'summary.json').read_text()) for r in s['chemistry']]
            for label in references:
                right = f'{label}{seed}'
                path = reference_root / right / 'summary.json'
                baseline = [r for s in json.loads(path.read_text()) for r in s['chemistry']]
                result['reference_sources'][right] = str(path)
                result['conditions'][right] = {'chemistry': statistics(baseline)}
                result['comparisons'][left + ':' + right] = {'chemistry': comparison(rows, baseline)}
        results[name] = result
        (args.output / f'memory_{name}.json').write_text(json.dumps(result, indent=2) + '\n')
        (args.output / f'memory_{name}_instances.csv').write_text((root / 'instances.csv').read_text())

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 9, 'font.family': 'DejaVu Sans',
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.spines.left': False, 'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    fig, axes = plt.subplots(1, 2, figsize=(8.1, 3.4), sharex=True, sharey=True, layout='constrained')
    for ax, name, title in zip(axes, results, ('Initial cohort', 'New instances')):
        ax.axvline(0, color='#777777', linewidth=.8)
        for seed, color, offset in ((911, '#235789', .10), (912, '#3D8178', -.10)):
            rows = results[name]['return_lags'][f'capacity12_{seed}:capacity6_{seed}']
            for index, lag in enumerate((2, 4, 6)):
                row = rows[str(lag)]
                mean = row['mean_action_difference']
                lo, hi = row['mean_action_difference_95ci']
                ax.errorbar(mean, index + offset, xerr=[[mean-lo], [hi-mean]],
                    fmt='o', markersize=4.5, capsize=3, color=color,
                    label=f'Continuation {seed-910}' if index == 0 else None)
        ax.set_title(title, loc='left', fontweight='bold', pad=12)
        ax.set_yticks([0, 1, 2], ['2 intervening tasks', '4 intervening tasks', '6 intervening tasks'])
        ax.tick_params(axis='y', length=0)
        ax.grid(axis='x', color='#E6E8EB', linewidth=.6)
        ax.set_axisbelow(True)
        ax.set_ylim(-.6, 2.6)
    axes[0].legend(loc='lower left', frameon=False)
    fig.supxlabel('Change in native operations (12 retained task ledgers − 6)', fontsize=9)
    for suffix in ('pdf', 'svg', 'png'):
        fig.savefig(args.output / f'memory_capacity.{suffix}', dpi=180)
    plt.close(fig)
    print(json.dumps({name: result['conditions'] for name, result in results.items()}, indent=2))


if __name__ == '__main__':
    main()
