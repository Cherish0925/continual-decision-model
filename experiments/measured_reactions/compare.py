"""Compare paired response curves across seeds and experimental groups."""
import argparse
import json
from pathlib import Path

import numpy as np

from .study import ALL_LEARNED, FEEDBACK_CONTROLS, REFERENCES


def grouped_curves(path, laboratory):
    with np.load(path, allow_pickle=False) as f:
        labels = f['conditions'].tolist()
        if laboratory == 'roche':
            values, screen, start = f['best'], f['screen'], f['start']
            family, notebook = f['family'], f['notebook']
            groups = sorted(set(family))
            means = []
            for group in groups:
                means.append(np.mean([values[:, (family == group) & (notebook == n)].mean(1)
                                      for n in sorted(set(notebook[family == group]))], axis=0))
        else:
            values, screen, start = f[f'{laboratory}_best'], f[f'{laboratory}_screen'], f[f'{laboratory}_start']
            groups = sorted(set(screen))
            means = [values[:, screen == group].mean(1) for group in groups]
        if len(set(zip(screen, start))) != len(screen):
            raise ValueError('Repeated screen/start pair')
        if not np.allclose(values[:, :, 0], values[0, :, 0], atol=1e-5):
            raise ValueError('Initial measurements must match')
        curves = np.stack(means, axis=1)
    return dict(zip(labels, curves)), groups


def paired_interval(difference, draws=10000):
    """Resample training seeds and reaction families (or external screens)."""
    rng = np.random.default_rng(49031)
    n, g = difference.shape
    seeds = rng.integers(n, size=(draws, n))
    groups = rng.integers(g, size=(draws, g))
    sampled = difference[seeds[:, :, None], groups[:, None, :]].mean((1, 2))
    return dict(mean=float(difference.mean()), interval_95=np.quantile(sampled, [.025, .975]).tolist(),
                positive_seeds=int((difference.mean(1) > 0).sum()), seeds=n, groups=g,
                per_seed=difference.mean(1).tolist())


def compare(inputs, seeds, output):
    result, plot_data = {}, {}
    for lab in ('roche', 'ahneman', 'perera'):
        curves, groups = {}, None
        for root in inputs:
            filename = 'development_curves.npz' if lab == 'roche' else 'external_curves.npz'
            source, keys = grouped_curves(root / filename, lab)
            if groups is not None and groups != keys:
                raise ValueError('Experimental groups differ between comparisons')
            groups = keys
            for name, values in source.items():
                if name in curves and not np.allclose(curves[name], values, atol=1e-4):
                    raise ValueError(f'Duplicate method with different outcomes: {name}')
                curves[name] = values
        modes = [m for m in ALL_LEARNED if all(f'{m}{s}' in curves for s in seeds)]
        scores = {m: np.stack([curves[f'{m}{s}'] for s in seeds]) for m in modes}
        scores.update({m: curves[m][None] for m in REFERENCES})
        comparisons = {}
        reference = {'roche': 'kernel_local', 'ahneman': 'kernel_diverse', 'perera': 'kernel_ucb'}[lab]
        for mode in modes:
            comparisons[f'{mode}_vs_classical_reference'] = paired_interval(
                scores[mode][:, :, -1] - scores[reference][:, :, -1])
            if mode in FEEDBACK_CONTROLS:
                hidden = np.stack([curves[f'{mode}{s}_without_readings'] for s in seeds])
                comparisons[f'{mode}_feedback'] = paired_interval((scores[mode] - hidden)[:, :, -1])
        for left, right in (('conditional', 'cdm'), ('relational', 'conditional'),
                            ('relational', 'relational_gru'), ('relational', 'relational_kernel'),
                            ('relational', 'relational_blind')):
            if left in scores and right in scores:
                comparisons[f'{left}_vs_{right}'] = paired_interval((scores[left] - scores[right])[:, :, -1])
        result[lab] = dict(grouping='reaction family' if lab == 'roche' else 'substrate screen',
                           classical_reference=reference,
                           scores={m: dict(best_8=float(v[:, :, -1].mean()),
                                           per_seed=v[:, :, -1].mean(1).tolist()) for m, v in scores.items()},
                           comparisons=comparisons)
        plot_data[lab] = scores
    output.mkdir(parents=True, exist_ok=True)
    (output / 'comparison.json').write_text(json.dumps(result, indent=2) + '\n')
    return result, plot_data


def plot(data, output):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    names = {'cdm': 'CDM', 'conditional': 'Conditional CDM', 'relational': 'Relational CDM',
             'relational_gru': 'Relational GRU', 'relational_kernel': 'Learned kernel'}
    colors = dict(zip(names, ('#727780', '#cf7728', '#168477', '#3b69b1', '#8e65a5')))
    fig, axes = plt.subplots(1, 3, figsize=(11.8, 3.5))
    for ax, (lab, scores) in zip(axes, data.items()):
        for mode, name in names.items():
            if mode in scores:
                ax.plot(np.arange(1, 9), scores[mode].mean((0, 1)), label=name, color=colors[mode], linewidth=1.8)
        ref = {'roche': 'kernel_local', 'ahneman': 'kernel_diverse', 'perera': 'kernel_ucb'}[lab]
        ax.plot(np.arange(1, 9), scores[ref].mean((0, 1)), '--', color='#202124', label='Classical reference')
        ax.set_title({'roche': 'Roche development', 'ahneman': 'Ahneman', 'perera': 'Perera'}[lab])
        ax.set_xlabel('Measurements requested')
        ax.set_xticks((1, 2, 4, 6, 8))
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', alpha=.18)
    axes[0].set_ylabel('Best measured response')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, frameon=False)
    fig.tight_layout(rect=(0, .16, 1, 1))
    fig.savefig(output / 'comparison.svg')
    fig.savefig(output / 'comparison.pdf')
    plt.close(fig)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, action='append', required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--seeds', default='101,102,103,104,105,106,107,108')
    p.add_argument('--plot', action='store_true')
    args = p.parse_args()
    summary, curves = compare(args.input, [int(s) for s in args.seeds.split(',')], args.out)
    if args.plot:
        plot(curves, args.out)
