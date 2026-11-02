"""Plot fixed-cohort performance and the paired action-representation comparison."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    data = json.loads(args.input.read_text())
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9,
        'axes.titlesize': 10, 'axes.labelsize': 9, 'axes.spines.top': False,
        'axes.spines.right': False, 'svg.fonttype': 'none', 'pdf.fonttype': 42})
    confirmation = 'core911' in data['conditions']
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 5.0), gridspec_kw={'width_ratios': [1.1, 1]})
    colors = {911: '#23648e', 912: '#b45d24'}
    labels = ([('Selector removed', 'core'), ('Inherited', 'inherited'), ('Frozen parent', 'parent')]
              if confirmation else [('Hypotheses', 'hypotheses'), ('Marginal', 'marginal'),
              ('Inherited', 'inherited'), ('Selector removed*', 'no_selector'),
              ('Frozen parent', 'parent'), ('Direct replay', 'direct')])
    ax = axes[0]
    for y, (label, key) in enumerate(labels):
        for seed, offset in ((911, -.13), (912, .13)):
            item = data['conditions'][f'{key}{seed}']['all']
            ax.scatter(item['operations'], y + offset, c=colors[seed], s=25, zorder=3)
            ax.annotate(f"{item['completed']}/{item['tasks']}",
                (item['operations'], y + offset), xytext=(5, 0), textcoords='offset points',
                va='center', fontsize=8, color=colors[seed])
    item = data['conditions']['belief']['all']
    ax.scatter(item['operations'], len(labels), color='#645582', s=35, marker='D', zorder=3)
    ax.annotate(f"{item['completed']}/{item['tasks']}", (item['operations'], len(labels)),
                xytext=(5, 0), textcoords='offset points', va='center', fontsize=8, color='#645582')
    ax.set_yticks(range(len(labels) + 1), [v[0] for v in labels] + ['Known-formula reference'])
    ax.invert_yaxis()
    maximum = max(data['conditions'][f'{key}{s}']['all']['operations'] for _, key in labels for s in colors)
    ax.set_xlim(0, max(maximum, item['operations']) * 1.36)
    ax.set_xlabel('Mean operations, including failed tasks')
    ax.set_title('A  Independent confirmation' if confirmation else 'A  Primary comparison', loc='left', pad=12)
    ax.grid(axis='x', color='#e5e8eb', linewidth=.6)
    ax.set_axisbelow(True)

    groups = [('all', 'All tasks'), ('group:new_parameters', 'Independent streams'),
              ('group:wider_parameters', 'Wider parameter range'),
              ('group:heldout_combination', 'Withheld combination'),
              ('exposure:new', 'New within stream'), ('exposure:return', 'Returning relationship'),
              ('exposure:changed', 'Changed relationship')]
    ax = axes[1]
    for seed, offset in ((911, -.12), (912, .12)):
        source = data['comparisons'][f'core{seed}:parent{seed}' if confirmation
                                     else f'hypotheses{seed}:marginal{seed}']
        for y, (key, label) in enumerate(groups):
            item = source[key]
            mean = item['mean_difference'][0]
            low, high = item['interval95'][0]
            ax.plot([low, high], [y + offset] * 2, color=colors[seed], linewidth=1.4)
            ax.scatter(mean, y + offset, color=colors[seed], s=20, zorder=3)
    ax.axvline(0, color='#929aa1', linewidth=.9, linestyle='--')
    ax.set_yticks(range(len(groups)), [x[1] for x in groups])
    ax.invert_yaxis()
    ax.set_xlabel('Operations: continued core minus frozen parent' if confirmation
                  else 'Operations: hypotheses minus marginal')
    ax.set_title('B  Joint continuation and task efficiency' if confirmation
                 else 'B  Effect of individual predictions', loc='left', pad=12)
    ax.grid(axis='x', color='#e5e8eb', linewidth=.6)
    ax.set_axisbelow(True)
    handles = [plt.Line2D([0], [0], marker='o', color=color, linewidth=0,
                         label=f'Continuation {i + 1}') for i, color in enumerate(colors.values())]
    fig.legend(handles=handles, loc='lower center', ncol=2, frameon=False, bbox_to_anchor=(.53, .035))
    note = ('Fixed models; new task streams. ' if confirmation else '*Post-hoc selector removal. ')
    fig.text(.02, .015, note + 'Intervals resample complete task streams (95%). Counts show completion.',
             fontsize=8, color='#555e66')
    fig.tight_layout(rect=(0, .10, 1, 1), w_pad=2.6)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for extension in ('pdf', 'svg', 'png'):
        fig.savefig(args.output.with_suffix('.' + extension), dpi=180, bbox_inches='tight')
    svg = args.output.with_suffix('.svg')
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines()) + '\n')


if __name__ == '__main__':
    main()
