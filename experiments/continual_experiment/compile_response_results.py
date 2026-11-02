"""Export complete response-family comparisons and their fixed task definitions."""
import argparse
import csv
import json
from pathlib import Path
import shutil

from .evaluate_memory import summarize as summarize_native
from .summarize_response_families import summarize


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, default=Path(__file__).parent / 'results')
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    pairs = [(f'hypotheses{s}', f'{other}{s}') for s in (911, 912)
             for other in ('marginal', 'inherited', 'parent', 'disabled', 'direct', 'no_selector')]
    pairs += [(f'no_selector{s}', f'{other}{s}') for s in (911, 912)
              for other in ('marginal', 'inherited', 'parent')]
    pairs += [(f'parent{s}', f'direct{s}') for s in (911, 912)]
    summarize(args.root / 'test', json.loads((args.root / 'cohorts/test.json').read_text()), pairs)
    confirmation = json.loads((args.root / 'confirmation/plan.json').read_text())
    summarize(args.root / 'confirmation', json.loads((args.root / 'cohorts/confirmation.json').read_text()),
              confirmation['paired_conditions'])
    native = json.loads((args.root / 'native/plan.json').read_text())
    native['jobs'] += native['references'] + [dict(name=f'no_selector{s}') for s in (911, 912)]
    native['paired_conditions'] += [(f'no_selector{s}', f'{other}{s}') for s in (911, 912)
                                    for other in ('hypotheses', 'parent', 'marginal', 'inherited')]
    native['followups'] = json.loads((args.root / 'selector_effect_plan.json').read_text())
    summarize_native(args.root / 'native', native)
    for source, name in (('test', 'response_families'), ('confirmation', 'response_families_confirmation'),
                         ('native', 'response_families_native')):
        for src, dst in (('comparisons.json', name + '.json'), ('instances.csv', name + '_instances.csv')):
            if dst.endswith('.csv'):
                (args.output / dst).write_text((args.root / source / src).read_text())
            else:
                shutil.copyfile(args.root / source / src, args.output / dst)
    cohort = json.loads((args.root / 'cohorts/confirmation.json').read_text())
    # One task per line keeps the complete fixed laws readable without a large
    # indentation expansion. No values or identities are removed.
    lines = ['{', f'  "seed": {cohort["seed"]},', '  "split": ' + json.dumps(cohort['split']) + ',', '  "streams": [']
    for i, stream in enumerate(cohort['streams']):
        lines += ['    {', '      "name": ' + json.dumps(stream['name']) + ',',
                  '      "group": ' + json.dumps(stream['group']) + ',', '      "tasks": [']
        lines += ['        ' + json.dumps(task, separators=(',', ':')) + (',' if j + 1 < len(stream['tasks']) else '')
                  for j, task in enumerate(stream['tasks'])]
        lines += ['      ]', '    }' + (',' if i + 1 < len(cohort['streams']) else '')]
    lines += ['  ]', '}']
    (args.output / 'response_families_confirmation_cohort.json').write_text('\n'.join(lines) + '\n')
    curves = []
    for file in sorted((args.root / 'train').glob('*/training.jsonl')):
        finished = json.loads((file.parent / 'finished.json').read_text())
        if finished['updates'] != 256:
            raise ValueError('Export requires the declared final endpoints')
        for line in file.read_text().splitlines():
            row = json.loads(line)
            curves.append(dict(condition=file.parent.name, **{k: row[k] for k in (
                'update', 'loss', 'physical_loss', 'chemical_loss', 'chemical_nll', 'gradient_norm',
                'physical_requests', 'chemical_returns', 'chemical_episodes_ended', 'seconds', 'peak_allocated_gib')}))
    with (args.output / 'response_families_training.csv').open('w') as file:
        writer = csv.DictWriter(file, fieldnames=list(curves[0]), lineterminator='\n')
        writer.writeheader(); writer.writerows(curves)
    print('Exported primary, independent confirmation, native retention, fixed tasks and all training curves.')


if __name__ == '__main__':
    main()
