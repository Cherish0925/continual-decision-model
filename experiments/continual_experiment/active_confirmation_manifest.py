"""Select untouched native A-B-A chemistry instances using generator identity.

This evaluator selects repetition structure, never a model input or target.
It runs before checkpoint evaluation and never reads model outcomes.
"""
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
import os
from pathlib import Path

from experiments.continual_chemistry.exploration_plan import identify


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--workers', type=int, default=12)
    args = p.parse_args()
    if os.environ.get('PYTHONHASHSEED') != '0':
        raise ValueError('Use PYTHONHASHSEED=0 to match the native interface')
    args.output.mkdir(parents=True, exist_ok=False)
    jobs = [(difficulty, seed, str(args.output / 'generation'))
            for difficulty in ('Normal', 'Challenge') for seed in range(230000, 230240)]
    with ProcessPoolExecutor(args.workers, mp_context=multiprocessing.get_context('spawn')) as pool:
        identified = list(pool.map(identify, jobs, chunksize=4))
    pools = defaultdict(lambda: defaultdict(list))
    for row in identified:
        pools[row['difficulty']][row['recipe_group']].append(row)
    used, streams = set(), []
    previous = json.loads(Path('experiments/continual_experiment/shared_confirmation.json').read_text())
    for index, old in enumerate(previous['streams']):
        first, other = ('Normal', 'Challenge') if index % 2 == 0 else ('Challenge', 'Normal')

        def choose(difficulty, count):
            options = [(group, [r for r in rows if (difficulty, r['seed']) not in used])
                       for group, rows in sorted(pools[difficulty].items())]
            options = [(group, rows) for group, rows in options if len(rows) >= count]
            _, rows = options[(index // 2) % len(options)]
            selection = rows[:count]
            used.update((difficulty, r['seed']) for r in selection)
            return [{k: r[k] for k in ('difficulty', 'seed', 'recipe_group')} for r in selection]

        a, b = choose(first, 4), choose(other, 2)
        streams.append(dict(name=f'active_{index:02d}', seeds=[s + 110000 for s in old['seeds']],
            segment_lengths=[2, 2, 2], chemistry=[a[0], a[1], b[0], b[1], a[2], a[3]]))
    manifest = dict(purpose='Fixed native comparisons after joint active-feedback training',
        scope='new instances of known physical and chemical simulator families',
        chemical_schedule='two A, two B, two A tasks; distinct seeds; evaluator-only recipe identity',
        budgets=dict(physics=400, chemistry=1000), streams=streams)
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'streams': len(streams), 'physics_instances': 6 * len(streams),
                      'chemistry_instances': len(used)}), flush=True)


if __name__ == '__main__':
    main()
