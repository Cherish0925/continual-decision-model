"""Choose new native task streams before evaluating experimental memory.

Generator identities establish novelty and recurrence for analysis only. The
policy receives the ordinary public task interface and its returned results.
"""
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
import os
from pathlib import Path

from experiments.continual_chemistry.exploration_plan import identify, proportion_structure


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--training-data', type=Path,
                   default=Path('runs/continual_physics/chemistry_exploration/manifest.json'))
    p.add_argument('--workers', type=int, default=20)
    p.add_argument('--instances-per-difficulty', type=int, default=600)
    p.add_argument('--streams', type=int, default=20)
    args = p.parse_args()
    if os.environ.get('PYTHONHASHSEED') != '0':
        raise ValueError('Use PYTHONHASHSEED=0')
    args.output.mkdir(parents=True, exist_ok=False)
    training = json.loads(args.training_data.read_text())['episodes']
    known = {r['recipe_group'] for r in training if r['split'] == 'train'}
    known_structures = {proportion_structure(group) for group in known}
    jobs = [(difficulty, seed, str(args.output / 'generation'))
            for difficulty in ('Normal', 'Challenge')
            for seed in range(310000, 310000 + args.instances_per_difficulty)]
    with ProcessPoolExecutor(args.workers, mp_context=multiprocessing.get_context('spawn')) as pool:
        identified = list(pool.map(identify, jobs, chunksize=4))
    (args.output / 'identified.json').write_text(json.dumps(identified, indent=2) + '\n')
    pools = defaultdict(lambda: defaultdict(list))
    for row in identified:
        pools[row['difficulty']][row['recipe_group']].append(row)
    order = [0, 1, 2, 0, 3, 4, 1, 5, 0, 2, 5, 3]
    counts = {i: order.count(i) for i in set(order)}
    streams, used = [], set()
    for index in range(args.streams):
        difficulty = ('Normal', 'Challenge')[index % 2]
        eligible = {group: [r for r in rows if (difficulty, r['seed']) not in used]
                    for group, rows in pools[difficulty].items()}
        groups = sorted(group for group, rows in eligible.items() if len(rows) >= 3)
        # Rotate choices, alternating unseen and trained named compositions.
        unseen, seen = ([g for g in groups if g not in known], [g for g in groups if g in known])
        offset = index // 2
        unseen = unseen[offset % max(1, len(unseen)):] + unseen[:offset % max(1, len(unseen))]
        seen = seen[offset % max(1, len(seen)):] + seen[:offset % max(1, len(seen))]
        selected = []
        for position in range(6):
            preferred = unseen if position % 2 == 0 else seen
            remaining = [g for g in preferred + groups if g not in selected]
            if not remaining:
                raise ValueError(f'Too few distinct groups in {difficulty}')
            selected.append(remaining[0])
        histories = {i: iter(eligible[group][:counts[i]]) for i, group in enumerate(selected)}
        chemistry, last = [], {}
        for position, group_index in enumerate(order):
            row = next(histories[group_index])
            group = row['recipe_group']
            used.add((difficulty, row['seed']))
            chemistry.append({k: row[k] for k in ('difficulty', 'seed', 'recipe_group')} | {
                'exposure': 'return' if group_index in last else 'new',
                'intervening_tasks': position - last[group_index] - 1 if group_index in last else None,
                'training_regime': 'trained_named_composition' if group in known else 'untrained_named_composition',
                'untrained_proportion_structure': proportion_structure(group) not in known_structures})
            last[group_index] = position
        streams.append(dict(name=f'memory_{index:02d}',
            seeds=[390000 + index * 1000 + position * 5 + group % 5
                   for position, group in enumerate(order)], chemistry=chemistry))
    manifest = dict(
        purpose='Future experiment utility after variable interference and recurrence',
        schedule=order, streams=streams,
        model_inputs='Public observations, returned experiments, public ingredient availability; no task identity',
        split='New native instances; named compositions may be held out, but cosine response family is unchanged',
        budgets=dict(physics=400, chemistry=1000),
        selection='Fixed seed ranges and recurrence before model outcomes; no selection by success')
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(dict(streams=len(streams), chemistry_instances=len(used),
        return_instances=sum(t['exposure'] == 'return' for s in streams for t in s['chemistry']),
        untrained_compositions=sum(t['training_regime'] == 'untrained_named_composition'
                                   for s in streams for t in s['chemistry']))), flush=True)


if __name__ == '__main__':
    main()
