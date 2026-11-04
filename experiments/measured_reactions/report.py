"""Aggregate matched development runs and preserve their complete best-so-far curves."""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .study import LEARNED, REFERENCES, summarize


def collect(root, data, seeds, out):
    with np.load(data, allow_pickle=False) as f:
        meta = json.loads(str(f['meta']))
    summary = {'source_revision': meta['revision'], 'source_sha256': meta['sha256'],
               'raw_rows': meta['raw_rows'], 'retained_rows': meta['retained_rows'],
               'condition_values': meta['conditions'], 'splits': {}, 'runs': {}, 'training': {}}
    for split in ('train', 'dev', 'test'):
        screens = [s for s in meta['screens'] if s['split'] == split]
        summary['splits'][split] = {'screens': len(screens), 'notebooks': len({s['notebook'] for s in screens}),
                                    'families': sorted({s['family'] for s in screens})}
    expected = [(m, None, False) for m in REFERENCES]
    for seed in seeds:
        expected.extend((m, seed, False) for m in LEARNED)
        expected.extend((m, seed, True) for m in ('cdm', 'conditional'))
    metadata, labels, all_curves = None, [], []
    for mode, seed, masked in expected:
        name = mode if seed is None else f'{mode}{seed}'
        run = root / name
        label = name + ('_without_readings' if masked else '')
        with (run / ('dev_without_readings.csv' if masked else 'dev.csv')).open() as f:
            rows = sorted(csv.DictReader(f), key=lambda r: (r['screen'], int(r['start'])))
        if seed is not None and not masked:
            with (run / 'train.json').open() as f:
                cfg = json.load(f)
            summary['training'][name] = {k: cfg[k] for k in (
                'mode', 'seed', 'device', 'width', 'slots', 'updates', 'batch', 'budget', 'lr', 'queries', 'init_native')}
        keys = [(r['screen'], int(r['start']), r['family'], r['notebook']) for r in rows]
        if len(set(keys)) != len(keys) or (metadata is not None and keys != metadata):
            raise ValueError(f'Screens or starting conditions do not match: {name}')
        metadata = keys
        budget = sum(k.startswith('best_') for k in rows[0])
        for row in rows:
            for j in range(1, budget + 1):
                row[f'best_{j}'] = float(row[f'best_{j}'])
        curves = np.array([[r[f'best_{i+1}'] for i in range(budget)] for r in rows], dtype=np.float32)
        if all_curves and not np.array_equal(curves[:, 0], all_curves[0][:, 0]):
            raise ValueError(f'First returned observations differ: {name}')
        families = defaultdict(list)
        for row in rows:
            families[row['family']].append(row)
        summary['runs'][label] = dict(**summarize(rows, budget), by_family={
            k: summarize(v, budget)[f'best_{budget}'] for k, v in sorted(families.items())})
        labels.append(label)
        all_curves.append(curves)
    summary['measured_training_queries'] = sum(v['queries'] for v in summary['training'].values())
    summary['matched_dev_rollouts_per_condition'] = len(metadata)
    summary['development_evaluations'] = len(expected) * len(metadata)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'development.json').write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n')
    screen, start, family, notebook = zip(*metadata)
    np.savez_compressed(out / 'development_curves.npz', conditions=labels, screen=screen, start=start,
                        family=family, notebook=notebook, best=np.stack(all_curves))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs', type=Path, required=True)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--seeds', default='41,73,109,127,149,173,197,223')
    a = p.parse_args()
    collect(a.runs, a.data, [int(s) for s in a.seeds.split(',')], a.out)
