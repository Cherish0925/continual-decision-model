"""Combine disjoint seed evaluations from independent machines."""
import argparse
import json
from pathlib import Path

import numpy as np


def merge(inputs, output):
    report, records, coordinates = {}, {}, {}
    for root in inputs:
        partial = json.loads((root / 'external.json').read_text())
        with np.load(root / 'external_curves.npz', allow_pickle=False) as arrays:
            for lab, item in partial.items():
                meta = {k: v for k, v in item.items() if k != 'runs'}
                if lab in report and meta != {k: v for k, v in report[lab].items() if k != 'runs'}:
                    raise ValueError('External data sources differ')
                report.setdefault(lab, {**meta, 'runs': {}})
                for suffix in ('screen', 'start'):
                    key = f'{lab}_{suffix}'
                    if key in coordinates and not np.array_equal(coordinates[key], arrays[key]):
                        raise ValueError('Screen and start order must match')
                    coordinates[key] = arrays[key].copy()
            for i, label in enumerate(arrays['conditions'].tolist()):
                curves = {lab: arrays[f'{lab}_best'][i].copy() for lab in partial}
                if label in records:
                    if any(not np.allclose(records[label][lab], curve, atol=1e-5)
                           for lab, curve in curves.items()):
                        raise ValueError(f'Conflicting results for {label}')
                else:
                    records[label] = curves
                for lab in partial:
                    report[lab]['runs'][label] = partial[lab]['runs'][label]
    output.mkdir(parents=True, exist_ok=True)
    labels = list(records)
    (output / 'external.json').write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    np.savez_compressed(output / 'external_curves.npz', conditions=labels, **coordinates,
                        **{f'{lab}_best': np.stack([records[k][lab] for k in labels]) for lab in report})
    print(f'Combined {len(labels)} policies and feedback controls across {len(report)} laboratories')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', action='append', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    merge(args.input, args.out)
