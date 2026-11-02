"""Paired task-stream comparisons for the digital response-law extension."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np


def paired(left, right):
    a = {(r['stream'], r['task_index']): r for r in left}
    b = {(r['stream'], r['task_index']): r for r in right}
    if a.keys() != b.keys():
        raise ValueError('Comparison requires the same complete task identities')
    streams = sorted({k[0] for k in a})
    differences = np.array([[np.sum([a[k][metric] - b[k][metric] for k in a if k[0] == s])
                             for metric in ('native_actions', 'completed')] for s in streams])
    counts = np.array([sum(k[0] == s for k in a) for s in streams])
    rng = np.random.default_rng(20261102)
    index = rng.integers(len(streams), size=(10000, len(streams)))
    samples = differences[index].sum(1) / counts[index].sum(1)[:, None]
    return dict(tasks=len(a), streams=len(streams), mean_difference=(differences.sum(0) / counts.sum()).tolist(),
                interval95=np.quantile(samples, [.025, .975], axis=0).T.tolist(),
                columns=['operations', 'completion_fraction'], unit='complete task stream')


def summarize(directory, manifest, pairs):
    expected = {(s['name'], i) for s in manifest['streams'] for i, _ in enumerate(s['tasks'])}
    records, stats, rows = {}, {}, []
    for f in sorted(directory.glob('*/summary.json')):
        result = json.loads(f.read_text())
        data = result['records']
        if len(data) != len(expected) or {(r['stream'], r['task_index']) for r in data} != expected:
            raise ValueError(f'Incomplete or duplicated cohort: {f}')
        records[f.parent.name] = data
        groups = {'all': data}
        for field in ('group', 'exposure', 'family'):
            groups.update({f'{field}:{value}': [r for r in data if r[field] == value]
                           for value in sorted({r[field] for r in data})})
        stats[f.parent.name] = {name: dict(tasks=len(group), completed=sum(r['completed'] for r in group),
            operations=float(np.mean([r['native_actions'] for r in group]))) for name, group in groups.items()}
        rows.extend(dict(condition=f.parent.name, **{k: r[k] for k in (
            'stream', 'task_index', 'group', 'family', 'exposure', 'intervening_tasks',
            'native_actions', 'completed', 'returned_experiments', 'inference_seconds')}) for r in data)
    comparisons = {}
    for left, right in pairs:
        if left not in records or right not in records:
            continue
        comparisons[left + ':' + right] = {'all': paired(records[left], records[right])}
        for field in ('group', 'exposure'):
            for value in sorted({r[field] for r in records[left]}):
                comparisons[left + ':' + right][f'{field}:{value}'] = paired(
                    [r for r in records[left] if r[field] == value], [r for r in records[right] if r[field] == value])
    output = dict(conditions=stats, comparisons=comparisons,
                  scope='Digital extension, fixed streams; known-family reference has generator-formula knowledge')
    (directory / 'comparisons.json').write_text(json.dumps(output, indent=2) + '\n')
    with (directory / 'instances.csv').open('w') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader(); writer.writerows(rows)
    return output


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--manifest', type=Path, required=True)
    args = p.parse_args()
    pairs = [(f'hypotheses{s}', f'{other}{s}') for s in (911, 912)
             for other in ('marginal', 'inherited', 'parent', 'disabled', 'direct', 'no_selector')]
    pairs += [(f'parent{s}', f'direct{s}') for s in (911, 912)]
    result = summarize(args.input, json.loads(args.manifest.read_text()), pairs)
    print(json.dumps({k: v['all'] for k, v in result['conditions'].items()}, indent=2))


if __name__ == '__main__':
    main()
