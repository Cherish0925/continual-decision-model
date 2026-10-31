"""Separate response-update transfer to new mixtures from repeated experiments."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F


def summarize(rows):
    if not rows:
        return {'predictions': 0}
    changes = np.asarray([r['with_update'] - r['without_update'] for r in rows])
    groups = sorted({r['recipe_group'] for r in rows})
    totals = np.asarray([sum(d for r, d in zip(rows, changes) if r['recipe_group'] == g) for g in groups])
    counts = np.asarray([sum(r['recipe_group'] == g for r in rows) for g in groups])
    samples = np.random.default_rng(1031).integers(len(groups), size=(10000, len(groups)))
    means = totals[samples].sum(1) / counts[samples].sum(1)
    return dict(predictions=len(rows), recipe_groups=len(groups),
                with_update_nll=float(np.mean([r['with_update'] for r in rows])),
                without_update_nll=float(np.mean([r['without_update'] for r in rows])),
                mean_difference=float(changes.mean()), interval95=np.quantile(means, [.025, .975]).tolist(),
                improved=int((changes < -1e-6).sum()), worsened=int((changes > 1e-6).sum()),
                unchanged_within_1e_6=int((np.abs(changes) <= 1e-6).sum()))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study', required=True, type=Path)
    p.add_argument('--output', type=Path,
                   default=Path('experiments/continual_chemistry/results/response_transfer.json'))
    args = p.parse_args()
    torch.set_num_threads(1)
    result = {'comparison': 'same fitted weights, response fitting present versus removed',
              'scope': 'existing development histories; no new model selection',
              'uncertainty': '10000 bootstrap resamples of whole named recipe groups',
              'distance': 'Euclidean distance between L2-normalized mixture proportions',
              'repetitions': {}}
    for seed in (911, 912):
        cache = torch.load(args.study / f'prefixes{seed}.pt', weights_only=True)
        rows = [r for r in cache['rows'] if r['split'] != 'train']
        yes = json.loads((args.study / f'relation{seed}/development.json').read_text())['predictions']
        no = json.loads((args.study / f'relation{seed}/without_update.json').read_text())['predictions']
        classified = []
        for r, a, b in zip(rows, yes, no, strict=True):
            assert (r['episode'], r['position']) == (a['episode'], a['position']) == (b['episode'], b['position'])
            distance = None
            kind = 'no_previous_return'
            if len(r['support_inputs']):
                q = F.normalize(r['query_inputs'].float(), dim=-1)
                s = F.normalize(r['support_inputs'].float(), dim=-1)
                distance = float((q[:, None] - s[None]).square().sum(-1).sqrt().min())
                kind = 'previously_tested_mixture' if distance < 1e-6 else 'different_mixture'
            classified.append(dict(kind=kind, distance=distance, recipe_group=r['recipe_group'],
                                   with_update=a['updated_nll'], without_update=b['updated_nll']))
        result['repetitions'][str(seed)] = {
            'by_history': {kind: summarize([r for r in classified if r['kind'] == kind])
                           for kind in ('no_previous_return', 'previously_tested_mixture', 'different_mixture')},
            'by_nearest_distance': {
                f'[{low},{high})': summarize([r for r in classified
                    if r['distance'] is not None and low <= r['distance'] < high])
                for low, high in ((0, .1), (.1, .25), (.25, .5), (.5, 2))}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v['by_history'] for k, v in result['repetitions'].items()}, indent=2))


if __name__ == '__main__':
    main()
