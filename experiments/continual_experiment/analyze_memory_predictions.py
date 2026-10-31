"""Compare memory and fresh predictions on the same actually chosen experiments."""
import argparse
from collections import defaultdict
import csv
import json
from math import gcd
from pathlib import Path

import numpy as np


def proportions(counts):
    """Identify the same mixture independently of total quantity."""
    divisor = gcd(*map(int, counts))
    return tuple(int(x) // divisor for x in counts)


def interval(rows, key, samples=10000):
    groups = defaultdict(list)
    for row in rows:
        groups[row['stream']].append(row[key])
    names = sorted(groups)
    totals = np.array([sum(groups[n]) for n in names])
    counts = np.array([len(groups[n]) for n in names])
    if len(names) < 2:
        return dict(predictions=len(rows), streams=len(names), mean=float(totals.sum() / counts.sum()),
                    interval95=None)
    draw = np.random.default_rng(591).integers(len(names), size=(samples, len(names)))
    means = totals[draw].sum(1) / counts[draw].sum(1)
    return dict(predictions=len(rows), streams=len(names), mean=float(totals.sum() / counts.sum()),
                interval95=np.quantile(means, [.025, .975]).tolist())


def predictions(root, manifest):
    rows = []
    for stream in manifest['streams']:
        archives = [None] * 6
        cursor = 0
        for index, task in enumerate(stream['chemistry']):
            path = root / stream['name'] / f'chemistry{index:03d}_seed{task["seed"]}' / 'experiments.jsonl'
            evidence = np.zeros(7)
            returned = []
            for trial in map(json.loads, path.read_text().splitlines()):
                if 'component_counts' not in trial:
                    continue
                cached = trial['experimental_memory']
                lp = np.asarray(cached['expert_return_log_probabilities'])[:, trial['rust_after']]
                weights = np.asarray(cached['weights'])
                full = -trial['model_outcome_logits'][trial['rust_after']]
                fresh = -lp[0]
                valid = np.array([a is not None for a in archives])
                scores = np.r_[1., np.where(valid,
                    evidence[1:] - evidence[0] - np.log(max(1, valid.sum())), -np.inf)]
                scores = scores - np.logaddexp.reduce(scores)
                fixed = -np.logaddexp.reduce(lp + scores)
                relevant = np.array([a is not None and a['group'] == task['recipe_group'] for a in archives])
                query = proportions(trial['component_counts'])
                seen_old = any(a is not None and query in a['queries'] for a in archives)
                seen_relevant = any(a is not None and a['group'] == task['recipe_group']
                                    and query in a['queries'] for a in archives)
                rows.append(dict(stream=stream['name'], episode_index=index, seed=task['seed'],
                    exposure=task['exposure'], training_regime=task['training_regime'],
                    prior_observations=len(returned), archives=int(valid.sum()),
                    applicable_archive_available=bool(relevant.any()),
                    query_seen_current=query in returned,
                    query_seen_archive=seen_old,
                    query_seen_applicable_archive=seen_relevant,
                    old_weight=float(weights[1:].sum()),
                    relevant_old_weight=float(weights[1:][relevant].sum()),
                    full_nll=float(full), fresh_nll=float(fresh), fixed_evidence_nll=float(fixed),
                    memory_minus_fresh=float(full-fresh), memory_minus_fixed=float(full-fixed)))
                evidence += lp
                returned.append(query)
            if returned:
                archives[cursor % 6] = {'group': task['recipe_group'], 'queries': set(returned[-32:])}
                cursor += 1
    return rows


def summarize(rows):
    groups = dict(all=rows, new=[r for r in rows if r['exposure'] == 'new'],
        returned=[r for r in rows if r['exposure'] == 'return'],
        supported_return=[r for r in rows if r['applicable_archive_available']],
        untested_query=[r for r in rows if not r['query_seen_current'] and not r['query_seen_archive']],
        return_untested_query=[r for r in rows if r['applicable_archive_available']
                              and not r['query_seen_current'] and not r['query_seen_archive']],
        first_observation=[r for r in rows if r['prior_observations'] == 0],
        after_observations=[r for r in rows if r['prior_observations'] > 0])
    result = {}
    for label, values in groups.items():
        if not values:
            continue
        result[label] = {key: interval(values, key) for key in ('memory_minus_fresh', 'memory_minus_fixed')}
        result[label].update({key: float(np.mean([r[key] for r in values]))
            for key in ('full_nll', 'fresh_nll', 'fixed_evidence_nll', 'old_weight', 'relevant_old_weight')})
        delta = np.array([r['memory_minus_fresh'] for r in values])
        result[label].update(improved=int((delta < -1e-7).sum()), worsened=int((delta > 1e-7).sum()),
            unchanged=int((abs(delta) <= 1e-7).sum()))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--native-root', type=Path, required=True)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    manifest = json.loads(args.manifest.read_text())
    result, flat = {}, []
    for seed in (911, 912):
        rows = predictions(args.native_root / f'memory{seed}', manifest)
        result[str(seed)] = summarize(rows)
        flat.extend(dict(continuation=seed, **r) for r in rows)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'memory_predictions.json').write_text(json.dumps(dict(
        method='same chosen experiments; cached predictions precede the scored return',
        uncertainty='whole-stream paired bootstrap; current policies determine query distribution',
        fixed_reference='initialized reader: current log prior 1, old total log prior 0; preceding log likelihood weights',
        results=result), indent=2) + '\n')
    with (args.output / 'memory_predictions.csv').open('w') as file:
        writer = csv.DictWriter(file, fieldnames=list(flat[0])); writer.writeheader(); writer.writerows(flat)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
