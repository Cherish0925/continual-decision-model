"""Compare the compiled execution tool with already recorded native requests.

This replays fixed public requests and received readings. Reconstructed private
resonances are used only to check the environment's activation transition. It
neither trains a policy nor treats these verification episodes as training data.
"""
from __future__ import annotations
import argparse
import contextlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch

from experiments.discoveryworld.calibration import FEATURE_QUANTITIES
from .native_execution import NativeToolExecution
from .prepare_execution import build_geometry


def verify(path, temporary):
    record = json.loads(path.read_text())
    seed = record['seed']
    order = record['bootstrap']['public_object_order']
    with open(os.devnull, 'w') as sink, contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
        from discoveryworld.DiscoveryWorldAPI import DiscoveryWorldAPI
        api = DiscoveryWorldAPI(threadID=os.getpid())
        api.loadScenario('Reactor Lab', 'Normal', randomSeed=seed, numUserAgents=1)
    objects = {o.uuid: o for outer in api.world.getAllWorldObjects()
               for o in outer.getAllContainedObjectsAndParts()}
    crystals = [objects[x['public_handle']] for x in order]
    values = np.asarray([[o.attributes['density'], o.attributes['temperatureC'],
                          o.attributes['materials'][0]['radiationusvh'], o.attributes['quantumSize'],
                          *o.attributes['materials'][0]['spectrum']] for o in crystals])
    targets = torch.tensor([[o.attributes['resonanceFreq'] for o in crystals]], dtype=torch.float64)
    _, geometry = build_geometry((0, seed, values, np.ones((4, 9), dtype=bool)))
    geometry_path = temporary / f'geometry_{seed}.npz'
    np.savez_compressed(geometry_path, **{k: v[None] for k, v in geometry.items()})
    executor = NativeToolExecution(geometry_path, 'cpu')
    table = SimpleNamespace(batch=1, device=torch.device('cpu'), indices=torch.tensor([0]),
        permutation=torch.arange(4)[None], query=torch.tensor([2]),
        initial_support=torch.tensor([[True, True, False, False]]), target_mask=torch.tensor([[False, False, True, True]]),
        known=torch.tensor([[True, True, False, False]]),
        known_hz=torch.tensor([record['bootstrap']['known_support_hz'] + [0., 0.]], dtype=torch.float64),
        targets=targets, steps=torch.zeros(1, dtype=torch.long), max_steps=160,
        counts=torch.zeros(1, 4, 9, dtype=torch.long), values=torch.zeros(1, 4, 9),
        group_counts=torch.zeros(1, 4, 5, dtype=torch.long),
        samples=torch.zeros(1, 4, 5, 64, 9), sample_present=torch.zeros(1, 4, 5, 64, 9, dtype=torch.bool),
        mixture=SimpleNamespace(x_mean=torch.zeros(9), x_scale=torch.ones(9)),
        event=torch.zeros(1, 36), acquisitions=torch.zeros(1), tests=torch.zeros(1), successes=torch.zeros(1),
        best_error=torch.full((1, 4), torch.inf), alive=torch.ones(1, dtype=torch.bool))
    table.observation = lambda: {'action_mask': torch.ones(1, 29, dtype=torch.bool)}
    for key, value in executor.initial(table).items():
        setattr(table, key, value)
    counts = np.zeros((4, 5), dtype=int)
    handles = {x['public_handle']: i for i, x in enumerate(order)}
    for decision in record['decisions']:
        request = decision['result']['request']
        if request['kind'] != 'measure':
            continue
        obj, group = handles[request['target_handle']], request['group_index']
        for reading in decision['result']['measurements']:
            q = FEATURE_QUANTITIES.index(reading['quantity'])
            table.samples[0, obj, group, counts[obj, group], q] = reading['value']
            table.sample_present[0, obj, group, counts[obj, group], q] = True
        counts[obj, group] += 1
    comparisons = []
    for decision in record['decisions']:
        request = decision['result']['request']
        prediction = {'means': torch.full((1, 9), float(request.get('mean_hz', 0.))),
                      'std': torch.full((1, 9), float(request.get('std_hz', 0.)))}
        result = executor(table, torch.tensor([decision['action_index']]), prediction)
        actual_attempts = decision['result'].get('attempts', [])
        predicted_attempts = [{'frequency_hz': float(f), 'activated': bool(a)} for f, a in zip(
            result['frequencies'][0, result['mask'][0]], result['activated'][0, result['mask'][0]])]
        # Native budget/termination finalization stores the final dialogue rather
        # than the intermediate attempt list. Cost and success are still checked.
        attempts_equal = actual_attempts == predicted_attempts if actual_attempts else None
        comparisons.append({'seed': seed, 'decision': decision['decision'], 'action_index': decision['action_index'],
            'expected_actions': decision['native_actions'], 'predicted_actions': int(result['cost'][0]),
            'expected_success': decision['success'], 'predicted_success': bool(result['success'][0]),
            'attempts_equal': attempts_equal, 'expected_attempts': len(actual_attempts),
            'predicted_attempts': len(predicted_attempts)})
    return comparisons


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('summaries', nargs='+', type=Path)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=True)
    records = [row for path in args.summaries for row in verify(path, args.output)]
    errors = [r for r in records if r['expected_actions'] != r['predicted_actions']
              or r['expected_success'] != r['predicted_success'] or r['attempts_equal'] is False]
    result = {'episodes': len(args.summaries), 'requests': len(records), 'mismatches': len(errors),
              'cost_absolute_error': sum(abs(r['expected_actions'] - r['predicted_actions']) for r in records),
              'comparisons': records}
    (args.output / 'comparison.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'comparisons'}))
    for error in errors[:20]:
        print(json.dumps(error))

if __name__ == '__main__':
    main()
