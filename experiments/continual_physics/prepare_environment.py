"""Reconstruct training-environment truth without adding it to observations.

Native activation is abs(setting - resonance) < 2 Hz. A previously observed
successful setting specifies an interval around that resonance, not its exact
value. Reconstructing the original training world lets the replay environment
apply the actual activation predicate. Policies continue to receive only
requested measurements and actual successful/unsuccessful setting feedback.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import contextlib
import json
import multiprocessing
import os
from pathlib import Path
import time

import numpy as np

from experiments.discoveryworld.adapter import UPSTREAM_REVISION


def reconstruct(job):
    index, seed, readings, present, working, labels = job
    # Upstream emits scenario/debug output during construction. Exceptions
    # propagate; no simulator actions or renders are requested here.
    with open(os.devnull, "w") as sink, contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
        from discoveryworld.DiscoveryWorldAPI import DiscoveryWorldAPI
        api = DiscoveryWorldAPI(threadID=os.getpid())
        if not api.loadScenario("Reactor Lab", "Normal", randomSeed=int(seed), numUserAgents=1):
            raise RuntimeError(f"Cannot reconstruct training world {seed}")
    crystals = {obj.uuid: obj for outer in api.world.getAllWorldObjects()
                for obj in outer.getAllContainedObjectsAndParts()
                if obj.name.startswith("quantum crystal")}
    if len(crystals) != 4:
        raise ValueError(f"Expected four training crystals in world {seed}")
    objects = list(crystals.values())
    features = np.asarray([[obj.attributes['density'], obj.attributes['temperatureC'],
                            obj.attributes['materials'][0]['radiationusvh'],
                            obj.attributes['quantumSize'],
                            *obj.attributes['materials'][0]['spectrum']]
                           for obj in objects], dtype=np.float64)
    deterministic = present.copy()
    deterministic[:, 2] = False  # Radiation returns include measurement noise.
    if not deterministic.any(-1).all():
        raise ValueError("Object matching needs received deterministic measurements")
    difference = np.abs(readings[:, None] - features[None])
    matches = ((difference < 5e-4) | ~deterministic[:, None]).all(-1)
    if not (matches.sum(-1) == 1).all() or not (matches.sum(0) == 1).all():
        raise ValueError(f"Recorded object identities do not match reconstructed world {seed}")
    order = matches.argmax(-1)
    resonance = np.asarray([objects[j].attributes['resonanceFreq'] for j in order], dtype=np.float64)
    # The public label may be near either edge of the successful interval.
    # Float32 storage of the recorded label permits sub-millihertz rounding.
    label_errors = np.abs(working[labels] - resonance[labels])
    if (label_errors >= 2.002).any():
        raise ValueError(f"A recorded activation contradicts reconstructed world {seed}: {label_errors.tolist()}")
    return index, resonance, float(label_errors.max(initial=0.))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True, help='Recorded training NPZ; never a confirmation set')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--workers', type=int, default=16)
    p.add_argument('--limit', type=int, help='Optional small construction check')
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    with np.load(args.data) as source:
        count = len(source['scenario_seed'])
        count = min(count, args.limit) if args.limit is not None else count
        seeds = source['scenario_seed'][:count].copy()
        identities = source['source'][:count].copy()
        jobs = [(i, seeds[i], source['x'][i], source['reading_present'][i],
                 source['working_frequency_hz'][i], source['label_present'][i]) for i in range(count)]
    started = time.monotonic()
    truth = np.empty((count, 4), dtype=np.float64)
    errors = []
    with ProcessPoolExecutor(args.workers, mp_context=multiprocessing.get_context('spawn')) as pool:
        for done, (index, resonance, maximum) in enumerate(pool.map(reconstruct, jobs), 1):
            truth[index] = resonance
            errors.append(maximum)
            if done % 100 == 0 or done == count:
                print(json.dumps({'worlds_reconstructed': done, 'total': count}), flush=True)
    np.savez_compressed(args.output / 'environment.npz', resonance_hz=truth,
                        scenario_seed=seeds, source=identities)
    summary = {'source_data': str(args.data), 'worlds': count, 'objects': int(truth.size),
               'upstream_revision': UPSTREAM_REVISION,
               'activation_rule': 'abs(actual_setting_hz - resonance_hz) < 2',
               'access': 'environment transitions and outer training supervision only; excluded from actor and online-update inputs',
               'object_alignment': 'unique agreement of all available deterministic public measurements',
               'maximum_recorded_success_error_hz': max(errors),
               'seconds': time.monotonic() - started}
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
