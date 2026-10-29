"""Prepare environment-only execution geometry for recorded training worlds.

No geometry, relation identity or unrequested measurements enter model inputs.
Coordinates and public teleport rules reproduce the fixed native execution tool;
recorded measurements remain the only measurements exposed by training tasks.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor
import contextlib
import json
import multiprocessing
import os
from pathlib import Path
import numpy as np

from experiments.discoveryworld.reference import INSTRUMENTS

FACES = ('north', 'east', 'south', 'west')


def build_geometry(job):
    index, seed, readings, present = job
    with open(os.devnull, 'w') as sink, contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
        from discoveryworld.DiscoveryWorldAPI import DiscoveryWorldAPI
        api = DiscoveryWorldAPI(threadID=os.getpid())
        if not api.loadScenario('Reactor Lab', 'Normal', randomSeed=int(seed), numUserAgents=1):
            raise RuntimeError(f'Cannot construct training world {seed}')
    world = api.world
    objects = {o.uuid: o for outer in world.getAllWorldObjects()
               for o in outer.getAllContainedObjectsAndParts()}
    crystals = [o for o in objects.values() if o.name.startswith('quantum crystal')]
    features = np.asarray([[o.attributes['density'], o.attributes['temperatureC'],
                           o.attributes['materials'][0]['radiationusvh'], o.attributes['quantumSize'],
                           *o.attributes['materials'][0]['spectrum']] for o in crystals])
    present = present.copy(); present[:, 2] = False
    matches = ((np.abs(readings[:, None] - features[None]) < 5e-4) | ~present[:, None]).all(-1)
    if not (matches.sum(-1) == 1).all() or not (matches.sum(0) == 1).all():
        raise ValueError('Training object identity is ambiguous')
    crystals = [crystals[j] for j in matches.argmax(-1)]
    reactors = {o.attributes['reactorNum']: o for o in objects.values() if 'reactorNum' in o.attributes}
    instruments = [next(o for o in objects.values() if o.name == name) for name in INSTRUMENTS]
    # Exact deterministic TELEPORT_TO_OBJECT fallback: first passable N/E/S/W.
    approach = np.zeros((world.sizeX, world.sizeY, 3), dtype=np.int16)
    approach[:, :, :2] = -1
    for x in range(world.sizeX):
        for y in range(world.sizeY):
            for dx, dy, face in ((0, -1, 2), (1, 0, 3), (0, 1, 0), (-1, 0, 1)):
                if world.isWithinBounds(x + dx, y + dy) and world.isPassable(x + dx, y + dy)[0]:
                    approach[x, y] = x + dx, y + dy, face
                    break
    agent = world.getUserAgents()[0]
    places = world.getTeleportLocations()
    location = next(name for name in places if 'instruments' in name)
    agent.actionTeleportAgentToLocation(location)
    def nearby():
        return agent.getNearbyVisibleObjects(maxDistance=3, includeUUID=True)[2]
    grasses = [(o['distance'], direction, o['uuid']) for direction, values in nearby().items()
               for o in values if o['name'] == 'grass' and direction != 'same_location']
    grass = min(grasses, key=lambda row: row[:2])[2]
    agent.actionTeleportAgentToObject(grass)
    count, visited = 2, {grass}  # One location and the first grass teleport.
    for _ in range(24):
        x, y = agent.getWorldLocation()
        distance = min(max(abs(x - p['gridX']), abs(y - p['gridY'])) for p in places.values())
        if distance >= 9:
            break
        choices = []
        for direction, values in nearby().items():
            if direction == 'same_location':
                continue
            axes = direction.split('-')
            for obj in values:
                if obj['name'] != 'grass' or obj['uuid'] in visited:
                    continue
                span = obj['distance'] / len(axes)
                dx = span * (('east' in axes) - ('west' in axes))
                dy = span * (('south' in axes) - ('north' in axes))
                separation = min(max(abs(x + dx - p['gridX']), abs(y + dy - p['gridY']))
                                 for p in places.values())
                choices.append((separation, obj['distance'], obj['uuid']))
        grass = max(choices, key=lambda row: row[:2])[2]
        agent.actionTeleportAgentToObject(grass)
        count += 1; visited.add(grass)
    else:
        raise RuntimeError('Public radiation-isolation route did not terminate')
    pose = (*agent.getWorldLocation(), FACES.index(agent.attributes['faceDirection']))
    return index, {
        'instrument_xy': np.asarray([o.getWorldLocation() for o in instruments], dtype=np.int16),
        'crystal_xy': np.asarray([o.getWorldLocation() for o in crystals], dtype=np.int16),
        'reactor_xy': np.asarray([reactors[int(o.name.rsplit(' ', 1)[-1])].getWorldLocation()
                                 for o in crystals], dtype=np.int16),
        'approach_pose': approach,
        'isolation_pose': np.asarray(pose, dtype=np.int16),
        'isolation_actions': np.asarray(count, dtype=np.int16),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--workers', type=int, default=12)
    p.add_argument('--limit', type=int)
    args = p.parse_args()
    with np.load(args.data) as source:
        n = min(args.limit or len(source['x']), len(source['x']))
        jobs = [(i, int(source['scenario_seed'][i]), source['x'][i], source['reading_present'][i]) for i in range(n)]
    output = {}
    with ProcessPoolExecutor(args.workers, mp_context=multiprocessing.get_context('spawn')) as workers:
        for index, record in workers.map(build_geometry, jobs):
            if not output:
                output = {k: np.empty((n, *v.shape), dtype=v.dtype) for k, v in record.items()}
            for key, value in record.items():
                output[key][index] = value
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **output, scenario_seed=np.asarray([x[1] for x in jobs]))
    print(json.dumps({'worlds': n, 'isolation_action_counts': sorted(set(output['isolation_actions'].tolist()))}))

if __name__ == '__main__':
    main()
