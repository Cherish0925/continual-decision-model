"""Evaluate cost-ordered, privileged and known-family digital references."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing
from pathlib import Path
from time import perf_counter

import torch

from experiments.continual_chemistry.family_reference import FamilyBelief
from experiments.continual_chemistry.response_families import ResponseFamilyEnvironment, response_levels


def run(job):
    stream, options = job
    torch.set_num_threads(1)
    env = ResponseFamilyEnvironment(options['data'], 1, 17, 'cpu', manifest=[stream])
    belief = FamilyBelief(stream['tasks'][0]['law']['components'],
                          capacity=options['capacity'], reuse=options['reuse'], selection=options['selection'])
    records = []
    started = perf_counter()
    decision_seconds = 0.
    while len(records) < len(stream['tasks']):
        boundary, _ = (env.new_task, env.new_lifetime) if env.dispatched == 0 else env.reset_finished()
        if boundary[0]:
            belief.start()
        obs = env.observation()
        tick = perf_counter()
        if env.rust[0] == 0:
            action = env.leave_index
        elif options['baseline'] == 'belief':
            action = belief.choose(int(env.rust[0]), obs['action_mask'][0, :len(belief.recipes)].numpy())
        else:
            valid = obs['action_mask'][0].clone()
            if options['baseline'] == 'oracle':
                valid &= response_levels(env.counts[0], env.current_laws[0]) == 0
            cost = 6 + env.counts[0].sum(-1) + (env.counts[0] > 0).sum(-1)
            action = int(cost.masked_fill(~valid, torch.inf).argmin())
        decision_seconds += perf_counter() - tick
        result = env.step(torch.tensor([action]))
        if options['baseline'] == 'belief' and result['returned'][0] and not result['leave'][0]:
            belief.receive(action, int(result['before'][0]), int(result['after'][0]))
        if env.task_rows:
            records.extend({**r, 'stream': stream['name'], 'group': stream['group'],
                            'inference_seconds': decision_seconds} for r in env.task_rows)
            env.task_rows.clear(); decision_seconds = 0.
    return records, perf_counter() - started


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--data', default='runs/continual_physics/chemistry_exploration')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--baseline', choices=('belief', 'enumeration', 'oracle'), default='belief')
    p.add_argument('--selection', choices=('two_step', 'greedy', 'information'), default='two_step')
    p.add_argument('--capacity', type=int, default=12)
    p.add_argument('--reuse', type=float, default=.45)
    p.add_argument('--workers', type=int, default=8)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads(args.manifest.read_text())
    options = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    rows = []; start = perf_counter()
    with ProcessPoolExecutor(args.workers, mp_context=multiprocessing.get_context('spawn')) as pool:
        for future in as_completed([pool.submit(run, (stream, options)) for stream in manifest['streams']]):
            result, seconds = future.result(); rows.extend(result)
            print(json.dumps(dict(stream=result[0]['stream'], completed=sum(r['completed'] for r in result), seconds=seconds)), flush=True)
    rows.sort(key=lambda r: (r['stream'], r['task_index']))
    result = dict(configuration=options, tasks=len(rows), completed=sum(r['completed'] for r in rows),
        mean_operations=sum(r['native_actions'] for r in rows)/len(rows), seconds=perf_counter()-start,
        scope='Digital extension; finite-grid belief knows generator formulas, not actual current law; oracle alone knows it',
        records=rows)
    (args.output / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
