"""Run physical/chemical task sequences with one loaded core and live state."""
from __future__ import annotations

import argparse
from dataclasses import replace
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing
from pathlib import Path

import torch

from experiments.continual_chemistry.native import run as run_chemistry
from experiments.continual_physics.evaluate_native import run_stream as run_physics
from experiments.continual_physics.runtime import ContinualPhysicsSession
from .shared_tasks import load_shared_checkpoint


def run_stream(job):
    options, stream = job
    torch.set_num_threads(options['threads'])
    model, saved = load_shared_checkpoint(options['checkpoint'])
    model.eval()
    physical, chemical = model.tasks['physics'], model.tasks['chemistry']
    if options.get('disable_response_replay', False):
        chemical.response_replay.config = replace(chemical.response_replay.config, mode='calibration')
    if options.get('memory_mode'):
        chemical.experimental_memory.config = replace(chemical.experimental_memory.config, mode=options['memory_mode'])
    assert physical.core is chemical.core
    session = ContinualPhysicsSession(physical)
    chemical_state = chemical.initial(1, 'cpu')
    rows = []
    mode = options.get('state_sharing') or saved['training']['state_sharing']
    output = Path(options['output'])

    def after_physics(index, physical_session):
        nonlocal chemical_state
        task = stream['chemistry'][index]
        prior_h, prior_z = chemical_state['h'], chemical_state['z']
        reset_chemical = options.get('reset_chemical_state', False)
        chemical_state = chemical.next_apparatus(chemical_state, torch.ones(1, dtype=torch.bool),
            torch.tensor([index == 0 or reset_chemical]))
        if mode == 'continuing':
            chemical_state.update(h=physical_session.state['h'], z=physical_session.state['z'])
        elif not reset_chemical:
            chemical_state.update(h=prior_h, z=prior_z)
        if options.get('reset_chemical_neural_state', False):
            fresh = chemical.initial(1, 'cpu')
            chemical_state.update(h=fresh['h'], z=fresh['z'])
        start = {k: chemical_state[k].clone() for k in ('h', 'z')}
        summary, chemical_state = run_chemistry({
            'checkpoint': options['checkpoint'], 'seed': task['seed'],
            'difficulty': task['difficulty'], 'policy': 'model',
            'output': str(output / stream['name'] / f"chemistry{index:03d}_seed{task['seed']}"),
            'max_actions': 1000, 'max_experiments': 209, 'threads': options['threads'],
            'initial_rng_root': options['initial_rng_root'], 'primitive_candidates': True,
            'discard_response_updates': options.get('discard_response_updates', False),
            'direct_retrieval': options.get('direct_retrieval', False),
        }, model=chemical, state=chemical_state, return_state=True)
        if mode == 'continuing':
            physical_session.state = {**physical_session.state,
                                      'h': chemical_state['h'], 'z': chemical_state['z']}
        row = {**summary, 'stream': stream['name'], 'episode_index': index,
               'state_sharing': mode,
               'cross_domain_state_change': {k: float((chemical_state[k] - start[k]).detach().norm()) for k in start}}
        row.update({k: task[k] for k in ('exposure', 'intervening_tasks', 'training_regime',
                   'untrained_proportion_structure') if k in task})
        rows.append(row)
        (output / stream['name'] / 'chemistry_summary.json').write_text(json.dumps(rows, indent=2) + '\n')
        if summary['error']:
            raise RuntimeError(summary['error'])

    physical_options = {**options, 'state_mode': 'carry', 'discard_physical_updates': False,
        'discard_physical_memory': False, 'feedback_delay_ticks': 2, 'tolerance_hz': 2.,
        'max_actions': 400, 'max_decisions': 160, 'device': 'cpu'}
    if options.get('chemistry_only', False):
        if mode != 'separated':
            raise ValueError('Skipping physical executions requires separate domain states')
        (output / stream['name']).mkdir(parents=True)
        for index in range(len(stream['chemistry'])):
            after_physics(index, session)
        result = {'episodes': []}
    else:
        result = run_physics((physical_options, stream), model=physical,
                             session=session, after_episode=after_physics)
    if any(row['error'] for row in result['episodes']):
        raise RuntimeError('Physical execution error')
    return {'stream': stream['name'], 'state_sharing': mode,
            'physics': result['episodes'], 'chemistry': rows}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--initial-rng-root', required=True)
    p.add_argument('--state-sharing', choices=('continuing', 'separated'))
    p.add_argument('--workers', type=int, default=6)
    p.add_argument('--threads', type=int, default=1)
    p.add_argument('--discard-response-updates', action='store_true')
    p.add_argument('--disable-response-replay', action='store_true')
    p.add_argument('--reset-chemical-state', action='store_true')
    p.add_argument('--memory-mode', choices=('learned', 'disabled', 'evidence', 'recent'))
    p.add_argument('--chemistry-only', action='store_true',
                   help='For same-weight memory controls with separated domain states')
    p.add_argument('--reset-chemical-neural-state', action='store_true')
    p.add_argument('--direct-retrieval', action='store_true')
    args = p.parse_args()
    streams = json.loads(args.manifest.read_text())['streams']
    options = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'configuration.json').write_text(json.dumps(options, indent=2) + '\n')
    results = []
    with ProcessPoolExecutor(args.workers, mp_context=multiprocessing.get_context('spawn')) as pool:
        futures = [pool.submit(run_stream, (options, stream)) for stream in streams]
        for future in as_completed(futures):
            result = future.result(); results.append(result)
            (args.output / 'summary.json').write_text(json.dumps(results, indent=2) + '\n')
            print(json.dumps({'stream': result['stream'], **{
                task: {'instances': len(result[task]),
                       'completed': sum(r['completed_successfully'] for r in result[task]),
                       'actions': sum(r['native_actions'] for r in result[task])}
                for task in ('physics', 'chemistry')}}), flush=True)
    (args.output / 'finished.json').write_text(json.dumps({'streams': len(results)}) + '\n')


if __name__ == '__main__':
    main()
