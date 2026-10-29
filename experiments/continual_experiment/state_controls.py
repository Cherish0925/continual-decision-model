"""Disentangle continuing neural state from outcome replay and online adapters.

Reuse both v0.10 full-model checkpoints, the original ordered native streams,
and their saved initial simulator RNG. These are post-hoc mechanism controls,
not additional training or independent confirmation of a new method.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing
from pathlib import Path
import time

from experiments.discoveryworld.evaluate_active_stream import run_stream


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, default=Path('runs/discoveryworld/continual_experiment/native'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--state-modes', nargs='+', choices=('neural_only', 'learning_only', 'h_only', 'z_only'),
                        default=('neural_only', 'learning_only'))
    parser.add_argument('--pilot', action='store_true', help='One original stream per arm, plus unchanged carry replay')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    jobs, configs = [], {}
    for seed in (201, 202):
        source = json.loads((args.reference / f'full_seed{seed}' / 'configuration.json').read_text())
        streams = source['streams'][:1] if args.pilot else source['streams']
        rng = Path(source['initial_rng_root'])
        if any(not (rng / f'seed{number}.json.gz').is_file() for stream in streams for number in stream['seeds']):
            raise FileNotFoundError('The original initial RNG is required for every instance')
        modes = (*args.state_modes, 'carry') if args.pilot else args.state_modes
        for mode in modes:
            name = f'{mode}_seed{seed}'
            output = args.output / name
            output.mkdir()
            options = {**source, 'state_mode': mode, 'output': str(output), 'workers': args.workers,
                       'device': 'cpu', 'threads': 1, 'save_boundary_states': False}
            configs[name] = {**options, 'streams': streams,
                             'reference_condition': f'full_seed{seed}',
                             'source_release': 'v0.10.0',
                             'comparison': 'Task-boundary selective state reset; all other execution settings unchanged'}
            (output / 'configuration.json').write_text(json.dumps(configs[name], indent=2) + '\n')
            jobs.extend((name, options, stream) for stream in streams)
    plan = {'pilot': args.pilot, 'reference': str(args.reference),
            'conditions': list(configs), 'native_instances': sum(len(stream['seeds']) for _, _, stream in jobs),
            'interventions': {'neural_only': 'Carry h/z; reset replay contents, replay cursor and fast/slow adapters',
                              'learning_only': 'Reset h/z; carry replay contents, replay cursor and fast/slow adapters',
                              'h_only': 'Carry h; reset z, replay contents, replay cursor and fast/slow adapters',
                              'z_only': 'Carry z; reset h, replay contents, replay cursor and fast/slow adapters'},
            'unchanged': ['model weights', 'initial simulator RNG', 'ordered streams', 'tools and observations',
                          'actual action budget', 'terminal feedback delivery', 'pending and activity counters'],
            'max_cpu_workers': args.workers}
    (args.output / 'plan.json').write_text(json.dumps(plan, indent=2) + '\n')
    results = {name: [] for name in configs}
    started = time.monotonic()
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context('spawn')) as pool:
        future_jobs = {pool.submit(run_stream, (options, stream)): name for name, options, stream in jobs}
        for future in as_completed(future_jobs):
            name = future_jobs[future]
            result = future.result()
            results[name].append(result)
            (args.output / name / 'summary.json').write_text(json.dumps(results[name], indent=2) + '\n')
            completed = sum(len(value) for value in results.values())
            print(json.dumps({'condition': name, 'stream': result['stream'], 'completed_streams': completed,
                              'total_streams': len(jobs), 'seconds': time.monotonic() - started}), flush=True)
    errors = [row for values in results.values() for stream in values for row in stream['episodes'] if row.get('error')]
    completion = {'seconds': time.monotonic() - started, 'execution_errors': len(errors),
                  'instances': sum(stream['instances'] for value in results.values() for stream in value)}
    (args.output / 'finished.json').write_text(json.dumps(completion, indent=2) + '\n')
    if errors:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
