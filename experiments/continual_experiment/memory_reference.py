"""Native Bayesian experimental memory with the known reaction family.

The reference carries posterior distributions from actual prior experiments.
It never reads the generator's recipe identity. A fixed half-uniform prior
allows new contexts; six old task posteriors provide the other half.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stdout
import fcntl
import json
import multiprocessing
import os
from pathlib import Path
from time import perf_counter

import numpy as np

from experiments.continual_chemistry.bayesian_reference import FiniteChemicalBelief
from experiments.continual_chemistry.native import BoundedChemistryExperiments, ActionBudgetReached
from experiments.discoveryworld.adapter import TrajectoryWriter
from experiments.discoveryworld.structured_observation import StructuredDiscoveryWorldEnv


class RememberingBelief(FiniteChemicalBelief):
    def __init__(self, components, archive):
        super().__init__(components)
        self.prior = np.ones(len(self.recipes)) / len(self.recipes)
        if archive:
            self.prior = .5 * self.prior + .5 * np.mean(archive, axis=0)

    def posterior(self, history):
        compatible = super().posterior(history) > 0
        posterior = self.prior * compatible
        return posterior / posterior.sum()


def run_stream(job):
    options, stream, mode = job
    root = Path(options['output']) / mode / stream['name']
    root.mkdir(parents=True)
    archives, rows = {}, []
    for index, task in enumerate(stream['chemistry']):
        output = root / f'chemistry{index:03d}_seed{task["seed"]}'
        output.mkdir()
        started = perf_counter()
        inference_seconds = execution_seconds = 0.
        error, ending = None, 'experiment_budget'
        with (output / 'native.log').open('w') as log, redirect_stdout(log):
            env = StructuredDiscoveryWorldEnv(output, 1000)
            env.reset('Combinatorial Chemistry', task['difficulty'], task['seed'])
            rng_root = Path(options['initial_rng_root'])
            rng_root.mkdir(parents=True, exist_ok=True)
            snapshot = rng_root / f'{task["difficulty"].lower()}_seed{task["seed"]}.json.gz'
            with snapshot.with_suffix('.lock').open('a') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                rng = env.restore_initial_rng(snapshot) if snapshot.exists() else env.save_initial_rng(snapshot)
            metadata = {**env.configuration, 'policy': mode,
                'prior': 'known cosine family; no instance identities; half uniform plus half past posterior mean',
                'memory_capacity': 6, 'initial_rng': rng}
            with TrajectoryWriter(output, metadata) as writer, (output / 'experiments.jsonl').open('w') as trials:
                executor = BoundedChemistryExperiments(env, writer)
                try:
                    tick = perf_counter()
                    public = executor.prepare()
                    execution_seconds += perf_counter() - tick
                    n = len(public['components'])
                    bank = archives.setdefault(n, [])
                    belief = RememberingBelief(n, bank if mode == 'bayes_memory' else [])
                    for _ in range(209):
                        if not public['rust_level']:
                            tick = perf_counter()
                            executor.leave()
                            execution_seconds += perf_counter() - tick
                            ending = 'native_termination'
                            break
                        tick = perf_counter()
                        recipe, prediction = belief.choose(public, 'two_step')
                        inference_seconds += perf_counter() - tick
                        tick = perf_counter()
                        try:
                            record = executor.experiment(recipe)
                        finally:
                            execution_seconds += perf_counter() - tick
                        record['prediction'] = prediction
                        trials.write(json.dumps(record) + '\n'); trials.flush()
                        public = executor.state()
                        if env.ended:
                            ending = 'native_action_budget'
                            break
                    # Archive the likelihood supported by this task's results,
                    # without recursively counting its inherited prior again.
                    if executor.history:
                        bank.append(FiniteChemicalBelief(n).posterior(executor.history))
                        del bank[:-6]
                except ActionBudgetReached:
                    ending = 'native_action_budget'
                except (ValueError, RuntimeError) as exc:
                    error, ending = f'{type(exc).__name__}: {exc}', 'execution_error'
            result = {**task, 'stream': stream['name'], 'episode_index': index,
                'condition': mode, 'native_actions': env.steps,
                'completed_successfully': all(r['completed_successfully'] for r in env.metrics),
                'experiments': len(executor.history), 'inference_seconds': inference_seconds,
                'execution_seconds': execution_seconds, 'seconds': perf_counter() - started,
                'ending': ending, 'error': error}
            (output / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
        rows.append(result)
        (root / 'chemistry_summary.json').write_text(json.dumps(rows, indent=2) + '\n')
        if error:
            raise RuntimeError(error)
    return dict(condition=mode, stream=stream['name'], chemistry=rows, physics=[])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--initial-rng-root', required=True)
    p.add_argument('--workers', type=int, default=20)
    args = p.parse_args()
    if os.environ.get('PYTHONHASHSEED') != '0':
        raise ValueError('Use PYTHONHASHSEED=0')
    args.output.mkdir(parents=True, exist_ok=False)
    streams = json.loads(args.manifest.read_text())['streams']
    options = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    (args.output / 'configuration.json').write_text(json.dumps(options, indent=2) + '\n')
    results = {mode: [] for mode in ('bayes_memory', 'bayes_reset')}
    jobs = [(options, stream, mode) for mode in results for stream in streams]
    with ProcessPoolExecutor(args.workers, mp_context=multiprocessing.get_context('spawn')) as pool:
        for future in as_completed([pool.submit(run_stream, job) for job in jobs]):
            r = future.result(); results[r['condition']].append(r)
            (args.output / r['condition'] / 'summary.json').write_text(json.dumps(results[r['condition']], indent=2) + '\n')
            print(json.dumps(dict(condition=r['condition'], stream=r['stream'],
                completed=sum(t['completed_successfully'] for t in r['chemistry']),
                actions=sum(t['native_actions'] for t in r['chemistry']))), flush=True)
    (args.output / 'finished.json').write_text(json.dumps(dict(streams=len(jobs))) + '\n')


if __name__ == '__main__':
    main()
