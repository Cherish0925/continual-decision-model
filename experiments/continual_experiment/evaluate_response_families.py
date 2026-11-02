"""Evaluate a fixed learned policy on declared digital response-family streams."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
import json
import multiprocessing
from pathlib import Path
from time import perf_counter

import torch

from cdm.continual_experiment import select_rows
from experiments.continual_chemistry.active_training import choose_activity, proposed_experiments, start_apparatus
from experiments.continual_chemistry.response_families import ResponseFamilyEnvironment, response_levels
from .shared_tasks import load_shared_checkpoint


def worker(job):
    checkpoint, manifest, options = job
    torch.set_num_threads(1)
    torch.manual_seed(43)
    model, _ = load_shared_checkpoint(checkpoint, 'cpu')
    return evaluate(model, manifest, **options)


@torch.no_grad()
def evaluate(model, manifest, *, root, device='cpu', batch=8, memory='evidence', slots=12,
             baseline='model', record_predictions=False, disable_selector=False):
    model.eval()
    policy = model.tasks['chemistry']
    if disable_selector:
        policy.experiment_value = None
    policy.experimental_memory.config = replace(policy.experimental_memory.config, mode=memory, slots=slots)
    rows, predictions = [], []
    started = perf_counter()
    for start in range(0, len(manifest['streams']), batch):
        streams = manifest['streams'][start:start + batch]
        env = ResponseFamilyEnvironment(root, len(streams), manifest['seed'], device, manifest=streams)
        state = policy.initial(len(streams), device)
        complete = torch.zeros(len(streams), dtype=torch.bool, device=device)
        counts = torch.zeros_like(complete, dtype=torch.long)
        inference = torch.zeros(len(streams), device=device)
        while not complete.all():
            boundary, lifetime = (env.new_task, env.new_lifetime) if env.dispatched == 0 else env.reset_finished()
            state = start_apparatus(policy, state, boundary, lifetime)
            obs = env.observation()
            if device.startswith('cuda'):
                torch.cuda.synchronize(device)
            tick = perf_counter()
            state = policy.receive(state, obs)
            state, _, _, internal, _ = choose_activity(policy, state, obs, sample=False)
            if baseline in ('model', 'retrieval'):
                action = proposed_experiments(policy, state, obs).argmax(-1)
                if baseline == 'retrieval':
                    from experiments.continual_chemistry.episodic_retrieval import previous_success
                    for lane in range(len(streams)):
                        remembered = previous_success({k: v[lane:lane + 1] for k, v in state.items()},
                            obs['candidates'][lane:lane + 1], obs['action_mask'][lane:lane + 1])
                        if remembered is not None:
                            action[lane] = remembered
            elif baseline == 'enumeration':
                # Public low-cost order; no response-law parameters.
                cost = 6 + env.counts.sum(-1) + (env.counts > 0).sum(-1)
                action = cost.masked_fill(~obs['action_mask'], torch.inf).argmin(-1)
            elif baseline == 'oracle':
                # Privileged lower reference knows the current law exactly.
                levels = torch.stack([response_levels(q, law) for q, law in zip(env.counts, env.current_laws)])
                feasible = obs['action_mask'] & (levels == 0)
                feasible[:, env.leave_index] = env.rust == 0
                cost = 6 + env.counts.sum(-1) + (env.counts > 0).sum(-1)
                action = cost.masked_fill(~feasible, torch.inf).argmin(-1)
            else:
                raise ValueError(baseline)
            index = torch.arange(len(streams), device=device)
            out = policy.readout(state, obs['candidates'][index, action][:, None],
                                 torch.ones(len(streams), 1, dtype=torch.bool, device=device))
            if device.startswith('cuda'):
                torch.cuda.synchronize(device)
            inference += (perf_counter() - tick) / len(streams)
            zero = torch.zeros_like(action)
            state = policy.begin_action(state, out['features'], zero)
            result = env.step(action)
            valid = result['returned'] & ~result['leave']
            if record_predictions:
                for lane in (valid & ~complete).nonzero().flatten().tolist():
                    task = env.task_metadata[lane]
                    predictions.append(dict(stream=streams[lane]['name'], task=int(counts[lane]),
                        family=task['law']['family'], exposure=task['exposure'],
                        before=int(result['before'][lane]), after=int(result['after'][lane]),
                        counts=env.counts[lane, action[lane]].tolist(),
                        probabilities=out['outcomes'][lane, 0].exp().tolist(),
                        operations=int(result['cost'][lane]), internal_activities=int(internal[lane])))
            state = policy.feedback(state, result['targets'], result['returned'])
            updated = policy.record_response(state, out, zero, result['before'], result['after'])
            state = select_rows(state, updated, valid)
            for row in env.task_rows:
                lane = row['lane']
                if counts[lane] >= len(streams[lane]['tasks']):
                    continue
                rows.append({**row, 'stream': streams[lane]['name'], 'group': streams[lane]['group'],
                             'inference_seconds': float(inference[lane])})
                counts[lane] += 1
                inference[lane] = 0
            env.task_rows.clear()
            complete = counts >= env.lifetime_length
        print(json.dumps(dict(streams=len(streams), tasks=len(rows), seconds=perf_counter() - started)), flush=True)
    return rows, predictions, perf_counter() - started


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--chemical-data', type=Path, default=Path('runs/continual_physics/chemistry_exploration'))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='cpu')
    p.add_argument('--batch', type=int, default=4)
    p.add_argument('--workers', type=int, default=1)
    p.add_argument('--memory-mode', choices=('evidence', 'disabled'), default='evidence')
    p.add_argument('--baseline', choices=('model', 'retrieval', 'enumeration', 'oracle'), default='model')
    p.add_argument('--record-predictions', action='store_true')
    p.add_argument('--disable-selector', action='store_true', help='Remove only the new action-score residual')
    args = p.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(43)
    args.output.mkdir(parents=True, exist_ok=False)
    model, saved = load_shared_checkpoint(args.checkpoint, args.device)
    manifest = json.loads(args.manifest.read_text())
    options = dict(root=args.chemical_data, device=args.device, batch=args.batch,
                   memory=args.memory_mode, baseline=args.baseline, record_predictions=args.record_predictions,
                   disable_selector=args.disable_selector)
    if args.workers == 1:
        rows, predictions, seconds = evaluate(model, manifest, **options)
    else:
        if args.device != 'cpu':
            p.error('Use one process for GPU evaluation')
        jobs = [(args.checkpoint, {**manifest, 'streams': manifest['streams'][i:i + args.batch]}, options)
                for i in range(0, len(manifest['streams']), args.batch)]
        rows, predictions = [], []
        started = perf_counter()
        with ProcessPoolExecutor(args.workers, mp_context=multiprocessing.get_context('spawn')) as pool:
            for future in as_completed([pool.submit(worker, job) for job in jobs]):
                r, pred, _ = future.result()
                rows.extend(r); predictions.extend(pred)
        seconds = perf_counter() - started
    rows.sort(key=lambda r: (r['stream'], r['task_index']))
    result = dict(configuration={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        completed=sum(r['completed'] for r in rows), tasks=len(rows),
        mean_operations=sum(r['native_actions'] for r in rows) / len(rows), seconds=seconds,
        parameters=sum(v.numel() for v in model.parameters()), records=rows,
        scope='Digital extension; no hidden law inputs to model; oracle alone receives law parameters')
    (args.output / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    if predictions:
        with (args.output / 'predictions.jsonl').open('w') as f:
            for row in predictions:
                f.write(json.dumps(row) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'records'}), flush=True)


if __name__ == '__main__':
    main()
