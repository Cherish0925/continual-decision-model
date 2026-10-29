"""Fixed initial-policy rollout in the training execution environment.

This checks the untrained starting policy under actual execution costs. It uses
training-pool worlds, makes no checkpoint selection, and is not a held-out score.
"""
from __future__ import annotations
import json
from pathlib import Path
import torch
from .train import training_parser, initialize_experiment, observe


def main():
    parser = training_parser()
    args = parser.parse_args()
    args.mode = 'recurrent_reference'
    args.native_fraction = 1.
    model, _, _, environment, delayed = initialize_experiment(args)
    model.eval()
    state = model.initial(args.batch, args.device)
    obs, _ = observe(model, environment, state, delayed=delayed)
    state = model.receive(state, obs)
    done = torch.zeros(args.batch, device=args.device, dtype=torch.bool)
    while not done.all():
        with torch.no_grad():
            obs, _ = observe(model, environment, state, delayed=delayed)
            output = model.readout(state, obs['candidates'], obs['action_mask'])
            action = output['logits'].argmax(-1)
            selected = ~done & ~delayed.pending
            state = model.begin_action(state, output['features'], action, selected, physical=output)
            delayed.dispatch(action, {'means': output['physical_settings'], 'std': output['physical_std'],
                                      'weights': output['physical_weights']}, selected)
            receipt = delayed.advance()
            state = model.feedback(state, receipt['targets'], receipt['arrived'],
                                   physical_feedback=receipt['physical_feedback'])
            obs, _ = observe(model, environment, state, delayed=delayed)
            state = model.receive(state, obs, receipt['arrived'])
            done |= receipt['task_done']
    table = environment.table
    result = {'scope': 'unchanged initialization; native source worlds from the training pool',
              'seed': args.seed, 'streams': args.batch,
              'execution': 'native primitive actions' if args.execution_geometry else 'single setting proxy',
              'successes': table.successes.tolist(), 'queries': table.target_mask.sum(-1).tolist(),
              'complete_tasks': int((table.successes == table.target_mask.sum(-1)).sum()),
              'requests': table.steps.tolist(), 'tests': table.tests.tolist(),
              'measurements': table.acquisitions.tolist(),
              'source_worlds': table.indices.tolist(), 'permutations': table.permutation.tolist()}
    if hasattr(table, 'native_steps'):
        result['native_actions'] = table.native_steps.tolist()
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)

if __name__ == '__main__':
    main()
