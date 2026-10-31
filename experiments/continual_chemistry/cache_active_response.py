"""Collect actual selected responses for chronological update learning.

The already checked batched native-transition model supplies results for the
selected experiment only. Neither generating recipe nor alternatives' outcomes
are passed to the policy or exported training row.
"""
import argparse
import json
from pathlib import Path
from time import perf_counter

import torch

from cdm.continual_experiment import select_rows
from experiments.continual_experiment.shared_tasks import load_shared_checkpoint
from .active_environment import ActiveChemicalEnvironment
from .active_training import start_apparatus, choose_activity


@torch.no_grad()
def collect(model, root, split, steps, batch, seed, device):
    env = ActiveChemicalEnvironment(root, batch, seed, device, split=split)
    state = model.initial(batch, device)
    rows, started = [], perf_counter()
    for step in range(steps):
        boundary, lifetime = (env.new_task, env.new_lifetime) if step == 0 else env.reset_finished()
        state = start_apparatus(model, state, boundary, lifetime)
        obs = env.observation()
        state = model.receive(state, obs)
        state, *_ = choose_activity(model, state, obs, sample=False)
        out = model.readout(state, obs['candidates'], obs['action_mask'])
        logits = model.experiment_logits(out, obs)
        choice = logits.argmax(-1)
        if split == 'train':
            random_choice = torch.multinomial(obs['action_mask'].float(), 1).squeeze(1)
            choice = torch.where(torch.rand(batch, device=device) < .2, random_choice, choice)
        context = model.replay_context(state, obs['candidates'])
        # Capture support before requesting the next result. Only its returned
        # outcome is attached as an outer target after the environment step.
        captured = []
        for lane in range(batch):
            a = int(choice[lane]); n = int(state['response_valid'][lane].sum())
            size = model.response_adaptation.config.memory_size
            index = (torch.arange(n, device=device) + max(0, int(state['response_cursor'][lane]) - size)) % size
            positions = context['valid'][lane].nonzero().flatten()[-n:] if n else torch.empty(0, device=device, dtype=torch.long)
            task = env.task_index[lane] - 1
            entry = env.lifetime[lane][int(task % 6 in (2, 3))]
            row = dict(episode=f'active/{lane}/{task}', split=split, position=int(env.cursor[lane]),
                recipe_group=entry['recipe_group'], before=int(env.rust[lane]),
                query_base=out['response_base_logits'][lane, a:a+1], query_x=out['response_features'][lane, a:a+1],
                query_inputs=obs['candidates'][lane, a:a+1, 8:16], original_logp=out['outcomes'][lane, a],
                stored_base=state['response_base'][lane, index], stored_x=state['response_x'][lane, index],
                **{'support_' + k: v[lane, positions] for k, v in context.items() if k != 'valid'})
            captured.append({k: v.cpu().clone() if isinstance(v, torch.Tensor) else v for k, v in row.items()})
        state = model.begin_action(state, out['features'], choice)
        result = env.step(choice)
        valid = result['returned'] & ~result['leave']
        state = model.feedback(state, result['targets'], result['returned'])
        state = select_rows(state, model.record_response(state, out, choice, result['before'], result['after']), valid)
        for lane in valid.nonzero().flatten().tolist():
            rows.append({**captured[lane], 'label': int(result['after'][lane])})
        if step % 32 == 0 or step + 1 == steps:
            print(json.dumps(dict(split=split, step=step + 1, returned=len(rows), seconds=perf_counter() - started)), flush=True)
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--steps', type=int, default=512)
    p.add_argument('--validation-steps', type=int, default=96)
    p.add_argument('--batch', type=int, default=32)
    p.add_argument('--seed', type=int, default=1041)
    p.add_argument('--device', default='cuda:0')
    args = p.parse_args()
    torch.set_num_threads(1); torch.manual_seed(args.seed)
    shared, _ = load_shared_checkpoint(args.checkpoint, args.device)
    model = shared.tasks['chemistry'].eval()
    rows = collect(model, Path('runs/continual_physics/chemistry_exploration'), 'train', args.steps,
                   args.batch, args.seed, args.device)
    rows += collect(model, Path('runs/continual_physics/chemistry_data'), 'validation', args.validation_steps,
                    args.batch, args.seed + 100, args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(rows=rows, checkpoint=str(args.checkpoint),
        scope='selected responses from exact native-transition model; frozen parent; 20 percent train exploration; greedy development'), args.output)


if __name__ == '__main__':
    main()
