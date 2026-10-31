"""Cache chronological prediction prefixes for response-only update comparisons.

The inherited core and task interface remain frozen. A row's support consists
only of already returned experiments; its label is the next actual return.
This separates update fitting from core retraining and action-policy changes.
"""
import argparse
import json
from pathlib import Path
from time import perf_counter

import torch

from experiments.continual_experiment.shared_tasks import load_shared_checkpoint
from .active_training import choose_activity
from .observations import feedback, observe


def response_context(model, state, candidates):
    return model.replay_context(state, candidates)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True, type=Path)
    p.add_argument('--data', type=Path, default=Path('runs/continual_physics/chemistry_exploration'))
    p.add_argument('--validation-data', type=Path, default=Path('runs/continual_physics/chemistry_data'))
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--limit', type=int)
    args = p.parse_args()
    torch.set_num_threads(1)
    shared, _ = load_shared_checkpoint(args.checkpoint, args.device)
    model = shared.tasks['chemistry'].eval()
    entries = [(args.data, e) for e in json.loads((args.data / 'manifest.json').read_text())['episodes']
               if e['split'] == 'train']
    entries += [(args.validation_data, e) for e in
                json.loads((args.validation_data / 'manifest.json').read_text())['episodes']
                if e['split'] == 'validation']
    if args.limit:
        entries = entries[:args.limit]
    cached, started = [], perf_counter()
    with torch.no_grad():
        for index, (root, entry) in enumerate(entries):
            records = [json.loads(s) for s in (root / entry['directory'] / 'experiments.jsonl').read_text().splitlines()]
            state = model.initial(1, args.device)
            for position, record in enumerate(records):
                counts = tuple(record['component_counts'])
                obs, _ = observe(record['state'], elapsed_actions=record['transition_start'],
                                  device=args.device, candidate_recipes=[counts])
                state = model.receive(state, obs)
                state, *_ = choose_activity(model, state, obs, sample=False)
                out = model.readout(state, obs['candidates'], obs['action_mask'])
                support = response_context(model, state, obs['candidates'])
                old_mask = state['response_valid'][0]
                old_n = int(old_mask.sum())
                # The configured memory is 64 returns; full raw history is
                # retained separately. Align stored and refreshed comparisons.
                if old_n:
                    order = torch.arange(old_n, device=args.device)
                    if int(state['response_cursor'][0]) > len(old_mask):
                        order = (order + state['response_cursor'][0]) % len(old_mask)
                    stored_x = state['response_x'][0, order]
                    stored_base = state['response_base'][0, order]
                else:
                    stored_x = state['response_x'][0, :0]
                    stored_base = state['response_base'][0, :0]
                n = old_n
                row = dict(episode=entry['directory'], split=entry['split'], position=position,
                    recipe_group=entry['recipe_group'], label=record['rust_after'], before=record['rust_before'],
                    query_base=out['response_base_logits'][0], query_x=out['response_features'][0],
                    query_inputs=obs['candidates'][0, :, 8:16],
                    original_logp=out['outcomes'][0, 0],
                    stored_base=stored_base, stored_x=stored_x,
                    support_base=support['base'][0, -n:] if n else support['base'][0, :0],
                    support_x=support['x'][0, -n:] if n else support['x'][0, :0],
                    support_inputs=support['inputs'][0, -n:] if n else support['inputs'][0, :0],
                    support_before=support['before'][0, -n:] if n else support['before'][0, :0],
                    support_after=support['after'][0, -n:] if n else support['after'][0, :0])
                if sum(counts):
                    cached.append({k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in row.items()})
                action = torch.zeros(1, dtype=torch.long, device=args.device)
                state = model.begin_action(state, out['features'], action)
                state = model.feedback(state, feedback(record, args.device))
                if sum(counts):
                    state = model.record_response(state, out, action,
                        torch.tensor([record['rust_before']], device=args.device),
                        torch.tensor([record['rust_after']], device=args.device))
            if index % 20 == 0 or index + 1 == len(entries):
                print(json.dumps(dict(episodes=index + 1, transitions=len(cached), seconds=perf_counter() - started)), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(rows=cached, checkpoint=str(args.checkpoint), seconds=perf_counter() - started,
        scope='frozen parent on recorded chronological prefixes; support excludes current return'), args.output)


if __name__ == '__main__':
    main()
