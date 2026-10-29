"""Joint sequence training after conditional-response warmup.

Every epoch covers every returned training transition exactly once. Batched
windows reconstruct their complete public prefix with the current core; the
supervised window backpropagates through continuing h/z and response prediction.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from time import perf_counter

import torch
from torch.nn import functional as F

from .observations import feedback
from .objectives import feedback_statistics, scaled_feedback_loss, training_parameter_groups
from .reaction import load_reaction_checkpoint, permute_observation_components
from .warmup_reaction import batch_observations, query_observation


def advance_rows(model, state, rows, records, permutations):
    obs = batch_observations([permute_observation_components(query_observation(row), permutation)
                              for row, permutation in zip(records, permutations)])
    part = {key: value[rows] for key, value in state.items()}
    part = model.receive(part, obs)
    out = model.readout(part, obs['candidates'], torch.ones(len(rows), 1, dtype=torch.bool))
    targets = torch.tensor([row['rust_after'] for row in records])
    returned = torch.cat([feedback(row) for row in records])
    part = model.begin_action(part, out['features'], torch.zeros(len(rows), dtype=torch.long))
    part = model.feedback(part, returned)
    state = {key: value.index_copy(0, rows, part[key]) for key, value in state.items()}
    return state, out, targets, returned


def train_window(model, episodes, windows, stats):
    b = len(windows)
    permutations = [torch.randperm(len(episodes[e][0]['component_counts'])) for e, _, _ in windows]
    prefix_state = model.initial(b, 'cpu')
    with torch.no_grad():
        for step in range(max(start for _, start, _ in windows)):
            indices = [i for i, (_, start, _) in enumerate(windows) if step < start]
            records = [episodes[windows[i][0]][step] for i in indices]
            prefix_state, _, _, _ = advance_rows(model, prefix_state, torch.tensor(indices), records,
                                                  [permutations[i] for i in indices])
    # New-episode initial slots retain their gradient. Other rows start from a
    # fully reconstructed prefix, which is the truncation boundary.
    state = model.initial(b, 'cpu')
    prefixed = torch.tensor([i for i, (_, start, _) in enumerate(windows) if start > 0], dtype=torch.long)
    if len(prefixed):
        state = {key: value.index_copy(0, prefixed, prefix_state[key][prefixed]) for key, value in state.items()}
    losses, nll_sum, seen = [], 0., 0
    for offset in range(max(stop - start for _, start, stop in windows)):
        indices = [i for i, (_, start, stop) in enumerate(windows) if start + offset < stop]
        records = [episodes[windows[i][0]][windows[i][1] + offset] for i in indices]
        state, out, targets, returned = advance_rows(model, state, torch.tensor(indices), records,
                                                     [permutations[i] for i in indices])
        nll = F.nll_loss(out['outcomes'][:, 0], targets)
        loss = nll + scaled_feedback_loss(out['predictions'][:, 0], returned, stats)
        losses.append(loss * len(indices)); seen += len(indices); nll_sum += float(nll.detach()) * len(indices)
    return torch.stack(losses).sum() / seen, nll_sum / seen, seen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, default=Path('runs/continual_physics/chemistry_exploration'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--sequence-length', type=int, default=8)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--core-lr', type=float, default=3e-5)
    parser.add_argument('--interface-lr', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=753)
    args = parser.parse_args()
    torch.set_num_threads(args.threads); torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    model, saved = load_reaction_checkpoint(args.checkpoint)
    model.train()
    manifest = json.loads((args.data_root / 'manifest.json').read_text())
    entries = [row for row in manifest['episodes'] if row['split'] == 'train']
    episodes = [[json.loads(line) for line in (args.data_root / row['directory'] / 'experiments.jsonl').read_text().splitlines()]
                for row in entries]
    stats = feedback_statistics(row for episode in episodes for row in episode)
    windows = [(i, start, min(len(rows), start + args.sequence_length))
               for i, rows in enumerate(episodes) for start in range(0, len(rows), args.sequence_length)]
    optimizer = torch.optim.AdamW(training_parameter_groups(model, core_lr=args.core_lr,
                                                            interface_lr=args.interface_lr))
    metadata = {**saved['training'], 'phase': 'joint continuing-core conditional response learning',
                'warmup_checkpoint': args.checkpoint.name, 'warmup_updates': saved['training']['updates'],
                'joint_epochs': args.epochs, 'seed': args.seed, 'core_lr': args.core_lr,
                'interface_lr': args.interface_lr, 'sequence_length': args.sequence_length,
                'batch_size': args.batch_size, 'feedback_statistics': stats,
                'ingredient_augmentation': 'consistent permutation throughout each reconstructed prefix and window',
                'window_schedule': 'all disjoint windows each epoch, current-weight public prefix replay',
                'selection_protocol': 'fixed final epoch; no development checkpoint selection'}
    args.output.mkdir(parents=True, exist_ok=True)

    def save(name, epoch, updates):
        torch.save({'kind': 'chemistry_reaction_v1', **model.configuration(), 'model': model.state_dict(),
                    'training': {**metadata, 'completed_joint_epochs': epoch, 'updates': updates}}, args.output / name)

    save('initial.pt', 0, 0)
    started = perf_counter(); updates = 0
    with (args.output / 'training.jsonl').open('w') as log:
        for epoch in range(args.epochs):
            rng.shuffle(windows)
            nll_total = loss_total = count = 0
            for start in range(0, len(windows), args.batch_size):
                batch = windows[start:start + args.batch_size]
                optimizer.zero_grad(set_to_none=True)
                loss, nll, n = train_window(model, episodes, batch, stats)
                loss.backward()
                gradients = {name: float(torch.stack([p.grad.square().sum() for p in module.parameters()
                              if p.grad is not None]).sum().sqrt())
                             for name, module in [('gru', model.core.gru), ('workspace', model.core.block), ('response', model.response)]}
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                optimizer.step(); updates += 1
                count += n; nll_total += nll * n; loss_total += float(loss.detach()) * n
                row = {'epoch': epoch + 1, 'update': updates, 'supervised_transitions': n,
                       'rust_nll': nll, 'loss': float(loss.detach()), 'module_gradient_norms': gradients,
                       'seconds': perf_counter() - started}
                log.write(json.dumps(row) + '\n'); log.flush()
                if updates % 20 == 0:
                    print(json.dumps(row), flush=True)
            print(json.dumps({'epoch': epoch + 1, 'transitions': count, 'rust_nll': nll_total / count,
                              'loss': loss_total / count, 'seconds': perf_counter() - started}), flush=True)
    save('joint.pt', args.epochs, updates)
    (args.output / 'completed.json').write_text(json.dumps({'training': metadata, 'updates': updates,
        'seconds': perf_counter() - started}, indent=2) + '\n')


if __name__ == '__main__':
    main()
