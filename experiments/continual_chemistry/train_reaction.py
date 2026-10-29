"""Train conditional experimental responses and the continuing core jointly.

Only the actually executed candidate receives an outcome label. Every received
prefix is reconstructed; no unexecuted outcomes or generating recipes are used.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from time import perf_counter

import torch
from torch.nn import functional as F

from cdm.continual_experiment import detach_state
from .dataset import load_episode
from .observations import feedback
from .objectives import (feedback_statistics, initialize_feedback_predictor,
                         scaled_feedback_loss, training_parameter_groups)
from .reaction import reaction_from_core_checkpoint, initialize_response_prior, permute_observation_components


def selected_prediction(model, state, obs, target):
    index = int(target['action'][0])
    # Full history updates the core; the observation label concerns one query.
    # Computing unlabelled candidate graphs would add cost without a loss term.
    candidate = obs['candidates'][:, index:index + 1]
    return model.readout(state, candidate, torch.ones(1, 1, dtype=torch.bool, device=candidate.device))


def load_training(data_root, device):
    manifest = json.loads((data_root / 'manifest.json').read_text())
    episodes = [row for row in manifest['episodes'] if row['split'] == 'train']
    sequences = [load_episode(data_root / row['directory'] / 'experiments.jsonl', device=device) for row in episodes]
    return episodes, sequences


def train(args):
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    episodes, sequences = load_training(args.data_root, args.device)
    records = [row for sequence in sequences for _, _, row in sequence]
    model = reaction_from_core_checkpoint(args.checkpoint, device=args.device, initialization_seed=args.seed)
    statistics = feedback_statistics(records)
    initialize_feedback_predictor(model, statistics)
    initialize_response_prior(model, records)
    model.train()
    optimizer = torch.optim.AdamW(training_parameter_groups(model, core_lr=args.core_lr,
                                                            interface_lr=args.interface_lr))
    args.output.mkdir(parents=True, exist_ok=True)
    metadata = {'seed': args.seed, 'objective': 'proper censored response likelihood and scaled public feedback regression',
                'feedback_statistics': statistics, 'core_lr': args.core_lr, 'interface_lr': args.interface_lr,
                'sequence_length': args.sequence_length, 'training_episode_directories': [r['directory'] for r in episodes],
                'selection_protocol': 'fixed final update; no development selection',
                'response_prior': 'training-only component-count/current-rust conditional frequency',
                'initial_response_prior': model.response.prior.detach().cpu().tolist(),
                'window_schedule': 'seeded per-episode window permutation with full observed-prefix replay',
                'ingredient_augmentation': 'consistent random public ingredient renaming throughout each prefix and window',
                'source_core': Path(args.checkpoint).name}

    def save(name, updates):
        torch.save({'kind': 'chemistry_reaction_v1', **model.configuration(), 'model': model.state_dict(),
                    'training': {**metadata, 'updates': updates}}, args.output / name)

    save('initial.pt', 0)
    rng = random.Random(args.seed)
    orders = {i: [] for i in range(len(sequences))}
    episode_order = list(range(len(sequences)))
    rng.shuffle(episode_order)
    started = perf_counter()
    with (args.output / 'training.jsonl').open('w') as log:
        for update in range(args.updates):
            episode_index = episode_order[update % len(sequences)]
            sequence = sequences[episode_index]
            n = len(sequence[0][2]['component_counts'])
            permutation = torch.randperm(n, device=args.device)
            sequence = [(permute_observation_components(obs, permutation), target, record)
                        for obs, target, record in sequence]
            if not orders[episode_index]:
                starts = list(range(0, len(sequence), args.sequence_length))
                rng.shuffle(starts)
                orders[episode_index] = starts
            start = orders[episode_index].pop()
            stop = min(len(sequence), start + args.sequence_length)
            state = model.initial(1, args.device)
            with torch.no_grad():
                for obs, target, record in sequence[:start]:
                    state = model.receive(state, obs)
                    out = selected_prediction(model, state, obs, target)
                    state = model.begin_action(state, out['features'], target['action'] * 0)
                    state = model.feedback(state, feedback(record, args.device))
            state = detach_state(state)
            optimizer.zero_grad(set_to_none=True)
            losses, nlls, progress_events = [], [], 0
            for obs, target, record in sequence[start:stop]:
                state = model.receive(state, obs)
                out = selected_prediction(model, state, obs, target)
                # out['outcomes'] are normalized log probabilities of observed
                # rust. Using nll_loss preserves the censored tail exactly.
                nll = F.nll_loss(out['outcomes'][:, 0], target['rust_after'])
                returned = feedback(record, args.device)
                loss = nll + scaled_feedback_loss(out['predictions'][:, 0], returned, statistics)
                losses.append(loss); nlls.append(nll)
                progress_events += record['rust_after'] < record['rust_before']
                state = model.begin_action(state, out['features'], target['action'] * 0)
                state = model.feedback(state, returned)
            loss = torch.stack(losses).mean()
            loss.backward()
            gradients = {}
            for name, module in [('gru', model.core.gru), ('workspace', model.core.block), ('response', model.response)]:
                squares = [p.grad.square().sum() for p in module.parameters() if p.grad is not None]
                gradients[name] = float(torch.stack(squares).sum().sqrt()) if squares else 0.
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            row = {'update': update + 1, 'episode': episodes[episode_index]['directory'],
                   'window_start': start, 'window_stop': stop, 'replayed_prefix': start,
                   'transitions': len(losses), 'improvements': progress_events,
                   'loss': float(loss.detach()), 'rust_nll': float(torch.stack(nlls).mean().detach()),
                   'gradient_norm': float(gradient_norm), 'module_gradient_norms': gradients,
                   'seconds': perf_counter() - started}
            log.write(json.dumps(row) + '\n'); log.flush()
            if (update + 1) % 10 == 0 or update + 1 == args.updates:
                print(json.dumps(row), flush=True)
    save('last.pt', args.updates)
    (args.output / 'completed.json').write_text(json.dumps({'updates': args.updates, 'seconds': perf_counter() - started,
                                                          'training': metadata}, indent=2) + '\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', default='runs/physical_belief/joint_20261029/carry_seed221/policy.pt')
    p.add_argument('--data-root', type=Path, default=Path('runs/continual_physics/chemistry_data'))
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--updates', type=int, default=100)
    p.add_argument('--sequence-length', type=int, default=8)
    p.add_argument('--core-lr', type=float, default=3e-5)
    p.add_argument('--interface-lr', type=float, default=1e-3)
    p.add_argument('--device', default='cpu')
    p.add_argument('--threads', type=int, default=1)
    p.add_argument('--seed', type=int, default=751)
    train(p.parse_args())


if __name__ == '__main__':
    main()
