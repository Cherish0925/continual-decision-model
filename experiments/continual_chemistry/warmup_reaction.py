"""Fit conditional responses over complete epochs, using frozen core features.

Core features are computed from actual received prefixes once. All mixture
history encoders and the core-feature readout train with uniform shuffled
transition batches and proper observed-response likelihood. Joint continuing
training follows this warmup; frozen features do not establish a core benefit.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import torch
from torch.nn import functional as F

from .observations import observe, feedback
from .objectives import feedback_statistics, initialize_feedback_predictor, scaled_feedback_loss
from .reaction import reaction_from_core_checkpoint, initialize_response_prior


def query_observation(row):
    return observe(row['state'], elapsed_actions=row['transition_start'],
                   candidate_recipes=[tuple(row['component_counts'])])[0]


def batch_observations(observations):
    result = {}
    for key in observations[0]:
        values = [obs[key][0] for obs in observations]
        if key in {'evidence', 'evidence_mask'}:
            result[key] = torch.nn.utils.rnn.pad_sequence(values, batch_first=True)
        else:
            result[key] = torch.stack(values)
    return result


@torch.no_grad()
def cache_features(model, episodes, batch_size=16):
    """Batch live prefixes; histories in the observation never include the query's return."""
    cached = []
    for start in range(0, len(episodes), batch_size):
        group = episodes[start:start + batch_size]
        states = model.base.initial(len(group), 'cpu')
        for step in range(max(map(len, group))):
            indices = torch.tensor([i for i, records in enumerate(group) if step < len(records)])
            records = [group[i][step] for i in indices.tolist()]
            observations = [query_observation(row) for row in records]
            obs = batch_observations(observations)
            partial = {key: value[indices] for key, value in states.items()}
            revised = model.base.advance(obs['event'], obs['evidence'], obs['evidence_mask'], partial)
            states = {key: value.index_copy(0, indices, revised[key]) for key, value in states.items()}
            features = model.core.state_features(revised['h'], revised['z'])
            for i, row in enumerate(records):
                cached.append((observations[i], features[i].clone(), row))
        print(json.dumps({'cached_episodes': min(start + batch_size, len(episodes)),
                          'cached_transitions': len(cached)}), flush=True)
    return cached


def batch_predictions(model, cached):
    obs = batch_observations([row[0] for row in cached])
    f = torch.stack([row[1] for row in cached])
    candidates = obs['candidates']
    component_mask = candidates[:, 0, 16:24].bool()
    history = obs['evidence'][:, 1:]
    mask = obs['evidence_mask'][:, 1:]
    if history.shape[1] == 0:
        history = history.new_zeros(len(cached), 1, 20)
        mask = mask.new_zeros(len(cached), 1)
    q = history[..., :8] / history[..., :8].sum(-1, keepdim=True).clamp_min(1e-8)
    response = model.response(candidates[..., 8:16], component_mask,
                              (candidates[..., 25] * 3).round().long(), q,
                              (history[..., 16] * 3).round().long(),
                              (history[..., 17] * 3).round().long(), mask, f)
    query = model.base.state_query(f)[:, None]
    c = model.base.candidate(candidates)
    joint = model.base.interaction(torch.cat((query, c, query * c), -1))
    x = F.normalize(model.adapter_features(joint).float(), dim=-1)
    prediction = model.predictor(x)  # No runtime adaptation occurs in this fixed-path warmup.
    target = torch.tensor([row[2]['rust_after'] for row in cached])
    returned = torch.cat([feedback(row[2]) for row in cached])
    return response['log_probabilities'][:, 0], prediction[:, 0], target, returned


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', default='runs/physical_belief/joint_20261029/carry_seed221/policy.pt')
    parser.add_argument('--data-root', type=Path, default=Path('runs/continual_physics/chemistry_exploration'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=12)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--cache-batch-size', type=int, default=16)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=753)
    args = parser.parse_args()
    torch.set_num_threads(args.threads); torch.manual_seed(args.seed)
    manifest = json.loads((args.data_root / 'manifest.json').read_text())
    entries = [row for row in manifest['episodes'] if row['split'] == 'train']
    episodes = [[json.loads(line) for line in (args.data_root / row['directory'] / 'experiments.jsonl').read_text().splitlines()]
                for row in entries]
    records = [row for episode in episodes for row in episode]
    model = reaction_from_core_checkpoint(args.checkpoint, initialization_seed=args.seed)
    stats = feedback_statistics(records)
    initialize_feedback_predictor(model, stats); initialize_response_prior(model, records)
    args.output.mkdir(parents=True, exist_ok=True)
    initial_prior = model.response.prior.detach().cpu().tolist()
    metadata = {'seed': args.seed, 'phase': 'frozen-core conditional response warmup', 'epochs': args.epochs,
                'batch_size': args.batch_size, 'feedback_statistics': stats,
                'initial_response_prior': initial_prior, 'uniform_transition_sampling': True,
                'training_episode_directories': [row['directory'] for row in entries],
                'source_core': Path(args.checkpoint).name, 'selection_protocol': 'fixed final epoch; no confirmation selection'}

    def save(name, epoch, updates):
        torch.save({'kind': 'chemistry_reaction_v1', **model.configuration(), 'model': model.state_dict(),
                    'training': {**metadata, 'completed_epochs': epoch, 'updates': updates}}, args.output / name)

    save('initial.pt', 0, 0)
    model.eval(); started = perf_counter()
    cached = cache_features(model, episodes, args.cache_batch_size)
    cache_seconds = perf_counter() - started
    for parameter in model.core.parameters():
        parameter.requires_grad_(False)
    model.train()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    updates = 0
    with (args.output / 'training.jsonl').open('w') as log:
        for epoch in range(args.epochs):
            order = torch.randperm(len(cached)).tolist()
            nll_total = loss_total = n = 0
            for start in range(0, len(order), args.batch_size):
                batch = [cached[i] for i in order[start:start + args.batch_size]]
                optimizer.zero_grad(set_to_none=True)
                logp, prediction, target, returned = batch_predictions(model, batch)
                nll = F.nll_loss(logp, target)
                loss = nll + scaled_feedback_loss(prediction, returned, stats)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                optimizer.step(); updates += 1
                n += len(batch); nll_total += float(nll.detach()) * len(batch); loss_total += float(loss.detach()) * len(batch)
            row = {'epoch': epoch + 1, 'updates': updates, 'transitions': n,
                   'rust_nll': nll_total / n, 'loss': loss_total / n, 'seconds': perf_counter() - started,
                   'cache_seconds': cache_seconds}
            print(json.dumps(row), flush=True); log.write(json.dumps(row) + '\n'); log.flush()
    save('warmup.pt', args.epochs, updates)
    (args.output / 'completed.json').write_text(json.dumps({'training': metadata, 'updates': updates,
        'cache_seconds': cache_seconds, 'seconds': perf_counter() - started}, indent=2) + '\n')


if __name__ == '__main__':
    main()
