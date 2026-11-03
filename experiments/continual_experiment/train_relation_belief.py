"""Learn coherent and marginal predictors from matched measured sequences."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import math
from pathlib import Path
import random
from time import perf_counter

import torch

from cdm.conditional_response import observable_log_probabilities
from cdm.relation_belief import ConditionalRelation, RelationBelief, RelationBeliefConfig
from experiments.continual_chemistry.recipes import primitive_recipes
from experiments.continual_chemistry.response_families import TRAIN_FAMILIES, sample_law, response_levels


def sample_records(item):
    seed, n, observations = item
    torch.set_num_threads(1)
    rng = random.Random(seed)
    law = sample_law(rng, n, TRAIN_FAMILIES)
    counts = torch.tensor(primitive_recipes(n))
    levels = response_levels(counts, law)
    indices, before, after, cost = [], [], [], []
    current = 3
    chosen = rng.sample(range(len(counts)), observations)
    for i in range(observations):
        # Replicate apparatuses share the law. Each six-measurement block
        # starts with a new device; early completion also starts another one.
        fresh = i % 6 == 0 or current == 0
        if fresh:
            current = 3
        action = chosen[i]
        result = min(current, int(levels[action]))
        indices.append(action); before.append(current); after.append(result)
        cost.append(6 + int(counts[action].sum()) + int((counts[action] > 0).sum()) +
                    (14 if fresh else 2) + (4 if result == 0 else 0))
        current = result
    return dict(indices=indices, before=before, after=after, cost=cost)


def prepare(output, tasks, observations, workers):
    output.mkdir(parents=True, exist_ok=False)
    for split, size, seed in (('train', tasks, 733091), ('validation', 512, 833091)):
        result = {}
        for n in (3, 4):
            items = [(seed + 100003 * n + i, n, observations) for i in range(size)]
            with ProcessPoolExecutor(workers) as pool:
                rows = list(pool.map(sample_records, items, chunksize=32))
            result[n] = {key: torch.tensor([r[key] for r in rows]) for key in rows[0]}
        torch.save(result, output / f'{split}.pt')
    (output / 'README.json').write_text(json.dumps(dict(
        training_laws_per_component_count=tasks, component_counts=[3, 4], observations=observations,
        families=TRAIN_FAMILIES, public_data='Chosen composition, pre/post response and charged operations',
        protocol='Independent replicate apparatus every six observations, or after completion; no formula labels',
        split='Independent generator seeds; discrete targets may recur; withheld combinations not generated'), indent=2)+'\n')


def candidate_set(n, device):
    counts = torch.zeros(len(primitive_recipes(n)), 8, device=device)
    counts[:, :n] = torch.tensor(primitive_recipes(n), device=device)
    return counts / counts.sum(-1, keepdim=True), torch.arange(8, device=device) < n


def loss(model, kind, batch, q, components, context):
    actions, before, after = (batch[k] for k in ('indices', 'before', 'after'))
    b, t = actions.shape
    queries = q[actions]
    if kind == 'functions':
        bank = model(q, components)
        # B,K,T,L: a single particle explains the complete measurement sequence.
        selected = bank[:, actions].permute(1, 0, 2, 3)
        observed = observable_log_probabilities(selected, before[:, None])
        labels = after[:, None, :, None].expand(-1, bank.shape[0], -1, 1)
        ll = observed.gather(-1, labels).squeeze(-1)
        prior = model.log_prior(components)
        support = prior[None] + ll[:, :, :context].sum(-1)
        posterior = support.log_softmax(-1)
        future = ll[:, :, context:]
        joint = -(torch.logsumexp(posterior + future.sum(-1), -1) / (t - context)).mean()
        marginal = -torch.logsumexp(posterior[:, :, None] + future, 1).mean()
        # Sequential likelihood is the objective. Marginal future loss is only
        # a reported diagnostic; it never supplies extra training labels.
        return joint, marginal
    present = components[None].expand(b, -1)
    valid = torch.arange(t, device=q.device)[None].expand(b, -1) < context
    logits = model(queries[:, context:], present, queries, before, after, valid)
    observed = observable_log_probabilities(logits, before[:, context:])
    nll = -observed.gather(-1, after[:, context:, None]).squeeze(-1).mean()
    return nll, nll


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--prepare', action='store_true')
    p.add_argument('--tasks', type=int, default=8192)
    p.add_argument('--observations', type=int, default=48)
    p.add_argument('--workers', type=int, default=16)
    p.add_argument('--output', type=Path)
    p.add_argument('--kind', choices=('functions', 'conditional', 'attentive'), default='functions')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--seed', type=int, default=31)
    p.add_argument('--updates', type=int, default=6000)
    p.add_argument('--batch', type=int, default=128)
    p.add_argument('--particles', type=int, default=512)
    p.add_argument('--decoder', choices=('additive', 'modulated'), default='additive')
    args = p.parse_args()
    if args.prepare:
        prepare(args.data, args.tasks, args.observations, args.workers)
        return
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    data_rng = torch.Generator(device=args.device).manual_seed(args.seed + 100000)
    torch.set_float32_matmul_precision('high')
    args.output.mkdir(parents=True, exist_ok=False)
    data = {k: {name: x.to(args.device) for name, x in v.items()} for k, v in
            torch.load(args.data / 'train.pt', weights_only=True).items()}
    validation = {k: {name: x.to(args.device) for name, x in v.items()} for k, v in
                  torch.load(args.data / 'validation.pt', weights_only=True).items()}
    model = (RelationBelief(RelationBeliefConfig(particles=args.particles,decoder=args.decoder)) if args.kind == 'functions'
             else ConditionalRelation(attentive=args.kind == 'attentive')).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    candidates = {n: candidate_set(n, args.device) for n in (3, 4)}
    started = perf_counter()
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config['parameters'] = sum(p.numel() for p in model.parameters())
    (args.output / 'configuration.json').write_text(json.dumps(config, indent=2)+'\n')
    def save(name, update):
        torch.save(dict(kind=args.kind, model=model.state_dict(), configuration=config, update=update,
                        model_config=model.configuration() if args.kind == 'functions' else
                        dict(width=192, attentive=args.kind == 'attentive')),
                   args.output / name)
    save('initial.pt', 0)
    with (args.output / 'training.jsonl').open('w') as log:
        for update in range(1, args.updates + 1):
            n = 3 + update % 2
            index = torch.randint(len(data[n]['indices']), (args.batch,), device=args.device, generator=data_rng)
            batch = {key: value[index] for key, value in data[n].items()}
            context = (0, 2, 4, 8, 16, 24)[(update // 2) % 6]
            optimizer.zero_grad(set_to_none=True)
            objective, marginal = loss(model, args.kind, batch, *candidates[n], context)
            objective.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
            lr = 3e-4 * (.1 + .9 * .5 * (1 + math.cos(math.pi * update / args.updates)))
            for group in optimizer.param_groups:
                group['lr'] = lr
            if update == 1 or update % 100 == 0:
                row = dict(update=update, objective=float(objective.detach()), marginal=float(marginal.detach()),
                    gradient=float(norm), seconds=perf_counter()-started)
                if update % 500 == 0 or update == args.updates:
                    with torch.no_grad():
                        row['validation'] = {f'{size}_{c}': [float(v) for v in loss(model,args.kind,
                            {k:x[:128] for k,x in validation[size].items()},*candidates[size],c)]
                            for size in (3,4) for c in (0,4,16)}
                    save('last.pt', update)
                log.write(json.dumps(row)+'\n'); log.flush()
                print(json.dumps(row),flush=True)
    save('last.pt', args.updates)
    (args.output / 'finished.json').write_text(json.dumps(dict(updates=args.updates,
        seconds=perf_counter()-started, parameters=config['parameters']))+'\n')


if __name__ == '__main__':
    main()
