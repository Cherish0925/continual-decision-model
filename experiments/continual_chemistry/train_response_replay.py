"""Train response updates against later returns, holding the original core fixed."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch.nn import functional as F

from cdm.response_replay import ResponseReplay, ResponseReplayConfig
from cdm.conditional_response import observable_log_probabilities


def batch(rows, device, mode):
    result = {k: torch.stack([r[k] for r in rows]).to(device)
              for k in ('query_base', 'query_x', 'query_inputs')}
    support_keys = ('support_base', 'support_x', 'support_inputs', 'support_before', 'support_after')
    for k in support_keys:
        source = k.replace('support_', 'stored_') if mode == 'stored' and k in ('support_base', 'support_x') else k
        result[k] = torch.nn.utils.rnn.pad_sequence([r[source] for r in rows], batch_first=True).to(device)
    n = result['support_x'].shape[1]
    if n == 0:
        for k in support_keys:
            x = result[k]
            result[k] = x.new_zeros(len(rows), 1, *x.shape[2:])
        n = 1
    result['valid'] = torch.arange(n, device=device)[None] < torch.tensor([len(r['support_x']) for r in rows], device=device)[:, None]
    return result, torch.tensor([r['label'] for r in rows], device=device), torch.tensor([r['before'] for r in rows], device=device)


def predict(model, data):
    return model(data['query_base'], data['query_x'], data['query_inputs'],
                 data['support_base'], data['support_x'], data['support_inputs'],
                 data['support_before'], data['support_after'], data['valid'])[:, 0]


@torch.no_grad()
def evaluate(model, rows, device, size):
    results = []
    for start in range(0, len(rows), size):
        group = rows[start:start + size]
        data, label, before = batch(group, device, model.config.mode)
        logp = observable_log_probabilities(predict(model, data), before)
        base = observable_log_probabilities(data['query_base'][:, 0], before)
        nll = F.nll_loss(logp, label, reduction='none').cpu().tolist()
        base_nll = F.nll_loss(base, label, reduction='none').cpu().tolist()
        for r, a, b in zip(group, nll, base_nll):
            results.append(dict(episode=r['episode'], recipe_group=r['recipe_group'], position=r['position'],
                updated_nll=a, base_nll=b, original_nll=-float(r['original_logp'][r['label']]), support=len(r['support_x'])))
    groups = sorted({r['recipe_group'] for r in results})
    totals = np.array([[sum(r[k] for r in results if r['recipe_group'] == g)
                       for k in ('updated_nll', 'base_nll', 'original_nll')] for g in groups])
    counts = np.array([sum(r['recipe_group'] == g for r in results) for g in groups])
    rng = np.random.default_rng(1031).integers(len(groups), size=(10000, len(groups)))
    means = totals[rng].sum(1) / counts[rng].sum(1)[:, None]
    return dict(transitions=len(results), recipe_groups=len(groups),
        mean_nll=dict(zip(('updated', 'base', 'original'), (totals.sum(0) / counts.sum()).tolist())),
        update_minus_base_95ci=np.quantile(means[:, 0] - means[:, 1], [.025, .975]).tolist(),
        predictions=results)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--mode', choices=['calibration', 'stored', 'refreshed', 'relation'], required=True)
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--batch', type=int, default=128)
    p.add_argument('--lr', type=float, default=.01)
    p.add_argument('--seed', type=int, default=1031)
    p.add_argument('--device', default='cuda:0')
    args = p.parse_args()
    torch.set_num_threads(1); torch.manual_seed(args.seed)
    data = torch.load(args.cache, weights_only=True)
    train = [r for r in data['rows'] if r['split'] == 'train']
    validation = [r for r in data['rows'] if r['split'] != 'train']
    model = ResponseReplay(config=ResponseReplayConfig(mode=args.mode)).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    args.output.mkdir(parents=True, exist_ok=False)
    started = perf_counter()
    with (args.output / 'training.jsonl').open('w') as log:
        for epoch in range(args.epochs):
            order = torch.randperm(len(train)).tolist()
            total = 0.
            for start in range(0, len(order), args.batch):
                rows = [train[i] for i in order[start:start + args.batch]]
                x, label, before = batch(rows, args.device, args.mode)
                loss = F.nll_loss(observable_log_probabilities(predict(model, x), before), label)
                optimizer.zero_grad(set_to_none=True); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                optimizer.step(); total += float(loss.detach()) * len(rows)
            record = dict(epoch=epoch + 1, train_nll=total / len(train), seconds=perf_counter() - started)
            log.write(json.dumps(record) + '\n'); log.flush()
            if epoch % 5 == 0 or epoch + 1 == args.epochs:
                print(json.dumps(record), flush=True)
    result = evaluate(model, validation, args.device, args.batch)
    result.update(config=asdict(model.config), train_transitions=len(train), epochs=args.epochs,
        seed=args.seed, seconds=perf_counter() - started,
        parent=data['checkpoint'], selection='fixed final epoch; development predictions only',
        learned={name: p.detach().cpu().tolist() for name, p in model.named_parameters()})
    torch.save(dict(config=asdict(model.config), model={k: v.detach().cpu() for k, v in model.state_dict().items()},
                    parent=data['checkpoint']), args.output / 'update.pt')
    (args.output / 'development.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'predictions'}), flush=True)


if __name__ == '__main__':
    main()
