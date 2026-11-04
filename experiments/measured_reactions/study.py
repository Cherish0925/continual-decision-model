"""Select measurements on held-out reaction screens with a CDM, GRU or classical policy."""
import argparse
import csv
import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from cdm.active_experiment import ActiveExperimentConfig, ActiveExperimentPolicy
from cdm.conditional_measurement import ConditionalMeasurement
from cdm.categorical_measurement import CategoricalMeasurement, relation_features
from cdm.core import ContinualCore, CoreConfig


FEATURES = 64
LEARNED = ('cdm', 'gru', 'blind', 'conditional', 'conditional_blind')
RELATIONAL = ('categorical', 'relational', 'relational_gru', 'relational_blind', 'relational_kernel')
ALL_LEARNED = LEARNED + RELATIONAL
FEEDBACK_CONTROLS = {'cdm': 'blind', 'conditional': 'conditional_blind',
                     'categorical': 'categorical_masked', 'relational': 'relational_blind',
                     'relational_gru': 'relational_gru_masked', 'relational_kernel': 'relational_kernel_masked'}
REFERENCES = ('random', 'prior', 'kernel_ucb', 'kernel_diverse', 'kernel_local')


def add_measurement(model, mode):
    width = model.config.action_width
    if mode in ('conditional', 'conditional_blind'):
        model.measurement = ConditionalMeasurement(FEATURES, width)
    elif mode in RELATIONAL:
        model.measurement = CategoricalMeasurement(FEATURES, width, named_prior=mode == 'categorical')
        if mode == 'relational_kernel':
            for p in model.parameters():
                p.requires_grad_(False)
            for p in model.measurement.parameters():
                p.requires_grad_(True)


class Screens:
    def __init__(self, path, device):
        with np.load(path, allow_pickle=False) as f:
            meta = json.loads(str(f['meta']))
            offsets, codes, y = f['offsets'], f['conditions'], f['peak_area'] / 100
        self._load(meta, offsets, codes, y, device)

    @classmethod
    def external(cls, meta, offsets, codes, y, device, source):
        data = cls.__new__(cls)
        if tuple(meta['roles']) != tuple(source.meta['roles']) or any(s['split'] != 'external' for s in meta['screens']):
            raise ValueError('External screens must use the source action schema and a separate split')
        data._load(meta, offsets, codes, y, device, source)
        return data

    def _load(self, meta, offsets, codes, y, device, source=None):
        self.meta = meta
        screens = meta['screens']
        self.split = {s: np.array([i for i, v in enumerate(screens) if v['split'] == s], dtype=int)
                      for s in sorted({v['split'] for v in screens})}
        maximum = np.diff(offsets).max()
        self.index = np.zeros((len(screens), maximum), np.int64)
        self.length = np.diff(offsets)
        train = np.zeros(len(codes), dtype=bool)
        for i, (a, b) in enumerate(zip(offsets, offsets[1:])):
            self.index[i, :b-a] = np.arange(a, b)
            if screens[i]['split'] == 'train':
                train[a:b] = True
        # Fixed name encodings and a train-only prior, shared by all reference runs.
        features = np.zeros((len(codes), FEATURES), np.float32)
        prior = np.zeros(len(codes), np.float32)
        self.prior_mean = source.prior_mean if source is not None else float(y[train].mean())
        self.prior_by_role = []
        for j, (role, vocab) in enumerate(zip(self.meta['roles'], self.meta['vocab'])):
            basis = np.zeros((len(vocab), FEATURES), np.float32)
            for i, name in enumerate(vocab):
                h = hashlib.sha256((role + '\0' + name).encode()).digest()
                basis[i, int.from_bytes(h[:4], 'big') % FEATURES] = 1 if h[4] & 1 else -1
            features += basis[codes[:, j]] / np.sqrt(codes.shape[1])
            if source is None:
                counts = np.bincount(codes[train, j], minlength=len(vocab))
                sums = np.bincount(codes[train, j], weights=y[train], minlength=len(vocab))
                means = (sums + 20 * self.prior_mean) / (counts + 20)
            else:
                means = np.array([source.prior_by_role[j].get(name, self.prior_mean) for name in vocab])
            self.prior_by_role.append(dict(zip(vocab, means)))
            prior += means[codes[:, j]] / codes.shape[1]
        self.x, self.y, self.codes, self.prior = (
            torch.as_tensor(x, device=device) for x in (features, y, codes, prior))
        self.device = device

    def batch(self, ids, ranks):
        # Sort independent random keys, placing padding last. Row 0 is a common random first probe.
        length = self.length[ids]
        m = length.max()
        mask = np.arange(m)[None] < length[:, None]
        order = np.argsort(np.where(mask, ranks[:, :m], np.inf), axis=1)
        ix = torch.as_tensor(np.take_along_axis(self.index[ids, :m], order, axis=1), device=self.device)
        present = torch.as_tensor(mask, device=self.device)
        return self.x[ix], self.y[ix], self.codes[ix], self.prior[ix], present


def make_policy(width, slots, init_native=None):
    config = ActiveExperimentConfig(event_dim=FEATURES + 2, evidence_dim=FEATURES + 2,
                                    candidate_dim=FEATURES, outcome_dim=1, action_width=width)
    if init_native:
        return ActiveExperimentPolicy.from_native_checkpoint(init_native, config)
    core = ContinualCore(CoreConfig(event_dim=FEATURES + 2, record_dim=1, actions=1, width=width,
                                    slots=slots, heads=4, hidden=width, action_width=width), torch.zeros(1, 1))
    return ActiveExperimentPolicy(config, core)


def kernel_score(codes, prior, history, observations, use_values):
    idx = torch.stack(history, 1)
    b = torch.arange(len(codes), device=codes.device)
    measured = codes[b[:, None], idx]
    def kernel(x, z):
        return (-4 * (x[:, :, None] != z[:, None, :]).float().mean(-1)).exp()
    k = kernel(codes, measured)
    kk = kernel(measured, measured) + .05 * torch.eye(len(history), device=codes.device)
    prior_at_probes = prior[b[:, None], idx]
    diff = (torch.stack(observations, 1) - prior_at_probes).unsqueeze(-1)
    if not use_values:
        diff = torch.zeros_like(diff)
    solved = torch.linalg.solve(kk, torch.cat((diff, k.transpose(1, 2)), -1))
    mean = prior + (k @ solved[:, :, :1]).squeeze(-1)
    variance = (1 - (k * solved[:, :, 1:].transpose(1, 2)).sum(-1)).clamp_min(0)
    return mean + .25 * variance.sqrt()


def rollout(batch, budget, mode, model=None, learning=False):
    x, table, codes, prior, available = batch
    b = torch.arange(len(x), device=x.device)
    selected = torch.zeros(len(b), device=x.device, dtype=torch.long)
    available = available.clone()
    best = torch.zeros_like(table[:, 0])
    curve, history, observed, events = [], [], [], []
    logs, values, predictions, targets, rewards, entropies = [], [], [], [], [], []
    state = model.initial(len(b), x.device) if model is not None else None
    categorical = mode.startswith(('categorical', 'relational'))
    relational = mode.startswith('relational')
    kernel_only = mode.startswith('relational_kernel')
    conditional = mode in ('conditional', 'conditional_blind') or categorical
    visible_values = mode not in ('blind', 'conditional_blind', 'relational_blind') and not mode.endswith('_masked')
    present = available.clone()
    encoded = (model.measurement.prepare(x, codes) if categorical else model.measurement.prepare(x)) if conditional else None
    if relational:
        x = relation_features(codes, present)
    for step in range(budget):
        if step and model is not None:
            if relational:
                shown = torch.stack(observed, 1)
                x = relation_features(codes, present, torch.stack(history, 1),
                                      shown if visible_values else torch.zeros_like(shown))
            if kernel_only:
                out = {'logits': x.new_zeros(x.shape[:2]).masked_fill(~available, -torch.inf),
                       'value': x.new_zeros(len(x))}
            elif mode in ('gru', 'relational_gru', 'relational_gru_masked'):
                h = model.core.fast_embeddings(state['h'], model.event(events[-1])[None])[0]
                state = {'h': h, 'z': torch.zeros_like(state['z'])}
            else:
                memory = torch.stack(events, 1)
                state = model.advance(events[-1], memory, torch.ones_like(memory[:, :, 0], dtype=torch.bool), state)
            if not kernel_only:
                out = model.readout(state, x, available)
            if conditional:
                mu, sd = model.measurement(encoded, torch.stack(history, 1), torch.stack(observed, 1),
                                           use_values=visible_values)
                score = out['logits'] + mu + .25 * sd
            else:
                mu = out['outcomes'][:, :, 0].sigmoid()
                score = out['logits'] + mu
            if learning:
                dist = torch.distributions.Categorical(logits=score)
                selected = dist.sample()
                logs.append(dist.log_prob(selected))
                entropies.append(dist.entropy())
                values.append(out['value'])
                predictions.append(mu[b, selected])
            else:
                selected = score.argmax(1)
        elif step:
            if mode in ('kernel_ucb', 'kernel_diverse', 'kernel_local'):
                base = torch.full_like(prior, .5) if mode == 'kernel_local' else prior
                score = kernel_score(codes, base, history, observed, use_values=mode != 'kernel_diverse')
            elif mode == 'prior':
                score = prior
            elif mode == 'random':
                # The menu is already in uniform random order.
                score = -torch.arange(x.shape[1], device=x.device, dtype=x.dtype).expand(len(x), -1)
            else:
                raise ValueError(mode)
            selected = score.masked_fill(~available, -torch.inf).argmax(1)
        y = table[b, selected]
        new_best = torch.maximum(best, y)
        if step and learning:
            rewards.append(new_best - best)
            targets.append(y)
        best = new_best
        curve.append(best)
        history.append(selected)
        observed.append(y)
        if model is not None:
            visible = y if visible_values else torch.zeros_like(y)
            events.append(torch.cat((x[b, selected], visible[:, None],
                                     torch.full_like(y[:, None], (budget - step - 1) / budget)), -1))
        available = available.scatter(1, selected[:, None], False)
    if not learning:
        return torch.stack(curve, 1)
    # With a common first probe, telescoping improvement rewards optimize the best
    # measurement at budget. Future chosen measurements, not the table optimum, supply returns.
    ret = torch.stack(rewards).flip([0]).cumsum(0).flip([0])
    val = torch.stack(values)
    return (-(torch.stack(logs) * (ret - val).detach()).mean() + .5 * F.mse_loss(val, ret)
            + F.mse_loss(torch.stack(predictions), torch.stack(targets)) - .01 * torch.stack(entropies).mean())


def evaluation(data, model, mode, budget, split, batch_size, starts):
    rows = []
    with torch.inference_mode():
        for first in range(starts):
            for ids in np.array_split(data.split[split], np.arange(batch_size, len(data.split[split]), batch_size)):
                screens = [data.meta['screens'][i] for i in ids]
                ranks = np.stack([np.random.default_rng(int(s['id'][:16], 16) ^ (104729 * first)).random(
                    data.index.shape[1]) for s in screens])
                curves = rollout(data.batch(ids, ranks), budget, mode, model).cpu().numpy()
                for screen, curve in zip(screens, curves):
                    rows.append(dict(screen=screen['id'], notebook=screen['notebook'], family=screen['family'],
                                     start=first, **{f'best_{i+1}': round(100 * float(y), 5)
                                                    for i, y in enumerate(curve)}))
    return rows


def summarize(rows, budget):
    result = {}
    for k in sorted({1, min(4, budget), budget}):
        notebooks = defaultdict(list)
        for row in rows:
            notebooks[row['family'], row['notebook']].append(row[f'best_{k}'])
        families = defaultdict(list)
        for (family, _), v in notebooks.items():
            families[family].append(float(np.mean(v)))
        result[f'best_{k}'] = round(float(np.mean([np.mean(v) for v in families.values()])), 3)
    return result


def load_policy(checkpoint, mode, device, budget):
    saved = torch.load(checkpoint, weights_only=True, map_location='cpu')
    if mode != saved['mode'] or budget != saved['budget']:
        raise ValueError('Mode and budget must match the trained policy')
    active = ActiveExperimentConfig(**saved['config']['active_config'])
    core_cfg = CoreConfig(**saved['config']['core_config'])
    model = ActiveExperimentPolicy(active, ContinualCore(core_cfg, saved['model']['core.action_values']))
    add_measurement(model, mode)
    model.load_state_dict(saved['model'])
    return model.to(device).eval()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['train', 'eval'])
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--mode', choices=ALL_LEARNED + REFERENCES, required=True)
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--device', default='cpu')
    p.add_argument('--threads', type=int, default=1)
    p.add_argument('--width', type=int, default=128)
    p.add_argument('--slots', type=int, default=8)
    p.add_argument('--batch', type=int, default=64)
    p.add_argument('--budget', type=int, default=8, choices=range(2, 13))
    p.add_argument('--updates', type=int, default=1000)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--starts', type=int, default=4)
    p.add_argument('--split', choices=['dev', 'test'], default='dev')
    p.add_argument('--without-readings', action='store_true',
                   help='Evaluate the same CDM or conditional checkpoint without returned signals')
    p.add_argument('--init-native', type=Path, help='Optional DigitalPolicy native checkpoint (core only)')
    a = p.parse_args()
    if a.without_readings and (a.action != 'eval' or a.mode not in FEEDBACK_CONTROLS):
        p.error('--without-readings needs a feedback-using learned evaluation policy')
    if min(a.threads, a.batch, a.starts, a.updates, a.slots) < 1 or a.width % 4:
        p.error('Require positive sizes and width divisible by 4')
    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    data = Screens(a.data, a.device)
    a.run.mkdir(parents=True, exist_ok=True)
    model = None
    ckpt = a.run / 'model.pt'
    if a.action == 'train':
        if a.mode not in ALL_LEARNED or ckpt.exists() or (a.run / 'train.json').exists():
            p.error('Training needs a learned policy and a fresh run directory')
        model = make_policy(a.width, a.slots, a.init_native)
        add_measurement(model, a.mode)
        model = model.to(a.device)
        optim = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=a.lr)
        started = time.perf_counter()
        with (a.run / 'learning.jsonl').open('x') as progress:
            for update in range(1, a.updates + 1):
                ids = rng.choice(data.split['train'], a.batch, replace=True)
                ranks = rng.random((a.batch, data.index.shape[1]))
                loss = rollout(data.batch(ids, ranks), a.budget, a.mode, model, learning=True)
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite training loss')
                optim.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optim.step()
                if update == 1 or update % 25 == 0 or update == a.updates:
                    row = dict(update=update, queries=update * a.batch * a.budget,
                               loss=float(loss.detach()), elapsed_seconds=time.perf_counter() - started)
                    progress.write(json.dumps(row) + '\n')
                    progress.flush()
                    print(json.dumps(row), flush=True)
        torch.save(dict(model=model.state_dict(), config=model.configuration(), mode=a.mode,
                        budget=a.budget, seed=a.seed), ckpt)
        cfg = {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()}
        cfg.update(queries=a.updates * a.batch * a.budget, last_training_loss=float(loss.detach()))
        cfg.update(elapsed_seconds=time.perf_counter() - started, torch_version=torch.__version__,
                   cuda_version=torch.version.cuda, parameters=sum(p.numel() for p in model.parameters()),
                   trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad))
        (a.run / 'train.json').write_text(json.dumps(cfg, indent=2) + '\n')
        print(f"trained {a.mode}: {cfg['queries']} queries")
        return
    if a.mode in ALL_LEARNED:
        model = load_policy(ckpt, a.mode, a.device, a.budget)
    mode = FEEDBACK_CONTROLS[a.mode] if a.without_readings else a.mode
    rows = evaluation(data, model, mode, a.budget, a.split, a.batch, a.starts)
    name = a.split + ('_without_readings' if a.without_readings else '')
    with (a.run / f'{name}.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    result = summarize(rows, a.budget)
    (a.run / f'{name}.json').write_text(json.dumps(result, indent=2) + '\n')
    print(a.mode, name, result)


if __name__ == '__main__':
    main()
