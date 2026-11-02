"""Digital experiment extension with changing response laws.

This is separate from official DiscoveryWorld. It retains its public action,
history, censoring and cost interfaces while varying the response generator.
Hidden law descriptions are used by the simulator and result stratification;
they are never included in policy observations.
"""
from copy import deepcopy
import random

import torch
from torch.nn import functional as F

from .active_environment import ActiveChemicalEnvironment
from .recipes import primitive_recipes


TRAIN_FAMILIES = ('cosine', 'metric', 'interaction', 'inhibition', 'multiple_solutions')


def response_levels(counts, law):
    """Deterministic intrinsic levels; public observations censor at current rust."""
    n = law['components']
    q = F.normalize(counts[..., :n].double(), dim=-1)
    target = torch.as_tensor(law['target'], device=q.device, dtype=q.dtype)
    t = F.normalize(target, dim=-1)
    family = law['family']
    cosine = (q * t).sum(-1)
    if family == 'metric':
        weights = torch.as_tensor(law['metric'], device=q.device, dtype=q.dtype).sqrt()
        cosine = (F.normalize(q * weights, dim=-1) * F.normalize(t * weights, dim=-1)).sum(-1)
    if family in {'interaction', 'interaction_inhibition'}:
        # Pair interactions alter the geometry while keeping target and query
        # in the same units. Cross terms make the response non-additive.
        pairs = torch.triu_indices(n, n, offset=1, device=q.device)
        qf = torch.cat((q, law['interaction'] * q[..., pairs[0]] * q[..., pairs[1]]), -1)
        tf = torch.cat((t, law['interaction'] * t[pairs[0]] * t[pairs[1]]), -1)
        cosine = (F.normalize(qf, dim=-1) * F.normalize(tf, dim=-1)).sum(-1)
    if family == 'multiple_solutions':
        other = F.normalize(torch.as_tensor(law['second_target'], device=q.device, dtype=q.dtype), dim=-1)
        cosine = torch.maximum(cosine, (q * other).sum(-1))
    thresholds = law.get('thresholds', (.99, .66, .33))
    level = torch.where(cosine >= thresholds[0], 0,
                        torch.where(cosine >= thresholds[1], 1, torch.where(cosine >= thresholds[2], 2, 3)))
    if family in {'inhibition', 'interaction_inhibition'}:
        p = counts[..., :n].double() / counts[..., :n].sum(-1, keepdim=True).clamp_min(1)
        tp = target / target.sum()
        inhibitor = law['inhibitor']
        inhibited = p[..., inhibitor] > tp[inhibitor] + law['tolerance']
        level = torch.where(inhibited, torch.full_like(level, 3), level)
    return level


def sample_law(rng, n, families, *, wider=False):
    choices = primitive_recipes(n)
    metric_range = (-1.4, 1.4) if wider else (-.9, .9)
    return dict(family=rng.choice(families), components=n, target=list(rng.choice(choices)),
        second_target=list(rng.choice(choices)), metric=[10 ** rng.uniform(*metric_range) for _ in range(n)],
        interaction=rng.uniform(3., 7.) if wider else rng.uniform(1., 4.),
        inhibitor=rng.randrange(n), tolerance=rng.uniform(.04, .18))


def sample_lifetime(rng, families, length=12, *, wider=False):
    n = rng.choice((3, 4))
    laws, seen, result = [], {}, []
    for i in range(length):
        mode = 'new' if i < 2 else rng.choices(('new', 'return', 'changed'), (.35, .45, .20))[0]
        if mode == 'new':
            law = sample_law(rng, n, families, wider=wider)
            laws.append(law)
            identity = len(laws) - 1
        else:
            identity = rng.randrange(len(laws))
            law = deepcopy(laws[identity])
            if mode == 'changed':
                # A nearby proportion is a changed law, not an exact return.
                candidates = primitive_recipes(n)
                target = torch.tensor(law['target'], dtype=torch.float64)
                q = torch.tensor(candidates, dtype=torch.float64)
                similarity = (F.normalize(q, dim=-1) * F.normalize(target, dim=-1)).sum(-1)
                nearest = [j for j in similarity.argsort(descending=True).tolist() if candidates[j] != tuple(law['target'])][:8]
                law['target'] = list(candidates[rng.choice(nearest)])
                laws.append(law)
                identity = len(laws) - 1
        gap = i - seen[identity] - 1 if identity in seen else None
        seen[identity] = i
        result.append(dict(law=law, exposure=mode, identity=identity, intervening_tasks=gap))
    return result


class ResponseFamilyEnvironment(ActiveChemicalEnvironment):
    def __init__(self, root, batch, seed, device, budget=1000, families=TRAIN_FAMILIES,
                 manifest=None, wider=False):
        self.law_rng = [random.Random(seed + 91827 + 104729 * lane) for lane in range(batch)]
        self.families, self.wider = tuple(families), wider
        self.law_streams = [None] * batch
        self.current_laws = [None] * batch
        self.task_metadata = [None] * batch
        self.fixed_manifest = manifest
        super().__init__(root, batch, seed, device, budget=budget, schedule='mixed')

    def load_entries(self, root, split):
        # These laws are generated here; no DiscoveryWorld trajectory files
        # contribute to their observations, transitions or preparation costs.
        return []

    def reset_finished(self):
        boundary = self.terminal.clone()
        lifetime = torch.zeros_like(boundary)
        for lane in boundary.nonzero().flatten().tolist():
            position = self.task_index[lane] % self.lifetime_length
            if position == 0:
                lifetime[lane] = True
                if self.fixed_manifest is None:
                    self.law_streams[lane] = sample_lifetime(self.law_rng[lane], self.families,
                                                           self.lifetime_length, wider=self.wider)
                else:
                    self.law_streams[lane] = self.fixed_manifest[lane]['tasks']
            item = self.law_streams[lane][position]
            self.current_laws[lane] = item['law']
            self.task_metadata[lane] = item
            n = item['law']['components']
            candidates = primitive_recipes(n)
            self.counts[lane].zero_()
            self.counts[lane, :len(candidates), :n] = torch.tensor(candidates, device=self.device)
            self.allowed[lane].zero_()
            self.allowed[lane, :len(candidates)] = True
            self.allowed[lane, self.leave_index] = True
            self.component_mask[lane].zero_()
            self.component_mask[lane, :n] = 1
            # Identical public preparation cost for every generator and method.
            self.actions[lane] = 14
            self.rust[lane] = 3
            self.cursor[lane] = 0
            self.history[lane].zero_()
            self.history_mask[lane].zero_()
            self.seen[lane].zero_()
            self.endpoint[lane].fill_(-1)
            self.task_index[lane] += 1
        self.terminal.zero_()
        self.new_task, self.new_lifetime = boundary, lifetime
        return boundary, lifetime

    def response(self, counts, recipe, before):
        levels = torch.stack([response_levels(q, law) for q, law in zip(counts, self.current_laws)])
        return torch.minimum(before, levels)

    def step(self, choice):
        start = len(self.task_rows)
        result = super().step(choice)
        for row in self.task_rows[start:]:
            item = self.task_metadata[row['lane']]
            row.update(family=item['law']['family'], exposure=item['exposure'],
                       intervening_tasks=item['intervening_tasks'])
        return result
