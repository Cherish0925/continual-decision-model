"""Learn joint response functions from observations, then condition them online.

The finite latent mixture is a predictive-model baseline. Its particles and
decoder are learned from experiment sequences; no generating formula or task
identity is an input. One latent index is shared across all candidate queries.
"""
from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .conditional_response import observable_log_probabilities


@dataclass(frozen=True)
class RelationBeliefConfig:
    components: int = 8
    levels: int = 4
    particles: int = 512
    latent: int = 64
    width: int = 192
    decoder: str = 'additive'


def query_features(q, present):
    unit = F.normalize(q.float(), dim=-1)
    return torch.cat((q, unit, q.square(), present.expand_as(q).float()), -1)


class RelationBelief(nn.Module):
    def __init__(self, config=RelationBeliefConfig()):
        super().__init__()
        self.config = config
        c = config
        self.particles = nn.Parameter(torch.randn(c.particles, c.latent))
        self.prior = nn.Parameter(torch.zeros(c.components + 1, c.particles))
        self.query = nn.Linear(4 * c.components, c.width)
        self.latent = nn.Linear(c.latent, c.width * (2 if c.decoder == 'modulated' else 1))
        self.decoder = nn.Sequential(nn.SiLU(), nn.Linear(c.width, c.width), nn.SiLU(),
                                     nn.Linear(c.width, c.levels))

    def forward(self, proportions, components):
        """Return K x A x L log probabilities for one public candidate set."""
        x = self.query(query_features(proportions, components))
        latent = self.latent(self.particles)
        if self.config.decoder == 'modulated':
            scale, shift = latent.chunk(2, -1)
            hidden = x[None] * (1 + scale[:, None]) + shift[:, None]
        else:
            hidden = x[None] + latent[:, None]
        return self.decoder(hidden).log_softmax(-1)

    def log_prior(self, components):
        return self.prior[components.sum(-1).long()].log_softmax(-1)

    def configuration(self):
        return asdict(self.config)

    def conditioned(self, queries, components, support, before, after, valid):
        """Predict a batch and return the observed sequence's marginal likelihood.

        One posterior explains every query in a lane. Padding and censored
        outcomes do not become observations of the hidden intrinsic response.
        """
        predictions, evidence = [], []
        for q, c, x, pre, post, mask in zip(queries, components, support, before, after, valid):
            bank = self(torch.cat((q, x[mask]), 0), c)
            prior = self.log_prior(c)
            if mask.any():
                observed = observable_log_probabilities(bank[:, len(q):], pre[mask][None])
                labels = post[mask][None, :, None].expand(len(prior), -1, 1)
                mass = prior + observed.gather(-1, labels).squeeze(-1).sum(-1)
            else:
                mass = prior
            evidence.append(mass.logsumexp(-1))
            predictions.append(torch.logsumexp(mass.log_softmax(-1)[:, None, None] + bank[:, :len(q)], 0))
        return torch.stack(predictions), torch.stack(evidence)


def condition(log_weights, log_likelihood):
    result = log_weights + log_likelihood
    return result - result.logsumexp(-1, keepdim=True)


def predict(log_weights, log_functions, current):
    """Marginal outcomes after preserving a shared latent across queries."""
    observed = observable_log_probabilities(log_functions, current)
    return torch.logsumexp(log_weights[..., :, None, None] + observed, -3)


def information(log_weights, log_functions, current):
    observed = observable_log_probabilities(log_functions, current)
    p = observed.exp()
    mixed = (log_weights.exp()[..., :, None, None] * p).sum(-3)
    entropy = -(mixed * mixed.clamp_min(1e-30).log()).sum(-1)
    within = -(p * observed.nan_to_num(neginf=0)).sum(-1)
    return entropy - (log_weights.exp()[..., :, None] * within).sum(-2)


class ConditionalRelation(nn.Module):
    """Conditional Neural Process reference with the same observed sequences."""
    def __init__(self, width=192, components=8, levels=4, attentive=False):
        super().__init__()
        self.width, self.components, self.levels = width, components, levels
        self.encoder = nn.Sequential(nn.Linear(4 * components + 2 * levels, width), nn.SiLU(),
                                     nn.Linear(width, width), nn.SiLU())
        self.decoder = nn.Sequential(nn.Linear(4 * components + width + 1, width), nn.SiLU(),
                                     nn.Linear(width, width), nn.SiLU(), nn.Linear(width, levels))
        self.attentive = attentive
        if attentive:
            self.keys = nn.Linear(4 * components, width)
            self.queries = nn.Linear(4 * components, width)
            self.attention = nn.MultiheadAttention(width, 4, batch_first=True)

    def forward(self, queries, components, support, before, after, valid):
        x = query_features(support, components[:, None])
        y = torch.cat((F.one_hot(before, self.levels), F.one_hot(after, self.levels)), -1).float()
        h = self.encoder(torch.cat((x, y), -1))
        count = valid.sum(-1, keepdim=True)
        pooled = (h * valid[..., None]).sum(-2) / count.clamp_min(1)
        q = query_features(queries, components[:, None])
        if self.attentive:
            keys = self.keys(x)
            # A zero slot represents an empty context. It is hidden when real
            # evidence exists, so no unobserved target contributes to attention.
            keys = torch.cat((torch.zeros_like(keys[:, :1]), keys), 1)
            values = torch.cat((torch.zeros_like(h[:, :1]), h), 1)
            mask = torch.cat((count > 0, ~valid), 1)
            pooled, _ = self.attention(self.queries(q), keys, values, key_padding_mask=mask, need_weights=False)
        else:
            pooled = pooled[:, None].expand(-1, q.shape[1], -1)
        count = count.float().log1p()[:, None].expand(-1, q.shape[1], -1)
        return self.decoder(torch.cat((q, pooled, count), -1))
