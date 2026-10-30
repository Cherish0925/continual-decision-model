"""Learn response distributions from the returned history of mixture experiments.

The module receives proportions and public responses, never a generating recipe
or a task identity. A candidate queries its relations to the previous mixtures.
Separate pools retain exact improvements and censored non-improvements without
changing their frequencies in the likelihood objective.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class ConditionalResponseConfig:
    components: int = 8
    levels: int = 4
    width: int = 48
    history_capacity: int = 256
    candidate_chunk: int = 32


def observable_log_probabilities(latent_logits, current):
    """Log P(Y=y), where a persistent response is Y=min(current, R).

    A non-improvement observes R >= current. In particular it is not an exact
    label R=current. logsumexp retains gradients even in small-probability tails.
    """
    logp = latent_logits.log_softmax(-1)
    levels = torch.arange(logp.shape[-1], device=logp.device)
    current = current.long()
    tail = torch.logsumexp(logp.masked_fill(levels < current[..., None], -torch.inf), -1)
    return torch.where(levels < current[..., None], logp,
                       torch.where(levels == current[..., None], tail[..., None], -torch.inf))


class ConditionalResponse(nn.Module):
    """Candidate-conditioned experimental evidence and continuing computation.

    Shared component functions make the direct relation representation invariant
    to a common permutation of ingredient identities. Scaling an entire recipe
    changes neither its proportions nor this response prediction. Dose-dependent
    cost remains the responsibility of a separate, explicitly trained readout.
    """

    def __init__(self, core_features: int, config=ConditionalResponseConfig()):
        super().__init__()
        self.config = config
        d, k = config.width, config.levels
        self.component = nn.Sequential(nn.Linear(4, d), nn.GELU(), nn.Linear(d, d))
        self.query_component = nn.Sequential(nn.Linear(2, d), nn.GELU(), nn.Linear(d, d))
        self.returned = nn.Linear(2 * k, d)
        self.relation_key = nn.Linear(d, d)
        self.relation_value = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, d))
        self.core_projection = nn.Sequential(nn.Linear(core_features, d), nn.LayerNorm(d))
        self.query = nn.Linear(2 * d, d)
        self.pool = nn.Sequential(nn.Linear(2 * k * (d + 1), d), nn.GELU())
        # The query-history interaction also changes how the persistent state is
        # read. Both paths receive gradient from the same returned response.
        self.output = nn.Sequential(nn.Linear(4 * d, d), nn.GELU(), nn.Linear(d, k))
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)
        self.prior = nn.Parameter(torch.zeros(config.components + 1, k, k))

    def forward(self, proportions, component_mask, current, history_proportions,
                history_before, history_after, history_mask, core_features):
        cfg = self.config
        core = self.core_projection(core_features)
        current = current.long()
        n = component_mask.sum(-1).long()
        before, after = history_before.long(), history_after.long()
        category = after + cfg.levels * (before == after).long()
        response = self.returned(torch.cat((F.one_hot(before, cfg.levels),
                                           F.one_hot(after, cfg.levels)), -1).float())
        groups = F.one_hot(category, 2 * cfg.levels).bool() & history_mask[..., None]
        pieces, hidden_pieces = [], []
        for start in range(0, proportions.shape[1], cfg.candidate_chunk):
            q = proportions[:, start:start + cfg.candidate_chunk]
            active = component_mask[:, None].expand_as(q)
            own = self.query_component(torch.stack((q, active.float()), -1))
            own = (own * active[..., None]).sum(-2) / n[:, None, None].clamp_min(1)
            query = self.query(torch.cat((own, core[:, None].expand_as(own)), -1))
            x, y = torch.broadcast_tensors(q[:, :, None], history_proportions[:, None])
            pair = torch.stack((x, y, x * y, (x - y).abs()), -1)
            encoded = self.component(pair)
            encoded = (encoded * component_mask[:, None, None, :, None]).sum(-2)
            encoded = encoded / n[:, None, None, None].clamp_min(1)
            score = (self.relation_key(encoded) * query[:, :, None]).sum(-1) / cfg.width**.5
            values = self.relation_value(torch.cat((encoded, response[:, None].expand_as(encoded)), -1))
            # Every response type has its own attention pool. An isolated useful
            # observation is not averaged away by many unchanged-rust readings.
            valid = groups[:, None].transpose(-1, -2)  # B,1,group,history
            weights = score[:, :, None].masked_fill(~valid, -1e4).softmax(-1) * valid
            weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-8)
            pooled = torch.einsum('bagh,bahd->bagd', weights, values)
            counts = groups.sum(1).float().log1p()[:, None, :, None].expand(*pooled.shape[:-1], 1)
            context = self.pool(torch.cat((pooled, counts), -1).flatten(-2))
            core_q = core[:, None].expand_as(context)
            hidden = self.output[:2](torch.cat((own, context, core_q, context * core_q), -1))
            residual = self.output[2](hidden)
            prior = self.prior[n[:, None], current[:, start:start + len(q[0])]]
            pieces.append(prior + residual)
            hidden_pieces.append(hidden)
        logits = torch.cat(pieces, 1)
        return {'latent_logits': logits, 'hidden': torch.cat(hidden_pieces, 1),
                'log_probabilities': observable_log_probabilities(logits, current)}
