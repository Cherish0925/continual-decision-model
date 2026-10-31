"""Fit a response correction in a representation shared by support and query.

Only earlier returned observations enter the inner updates. Outer training
scores the next experiment, so fitting a support observation is not its target.
The relation variant uses a kernel on experimental inputs instead of a global
hidden-state direction. These are functional-gradient baselines, not a claim
that kernel adaptation or meta-learning is new.
"""
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class ResponseReplayConfig:
    mode: str = 'refreshed'
    updates: int = 4


def recent_support(support, capacity):
    """Keep the newest returns of each stream, independent of batch padding."""
    valid = support['valid']
    lengths = valid.sum(-1)
    n = min(valid.shape[1], capacity)
    offsets = (lengths - n).clamp_min(0)[:, None]
    index = (offsets + torch.arange(n, device=valid.device)[None]).clamp_max(valid.shape[1] - 1)
    result = {}
    for key, value in support.items():
        gather = index.reshape(*index.shape, *([1] * (value.ndim - 2))).expand(-1, -1, *value.shape[2:])
        result[key] = value.gather(1, gather)
    result['valid'] = torch.arange(n, device=valid.device)[None] < lengths[:, None].clamp_max(n)
    return result


def response_gradient(logits, before, after):
    levels = torch.arange(logits.shape[-1], device=logits.device)
    conditional = logits.masked_fill(levels < before[..., None], -torch.inf).softmax(-1)
    target = torch.where((before == after)[..., None], conditional,
                         F.one_hot(after, logits.shape[-1]).to(logits.dtype))
    return logits.softmax(-1) - target


class ResponseReplay(nn.Module):
    def __init__(self, levels=4, config=ResponseReplayConfig()):
        super().__init__()
        self.config = config
        if config.mode not in {'calibration', 'stored', 'refreshed', 'relation'}:
            raise ValueError(config.mode)
        self.log_rates = nn.Parameter(torch.full((config.updates,), -.7))
        self.log_penalty = nn.Parameter(torch.tensor(-4.))
        self.log_length = nn.Parameter(torch.tensor(-1.))
        self.log_temperature = nn.Parameter(torch.tensor(0.))
        self.bias = nn.Parameter(torch.zeros(levels))

    def forward(self, query_base, query_x, query_inputs, support_base,
                support_x, support_inputs, before, after, valid):
        temperature = self.log_temperature.exp()
        base = query_base / temperature + self.bias
        if self.config.mode == 'calibration':
            return base
        support = support_base / temperature + self.bias
        if self.config.mode == 'relation':
            # A common permutation of ingredient names leaves these distances
            # unchanged. Kernel locality is learned from subsequent returns.
            s = F.normalize(support_inputs.float(), dim=-1)
            q = F.normalize(query_inputs.float(), dim=-1)
            scale = 2 * self.log_length.exp().square()
            kss = (-torch.cdist(s, s).square() / scale).exp()
            kqs = (-torch.cdist(q, s).square() / scale).exp()
        else:
            kss = torch.bmm(support_x, support_x.transpose(1, 2))
            kqs = torch.bmm(query_x, support_x.transpose(1, 2))
        coefficients = torch.zeros_like(support)
        denominator = valid.sum(-1)[:, None, None].clamp_min(1)
        for rate in self.log_rates.exp():
            logits = support + torch.bmm(kss, coefficients)
            gradient = response_gradient(logits, before, after) * valid[..., None]
            coefficients = coefficients - rate * (gradient / denominator +
                                                   self.log_penalty.exp() * coefficients)
        return base + torch.bmm(kqs, coefficients)
