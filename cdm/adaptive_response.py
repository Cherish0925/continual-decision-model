"""Differentiable online updates of a categorical, possibly censored response.

The update uses received outcomes only. A censored return y=c means R>=c;
its gradient is p(R)-p(R|R>=c), rather than a one-hot target at c. Subsequent
predictions train the representation and update rates through the update.
"""
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class ResponseAdaptationConfig:
    memory_size: int = 64
    fast_rate: float = .5
    slow_rate: float = .1


class AdaptiveResponse(nn.Module):
    def __init__(self, width, levels, config=ResponseAdaptationConfig()):
        super().__init__()
        self.width, self.levels, self.config = width + 1, levels, config
        self.fast_rate = nn.Parameter(torch.tensor(config.fast_rate).log())
        self.slow_rate = nn.Parameter(torch.tensor(config.slow_rate).log())

    def initial(self, batch, device):
        b, m, w, k = batch, self.config.memory_size, self.width, self.levels
        return {
            'response_fast': torch.zeros(b, w, k, device=device),
            'response_slow': torch.zeros(b, w, k, device=device),
            'response_x': torch.zeros(b, m, w, device=device),
            'response_base': torch.zeros(b, m, k, device=device),
            'response_before': torch.zeros(b, m, dtype=torch.long, device=device),
            'response_after': torch.zeros(b, m, dtype=torch.long, device=device),
            'response_valid': torch.zeros(b, m, dtype=torch.bool, device=device),
            'response_cursor': torch.zeros(b, dtype=torch.long, device=device),
        }

    def features(self, hidden):
        # A constant feature permits class-prior correction. Normalization
        # bounds the per-observation update without discarding direction.
        return torch.cat((F.normalize(hidden.float(), dim=-1),
                          torch.ones_like(hidden[..., :1])), -1) / 2**.5

    def predict(self, state, base, features):
        return base + torch.bmm(features, state['response_fast'] + state['response_slow'])

    def record(self, state, features, base, before, after):
        slot = F.one_hot(state['response_cursor'] % self.config.memory_size,
                         self.config.memory_size).bool()
        return {**state,
            'response_x': torch.where(slot[..., None], features[:, None], state['response_x']),
            'response_base': torch.where(slot[..., None], base[:, None], state['response_base']),
            'response_before': torch.where(slot, before[:, None], state['response_before']),
            'response_after': torch.where(slot, after[:, None], state['response_after']),
            'response_valid': state['response_valid'] | slot,
            'response_cursor': state['response_cursor'] + 1}

    def update(self, state, name, selected, *, retain=True):
        x = state['response_x']
        logits = self.predict(state, state['response_base'], x)
        before, after = state['response_before'], state['response_after']
        levels = torch.arange(self.levels, device=x.device)
        tail = logits.masked_fill(levels < before[..., None], -torch.inf).softmax(-1)
        target = torch.where((before == after)[..., None], tail,
                             F.one_hot(after, self.levels).to(logits.dtype))
        error = (logits.softmax(-1) - target) * state['response_valid'][..., None]
        gradient = torch.bmm(x.transpose(1, 2), error)
        gradient = gradient / state['response_valid'].sum(-1)[:, None, None].clamp_min(1)
        key = 'response_' + name
        rate = getattr(self, name + '_rate').exp()
        updated = state[key] - rate * gradient
        if not retain:
            updated = state[key]
        return {**state, key: torch.where(selected[:, None, None], updated, state[key])}
