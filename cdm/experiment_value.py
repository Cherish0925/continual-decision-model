"""Action values from a set of competing experimental response functions.

An expert is represented by its predictions across the available experiments,
then queried by each candidate action. This retains cross-action structure
that is lost when the expert dimension is marginalized before action selection.
The final zero-initialized residual preserves the inherited policy exactly.
"""
from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class ExperimentValueConfig:
    width: int = 48
    mode: str = 'hypotheses'


class ExperimentValue(nn.Module):
    def __init__(self, action_features, config=ExperimentValueConfig()):
        super().__init__()
        self.config = config
        d = config.width
        self.function = nn.Sequential(nn.Linear(34, d), nn.SiLU(), nn.Linear(d, d))
        self.candidate = nn.Sequential(nn.Linear(34 + d, d), nn.SiLU(), nn.Linear(d, d))
        self.output = nn.Sequential(nn.Linear(2 * d + action_features + 5, d), nn.SiLU(), nn.Linear(d, 1))
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)

    def forward(self, candidates, mask, experts, weights, action_features):
        # B,A,K,Y. The same channel covers the fresh predictor and old ledgers.
        p = experts.exp()
        if self.config.mode == 'marginal':
            p = (p * weights[:, None, :, None]).sum(2, keepdim=True)
            weights = torch.ones_like(weights[:, :1])
        q = candidates[:, :, None].expand(-1, -1, p.shape[2], -1)
        local = torch.cat((q, p), -1)
        function = self.function(local)
        count = mask.sum(1).clamp_min(1)[:, None, None]
        # Normalize over actions, not archive slots. A common permutation of
        # actions or expert slots preserves the corresponding action values.
        summary = (function * mask[:, :, None, None]).sum(1) / count
        hidden = self.candidate(torch.cat((local, summary[:, None].expand_as(function)), -1))
        mean = (hidden * weights[:, None, :, None]).sum(2)
        spread = ((hidden - mean[:, :, None]).square() * weights[:, None, :, None]).sum(2)
        mixture = (p * weights[:, None, :, None]).sum(2)
        entropy = -(p * p.clamp_min(1e-30).log()).sum(-1)
        information = -(mixture * mixture.clamp_min(1e-30).log()).sum(-1)
        information -= (entropy * weights[:, None]).sum(-1)
        x = torch.cat((mean, spread, action_features, mixture, information[..., None]), -1)
        return self.output(x).squeeze(-1)
