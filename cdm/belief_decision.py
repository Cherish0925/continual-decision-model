"""Learn action returns from current and possible future response beliefs."""
from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class BeliefDecisionConfig:
    width: int = 128
    learning: str = 'joint'


class BeliefDecision(nn.Module):
    def __init__(self, action_features, latent_features, config=BeliefDecisionConfig()):
        super().__init__()
        self.config = config
        # Four observable outcomes, each with a possible posterior mean.
        self.output = nn.Sequential(nn.Linear(action_features + 4 * latent_features + 5, config.width),
                                    nn.SiLU(), nn.Linear(config.width, 1))
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)

    def forward(self, action_features, functions, weights, particles):
        if self.config.learning == 'prediction_only':
            functions, weights, particles = functions.detach(), weights.detach(), particles.detach()
        probabilities = functions.exp()
        joint = probabilities * weights.exp()[:, None, :, None]
        marginal = joint.sum(2)
        future = torch.einsum('baky,kd->bayd', joint, particles)
        future = future / marginal[..., None].clamp_min(1e-12)
        current = weights.exp() @ particles
        change = (future - current[:, None, None]) * marginal[..., None]
        information = (joint * (functions.nan_to_num(neginf=0) -
                                 marginal.clamp_min(1e-30).log()[:, :, None])).sum((2, 3))
        value = torch.cat((action_features, change.flatten(-2), marginal, information[..., None]), -1)
        return self.output(value).squeeze(-1)
