"""Update predictions for untried experimental actions from earlier, returned measurements."""
import torch
from torch import nn
from torch.nn import functional as F


class ConditionalMeasurement(nn.Module):
    """Kernel conditioning with a learned action representation and a learned prior mean.

    Inputs must describe candidate actions before their outcomes are known. A
    measured action is identified by its index in the same menu. A missing
    reading may still reduce uncertainty without contributing a signed residual.
    """
    def __init__(self, features, width):
        super().__init__()
        self.encode = nn.Sequential(nn.Linear(features, width), nn.GELU(), nn.Linear(width, width))
        self.mean = nn.Sequential(nn.Linear(features, width), nn.GELU(), nn.Linear(width, 1))
        self.log_ridge = nn.Parameter(torch.tensor(-2.0))

    def prepare(self, x):
        return F.normalize(self.encode(x).float(), dim=-1), self.mean(x).float().squeeze(-1).sigmoid()

    def forward(self, encoded, indices, values, use_values=True):
        z, prior = encoded
        b = torch.arange(len(z), device=z.device)
        measured = z[b[:, None], indices]
        def kernel(left, right):
            similarity = left @ right.transpose(1, 2)
            return (-2 * (1 - similarity.clamp(-1, 1))).exp()
        k = kernel(z, measured)
        ridge = .01 + F.softplus(self.log_ridge)
        system = kernel(measured, measured) + ridge * torch.eye(indices.shape[1], device=z.device)
        residual = values - prior[b[:, None], indices]
        if not use_values:
            residual = torch.zeros_like(residual)
        solved = torch.linalg.solve(system, torch.cat((residual[:, :, None], k.transpose(1, 2)), -1))
        mean = prior + (k @ solved[:, :, :1]).squeeze(-1)
        variance = 1 - (k * solved[:, :, 1:].transpose(1, 2)).sum(-1)
        return mean, variance.clamp_min(1e-6).sqrt()
