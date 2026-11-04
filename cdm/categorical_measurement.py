"""Condition experimental predictions on equality of reagent roles.

Labels are opaque: changing every reagent name consistently leaves the
relation-only model unchanged. No unmeasured response enters these features.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F


class CategoricalMeasurement(nn.Module):
    def __init__(self, features, width, roles=8, named_prior=False):
        super().__init__()
        self.log_weights = nn.Parameter(torch.full((roles,), math.log(math.expm1(4 / roles))))
        self.log_ridge = nn.Parameter(torch.tensor(math.log(math.expm1(.04))))
        self.prior_logit = nn.Parameter(torch.tensor(0.))
        self.mean = (nn.Sequential(nn.Linear(features, width), nn.GELU(), nn.Linear(width, 1))
                     if named_prior else None)

    def prepare(self, x, codes):
        prior = (self.mean(x).squeeze(-1).sigmoid() if self.mean is not None
                 else self.prior_logit.sigmoid().expand(x.shape[:2]))
        return codes, prior

    def forward(self, encoded, indices, values, use_values=True):
        codes, prior = encoded
        b = torch.arange(len(codes), device=codes.device)
        measured = codes[b[:, None], indices]
        weights = F.softplus(self.log_weights)

        def kernel(left, right):
            different = (left[:, :, None] != right[:, None, :]).to(prior.dtype)
            return (-(different * weights).sum(-1)).exp()

        cross = kernel(codes, measured)
        system = kernel(measured, measured) + (.01 + F.softplus(self.log_ridge)) * torch.eye(
            indices.shape[1], device=codes.device)
        residual = values - prior[b[:, None], indices]
        if not use_values:
            residual = torch.zeros_like(residual)
        solved = torch.linalg.solve(system, torch.cat((residual[:, :, None], cross.transpose(1, 2)), -1))
        mean = prior + (cross @ solved[:, :, :1]).squeeze(-1)
        variance = 1 - (cross * solved[:, :, 1:].transpose(1, 2)).sum(-1)
        return mean, variance.clamp_min(1e-6).sqrt()


def relation_features(codes, present, indices=None, values=None):
    """Eight observable statistics per role, without reagent identities.

    Menu frequency; observed-match fraction, mean, maximum and minimum;
    equality with the last, best and worst observed conditions. Empty history
    has zero observation statistics. Pass zero values to hide all readings.
    """
    b, n, roles = codes.shape
    same_menu = (codes[:, :, None] == codes[:, None, :]).float()
    frequency = (same_menu * present[:, None, :, None]).sum(2) / present.sum(1)[:, None, None]
    if indices is None:
        return torch.cat((frequency, frequency.new_zeros(b, n, 7 * roles)), -1)
    rows = torch.arange(b, device=codes.device)
    observed = codes[rows[:, None], indices]
    matches = codes[:, :, None] == observed[:, None, :]
    count = matches.sum(2)
    y = values[:, None, :, None].expand_as(matches)
    mean = (matches * y).sum(2) / count.clamp_min(1)
    maximum = y.masked_fill(~matches, -torch.inf).amax(2).masked_fill(count == 0, 0)
    minimum = y.masked_fill(~matches, torch.inf).amin(2).masked_fill(count == 0, 0)
    last = (codes == observed[:, -1, None]).float()
    best = (codes == observed[rows, values.argmax(1)][:, None]).float()
    worst = (codes == observed[rows, values.argmin(1)][:, None]).float()
    return torch.cat((frequency, count / indices.shape[1], mean, maximum, minimum, last, best, worst), -1)
