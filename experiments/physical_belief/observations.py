"""Expose physical uncertainty to every decision through the shared workspace."""
from __future__ import annotations

import torch


def add_physical_context(observation, belief, normalization):
    """Append relation tokens to the public evidence consumed by the core.

    Object evidence has one of four object indicator bits set. Physical tokens
    leave those bits zero, use the same quantity indicator, and describe a
    predictive distribution. They contain no evaluator targets or task labels.
    This makes prior experience visible before a setting becomes executable,
    including when the next decision is which measurement to acquire.
    """
    n, quantities = belief["means"].shape
    original = observation["evidence"]
    if quantities != 9 or original.shape[-1] != 20:
        raise ValueError("This input adapter uses the nine-quantity native task encoding")
    tokens = original.new_zeros(n, quantities, 20)
    tokens[:, :, 4:13] = torch.eye(quantities, device=original.device)
    tokens[:, :, 13] = ((belief["means"] - normalization.y_mean) / normalization.y_scale).float()
    tokens[:, :, 14] = (belief["std"] / normalization.y_scale).log1p().float()
    tokens[:, :, 15] = belief["weights"].float()
    tokens[:, :, 16] = belief["support_counts"].float() / 4
    tokens[:, :, 17] = belief["eligible"].float()
    if "context_weights" in belief:
        tokens[:, :, 18] = belief["context_weights"][:, :1].float()
    else:
        tokens[:, :, 18] = 1.
    weights = belief["weights"].clamp_min(1e-20)
    tokens[:, :, 19] = (-(weights * weights.log()).sum(-1)[:, None]).float()
    return {**observation,
            "evidence": torch.cat((original, tokens), 1),
            "evidence_mask": torch.cat((observation["evidence_mask"],
                torch.ones(n, quantities, device=original.device, dtype=torch.bool)), 1)}
