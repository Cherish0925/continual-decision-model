"""Retain observed values inside a candidate deterministic task context."""
import math
import torch


def condition_on_observed_responses(logits, queries, inputs, before, after, valid,
                                   *, residual_mass=1e-6):
    """Keep exact/censored observations distinct from context applicability.

    The caller still infers whether this archived context applies to a new
    apparatus. This operation only concerns predictions conditional on that
    context. It assumes the experimental response is deterministic; residual
    mass keeps a contradicted context's prequential likelihood representable.
    Unobserved query proportions retain the learned prediction unchanged.
    """
    matching = ((queries[:, :, None] - inputs[:, None]).abs().amax(-1) < 1e-6) & valid[:, None]
    levels = torch.arange(logits.shape[-1], device=logits.device)
    allowed = torch.where((after < before)[..., None], levels == after[..., None],
                          levels >= before[..., None])
    support = (~matching[..., None] | allowed[:, None]).all(-2)
    if not support.any(-1).all():
        raise ValueError('A deterministic context contains contradictory returned observations')
    base = logits.log_softmax(-1)
    conditioned = base.masked_fill(~support, -torch.inf).log_softmax(-1)
    return torch.where(matching.any(-1)[..., None],
        torch.logaddexp(conditioned + math.log1p(-residual_mass), base + math.log(residual_mass)), base)
