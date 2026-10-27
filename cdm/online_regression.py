"""Instance-owned ridge residuals on features learned by an outer objective.

Each update replaces the complete currently available support set. It does not
append repeated versions of a label or mutate shared model parameters. Support
and query features may carry gradients; the kernel factorization uses FP64.
"""
from __future__ import annotations

from dataclasses import dataclass, fields

import torch
from torch import Tensor


@dataclass(frozen=True)
class RidgeState:
    features: Tensor                 # [B, S, D], original support features
    targets: Tensor                  # [B, S]
    prior_mean: Tensor               # [B, S], prior at each support
    present: Tensor                  # [B, S], padding mask
    ridge: Tensor                    # [B], positive regularization
    coefficients: Tensor             # [B, S], FP64 dual residual coefficients
    cholesky: Tensor                 # [B, S, S], FP64 kernel factor

    def detach(self) -> RidgeState:
        """Keep the deployed numerical state without an outer-training graph."""
        return RidgeState(**{field.name: getattr(self, field.name).detach() for field in fields(self)})


def ridge_update(features: Tensor, targets: Tensor, prior_mean: Tensor, ridge: Tensor | float,
                 *, present: Tensor | None = None) -> RidgeState:
    """Solve on the complete then-available support, returning a new state.

    Shapes are features [B,S,D], targets/prior_mean [B,S], and ridge scalar or
    [B]. A learned positive ridge can be passed as ``log_ridge.exp()``. Native
    value units and feature scaling belong to the caller. No previous state is
    read: replacing a snapshot never counts its earlier version twice.

    The solve and its gradients use double precision. A computed-and-discarded
    control calls this same function and keeps its previous state instead.
    """
    if features.ndim != 3 or targets.shape != features.shape[:2] or prior_mean.shape != targets.shape:
        raise ValueError("Expected features [B,S,D] and targets/prior_mean [B,S]")
    batch, count, _ = features.shape
    if targets.device != features.device or prior_mean.device != features.device:
        raise ValueError("Support features, targets and prior must share a device")
    if present is None:
        present = torch.ones_like(targets, dtype=torch.bool)
    elif present.shape != targets.shape or present.device != features.device:
        raise ValueError("Support present mask must match targets shape and device")
    present = present.bool()
    regularization = torch.as_tensor(ridge, device=features.device, dtype=torch.float64)
    if regularization.ndim == 0:
        regularization = regularization.expand(batch)
    if regularization.shape != (batch,) or not bool(torch.isfinite(regularization).all() & (regularization > 0).all()):
        raise ValueError("Ridge must be a finite positive scalar or tensor [B]")
    values = features.to(torch.float64).masked_fill(~present[..., None], 0)
    residual = (targets.to(torch.float64) - prior_mean.to(torch.float64)).masked_fill(~present, 0)
    if count:
        gram = values @ values.transpose(-1, -2)
        gram = gram + regularization[:, None, None] * torch.eye(count, dtype=torch.float64, device=features.device)
        factor = torch.linalg.cholesky(gram)
        coefficients = torch.cholesky_solve(residual[..., None], factor)[..., 0]
    else:
        factor = values.new_empty(batch, 0, 0)
        coefficients = residual
    return RidgeState(features, targets, prior_mean, present, regularization, coefficients, factor)


def _query(state: RidgeState, features: Tensor) -> tuple[Tensor, Tensor]:
    if features.ndim != 3 or features.shape[0] != state.features.shape[0] or features.shape[2] != state.features.shape[2]:
        raise ValueError("Query features [B,Q,D] must match support batch and feature dimensions")
    if features.device != state.features.device:
        raise ValueError("Query and support features must share a device")
    query = features.to(torch.float64)
    support = state.features.to(torch.float64).masked_fill(~state.present[..., None], 0)
    return query, query @ support.transpose(-1, -2)


def ridge_predict(state: RidgeState, query_features: Tensor, prior_mean: Tensor) -> Tensor:
    """Prior plus the learned-feature kernel's fitted residual, [B,Q]."""
    if prior_mean.shape != query_features.shape[:2] or prior_mean.device != query_features.device:
        raise ValueError("Query prior must match query shape [B,Q] and device")
    _, cross = _query(state, query_features)
    prediction = prior_mean.to(torch.float64) + (cross @ state.coefficients[..., None])[..., 0]
    dtype = torch.promote_types(torch.promote_types(prior_mean.dtype, query_features.dtype), torch.float32)
    return prediction.to(dtype)


def ridge_variance(state: RidgeState, query_features: Tensor) -> Tensor:
    """Conditional linear-kernel variance, excluding observation noise, [B,Q].

    This is k(q,q)-k(q,S)(K(S,S)+ridge I)^-1 k(S,q), in feature-prior units.
    It is not automatically a calibrated variance in the target's native units.
    With no support it returns the unchanged feature-prior variance k(q,q).
    """
    query, cross = _query(state, query_features)
    variance = query.square().sum(-1)
    if state.features.shape[1]:
        solved = torch.cholesky_solve(cross.transpose(-1, -2), state.cholesky).transpose(-1, -2)
        variance = variance - (cross * solved).sum(-1)
    return variance.clamp_min(0).to(torch.promote_types(query_features.dtype, torch.float32))
