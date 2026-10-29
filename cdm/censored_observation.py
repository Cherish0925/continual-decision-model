"""Gaussian belief updates from public failures of interval-valued tests.

If a setting succeeds exactly when |Y - setting| < tolerance, its failure
reveals that Y is outside that interval. Multiple failures exclude the UNION
of their intervals; they are not independent likelihood factors. This module
conditions a Gaussian or Gaussian mixture on that public event without reading
the unobserved target. Returned moments approximate each conditioned component
by a Gaussian only when a caller chooses to reuse those moments.
"""
from __future__ import annotations

import math

import torch
from torch import Tensor


def _complement(intervals: Tensor, mask: Tensor):
    """Sorted disjoint complement segments, with padded rows masked out."""
    batch, count, _ = intervals.shape
    if count == 0:
        low = intervals.new_full((batch, 1), -torch.inf)
        high = intervals.new_full((batch, 1), torch.inf)
        return low, high, torch.ones_like(low, dtype=torch.bool)
    valid = mask & (intervals[..., 0] < intervals[..., 1])
    low = intervals[..., 0].masked_fill(~valid, torch.inf)
    order = low.argsort(-1)
    low = low.gather(1, order)
    high = intervals[..., 1].masked_fill(~valid, -torch.inf).gather(1, order)
    valid = valid.gather(1, order)
    previous_high = torch.cat((high.new_full((batch, 1), -torch.inf),
                               high.cummax(-1).values[:, :-1]), -1)
    starts = valid & ((torch.arange(count, device=intervals.device)[None] == 0) | (low > previous_high))
    group = (starts.long().cumsum(-1) - 1).clamp_min(0)
    union_low = low.new_full(low.shape, torch.inf).scatter_reduce(1, group, low, reduce="amin")
    union_high = high.new_full(high.shape, -torch.inf).scatter_reduce(1, group, high, reduce="amax")
    length = starts.sum(-1)
    complement_low = torch.cat((low.new_full((batch, 1), -torch.inf), union_high), -1)
    complement_high = torch.cat((union_low, low.new_full((batch, 1), torch.inf)), -1)
    present = torch.arange(count + 1, device=intervals.device)[None] <= length[:, None]
    present &= complement_low < complement_high
    return complement_low, complement_high, present


def _log_interval_mass(low: Tensor, high: Tensor):
    """log(Phi(high)-Phi(low)), using the nearer Gaussian tail."""
    def log_cdf(value):
        finite = torch.isfinite(value)
        calculated = torch.special.log_ndtr(torch.where(finite, value, 0.))
        return torch.where(finite, calculated, torch.where(value < 0, -torch.inf, 0.))
    upper_tail = low > 0
    log_large = torch.where(upper_tail, log_cdf(-low), log_cdf(high))
    log_small = torch.where(upper_tail, log_cdf(-high), log_cdf(low))
    # expm1 preserves close CDF differences. The clamp covers floating-point
    # equality for an interval below representable relative resolution.
    difference = (log_small - log_large).clamp_max(-torch.finfo(low.dtype).eps)
    return log_large + (-difference.expm1()).log()


def _segment_moments(low: Tensor, high: Tensor, log_mass: Tensor):
    """Conditional standard-normal moments on one nonempty segment."""
    constant = .5 * math.log(2 * math.pi)
    finite_low, finite_high = torch.isfinite(low), torch.isfinite(high)
    log_phi_low = (-.5 * torch.where(finite_low, low, 0.).square() - constant).masked_fill(~finite_low, -torch.inf)
    log_phi_high = (-.5 * torch.where(finite_high, high, 0.).square() - constant).masked_fill(~finite_high, -torch.inf)
    ratio_low = (log_phi_low - log_mass).exp()
    ratio_high = (log_phi_high - log_mass).exp()
    mean = ratio_low - ratio_high
    low_term = torch.where(finite_low, low, 0.) * ratio_low
    high_term = torch.where(finite_high, high, 0.) * ratio_high
    variance = (1. + low_term - high_term - mean.square()).clamp_min(0.)

    # Beyond 20 standard deviations, direct subtraction in the one-sided tail
    # variance loses digits. These inverse-Mills expansions retain the small
    # variance while log mass remains computed by log_ndtr.
    right_tail = finite_low & ~finite_high & (low > 20)
    left_tail = ~finite_low & finite_high & (high < -20)
    tail = right_tail | left_tail
    a = torch.where(right_tail, low, torch.where(left_tail, -high, 21.))
    inv = a.reciprocal()
    inv2 = inv.square()
    mills = a + inv * (1. + inv2 * (-2. + inv2 * (10. + inv2 * (-74. + inv2 * (706. - 8162. * inv2)))))
    tail_variance = inv2 * (1. + inv2 * (-6. + inv2 * (50. + inv2 * (-518. + 6354. * inv2))))
    mean = torch.where(tail, torch.where(left_tail, -mills, mills), mean)
    variance = torch.where(tail, tail_variance, variance)

    # A sufficiently narrow finite segment is locally uniform. The leading
    # density-slope correction avoids cancellation of nearly equal boundary
    # densities; error is higher order in width * (1 + |center|).
    safe_low = torch.where(finite_low, low, 0.)
    safe_high = torch.where(finite_high, high, 0.)
    width = safe_high - safe_low
    center = (safe_low + safe_high) / 2
    narrow = finite_low & finite_high & (width * (1 + center.abs()) < 1e-4)
    narrow_variance = width.square() / 12
    mean = torch.where(narrow, center - center * narrow_variance, mean)
    variance = torch.where(narrow, narrow_variance, variance)
    return mean, variance


def censored_gaussian_observation(means: Tensor, std: Tensor, intervals: Tensor,
                                  mask: Tensor | None = None):
    """Condition component predictions on exclusion of a union of intervals.

    Args: means/std [B,K], intervals [B,L,2], optional mask [B,L]. Values use
    the target's physical units. Standard deviations must be strictly positive.
    Exclusion endpoints may be infinite; a fully excluded real line returns
    log_survival=-inf, possible=False and undefined (NaN) conditional moments.

    Computation uses FP64 and supports gradients to means and standard deviations.
    Empty/duplicate/overlapping failures do not change the meaning of the event.
    The caller applies this union to the original prediction once, rather than
    repeatedly conditioning already-truncated moments on overlapping failures.
    """
    if means.ndim != 2 or std.shape != means.shape:
        raise ValueError("Expected means and std [batch, components]")
    if intervals.ndim != 3 or intervals.shape[0] != means.shape[0] or intervals.shape[-1] != 2:
        raise ValueError("Expected intervals [batch, exclusions, 2]")
    if not bool(torch.isfinite(means).all() & torch.isfinite(std).all() & (std > 0).all()):
        raise ValueError("Component means must be finite and standard deviations positive")
    means, std = means.double(), std.double()
    intervals = intervals.to(device=means.device, dtype=torch.float64)
    mask = torch.ones(intervals.shape[:2], dtype=torch.bool, device=means.device) if mask is None else mask.to(device=means.device, dtype=torch.bool)
    low, high, present = _complement(intervals, mask)
    # Invalid padding receives a harmless finite interval before arithmetic.
    low = torch.where(present, low, means.new_tensor(-1.))
    high = torch.where(present, high, means.new_tensor(1.))
    finite_low, finite_high = torch.isfinite(low), torch.isfinite(high)
    a = (torch.where(finite_low, low, 0.)[:, None] - means[..., None]) / std[..., None]
    b = (torch.where(finite_high, high, 0.)[:, None] - means[..., None]) / std[..., None]
    a = torch.where(finite_low[:, None], a, low[:, None])
    b = torch.where(finite_high[:, None], b, high[:, None])
    log_mass = _log_interval_mass(a, b)
    segment_mean, segment_variance = _segment_moments(a, b, log_mass)
    log_mass = log_mass.masked_fill(~present[:, None], -torch.inf)
    log_survival = log_mass.logsumexp(-1)
    possible = present.any(-1)[:, None].expand_as(means)
    normalizer = torch.where(possible, log_survival, 0.)
    weight = (log_mass - normalizer[..., None]).exp()
    z_mean = (weight * segment_mean).sum(-1)
    z_variance = (weight * (segment_variance + (segment_mean - z_mean[..., None]).square())).sum(-1)
    conditional_mean = means + std * z_mean
    conditional_variance = std.square() * z_variance
    return {"log_survival": log_survival, "possible": possible,
            "mean": conditional_mean.masked_fill(~possible, torch.nan),
            "variance": conditional_variance.masked_fill(~possible, torch.nan)}


def condition_gaussian_mixture(prediction: dict, intervals: Tensor, mask: Tensor | None = None):
    """Apply the same public failure event to each component and its weight."""
    result = censored_gaussian_observation(prediction["means"], prediction["std"], intervals, mask)
    weights = prediction["weights"].double()
    log_weights = weights.log() + result["log_survival"]
    normalizer = log_weights.logsumexp(-1)
    weights = (log_weights - normalizer[:, None]).exp()
    mean = (weights * result["mean"]).sum(-1)
    variance = (weights * (result["variance"] + (result["mean"] - mean[:, None]).square())).sum(-1)
    return {**prediction, "means": result["mean"], "std": result["variance"].sqrt(),
            "weights": weights, "log_weights": log_weights - normalizer[:, None],
            "mixture_mean": mean, "mixture_std": variance.sqrt(),
            "log_survival": result["log_survival"], "log_event_probability": normalizer,
            "possible": result["possible"]}


def _segment_log_mass_in_units(means, std, low, high, present):
    low = torch.where(present, low, -1.)
    high = torch.where(present, high, 1.)
    finite_low, finite_high = torch.isfinite(low), torch.isfinite(high)
    a = (torch.where(finite_low, low, 0.)[:, None] - means[..., None]) / std[..., None]
    b = (torch.where(finite_high, high, 0.)[:, None] - means[..., None]) / std[..., None]
    a = torch.where(finite_low[:, None], a, low[:, None])
    b = torch.where(finite_high[:, None], b, high[:, None])
    return _log_interval_mass(a, b).masked_fill(~present[:, None], -torch.inf).logsumexp(-1)


def interval_log_probability(prediction: dict, interval: Tensor,
                             exclusions: Tensor | None = None, mask: Tensor | None = None):
    """Exact mixture log P(Y in interval | Y outside excluded union), [B].

    ``prediction`` is the ORIGINAL Gaussian mixture before these exclusions,
    and ``interval`` is [B,2], for example a later public successful setting's
    tolerance interval. This integrates the actual truncated mixture, rather
    than fitting a Gaussian to its moments. A disjoint event has probability
    zero. A completely excluded real line makes the conditional probability
    undefined and returns NaN.
    """
    means, std = prediction["means"].double(), prediction["std"].double()
    interval = torch.as_tensor(interval, dtype=means.dtype, device=means.device)
    if interval.shape != (means.shape[0], 2):
        raise ValueError("Expected one physical interval [batch, 2]")
    if exclusions is None:
        exclusions = means.new_empty(means.shape[0], 0, 2)
    else:
        exclusions = exclusions.to(device=means.device, dtype=means.dtype)
    mask = torch.ones(exclusions.shape[:2], dtype=torch.bool, device=means.device) if mask is None else mask.to(device=means.device, dtype=torch.bool)
    low, high, present = _complement(exclusions, mask)
    denominator = _segment_log_mass_in_units(means, std, low, high, present)
    intersection_low = torch.maximum(low, interval[:, :1])
    intersection_high = torch.minimum(high, interval[:, 1:])
    intersection_present = present & (intersection_low < intersection_high)
    numerator = _segment_log_mass_in_units(means, std, intersection_low, intersection_high, intersection_present)
    log_weights = prediction["weights"].double().log()
    log_numerator = (log_weights + numerator).logsumexp(-1)
    log_denominator = (log_weights + denominator).logsumexp(-1)
    return (log_numerator - log_denominator).clamp_max(0.)
