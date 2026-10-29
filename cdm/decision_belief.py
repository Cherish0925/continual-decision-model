"""Current relation probabilities after observed unsuccessful settings.

Every read starts from the ledger-conditioned Gaussian mixture. A union of
failed setting intervals supplies one additional event. Repeated readouts and
overlapping failures therefore do not multiply the same evidence twice.
"""
import torch

from .censored_observation import censored_gaussian_observation


def failure_conditioned_weights(means, std, weights, exclusions, mask):
    used = int(torch.where(
        mask, torch.arange(mask.shape[1], device=mask.device)[None] + 1, 0
    ).max().item()) if mask.numel() else 0
    if not used:
        return {"weights": weights, "log_survival": torch.zeros_like(means),
                "log_event_probability": means.new_zeros(len(means))}
    event = censored_gaussian_observation(means, std, exclusions[:, :used], mask[:, :used])
    log_joint = weights.double().log() + event["log_survival"]
    normalizer = log_joint.logsumexp(-1, keepdim=True)
    return {"weights": (log_joint - normalizer).exp(),
            "log_survival": event["log_survival"],
            "log_event_probability": normalizer.squeeze(-1)}
