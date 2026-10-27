"""Learned Bayesian priors over nine explicitly chosen affine hypotheses.

The affine basis is a task modeling assumption. Shared parameters learn from
training episodes; public support updates an instance's coefficient and expert
posteriors. This readout is separate from the original CDM A/B mechanism.
"""
from __future__ import annotations

import math
from pathlib import Path

import torch
from torch import nn

from cdm.online_regression import ridge_update
from .calibration import FEATURE_QUANTITIES


class AffineMixture(nn.Module):
    def __init__(self, normalization: dict, *, noise_floor_hz: float = 2.0, prior_mode: str = "typed"):
        super().__init__()
        if prior_mode not in ("typed", "shared"):
            raise ValueError("prior_mode must be typed or shared")
        self.normalization = normalization
        self.noise_floor_hz = float(noise_floor_hz)
        self.prior_mode = prior_mode
        self.register_buffer("x_mean", torch.tensor(normalization["x_mean"], dtype=torch.float64))
        self.register_buffer("x_scale", torch.tensor(normalization["x_scale"], dtype=torch.float64))
        self.register_buffer("y_mean", torch.tensor(normalization["y_mean_hz"], dtype=torch.float64))
        self.register_buffer("y_scale", torch.tensor(normalization["y_scale_hz"], dtype=torch.float64))
        if self.x_mean.shape != (9,) or self.x_scale.shape != (9,):
            raise ValueError("Nine named quantity normalization values are required")
        if not bool((self.x_scale > 0).all() & (self.y_scale > 0)) or noise_floor_hz <= 0:
            raise ValueError("Normalization scales and native noise floor must be positive")
        count = 9 if prior_mode == "typed" else 1
        self.prior_mean = nn.Parameter(torch.zeros(count, 2, dtype=torch.float64))
        self.log_coefficient_scale = nn.Parameter(torch.zeros(count, 2, dtype=torch.float64))
        self.log_noise_extra_hz = nn.Parameter(torch.full((count,), math.log(2.), dtype=torch.float64))
        if prior_mode == "typed":
            self.mixture_logits = nn.Parameter(torch.zeros(9, dtype=torch.float64))
        else:
            self.register_buffer("mixture_logits", torch.zeros(9, dtype=torch.float64))

    @property
    def device(self):
        return self.x_mean.device

    @property
    def noise_std_hz(self):
        return (self.noise_floor_hz ** 2 + (2 * self.log_noise_extra_hz).exp()).sqrt().expand(9)

    def _features(self, values):
        values = torch.as_tensor(values, dtype=torch.float64, device=self.device)
        if values.ndim != 2 or values.shape[1] != 9 or not bool(torch.isfinite(values).all()):
            raise ValueError("Expected finite quantity vectors [N,9]")
        standardized = (values - self.x_mean) / self.x_scale
        return torch.stack((torch.ones_like(standardized), standardized), -1).transpose(0, 1)

    def _prior(self):
        return {"coefficient_mean": self.prior_mean.expand(9, -1),
                "coefficient_covariance": torch.diag_embed((2 * self.log_coefficient_scale).exp().expand(9, -1)),
                "log_weights": self.mixture_logits.log_softmax(-1),
                "log_evidence": self.mixture_logits.new_zeros(9), "support_count": 0}

    def fit_posterior(self, support_x, support_hz) -> dict:
        """Pure replacement from the complete public support [S,9], [S].

        Every expert uses phi=[1,x_k]. Whitened coefficient-prior features
        make ridge=noise_variance in standardized y units. Evidence and
        coefficient covariance use the same FP64 factorization.
        """
        phi = self._features(support_x)
        targets = torch.as_tensor(support_hz, dtype=torch.float64, device=self.device)
        count = phi.shape[1]
        if targets.shape != (count,) or not bool(torch.isfinite(targets).all()):
            raise ValueError("Expected finite support labels [S] in Hz")
        if not count:
            return self._prior()
        y = ((targets - self.y_mean) / self.y_scale)[None].expand(9, -1)
        scales = self.log_coefficient_scale.exp().expand(9, -1)
        features = phi * scales[:, None, :]
        prior = (phi * self.prior_mean[:, None, :]).sum(-1)
        noise_variance = (self.noise_std_hz / self.y_scale).square()
        state = ridge_update(features, y, prior, noise_variance)
        white_mean = (features.transpose(1, 2) @ state.coefficients[..., None])[..., 0]
        solved = torch.cholesky_solve(features, state.cholesky)
        white_cov = torch.eye(2, dtype=torch.float64, device=self.device)[None] - features.transpose(1, 2) @ solved
        covariance = white_cov * scales[:, :, None] * scales[:, None, :]
        covariance = (covariance + covariance.transpose(-1, -2)) / 2
        residual = y - prior
        evidence = -.5 * ((residual * state.coefficients).sum(-1)
                         + 2 * state.cholesky.diagonal(dim1=-2, dim2=-1).log().sum(-1)
                         + count * math.log(2 * math.pi))
        weights = self.mixture_logits.log_softmax(-1) + evidence
        return {"coefficient_mean": self.prior_mean + scales * white_mean,
                "coefficient_covariance": covariance,
                "log_weights": weights - weights.logsumexp(-1),
                "log_evidence": evidence, "support_count": count}

    def predict(self, query_x, posterior: dict | None = None) -> dict:
        """Return all candidate distributions; no frequency or action selected."""
        posterior = self._prior() if posterior is None else posterior
        phi = self._features(query_x)
        means = (phi * posterior["coefficient_mean"][:, None, :]).sum(-1)
        variances = torch.einsum("kqi,kij,kqj->kq", phi, posterior["coefficient_covariance"], phi).clamp_min(0)
        means = (means * self.y_scale + self.y_mean).transpose(0, 1)
        variances = (variances * self.y_scale.square() + self.noise_std_hz[:, None].square()).transpose(0, 1)
        weights = posterior["log_weights"].exp()[None].expand(means.shape[0], -1)
        mixture_mean = (weights * means).sum(-1)
        # Center before summing to avoid cancellation near a narrow mixture.
        mixture_variance = (weights * (variances + (means - mixture_mean[:, None]).square())).sum(-1)
        choice = weights.argmax(-1)
        return {"expert_posterior": weights, "expert_log_posterior": posterior["log_weights"][None].expand_as(means),
                "expert_mean_hz": means, "expert_std_hz": variances.sqrt(),
                "mixture_mean_hz": mixture_mean, "mixture_std_hz": mixture_variance.sqrt(),
                "map_expert_mean_hz": means.gather(1, choice[:, None])[:, 0],
                "map_expert_index": choice}

    def predictive_nll(self, query_x, query_hz, posterior: dict | None = None):
        prediction = self.predict(query_x, posterior)
        y = torch.as_tensor(query_hz, dtype=torch.float64, device=self.device).reshape(-1, 1)
        std = prediction["expert_std_hz"]
        log_likelihood = -.5 * ((y - prediction["expert_mean_hz"]) / std).square() - std.log() - .5 * math.log(2 * math.pi)
        return -(prediction["expert_log_posterior"] + log_likelihood).logsumexp(-1)

    def prior_penalty(self, *, examples: int, concentration: float = 1.1):
        """Dirichlet prior on all nine weights, scaled for a mean NLL objective."""
        return -(concentration - 1) * self.mixture_logits.log_softmax(-1).sum() / examples


def save_mixture_checkpoint(model, path, *, source_checkpoint: str, source_sha256: str, training=None):
    torch.save({"kind": "discoveryworld_gaussian_affine_mixture_v1", "model": model.state_dict(),
                "normalization": model.normalization, "noise_floor_hz": model.noise_floor_hz,
                "prior_mode": model.prior_mode,
                "quantity_names": list(FEATURE_QUANTITIES), "source_checkpoint": source_checkpoint,
                "source_sha256": source_sha256, "training": training or {},
                "scope": "explicit_task_affine_hypothesis_prior_not_original_cdm_AB"}, path)


def load_mixture_checkpoint(path: str | Path, device="cpu"):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if saved.get("kind") != "discoveryworld_gaussian_affine_mixture_v1":
        raise ValueError("Expected a Gaussian affine mixture checkpoint")
    model = AffineMixture(saved["normalization"], noise_floor_hz=saved["noise_floor_hz"],
                          prior_mode=saved.get("prior_mode", "typed"))
    model.load_state_dict(saved["model"], strict=True)
    return model.to(device), saved
