"""Learned physical predictions with a bounded memory of experimental posteriors.

Each task fits its own affine coefficients. A previous task is a possible prior,
not an assertion that the coefficients or relation are unchanged. A learned
coefficient drift broadens remembered posteriors; current public measurements
select between those priors and a fresh shared prior. The marginal likelihood,
physical prediction, and remembered state are differentiable through tasks.

The feature count is unrestricted. Choosing affine single-feature relations is
an explicit model class, independent of the environment or its hidden labels.
"""
from __future__ import annotations

import math
from pathlib import Path

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class ExperimentalBelief(nn.Module):
    def __init__(self, normalization: dict, *, memory_slots: int = 16,
                 noise_floor_hz: float = 2.0):
        super().__init__()
        self.normalization = normalization
        self.memory_slots = int(memory_slots)
        self.noise_floor_hz = float(noise_floor_hz)
        self.register_buffer("x_mean", torch.tensor(normalization["x_mean"], dtype=torch.float64))
        self.register_buffer("x_scale", torch.tensor(normalization["x_scale"], dtype=torch.float64))
        self.register_buffer("y_mean", torch.tensor(normalization["y_mean_hz"], dtype=torch.float64))
        self.register_buffer("y_scale", torch.tensor(normalization["y_scale_hz"], dtype=torch.float64))
        self.features = len(self.x_mean)
        if self.memory_slots < 1 or self.features < 1:
            raise ValueError("Physical belief needs at least one feature and memory slot")
        if not bool((self.x_scale > 0).all() & (self.y_scale > 0)) or noise_floor_hz <= 0:
            raise ValueError("Normalization and observation noise scales must be positive")
        self.prior_mean = nn.Parameter(torch.zeros(self.features, 2, dtype=torch.float64))
        self.log_coefficient_scale = nn.Parameter(torch.zeros(self.features, 2, dtype=torch.float64))
        self.mixture_logits = nn.Parameter(torch.zeros(self.features, dtype=torch.float64))
        self.log_noise_extra_hz = nn.Parameter(torch.full((self.features,), math.log(2.), dtype=torch.float64))
        self.log_transition_scale = nn.Parameter(torch.full((self.features, 2), math.log(.25), dtype=torch.float64))
        self.memory_logit = nn.Parameter(torch.tensor(0., dtype=torch.float64))

    @property
    def device(self):
        return self.x_mean.device

    @property
    def noise_std_hz(self):
        return (self.noise_floor_hz ** 2 + (2 * self.log_noise_extra_hz.clamp(-12, 12)).exp()).sqrt()

    @torch.no_grad()
    def initialize_from_mixture(self, mixture):
        """Reuse an existing fitted numerical prior, with no target data read."""
        for name in ("prior_mean", "log_coefficient_scale", "mixture_logits", "log_noise_extra_hz"):
            getattr(self, name).copy_(getattr(mixture, name).expand_as(getattr(self, name)))
        for name in ("x_mean", "x_scale", "y_mean", "y_scale"):
            if not torch.equal(getattr(self, name), getattr(mixture, name).to(self.device)):
                raise ValueError("Numerical-prior normalization differs from this belief")
        return self

    def initial(self, batch: int, device=None):
        device = self.device if device is None else device
        shape = (batch, self.memory_slots, self.features)
        return {"means": torch.zeros(*shape, 2, dtype=torch.float64, device=device),
                "covariances": torch.zeros(*shape, 2, 2, dtype=torch.float64, device=device),
                "log_weights": torch.zeros(*shape, dtype=torch.float64, device=device),
                "valid": torch.zeros(batch, self.memory_slots, dtype=torch.bool, device=device),
                "cursor": torch.zeros(batch, dtype=torch.long, device=device)}

    def _priors(self, memory, retrieve):
        batch = memory["means"].shape[0]
        base_mean = self.prior_mean[None, None].expand(batch, 1, -1, -1)
        base_cov = torch.diag_embed((2 * self.log_coefficient_scale.clamp(-12, 8)).exp())
        base_cov = base_cov[None, None].expand(batch, 1, -1, -1, -1)
        drift = torch.diag_embed((2 * self.log_transition_scale.clamp(-12, 8)).exp())
        means = torch.cat((base_mean, memory["means"]), 1)
        covariances = torch.cat((base_cov, memory["covariances"] + drift[None, None]), 1)
        valid = memory["valid"] & retrieve
        has_memory = valid.any(-1)
        count = valid.sum(-1).clamp_min(1)
        base_log_mass = torch.where(has_memory, F.logsigmoid(-self.memory_logit), 0.)
        slot_log_mass = F.logsigmoid(self.memory_logit) - count.log()
        slot_log_mass = slot_log_mass[:, None].expand_as(valid).masked_fill(~valid, -torch.inf)
        base_log_weights = self.mixture_logits.log_softmax(-1)[None, None].expand(batch, 1, -1)
        slot_log_weights = memory["log_weights"].log_softmax(-1)
        log_weights = torch.cat((base_log_mass[:, None, None] + base_log_weights,
                                 slot_log_mass[..., None] + slot_log_weights), 1)
        return means, covariances, log_weights

    def posterior(self, values: Tensor, seen: Tensor, known_hz: Tensor, known: Tensor,
                  memory: dict, *, retrieve: bool = True):
        """Replace inference from a full public ledger; repeated calls add no data.

        Inputs are [batch, objects, features], [batch, objects]. Unseen values
        and unknown targets may contain arbitrary placeholders: they are masked
        before arithmetic. Every known target contributes one likelihood term
        to every candidate. An unmeasured feature uses the common standardized
        target reference N(0, 1), without updating that candidate's coefficients.
        This explicit missing-input likelihood avoids giving candidates with
        fewer measured supports an unearned likelihood advantage. It also keeps
        the posterior independent of the order of rows in the same ledger.
        """
        if values.ndim != 3 or values.shape[-1] != self.features:
            raise ValueError("Expected values [batch, objects, features]")
        seen, known = seen.bool(), known.bool()
        clean_x = torch.where(seen, values.double(), self.x_mean)
        x = (clean_x - self.x_mean) / self.x_scale
        y = (torch.where(known, known_hz.double(), self.y_mean) - self.y_mean) / self.y_scale
        means, cov, log_weights = self._priors(memory, retrieve)
        noise = (self.noise_std_hz / self.y_scale).square()[None, None]
        for obj in range(values.shape[1]):
            phi = torch.stack((torch.ones_like(x[:, obj]), x[:, obj]), -1)[:, None]
            cp = (cov @ phi[..., None])[..., 0]
            variance = ((phi * cp).sum(-1) + noise).clamp_min(1e-12)
            prediction = (phi * means).sum(-1)
            measured = seen[:, obj, None]
            likelihood_variance = torch.where(measured, variance, 1.)
            likelihood_mean = torch.where(measured, prediction, 0.)
            likelihood_error = y[:, obj, None, None] - likelihood_mean
            likelihood = -.5 * (likelihood_error.square() / likelihood_variance
                                 + likelihood_variance.log() + math.log(2 * math.pi))
            log_weights = log_weights + torch.where(known[:, obj, None, None], likelihood, 0.)
            valid = measured & known[:, obj, None, None]
            gain = cp / variance[..., None]
            residual = y[:, obj, None, None] - prediction
            means = means + torch.where(valid[..., None], gain * residual[..., None], 0.)
            # Joseph form keeps each covariance PSD under narrow observation noise.
            identity = torch.eye(2, dtype=cov.dtype, device=cov.device)
            update = identity - gain[..., :, None] * phi[..., None, :]
            new_cov = update @ cov @ update.transpose(-1, -2) + noise[..., None, None] * gain[..., :, None] * gain[..., None, :]
            cov = torch.where(valid[..., None, None], new_cov, cov)
        log_joint = log_weights - log_weights.flatten(1).logsumexp(-1)[:, None, None]
        log_relation = log_joint.logsumexp(1)
        conditional = (log_joint - log_relation[:, None]).exp()
        collapsed_mean = (conditional[..., None] * means).sum(1)
        difference = means - collapsed_mean[:, None]
        collapsed_cov = (conditional[..., None, None] *
                         (cov + difference[..., :, None] * difference[..., None, :])).sum(1)
        return {"coefficient_mean": collapsed_mean, "coefficient_covariance": collapsed_cov,
                "log_weights": log_relation, "component_means": means,
                "component_covariances": cov, "component_log_weights": log_joint,
                "context_weights": log_joint.exp().sum(-1),
                "support_counts": (seen & known[..., None]).sum(1),
                "known_count": known.sum(-1)}

    def predict(self, query_values: Tensor, query_seen: Tensor, posterior: dict):
        query_seen = query_seen.bool()
        x = (torch.where(query_seen, query_values.double(), self.x_mean) - self.x_mean) / self.x_scale
        phi = torch.stack((torch.ones_like(x), x), -1)
        means, cov = posterior["coefficient_mean"], posterior["coefficient_covariance"]
        location = (phi * means).sum(-1)
        variance = (phi[..., None, :] @ cov @ phi[..., None])[..., 0, 0].clamp_min(0)
        variance = variance + (self.noise_std_hz / self.y_scale).square()
        unseen_variance = cov[..., 0, 0] + cov[..., 1, 1] + means[..., 1].square() + (self.noise_std_hz / self.y_scale).square()
        location = torch.where(query_seen, location, means[..., 0])
        variance = torch.where(query_seen, variance, unseen_variance.clamp_min(1e-12))
        means_hz = location * self.y_scale + self.y_mean
        variances_hz = variance * self.y_scale.square()
        weights = posterior["log_weights"].exp()
        average = (weights * means_hz).sum(-1)
        total_variance = (weights * (variances_hz + (means_hz - average[:, None]).square())).sum(-1)
        return {"means": means_hz, "std": variances_hz.clamp_min(1e-12).sqrt(),
                "weights": weights, "log_weights": posterior["log_weights"],
                "mixture_mean": average, "mixture_std": total_variance.clamp_min(1e-12).sqrt(),
                "support_counts": posterior["support_counts"],
                "eligible": (posterior["support_counts"] >= 2) & query_seen,
                "context_weights": posterior["context_weights"]}

    def observe_predict(self, values, seen, known_hz, known, query, memory, *, retrieve=True):
        posterior = self.posterior(values, seen, known_hz, known, memory, retrieve=retrieve)
        rows = torch.arange(len(values), device=values.device)
        return self.predict(values[rows, query], seen[rows, query], posterior), posterior

    def nll(self, prediction: dict, target_hz: Tensor):
        target = target_hz.double().reshape(-1, 1)
        std = prediction["std"]
        likelihood = -.5 * ((target - prediction["means"]) / std).square() - std.log() - .5 * math.log(2 * math.pi)
        return -(prediction["log_weights"] + likelihood).logsumexp(-1)

    def remember(self, memory: dict, posterior: dict, selected: Tensor | None = None):
        """Store one completed task; the caller commits each task exactly once.

        Current task coefficients are never silently carried into the next fit.
        They become one uncertain memory candidate alongside the fresh prior.
        For a computed-and-discarded control call this method but keep memory.
        """
        valid = posterior["known_count"] >= 2
        if selected is not None:
            valid = valid & selected.bool()
        slots = torch.arange(self.memory_slots, device=memory["cursor"].device)
        replace = (slots[None] == memory["cursor"][:, None]) & valid[:, None]
        return {"means": torch.where(replace[..., None, None], posterior["coefficient_mean"][:, None], memory["means"]),
                "covariances": torch.where(replace[..., None, None, None], posterior["coefficient_covariance"][:, None], memory["covariances"]),
                "log_weights": torch.where(replace[..., None], posterior["log_weights"][:, None], memory["log_weights"]),
                "valid": memory["valid"] | replace,
                "cursor": torch.where(valid, (memory["cursor"] + 1) % self.memory_slots, memory["cursor"])}

    def configuration(self):
        return {"kind": "experimental_belief_v1", "normalization": self.normalization,
                "memory_slots": self.memory_slots, "noise_floor_hz": self.noise_floor_hz}


def save_experimental_belief_checkpoint(model, path, *, training=None):
    torch.save({**model.configuration(), "model": model.state_dict(), "training": training or {}}, path)


def load_experimental_belief_checkpoint(path: str | Path, device="cpu"):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if saved.get("kind") != "experimental_belief_v1":
        raise ValueError("Expected an experimental belief checkpoint")
    model = ExperimentalBelief(saved["normalization"], memory_slots=saved["memory_slots"],
                               noise_floor_hz=saved["noise_floor_hz"])
    model.load_state_dict(saved["model"], strict=True)
    return model.to(device), saved
