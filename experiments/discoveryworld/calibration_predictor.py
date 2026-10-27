"""A numerical query bridge into the existing CDM predictor and instance A/B.

This module predicts later publicly observed working settings. It does not
select native actions or claim that a small Hz error establishes activation.
"""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import torch
from torch import nn

from cdm.core import ContinualCore, CoreConfig, GaussianQuantityLikelihood, adapter_update
from cdm.digital import DigitalConfig, DigitalPolicy, observation_numeric

QUANTITIES = ("density_g_cm3", "temperature_c", "radiation_usv_h", "quantum_gap_nm",
              *(f"spectrum_channel_{i}" for i in range(1, 6)))


def query_rows(query) -> tuple[torch.Tensor, torch.Tensor]:
    """Keep every then-available condition group; object handles only join facts."""
    numeric, kinds = [], []
    for kind, quantity in enumerate(QUANTITIES):
        for summary in query.features.get(quantity, ()):
            row = asdict(summary) if not isinstance(summary, dict) else summary
            conditions = row["conditions"]
            inventory, accessible = conditions.get("inventory", []), conditions.get("accessible", [])
            ui = {"agentLocation": conditions.get("agent_location", {}),
                  "inventoryObjects": inventory, "accessibleEnvironmentObjects": accessible,
                  "nearbyObjects": {"objects": conditions.get("nearby", {})}}
            values = torch.tensor([row["count"], row["minimum"], row["maximum"], row["mean"],
                                   row["latest"], row["maximum"] - row["minimum"], row["last_change"],
                                   row["last_sequence"] - row["first_sequence"]], dtype=torch.float32)
            values = values.sign() * values.abs().log1p() / 10
            membership = [float(any(obj["uuid"] == query.object_handle for obj in group))
                          for group in (inventory, accessible)]
            numeric.append(torch.cat((values, torch.tensor(observation_numeric({"ui": ui}) + membership))))
            kinds.append(kind)
    return (torch.stack(numeric) if numeric else torch.empty(0, 42), torch.tensor(kinds, dtype=torch.long))


class QueryEncoder(nn.Module):
    """Read named quantity/condition groups without a hand-fitted scientific law."""
    def __init__(self, width: int, query_encoding: str = "late_binding"):
        super().__init__()
        if query_encoding not in ("late_binding", "typed_interaction"):
            raise ValueError("query_encoding must be late_binding or typed_interaction")
        self.query_encoding = query_encoding
        self.numeric = nn.Sequential(nn.Linear(42, width), nn.GELU(), nn.Linear(width, width))
        self.quantity = nn.Embedding(len(QUANTITIES), width)
        self.weight = nn.Linear(width, 1, bias=False)
        self.norm = nn.LayerNorm(width)

    def forward(self, queries):
        device = self.quantity.weight.device
        rows = [query_rows(query) for query in queries]
        length = max(1, max(len(values) for values, _ in rows))
        numbers = torch.zeros(len(rows), length, 42, device=device)
        kinds = torch.zeros(len(rows), length, dtype=torch.long, device=device)
        present = torch.zeros(len(rows), length, dtype=torch.bool, device=device)
        for i, (values, quantity) in enumerate(rows):
            numbers[i, :len(values)] = values.to(device)
            kinds[i, :len(values)] = quantity.to(device)
            present[i, :len(values)] = True
        valid = present.any(-1)
        present[~valid, 0] = True
        if self.query_encoding == "typed_interaction":
            tokens = self.numeric[2](self.numeric[1](self.numeric[0](numbers) + self.quantity(kinds)))
        else:
            tokens = self.numeric(numbers) + self.quantity(kinds)
        scores = self.weight(tokens).squeeze(-1).masked_fill(~present, -torch.inf)
        value = self.norm((scores.softmax(-1)[..., None] * tokens).sum(1))
        return value * valid[:, None]


class CalibrationPredictor(nn.Module):
    def __init__(self, source: DigitalPolicy, *, source_mode: str = "cdm",
                 likelihood: GaussianQuantityLikelihood | None = None, learning_rate: float = .05,
                 context_mode: str = "current", query_encoding: str = "late_binding"):
        super().__init__()
        if context_mode not in ("current", "object_only"):
            raise ValueError("context_mode must be current or object_only")
        self.source = source.requires_grad_(False).eval()
        self.source_mode, self.learning_rate = source_mode, learning_rate
        self.context_mode, self.query_encoding = context_mode, query_encoding
        self.likelihood = likelihood or GaussianQuantityLikelihood(
            scale=1000., noise_std=2., logvar_bounds=(-8., 20.))
        self.query_encoder = QueryEncoder(source.core.config.width, query_encoding)
        # Frozen perception, GRU, workspace and action path make cached h/z
        # contexts valid throughout numerical training. Predictor tensors are
        # the same objects owned by this source core, not a separate regressor.
        for name in ("pred_in", "pred_out", "selector"):
            module = getattr(self.core, name)
            if module is not None:
                module.requires_grad_(True)
        for name in ("log_step", "log_kappa"):
            parameter = getattr(self.core, name)
            if parameter is not None:
                parameter.requires_grad_(True)

    @property
    def core(self):
        return self.source.core

    def train(self, mode=True):
        super().train(mode)
        self.source.eval()
        return self

    def context(self, state: dict) -> torch.Tensor:
        if self.source_mode == "gru":
            return torch.cat((self.core.norm_h(state["h"]), torch.zeros_like(state["h"])), -1)
        return self.core.state_features(state["h"], state["z"])

    def inputs(self, queries, contexts: torch.Tensor) -> torch.Tensor:
        query = self.query_encoder(queries)
        if self.context_mode == "object_only":
            # A stationary object relation need not change with navigation or
            # dialogue state. The source still assimilates those observations
            # normally; only this numerical predictor's context input is masked.
            contexts = torch.zeros_like(contexts)
        return torch.cat((contexts.to(query.device), query), -1).unsqueeze(1)

    def raw_prediction(self, queries, contexts, adapters):
        return self.core.predict_raw(self.inputs(queries, contexts), adapters["A"], adapters["B"])

    def predict(self, queries, contexts, adapters) -> dict:
        mean, logvar = self.likelihood.distribution(*self.raw_prediction(queries, contexts, adapters))
        return {"mean_hz": mean[:, 0], "std_hz": (logvar[:, 0] / 2).exp()}

    def update(self, support_queries, support_contexts, support_hz, adapters, *, differentiable=False):
        inputs = self.inputs(support_queries, support_contexts).squeeze(1).unsqueeze(0)
        targets = torch.as_tensor(support_hz, device=inputs.device, dtype=torch.float32).reshape(1, -1)
        A, B, loss = adapter_update(self.core, adapters["A"], adapters["B"], inputs, targets,
                                    None, None, self.learning_rate, differentiable=differentiable,
                                    likelihood=self.likelihood)
        return {"A": A, "B": B}, loss


def save_calibration_checkpoint(model, source_metadata, destination, *, training_metadata=None,
                                initial_adapters=None):
    source = {**source_metadata, "model": model.source.state_dict()}
    checkpoint = {"kind": "discoveryworld_calibration_predictor_v1", "source": source,
                  "source_mode": model.source_mode, "context_mode": model.context_mode,
                  "query_encoding": model.query_encoding,
                  "learning_rate": model.learning_rate,
                  "likelihood": asdict(model.likelihood), "query_encoder": model.query_encoder.state_dict(),
                  "query_schema": "condition_groups_signedlog_v1", "quantities": list(QUANTITIES),
                  "training": training_metadata or {},
                  "scope": "preliminary_public_working_frequency_prediction_not_native_task_completion"}
    if initial_adapters is not None:
        checkpoint["initial_adapters"] = {key: initial_adapters[key].detach().cpu().clone() for key in ("A", "B")}
    torch.save(checkpoint, destination)


def load_calibration_checkpoint(path: str | Path, device="cpu") -> tuple[CalibrationPredictor, dict]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("kind") != "discoveryworld_calibration_predictor_v1":
        raise ValueError("Expected a calibration predictor checkpoint")
    source = checkpoint["source"]
    core = ContinualCore(CoreConfig(**source["core_config"]), source["model"]["core.action_values"])
    policy = DigitalPolicy(DigitalConfig(**source["digital_config"]), core)
    policy.load_state_dict(source["model"], strict=True)
    model = CalibrationPredictor(policy, source_mode=checkpoint["source_mode"],
                                 likelihood=GaussianQuantityLikelihood(**checkpoint["likelihood"]),
                                 learning_rate=checkpoint["learning_rate"],
                                 context_mode=checkpoint.get("context_mode", "current"),
                                 query_encoding=checkpoint.get("query_encoding", "late_binding"))
    model.query_encoder.load_state_dict(checkpoint["query_encoder"], strict=True)
    return model.to(device), checkpoint
