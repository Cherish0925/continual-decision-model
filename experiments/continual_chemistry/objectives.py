"""Fit new feedback interfaces in the units of recorded experiment returns.

The continuing model still stores and predicts physical progress and log cost.
Only the supervised regression loss is scaled; runtime adaptation keeps its
existing target units. Statistics are estimated from training records alone.
"""
from __future__ import annotations

import math
from collections.abc import Iterable

import torch
from torch.nn import functional as F


def feedback_statistics(records: Iterable[dict]) -> dict:
    values = torch.tensor([
        [(row["rust_before"] - row["rust_after"]) / 3,
         math.log1p(row["native_actions"])]
        for row in records
    ], dtype=torch.float64)
    if values.numel() == 0:
        raise ValueError("Feedback statistics require training observations")
    standard_deviation = values.std(0, correction=0)
    # A constant target has no empirical scale; retain its original unit.
    scale = torch.where(standard_deviation > 0, standard_deviation,
                        torch.ones_like(standard_deviation))
    return {"count": len(values), "mean": values.mean(0).tolist(),
            "scale": scale.tolist(),
            "units": ["fractional rust reduction", "log(1 + native operations)"]}


@torch.no_grad()
def initialize_feedback_predictor(model, statistics: dict) -> None:
    """Initialize a new task head to its train-only constant predictor.

    Call before saving the initial checkpoint and before optimizer creation.
    The inherited recurrent core and every task representation stay unchanged.
    """
    model.predictor.weight.zero_()
    model.predictor.bias.copy_(model.predictor.bias.new_tensor(statistics["mean"]))


def scaled_feedback_loss(prediction, returned, statistics: dict):
    scale = prediction.new_tensor(statistics["scale"])
    return F.smooth_l1_loss((prediction - returned) / scale,
                           torch.zeros_like(prediction))


def training_parameter_groups(model, *, core_lr=3e-5, interface_lr=1e-3):
    """Fine-tune inherited computation and learn new interfaces separately."""
    core_ids = {id(parameter) for parameter in model.core.parameters()}
    inherited, interfaces = [], []
    for parameter in model.parameters():
        if parameter.requires_grad:
            (inherited if id(parameter) in core_ids else interfaces).append(parameter)
    return [{"params": inherited, "lr": core_lr, "name": "inherited_core"},
            {"params": interfaces, "lr": interface_lr, "name": "task_interfaces"}]
