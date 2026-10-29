"""Optional context input to the GRU at a newly received observation.

The workspace path reads the previous computation; the evidence path reads the
current public tokens. Both start at zero so installing the connection preserves
the parent policy. No extra observation or internal state transition is created.
"""
from dataclasses import asdict, dataclass

import torch
from torch import nn
import torch.nn.functional as F


@dataclass(frozen=True)
class RecurrentFeedbackConfig:
    source: str = "workspace"


class RecurrentFeedback(nn.Module):
    def __init__(self, width, config=RecurrentFeedbackConfig()):
        super().__init__()
        if config.source not in {"workspace", "evidence", "both"}:
            raise ValueError("Feedback source must be workspace, evidence, or both")
        self.config = config
        self.workspace = nn.Parameter(torch.zeros(width, width)) if config.source != "evidence" else None
        self.evidence = nn.Parameter(torch.zeros(width, width)) if config.source != "workspace" else None

    def forward(self, workspace, tokens, mask):
        result = workspace.new_zeros(workspace.shape[0], workspace.shape[-1])
        if self.workspace is not None:
            summary = workspace.mean(1)
            summary = F.layer_norm(summary.float(), (summary.shape[-1],)).to(summary)
            result = result + F.linear(summary, self.workspace)
        if self.evidence is not None:
            weights = mask.to(tokens.dtype)
            summary = (tokens * weights[..., None]).sum(1) / weights.sum(1, keepdim=True).clamp_min(1)
            summary = F.layer_norm(summary.float(), (summary.shape[-1],)).to(summary)
            result = result + F.linear(summary, self.evidence)
        return result

    def configuration(self):
        return asdict(self.config)


def configure_recurrent_feedback(model, config=RecurrentFeedbackConfig()):
    """Install before constructing the optimizer; preserve a loaded connection."""
    existing = getattr(model, "recurrent_feedback", None)
    if existing is not None:
        if existing.config != config:
            raise ValueError("Do not replace a trained feedback source when resuming a checkpoint")
        return model
    parameter = next(model.parameters())
    model.recurrent_feedback = RecurrentFeedback(model.core.config.width, config).to(
        device=parameter.device, dtype=parameter.dtype)
    return model
