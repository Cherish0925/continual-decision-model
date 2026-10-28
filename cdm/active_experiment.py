"""Train the continuing core to choose experiments from partial observations.

The caller supplies received events, a visible evidence ledger and executable
candidate descriptors. Unrevealed measurements and outcome labels do not enter
this interface. The GRU and workspace are inherited from ContinualCore; all
parts used for experimental decisions are trainable.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import Tensor, nn

from .core import ContinualCore, CoreConfig


@dataclass(frozen=True)
class ActiveExperimentConfig:
    event_dim: int
    evidence_dim: int
    candidate_dim: int
    outcome_dim: int = 1
    action_width: int = 256
    mode: str = "persistent"


class ActiveExperimentPolicy(nn.Module):
    """Shared recurrent computation and dynamic experimental action readouts.

    ``ledger`` is a separately trained control: every decision receives the same
    visible ledger but starts a fresh GRU/workspace. It retains the architecture
    and parameter initialization of the persistent model.
    """

    def __init__(self, config: ActiveExperimentConfig, core: ContinualCore):
        super().__init__()
        if config.mode not in {"persistent", "ledger"}:
            raise ValueError("mode must be persistent or ledger")
        self.config, self.core = config, core
        w, a = core.config.width, config.action_width
        for parameter in core.parameters():
            parameter.requires_grad_(False)
        for name in ("gru", "block", "norm_h", "norm_z", "norm_slots"):
            for parameter in getattr(core, name).parameters():
                parameter.requires_grad_(True)
        core.slots.requires_grad_(True)
        if core.gate is not None:
            for parameter in core.gate.parameters():
                parameter.requires_grad_(True)
        self.event = nn.Sequential(nn.Linear(config.event_dim, w), nn.GELU(), nn.Linear(w, w))
        self.evidence = nn.Sequential(nn.Linear(config.evidence_dim, w), nn.GELU(), nn.Linear(w, w))
        self.candidate = nn.Sequential(nn.Linear(config.candidate_dim, a), nn.GELU(), nn.Linear(a, a))
        self.state_query = nn.Linear(2 * w, a)
        self.interaction = nn.Sequential(nn.Linear(3 * a, a), nn.GELU(), nn.Linear(a, a), nn.GELU())
        self.action = nn.Linear(a, 1)
        self.outcome = nn.Linear(a, config.outcome_dim)
        self.value = nn.Sequential(nn.Linear(2 * w, a), nn.GELU(), nn.Linear(a, 1))

    @classmethod
    def from_native_checkpoint(cls, path: str | Path, config: ActiveExperimentConfig):
        saved = torch.load(path, map_location="cpu", weights_only=True)
        core = ContinualCore(CoreConfig(**saved["core_config"]), saved["model"]["core.action_values"])
        core.load_state_dict({name.removeprefix("core."): value for name, value in saved["model"].items()
                              if name.startswith("core.")}, strict=True)
        return cls(config, core)

    def initial(self, batch: int, device) -> dict[str, Tensor]:
        return {"h": torch.zeros(batch, self.core.config.width, device=device),
                "z": self.core.slots.unsqueeze(0).expand(batch, -1, -1).clone()}

    def reset_rows(self, state: dict[str, Tensor], reset: Tensor) -> dict[str, Tensor]:
        initial = self.initial(len(reset), reset.device)
        return {key: torch.where(reset.view(-1, *([1] * (value.ndim - 1))), initial[key], value)
                for key, value in state.items()}

    def advance(self, event: Tensor, evidence: Tensor, evidence_mask: Tensor,
                state: dict[str, Tensor], refinements: Tensor | int = 1) -> dict[str, Tensor]:
        if self.config.mode == "ledger":
            state = self.initial(event.shape[0], event.device)
        h = self.core.fast_embeddings(state["h"], self.event(event).unsqueeze(0))[0]
        tokens = self.evidence(evidence)
        counts = (torch.full((event.shape[0],), refinements, device=event.device, dtype=torch.long)
                  if isinstance(refinements, int) else refinements)
        empty = event.new_empty(event.shape[0], 0, self.core.config.record_dim)
        z = self.core.refine(state["z"], h, empty, counts, tokens, evidence_mask)
        return {"h": h, "z": z}

    def readout(self, state: dict[str, Tensor], candidates: Tensor, action_mask: Tensor):
        features = self.core.state_features(state["h"], state["z"])
        query = self.state_query(features).unsqueeze(1).expand(-1, candidates.shape[1], -1)
        candidate = self.candidate(candidates)
        joint = self.interaction(torch.cat((query, candidate, query * candidate), -1))
        logits = self.action(joint).squeeze(-1).masked_fill(~action_mask, -torch.inf)
        return {"logits": logits, "value": self.value(features).squeeze(-1),
                "outcomes": self.outcome(joint)}

    def forward(self, event: Tensor, evidence: Tensor, evidence_mask: Tensor,
                candidates: Tensor, action_mask: Tensor, state: dict[str, Tensor],
                refinements: Tensor | int = 1):
        state = self.advance(event, evidence, evidence_mask, state, refinements)
        return {**self.readout(state, candidates, action_mask), "state": state}

    def configuration(self):
        return {"active_config": asdict(self.config), "core_config": asdict(self.core.config)}


def load_active_checkpoint(path: str | Path, device="cpu"):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    core = ContinualCore(CoreConfig(**saved["core_config"]), saved["model"]["core.action_values"])
    model = ActiveExperimentPolicy(ActiveExperimentConfig(**saved["active_config"]), core)
    model.load_state_dict(saved["model"], strict=True)
    return model.to(device), saved
