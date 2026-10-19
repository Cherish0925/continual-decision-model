"""Pretrained image/text perception for the persistent GRU/Transformer core.

Encoding and state updates are separate so frozen features can be cached.
The SigLIP 2 FixRes checkpoint uses Transformers' SiglipModel implementation.
This module supplies perception and state integration; the lifetime trainer's
environment adapter still supplies actions, records, rewards and learning steps.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import isqrt

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from cdm.core import CoreConfig, ContinualCore

ENCODER_ID = "google/siglip2-base-patch16-224"
ENCODER_REVISION = "75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2"


@dataclass
class EncodedEvent:
    tokens: Tensor             # [batch, tokens, width]
    present: Tensor            # [batch, tokens], True for received inputs


class MultimodalEncoder(nn.Module):
    """Frozen image/text towers, trainable projections and structured inputs.

    ``numeric`` is a task-defined vector containing observed values, their
    presence indicators, the last executed action and current task fields.
    ``timing`` contains elapsed time, observation delay and action age in seconds.
    Request identifiers remain in the event log; they are not continuous values.
    """

    def __init__(self, backbone: nn.Module, numeric_dim: int, width: int = 512,
                 image_grid: int = 4):
        super().__init__()
        self.backbone = backbone.requires_grad_(False).eval()
        self.width, self.image_grid = width, image_grid
        self.image = nn.Linear(backbone.config.vision_config.hidden_size, width)
        self.text = nn.Linear(backbone.config.text_config.hidden_size, width)
        self.numeric = nn.Sequential(nn.Linear(numeric_dim, width), nn.GELU(), nn.Linear(width, width))
        self.time = nn.Sequential(nn.Linear(3, width), nn.GELU(), nn.Linear(width, width))
        self.modality = nn.Parameter(0.02 * torch.randn(4, width))
        self.image_position = nn.Parameter(0.02 * torch.randn(image_grid ** 2, width))
        self.norm = nn.LayerNorm(width)

    @classmethod
    def from_pretrained(cls, numeric_dim: int, width: int = 512, image_grid: int = 4,
                        model_id: str = ENCODER_ID, revision: str = ENCODER_REVISION,
                        **load_kwargs):
        from transformers import AutoModel
        backbone = AutoModel.from_pretrained(model_id, revision=revision, **load_kwargs)
        return cls(backbone, numeric_dim, width, image_grid)

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()
        return self

    @torch.no_grad()
    def encode_image(self, pixel_values: Tensor) -> Tensor:
        """Return patch features before the trainable projection."""
        return self.backbone.vision_model(pixel_values=pixel_values, return_dict=True).last_hidden_state

    @torch.no_grad()
    def encode_text(self, input_ids: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        """Return token features; use the checkpoint processor for tokenization."""
        return self.backbone.text_model(input_ids=input_ids, attention_mask=attention_mask,
                                        return_dict=True).last_hidden_state

    def project(self, numeric: Tensor, timing: Tensor, image_features: Tensor | None = None,
                text_features: Tensor | None = None, image_present: Tensor | None = None,
                text_present: Tensor | None = None) -> EncodedEvent:
        """Project fresh or cached frozen features. Missing inputs are masked.

        For mixed batches, supply finite placeholders for absent images and
        ``image_present`` [batch]. ``text_present`` is [batch, text_tokens].
        Features must be from the same encoder revision used by the run.
        """
        n = numeric.shape[0]
        dtype = self.numeric[0].weight.dtype
        pieces = [self.numeric(numeric.to(dtype)).unsqueeze(1) + self.modality[0],
                  self.time(torch.log1p(timing.to(dtype))).unsqueeze(1) + self.modality[1]]
        masks = [torch.ones(n, 2, dtype=torch.bool, device=numeric.device)]
        if image_features is not None:
            side = isqrt(image_features.shape[1])
            if side * side != image_features.shape[1]:
                raise ValueError("FixRes image features must form a square patch grid")
            feature = image_features.detach().transpose(1, 2).reshape(n, -1, side, side)
            feature = F.adaptive_avg_pool2d(feature, (self.image_grid, self.image_grid)).flatten(2).transpose(1, 2)
            pieces.append(self.image(feature.to(dtype)) + self.image_position + self.modality[2])
            present = torch.ones(n, dtype=torch.bool, device=numeric.device) \
                if image_present is None else image_present.bool()
            masks.append(present[:, None].expand(-1, self.image_grid ** 2))
        if text_features is not None:
            pieces.append(self.text(text_features.detach().to(dtype)) + self.modality[3])
            masks.append(torch.ones(text_features.shape[:2], dtype=torch.bool, device=numeric.device)
                         if text_present is None else text_present.bool())
        present = torch.cat(masks, 1)
        tokens = self.norm(torch.cat(pieces, 1)).masked_fill(~present.unsqueeze(-1), 0)
        return EncodedEvent(tokens, present)

    def forward(self, numeric: Tensor, timing: Tensor, pixel_values: Tensor | None = None,
                input_ids: Tensor | None = None, attention_mask: Tensor | None = None,
                image_present: Tensor | None = None) -> EncodedEvent:
        images = None if pixel_values is None else self.encode_image(pixel_values)
        texts = None if input_ids is None else self.encode_text(input_ids, attention_mask)
        return self.project(numeric, timing, images, texts, image_present, attention_mask)


class MultimodalCore(nn.Module):
    """The existing continual core with a token-preserving perception path.

    Fast assimilation reads the event tokens with a query from the previous
    recurrent state. Refinement reads these same tokens directly alongside
    the recurrent state and recent records. It does not reconstruct an image
    from a pooled vector. The action heads can read the state before refinement.
    """

    def __init__(self, config: CoreConfig, action_values: Tensor, encoder: MultimodalEncoder):
        super().__init__()
        if config.width != encoder.width or config.event_dim != encoder.width:
            raise ValueError("core width and event_dim must equal the encoder projection width")
        self.encoder = encoder
        self.core = ContinualCore(config, action_values)
        self.ingest = nn.MultiheadAttention(config.width, config.heads, batch_first=True)
        self.query_norm = nn.LayerNorm(config.width)
        self.summary_norm = nn.LayerNorm(config.width)

    def initial(self, batch: int, device) -> dict[str, Tensor]:
        return self.core.initial(batch, device)

    def observe(self, state: dict[str, Tensor], event: EncodedEvent) -> dict[str, Tensor]:
        query = self.query_norm(state["h"]).unsqueeze(1)
        summary = self.ingest(query, event.tokens, event.tokens,
                              key_padding_mask=~event.present, need_weights=False)[0][:, 0]
        return {**state, "h": self.core.fast(state["h"], self.summary_norm(summary))}

    def refine(self, state: dict[str, Tensor], event: EncodedEvent, records: Tensor,
               counts: Tensor) -> dict[str, Tensor]:
        z = self.core.refine(state["z"], state["h"], records, counts, event.tokens, event.present)
        return {**state, "z": z}

    def readout(self, state: dict[str, Tensor], scalars: Tensor, goal: Tensor):
        """Process-task heads; ``goal`` [batch] is in the predictor target units.

        Returns action logits, activity logits, value, predicted mean and logvar.
        Other task schemas replace these output heads while retaining the core.
        """
        features = self.core.state_features(state["h"], state["z"])
        mean, logvar = self.core.predict_actions(features, state["A"], state["B"])
        offset = mean - goal.unsqueeze(-1)
        trunk = self.core.trunk(state["h"], state["z"], offset, logvar, scalars)
        action, activity, value = self.core.heads(trunk, offset, logvar)
        return action, activity, value, mean, logvar
