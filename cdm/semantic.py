"""An on-demand pretrained vision-language model and its latent connection to CDM.

The pretrained model keeps its own tokenizer, vision-language fusion and output
head. Its final hidden states can be cached; the trainable reader selects a
bounded set of context tokens for the persistent decision core. Scheduling and
task training are supplied by the caller.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn

from cdm.multimodal import EncodedEvent

SEMANTIC_ID = "Qwen/Qwen3.5-2B"
SEMANTIC_REVISION = "15852e8c16360a2fea060d615a32b45270f8a8fc"


class SemanticExpert(nn.Module):
    """Frozen pretrained model, with hidden-state reading and native generation."""

    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone.requires_grad_(False).eval()
        self.hidden_size = backbone.config.text_config.hidden_size

    @classmethod
    def from_pretrained(cls, model_id: str = SEMANTIC_ID,
                        revision: str = SEMANTIC_REVISION, **load_kwargs):
        from transformers import AutoModelForImageTextToText
        return cls(AutoModelForImageTextToText.from_pretrained(
            model_id, revision=revision, **load_kwargs))

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()
        return self

    @torch.no_grad()
    def encode(self, **inputs) -> Tensor:
        """Read the final layer without allocating vocabulary logits for every token.

        Inputs come from the model's AutoProcessor and chat template. The result
        represents the supplied sequence; it is not an executed reasoning trace.
        Any generated continuation must be supplied explicitly to encode it.
        """
        return self.backbone.model(**inputs, use_cache=False, return_dict=True).last_hidden_state

    @torch.no_grad()
    def generate(self, **kwargs):
        """Use the original pretrained decoder when language output is needed."""
        return self.backbone.generate(**kwargs)


class SemanticReader(nn.Module):
    """Read frozen expert features into a fixed number of CDM context tokens.

    ``age_seconds`` is the age of the source input at the current read, not just
    time since the expert finished. Empty rows represent results not yet available
    and have no effect after joining with observation tokens.
    """

    def __init__(self, source_width: int, width: int = 512, tokens: int = 8, heads: int = 8):
        super().__init__()
        self.queries = nn.Parameter(0.02 * torch.randn(tokens, width))
        self.source = nn.Sequential(nn.LayerNorm(source_width), nn.Linear(source_width, width))
        self.query = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width))
        self.read = nn.MultiheadAttention(width, heads, batch_first=True)
        self.age = nn.Sequential(nn.Linear(1, width), nn.GELU(), nn.Linear(width, width))
        self.kind = nn.Parameter(0.02 * torch.randn(width))
        self.norm = nn.LayerNorm(width)

    def compress(self, features: Tensor, present: Tensor, context: Tensor) -> EncodedEvent:
        """Compress once when a result arrives; cache these bounded tokens."""
        dtype = self.source[1].weight.dtype
        source = self.source(features.detach().to(dtype))
        mask = present.bool()
        available = mask.any(1)
        # A null key keeps attention defined for rows whose result has not arrived.
        source = torch.cat((source, torch.zeros_like(source[:, :1]) if source.shape[1]
                            else source.new_zeros(source.shape[0], 1, source.shape[-1])), 1)
        padding = torch.cat((~mask, available.unsqueeze(1)), 1)
        q = self.queries.unsqueeze(0) + self.query(context.to(dtype)).unsqueeze(1)
        read = self.read(q, source, source, key_padding_mask=padding, need_weights=False)[0]
        tokens = self.norm(q + read + self.kind)
        valid = available[:, None].expand(-1, self.queries.shape[0])
        return EncodedEvent(tokens.masked_fill(~valid.unsqueeze(-1), 0), valid)

    def at_age(self, memory: EncodedEvent, age_seconds: Tensor) -> EncodedEvent:
        """Refresh time features without rerunning the expert or full-sequence reader."""
        age = self.age(torch.log1p(age_seconds.to(self.age[0].weight.dtype)).reshape(-1, 1)).unsqueeze(1)
        tokens = self.norm(memory.tokens + age)
        return EncodedEvent(tokens.masked_fill(~memory.present.unsqueeze(-1), 0), memory.present)

    def forward(self, features: Tensor, present: Tensor, context: Tensor,
                age_seconds: Tensor) -> EncodedEvent:
        return self.at_age(self.compress(features, present, context), age_seconds)
