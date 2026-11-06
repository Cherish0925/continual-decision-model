"""Trainable text understanding for the existing dynamic decision policy."""
import torch
from torch import nn
import torch.nn.functional as F


class DecisionTextEncoder(nn.Module):
    def __init__(self, config: dict, trainable: bool):
        super().__init__()
        from transformers import BertConfig, BertModel
        self.backbone = BertModel(BertConfig(**config))
        self.trainable = trainable
        self.backbone.requires_grad_(trainable)
        if trainable:
            self.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        self.train(self.training)

    def train(self, mode=True):
        super().train(mode)
        self.backbone.train(mode and self.trainable)
        return self

    def forward(self, ids):
        """One embedding per line, or per multi-chunk candidate/record.

        Token zero is padding. Long text is split upstream without dropping
        its tail. Candidate chunks are combined by their actual token counts.
        Identical strings within a call share encoding and retain gradients.
        """
        shape = ids.shape[:-1]
        flat = ids.reshape(-1, ids.shape[-1]).long()
        unique, inverse = torch.unique(flat, dim=0, return_inverse=True)
        present = unique.ne(0)
        valid = present.any(-1)
        source = unique[valid]
        vectors = []
        for batch in source.split(64):
            mask = batch.ne(0)
            with torch.set_grad_enabled(torch.is_grad_enabled() and self.trainable):
                hidden = self.backbone(input_ids=batch, attention_mask=mask, return_dict=True).last_hidden_state
                vectors.append((hidden * mask[..., None]).sum(1) / mask.sum(1, keepdim=True))
        width = self.backbone.config.hidden_size
        values = self.backbone.embeddings.word_embeddings.weight.new_zeros(len(unique), width)
        if vectors:
            values = values.index_copy(0, valid.nonzero().flatten(), torch.cat(vectors).to(values.dtype))
        encoded = values[inverse].reshape(*shape, width)
        if ids.ndim == 4:
            weights = ids.ne(0).sum(-1).to(encoded.dtype)
            encoded = (encoded * weights[..., None]).sum(-2) / weights.sum(-1, keepdim=True).clamp_min(1)
        return F.normalize(encoded, dim=-1)
