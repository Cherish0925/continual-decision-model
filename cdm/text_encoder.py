"""Trainable text understanding for the existing dynamic decision policy."""
import torch
from torch import nn
import torch.nn.functional as F


class DecisionTextEncoder(nn.Module):
    def __init__(self, config: dict, trainable: bool, *, batch_size: int = 64, checkpointing: bool = True):
        super().__init__()
        from transformers import BertConfig, BertModel
        self.backbone = BertModel(BertConfig(**config))
        self.trainable = trainable
        if batch_size < 1:
            raise ValueError('Text encoding batch size must be positive')
        self.batch_size = batch_size
        self.backbone.requires_grad_(trainable)
        if trainable and checkpointing:
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
        for start in range(0, len(source), self.batch_size):
            batch = source[start:start + self.batch_size]
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

    def forward_many(self, inputs):
        """Encode independent CPU token arrays together, then restore each view.

        Each input retains its own duplicate-string grouping. In training, an
        occurrence in a different event therefore receives independent dropout;
        no learned representation is cached across optimizer updates. Grouping
        changes random-number assignment, not the distribution of dropout.
        Token rows attend only within their own string, never across events.
        """
        device = self.backbone.embeddings.word_embeddings.weight.device
        width = self.backbone.config.hidden_size
        sources, layouts, offset = [], [], 0
        for ids in inputs:
            if ids.device.type != 'cpu':
                raise ValueError('Window token indexing must run on CPU inputs')
            shape = ids.shape[:-1]
            unique, inverse = torch.unique(ids.reshape(-1, ids.shape[-1]).long(),
                                           dim=0, return_inverse=True)
            valid = unique.ne(0).any(-1)
            source = unique[valid]
            index = valid.long().cumsum(0) + offset
            index = index.masked_fill(~valid, 0)[inverse]
            weights = ids.ne(0).sum(-1) if ids.ndim == 4 else None
            layouts.append((shape, index, weights))
            sources.append(source)
            offset += len(source)
        if not inputs:
            return []
        source = torch.cat(sources)
        # Trimming only trailing padding retains absolute token positions,
        # including any masked positions inside a string.
        positions = torch.arange(1, source.shape[-1] + 1)
        lengths = (source.ne(0) * positions).amax(-1)
        buckets = ((lengths + 31) // 32 * 32).clamp_max(source.shape[-1])
        order = torch.argsort(buckets, stable=True)
        ordered = source[order].to(device)
        values = self.backbone.embeddings.word_embeddings.weight.new_zeros(1, width)
        vectors, start = [], 0
        for length, count in zip(*torch.unique_consecutive(buckets[order], return_counts=True)):
            length, count = int(length), int(count)
            for batch in ordered[start:start + count, :length].split(self.batch_size):
                mask = batch.ne(0)
                with torch.set_grad_enabled(torch.is_grad_enabled() and self.trainable):
                    hidden = self.backbone(input_ids=batch, attention_mask=mask,
                                           return_dict=True).last_hidden_state
                    vectors.append((hidden * mask[..., None]).sum(1) / mask.sum(1, keepdim=True))
            start += count
        if vectors:
            vectors = torch.cat(vectors).to(values.dtype)[torch.argsort(order).to(device)]
            values = torch.cat((values, vectors))
        result = []
        for shape, index, weights in layouts:
            encoded = values[index.to(device)].reshape(*shape, width)
            if weights is not None:
                weights = weights.to(device=device, dtype=encoded.dtype)
                encoded = (encoded * weights[..., None]).sum(-2) / weights.sum(-1, keepdim=True).clamp_min(1)
            result.append(F.normalize(encoded, dim=-1))
        return result
