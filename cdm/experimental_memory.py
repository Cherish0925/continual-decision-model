"""Retain returned experiments and learn their applicability to a new task.

Memory identity is a storage slot, never an environment's hidden task label.
Evidence is scored before the return used to update it. A fresh-task predictor
competes with archived predictors, so a new relationship need not be forced
into an old one.
"""
from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class ExperimentalMemoryConfig:
    slots: int = 6
    capacity: int = 32
    mode: str = 'learned'


class ExperimentalMemory(nn.Module):
    def __init__(self, config=ExperimentalMemoryConfig()):
        super().__init__()
        self.config = config
        self.log_evidence_scale = nn.Parameter(torch.tensor(0.))
        self.new_log_prior = nn.Parameter(torch.tensor(1.))
        self.recency = nn.Parameter(torch.tensor(0.))
        self.compatibility = nn.Sequential(nn.Linear(5, 32), nn.SiLU(), nn.Linear(32, 1))
        nn.init.zeros_(self.compatibility[-1].weight)
        nn.init.zeros_(self.compatibility[-1].bias)

    def initial(self, batch, device):
        k, h = self.config.slots, self.config.capacity
        return {
            'archive_history': torch.zeros(batch, k, h, 20, device=device),
            'archive_mask': torch.zeros(batch, k, h, dtype=torch.bool, device=device),
            'archive_components': torch.zeros(batch, k, 8, dtype=torch.bool, device=device),
            'archive_cursor': torch.zeros(batch, dtype=torch.long, device=device),
            'archive_evidence': torch.zeros(batch, k + 1, device=device),
            'archive_returns': torch.zeros(batch, device=device),
        }

    def archive(self, state, selected):
        """Write the latest actual ledger at an apparatus boundary."""
        length = state['reaction_mask'].sum(-1)
        write = selected & (length > 0)
        if not write.any():
            return state
        result = dict(state)
        for key in ('archive_history', 'archive_mask', 'archive_components', 'archive_cursor'):
            result[key] = state[key].clone()
        rows = write.nonzero().flatten()
        slots = state['archive_cursor'][rows] % self.config.slots
        n = min(self.config.capacity, state['reaction_history'].shape[1])
        start = (length[rows] - n).clamp_min(0)
        indices = start[:, None] + torch.arange(n, device=rows.device)[None]
        ledger = state['reaction_history'][rows[:, None], indices]
        valid = torch.arange(n, device=rows.device)[None] < length[rows, None].clamp_max(n)
        result['archive_history'][rows, slots, :n] = ledger.detach()
        result['archive_mask'][rows, slots, :n] = valid
        result['archive_components'][rows, slots] = ledger[:, 0, 8:16].bool()
        result['archive_cursor'][rows] += 1
        return result

    def weights(self, state, component_mask):
        evidence = state['archive_evidence']
        valid = state['archive_mask'].any(-1)
        valid &= (state['archive_components'] == component_mask[:, None]).all(-1)
        count = state['archive_returns'][:, None]
        relative = evidence[:, 1:] - evidence[:, :1]
        n = state['archive_mask'].sum(-1).float()
        slots = torch.arange(self.config.slots, device=evidence.device)[None]
        age = (state['archive_cursor'][:, None] - 1 - slots) % self.config.slots
        mode = self.config.mode
        if mode == 'recent':
            score = -age.float() * 8
            new = torch.zeros_like(evidence[:, :1])
        elif mode in ('learned', 'disabled'):
            x = torch.stack((relative / count.clamp_min(1), relative / 10,
                             count.expand_as(n).log1p(), n.log1p(), age.float() / self.config.slots), -1)
            score = self.log_evidence_scale.exp() * relative + self.compatibility(x).squeeze(-1)
            score = score - self.recency * age
            new = self.new_log_prior.expand(len(evidence), 1)
        elif mode == 'evidence':
            score = relative
            new = torch.zeros_like(evidence[:, :1])
        else:
            raise ValueError(mode)
        # A fixed total mass for the old-context branch avoids changing its
        # prior simply by adding duplicate memories of the same relationship.
        score = score - valid.sum(-1, keepdim=True).clamp_min(1).float().log()
        if mode == 'disabled':
            valid = torch.zeros_like(valid)
        logits = torch.cat((new, score.masked_fill(~valid, -torch.inf)), -1)
        return logits.log_softmax(-1)

    def record(self, state, log_probabilities, after):
        selected = log_probabilities.gather(-1, after[:, None, None].expand(-1, log_probabilities.shape[1], 1)).squeeze(-1)
        return {**state, 'archive_evidence': state['archive_evidence'] + selected,
                'archive_returns': state['archive_returns'] + 1}

    def combine(self, state, component_mask, probabilities):
        weights = self.weights(state, component_mask)
        return torch.logsumexp(probabilities + weights[:, None, :, None], dim=2)
