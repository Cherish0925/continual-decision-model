"""Batched DiscoveryWorld chemical experiments with native operation costs.

Environment-only recipe vectors come from the existing training split. Policies
receive the same public candidate/history tensors as the native executor. The
transition reproduces the official cosine thresholds and persistent rust rule;
native execution remains the final task evaluation.
"""
from __future__ import annotations

import json
import math
import random
from pathlib import Path

import torch
from .recipes import primitive_recipes


class ActiveChemicalEnvironment:
    def __init__(self, root, batch, seed, device, budget=1000, split='train', schedule='paired'):
        self.device, self.batch, self.budget = torch.device(device), batch, budget
        self.rng = random.Random(seed)
        self.schedule = schedule
        self.lifetime_length = 6 if schedule == 'paired' else 12
        manifest = json.loads((Path(root) / 'manifest.json').read_text())
        self.entries = [r for r in manifest['episodes'] if r['split'] == split]
        for entry in self.entries:
            with (Path(root) / entry['directory'] / 'experiments.jsonl').open() as stream:
                entry['prepare_actions'] = json.loads(next(stream))['transition_start']
        self.candidate_count = len(primitive_recipes(4)) + 1
        self.leave_index = self.candidate_count - 1
        self.capacity = 128  # At least eight operations per return, budget 1000.
        b, a, h = batch, self.candidate_count, self.capacity
        self.counts = torch.zeros(b, a, 8, device=device)
        self.allowed = torch.zeros(b, a, dtype=torch.bool, device=device)
        self.component_mask = torch.zeros(b, 8, device=device)
        self._recipe = torch.zeros(b, 8, dtype=torch.float64, device=device)
        self.rust = torch.full((b,), 3, dtype=torch.long, device=device)
        self.actions = torch.full((b,), 4, dtype=torch.long, device=device)
        self.history = torch.zeros(b, h, 20, device=device)
        self.history_mask = torch.zeros(b, h, dtype=torch.bool, device=device)
        self.seen = torch.zeros(b, a, dtype=torch.long, device=device)
        self.endpoint = torch.full((b, a), -1, dtype=torch.long, device=device)
        self.cursor = torch.zeros(b, dtype=torch.long, device=device)
        self.task_index = [0] * b
        self.lifetime = [None] * b
        self.new_task = torch.ones(b, dtype=torch.bool, device=device)
        self.new_lifetime = torch.ones_like(self.new_task)
        self.terminal = torch.ones_like(self.new_task)
        self.returned = self.dispatched = self.ended = self.total_actions = 0
        self.task_rows = []
        self.reset_finished()

    def reset_finished(self):
        self.new_task = self.terminal.clone()
        self.new_lifetime = torch.zeros_like(self.terminal)
        for lane in self.terminal.nonzero().flatten().tolist():
            position = self.task_index[lane] % self.lifetime_length
            if position == 0:
                if self.schedule == 'paired':
                    a, b = self.rng.sample(self.entries, 2)
                    self.lifetime[lane] = [a, a, b, b, a, a]
                elif self.schedule == 'mixed':
                    groups = {}
                    for r in self.entries:
                        groups.setdefault((r['difficulty'], r['recipe_group']), []).append(r)
                    keys = self.rng.sample(list(groups), 6)
                    sequence = [0, 1]
                    for i in range(2, self.lifetime_length):
                        # Context return and novelty have variable lags. No
                        # sequence position or hidden key is passed to the model.
                        seen = set(sequence)
                        unused = [j for j in range(6) if j not in seen]
                        choices = unused if unused and self.rng.random() < .45 else sorted(seen)
                        sequence.append(self.rng.choice(choices))
                    self.lifetime[lane] = [self.rng.choice(groups[keys[i]]) for i in sequence]
                else:
                    raise ValueError(self.schedule)
                self.new_lifetime[lane] = True
            row = self.lifetime[lane][position]
            n = 3 if row['difficulty'] == 'Normal' else 4
            target = dict(item.rsplit(':', 1) for item in row['recipe_group'].split('|'))
            vector = [int(target.get(f'Substance {chr(65+i)}', 0)) for i in range(n)]
            # Preserve each repeated relationship's ingredient identities. The
            # selected training instances determine all task content.
            candidates = primitive_recipes(n)
            self.counts[lane].zero_()
            self.counts[lane, :len(candidates), :n] = torch.tensor(candidates, device=self.device)
            self.allowed[lane].zero_(); self.allowed[lane, :len(candidates)] = True
            self.allowed[lane, self.leave_index] = True
            self.component_mask[lane].zero_(); self.component_mask[lane, :n] = 1
            self._recipe[lane].zero_(); self._recipe[lane, :n] = torch.tensor(vector, device=self.device)
            self.rust[lane] = 3; self.actions[lane] = row['prepare_actions']; self.cursor[lane] = 0
            self.history[lane].zero_(); self.history_mask[lane].zero_()
            self.seen[lane].zero_(); self.endpoint[lane].fill_(-1)
            self.task_index[lane] += 1
        self.terminal.zero_()
        return self.new_task, self.new_lifetime

    def observation(self):
        b, a = self.batch, self.candidate_count
        total = self.counts.sum(-1)
        candidate = self.counts.new_zeros(b, a, 30)
        candidate[..., :8] = self.counts / 6
        candidate[..., 8:16] = self.counts / total[..., None].clamp_min(1)
        candidate[..., 16:24] = self.component_mask[:, None]
        candidate[..., 24] = total / 6
        candidate[..., 25] = self.rust[:, None] / 3
        candidate[..., 26] = self.seen / (self.seen + 1)
        candidate[..., 27] = self.endpoint.clamp_min(0) / 3
        candidate[..., 28] = (self.seen > 0).float()
        candidate[..., 29] = total.log1p() / math.log(7)
        mask = self.allowed & (total > 0) & (self.rust[:, None] > 0)
        mask &= ~((self.seen > 0) & (self.endpoint >= self.rust[:, None]))
        mask[:, self.leave_index] = self.rust == 0
        used = max(1, int(self.cursor.max()))
        evidence = self.history.new_zeros(b, used + 1, 20)
        evidence[:, 0, 8:16] = self.component_mask
        evidence[:, 0, 16:18] = self.rust[:, None] / 3
        evidence[:, 0, 18] = self.actions / self.budget
        evidence[:, 0, 19] = 1
        evidence[:, 1:] = self.history[:, :used]
        index = torch.arange(1, used + 1, device=self.device)
        evidence[:, 1:, 19] = index / (self.cursor[:, None] + 1)
        evidence_mask = torch.cat((torch.ones(b, 1, dtype=torch.bool, device=self.device),
                                   self.history_mask[:, :used]), 1)
        event = self.history.new_zeros(b, 36)
        event[:, 8:16] = self.component_mask
        event[:, 16:18] = self.rust[:, None] / 3
        event[:, 20] = self.cursor / (self.cursor + 10)
        event[:, 22] = (self.rust == 0).float()
        event[:, 23] = self.actions / self.budget
        event[:, 35] = 1
        row = torch.arange(b, device=self.device)
        last = self.history[row, (self.cursor - 1).clamp_min(0)]
        present = self.cursor > 0
        event[:, :8] = last[:, :8]
        event[:, 16] = torch.where(present, last[:, 16], event[:, 16])
        event[:, 17] = torch.where(present, last[:, 17], event[:, 17])
        event[:, 18] = (last[:, 16] - last[:, 17]) * present
        event[:, 19] = last[:, 18]
        event[:, 21] = present.float()
        return dict(event=event, evidence=evidence, evidence_mask=evidence_mask,
                    candidates=candidate, action_mask=mask)

    @staticmethod
    def response(counts, recipe, before):
        q, r = counts.double(), recipe.double()
        cosine = (q * r).sum(-1) / (q.norm(dim=-1) * r.norm(dim=-1)).clamp_min(1e-30)
        intrinsic = torch.where(cosine >= .99, 0,
                    torch.where(cosine >= .66, 1, torch.where(cosine >= .33, 2, 3)))
        return torch.minimum(before, intrinsic)

    def step(self, choice):
        row = torch.arange(self.batch, device=self.device)
        counts = self.counts[row, choice]
        leave = choice == self.leave_index
        before = self.rust.clone()
        required = 6 + counts.sum(-1).long() + (counts > 0).sum(-1) + 2 * (self.cursor > 0)
        required = torch.where(leave, 4, required)
        cost = torch.minimum(required, self.budget - self.actions)
        returned = cost == required
        after = self.response(counts, self._recipe, before)
        after = torch.where(returned, after, before)
        success = leave & returned & (before == 0)
        self.actions += cost
        self.rust = after.clone()
        for lane in (returned & ~leave).nonzero().flatten().tolist():
            i = int(self.cursor[lane])
            self.history[lane, i, :8] = counts[lane] / 6
            self.history[lane, i, 8:16] = self.component_mask[lane]
            self.history[lane, i, 16] = before[lane] / 3
            self.history[lane, i, 17] = after[lane] / 3
            self.history[lane, i, 18] = cost[lane].float().log1p() / math.log(101)
            self.history_mask[lane, i] = True
            self.cursor[lane] += 1
            self.seen[lane, choice[lane]] += 1
            self.endpoint[lane, choice[lane]] = after[lane]
        self.terminal = success | (self.actions >= self.budget)
        self.dispatched += self.batch
        self.returned += int((returned & ~leave).sum())
        self.total_actions += int(cost.sum())
        for lane in self.terminal.nonzero().flatten().tolist():
            self.task_rows.append(dict(lane=lane, task_index=self.task_index[lane] - 1,
                native_actions=int(self.actions[lane]), completed=bool(success[lane]),
                returned_experiments=int(self.cursor[lane])))
        self.ended += int(self.terminal.sum())
        # Reward uses task completion and all executed native operations.
        reward = 2 * success.float() - cost.float() / 100
        targets = torch.stack(((before - after).float() / 3, cost.float().log1p()), -1)
        return dict(before=before, after=after, returned=returned, leave=leave,
                    cost=cost, targets=targets, reward=reward, terminal=self.terminal.clone())
