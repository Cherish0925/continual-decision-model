"""Batched Reactor Lab tuning with the deployed public scalar controller.

The requested mean is a control target, not one atomic simulator action.
Intermediate settings, the sixteen-action local probe, and dialogue entry/exit
all consume actions. Private resonances are consulted only by the environment
transition. Returned frequencies follow the official reactor's integer/snap rule.
"""
from __future__ import annotations

from collections import deque
from functools import lru_cache

import numpy as np
import torch


@lru_cache(maxsize=1)
def _integer_distances():
    # Signed decimal increments. The enlarged line contains a shortest path
    # between any two settings in the public 0..10000 interval.
    radius = 11000
    distances = np.full(radius * 2 + 1, 127, dtype=np.int16)
    distances[radius] = 0
    queue = deque([0])
    while queue:
        current = queue.popleft()
        for delta in (1, -1, 10, -10, 100, -100, 1000, -1000):
            target = current + delta
            if -radius <= target <= radius and distances[target + radius] == 127:
                distances[target + radius] = distances[current + radius] + 1
                queue.append(target)
    return distances[radius:radius + 10001].copy()


class ScalarExecution:
    """Native additive tuning and local probing, without rendering a world.

    Inputs are vectors [B]. ``budget`` includes TALK and EXIT; earlier travel,
    placement and restoration are charged by the enclosing environment.
    ``frequencies`` and ``activated`` contain only actually returned settings.
    """

    def __init__(self, device='cpu'):
        self.device = torch.device(device)
        self.grid = torch.arange(10001, device=device)
        self.distance = torch.as_tensor(_integer_distances(), device=device)
        # Native dialogue option order; stable sort implements the controller's
        # distance, absolute delta, option-number tie order.
        self.deltas = torch.tensor([1., -1., 10., -10., 100., -100., 1000., -1000.],
                                   dtype=torch.float64, device=device)

    def _step(self, current, target):
        proposed = (current[:, None] + self.deltas).clamp(0., 10000.)
        error = (proposed - target[:, None]).abs()
        choice = error.argmin(-1)
        value = proposed.gather(1, choice[:, None]).squeeze(1)
        improved = (value - target).abs() < (current - target).abs() - 1e-9
        return value, improved

    def _probe(self, current, mean, std, failed, remaining):
        grid = self.grid[None]
        origin = current.long()[:, None]
        direct = self.distance[(grid - origin).abs()]
        # Clipping an offered ±1000 action can reach a boundary more cheaply
        # than an exact signed-increment path. No other clipping exists.
        via_low = torch.div(origin + 999, 1000, rounding_mode='floor') + self.distance[grid]
        via_high = torch.div(10000 - origin + 999, 1000, rounding_mode='floor') + self.distance[10000 - grid]
        distance = torch.minimum(direct, torch.minimum(via_low, via_high))
        eligible = (~failed & (grid != origin) & (grid >= (mean - 2 * std)[:, None])
                    & (grid <= (mean + 2 * std)[:, None]) & (distance <= remaining[:, None]))
        minimum = distance.masked_fill(~eligible, 127).min(-1).values
        eligible &= distance == minimum[:, None]
        # argmin breaks exact mean-distance ties toward the lower Hz, as BFS.
        displacement = (grid - mean[:, None]).abs().masked_fill(~eligible, torch.inf)
        target = displacement.argmin(-1).to(current)
        return target, eligible.any(-1)

    @staticmethod
    def reactor_tick(frequency, resonance):
        clipped = frequency.clamp(0., 10000.)
        active = (clipped - resonance).abs() < 2.
        returned = torch.where(clipped.trunc() == resonance.trunc(), resonance, clipped.trunc())
        return returned, active

    @torch.no_grad()
    def __call__(self, current, mean, std, resonance, budget=None, *, terminate_on_success=None):
        current, mean, std, resonance = [x.to(device=self.device, dtype=torch.float64)
                                         for x in (current, mean, std, resonance)]
        batch = len(current)
        if terminate_on_success is None:
            terminate_on_success = torch.zeros(batch, dtype=torch.bool, device=self.device)
        rows = torch.arange(batch, device=self.device)
        if budget is None:
            budget = torch.full((batch,), 128, device=self.device, dtype=torch.long)
        else:
            budget = budget.to(device=self.device, dtype=torch.long)
        cost = (budget > 0).long()  # TALK
        active = budget > 0
        completed = torch.zeros(batch, dtype=torch.bool, device=self.device)
        success = torch.zeros_like(completed)
        started = torch.zeros_like(completed)
        probe_actions = torch.zeros_like(cost)
        failed = torch.zeros(batch, 10001, dtype=torch.bool, device=self.device)
        frequencies, activations, masks = [], [], []
        current, activated = self.reactor_tick(current, resonance)
        # At most 25 greedy adjustments in the 0..10000 integer domain plus
        # sixteen probe actions. The loop is bounded by actual budget as well.
        while active.any():
            frequencies.append(current.clone())
            activations.append(activated & active)
            masks.append(active.clone())
            success |= activated & active
            finish = active & activated
            proposed, improves = self._step(current, mean)
            start = active & ~activated & ~started & ~improves
            started |= start
            failed[rows[start], current[start].long()] = True
            probe_rows = active & ~activated & started
            if probe_rows.any():
                idx = probe_rows.nonzero().flatten()
                target, possible = self._probe(current[idx], mean[idx], std[idx], failed[idx],
                                                16 - probe_actions[idx])
                next_value, _ = self._step(current[idx], target)
                proposed[idx] = next_value
                finish[idx] |= ~possible
            # EXIT is an action; if the public budget ended after the previous
            # setting, there is no invented final exit or extra observation.
            can_act = active & (cost < budget)
            terminal_success = finish & activated & terminate_on_success
            completed |= (finish & can_act) | terminal_success
            cost += (finish & can_act & ~terminal_success).long()
            adjust = can_act & ~finish
            cost += adjust.long()
            revised, is_active = self.reactor_tick(proposed, resonance)
            current = torch.where(adjust, revised, current)
            activated = torch.where(adjust, is_active, activated)
            counted = adjust & started
            probe_actions += counted.long()
            newly_failed = counted & ~activated
            failed[rows[newly_failed], current[newly_failed].long()] = True
            # A setting returned on the last allowed adjustment is still valid
            # feedback, even though the request cannot subsequently exit.
            active = adjust
        return {"current_hz": current, "success": success, "actions": cost,
                "completed": completed, "probe_adjustments": probe_actions,
                "frequencies": torch.stack(frequencies, -1) if frequencies else current[:, None],
                "activated": torch.stack(activations, -1) if activations else success[:, None],
                "mask": torch.stack(masks, -1) if masks else completed[:, None]}
