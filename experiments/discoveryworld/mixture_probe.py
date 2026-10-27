"""Plan scalar experiments from all hypotheses and received inactive feedback.

The success tolerance is an explicit task/control assumption. No hidden target
or simulator state is read. Inactive settings exclude their success intervals;
the remaining mixture predicts which reachable experiment can succeed.
"""
from __future__ import annotations

from collections import deque
from decimal import Decimal

import torch

from .scalar_probe import ScalarProbe, public_scalar_control


class MixtureProbe(ScalarProbe):
    def __init__(self, weights, means, stds, *, tolerance_hz, max_adjustment_actions=16):
        super().__init__(max_adjustment_actions=max_adjustment_actions)
        self.weights, self.means, self.stds = [torch.as_tensor(x, dtype=torch.float64).detach().cpu().flatten()
                                             for x in (weights, means, stds)]
        if tolerance_hz <= 0 or not bool((self.stds > 0).all()):
            raise ValueError("Positive success tolerance and predictive scales are required")
        self.weights = self.weights / self.weights.sum()
        self.tolerance_hz = float(tolerance_hz)
        self.last_choice = None

    def _excluded(self):
        merged = []
        for value in sorted(self.failed_values):
            lower, upper = value - self.tolerance_hz, value + self.tolerance_hz
            if merged and lower <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], upper)
            else:
                merged.append([lower, upper])
        return merged

    def _mass(self, lower, upper):
        lower = torch.as_tensor(lower, dtype=torch.float64)[..., None]
        upper = torch.as_tensor(upper, dtype=torch.float64)[..., None]
        return (torch.special.ndtr((upper - self.means) / self.stds)
                - torch.special.ndtr((lower - self.means) / self.stds)).clamp_min(0)

    def remaining_mass(self, lower, upper):
        lower, upper = torch.as_tensor(lower, dtype=torch.float64), torch.as_tensor(upper, dtype=torch.float64)
        mass = self._mass(lower, upper)
        for left, right in self._excluded():
            lo, hi = lower.clamp_min(left), upper.clamp_max(right)
            mass -= self._mass(lo, hi)
        return mass.clamp_min(0)

    @torch.no_grad()
    def next_target(self, current, mean, std, public_dialog):
        if not self.started or self.activated or self._unconfirmed or not self.remaining_actions:
            return None
        control = public_scalar_control(public_dialog)
        if control is None:
            return None
        value = Decimal(str(current))
        # Integer coordinates preserve all offered decimal increments exactly.
        places = max(0, -min(x.as_tuple().exponent for x in
                            (value, control.lower_hz, control.upper_hz, *control.increments_hz)))
        scale = 10 ** places
        origin, lower, upper = (int(x * scale) for x in (value, control.lower_hz, control.upper_hz))
        steps = [int(x * scale) for x in control.increments_hz]
        distance, first_hop, queue = {origin: 0}, {}, deque([origin])
        while queue:
            point = queue.popleft()
            if distance[point] >= self.remaining_actions:
                continue
            for step in steps:
                reached = min(max(point + step, lower), upper)
                if reached in distance:
                    continue
                distance[reached] = distance[point] + 1
                first_hop[reached] = reached if point == origin else first_hop[point]
                queue.append(reached)
        candidates = sorted(first_hop)
        if not candidates:
            return None
        values = torch.tensor([x / scale for x in candidates], dtype=torch.float64)
        costs = torch.tensor([distance[x] for x in candidates], dtype=torch.float64)
        remaining = self.remaining_mass(float(control.lower_hz), float(control.upper_hz))
        normalizer = (remaining * self.weights).sum()
        if float(normalizer) <= 0:
            return None
        probability = (self.remaining_mass((values - self.tolerance_hz).clamp_min(float(control.lower_hz)),
                                           (values + self.tolerance_hz).clamp_max(float(control.upper_hz)))
                       * self.weights).sum(-1) / normalizer
        score = probability / costs
        selected = int(score.argmax())
        if float(score[selected]) <= 0:
            return None
        goal = candidates[selected]
        self.last_choice = {"target_hz": goal / scale, "first_hop_hz": first_hop[goal] / scale,
                            "shortest_actions": distance[goal], "success_probability": float(probability[selected]),
                            "remaining_hypothesis_weights": (remaining * self.weights / normalizer).tolist(),
                            "failed_settings": len(self.failed_values)}
        return first_hop[goal] / scale
