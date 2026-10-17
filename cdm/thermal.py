"""Two-state thermal processes behind the same interface as the instruments.

A heater warms a block that exchanges heat with a body; the body loses heat to
its surroundings, faster while coolant flows. The measured output is the body
temperature:

    C1 dT1/dt = q(h) - k12 (T1 - T2) - k1 (T1 - Ta)
    C2 dT2/dt = k12 (T1 - T2) - k2 (1 + kappa c) (T2 - Ta)

The first actuator sets the heater command h in [0, 1], which acts after a
transport delay of up to two ticks through a nonlinear efficiency
q(h) = eta h^gamma; the second sets the coolant flow c in [0, 1]. The ambient
temperature Ta drifts. A condition draws the coupling, losses, coolant effect,
heater shape, ambient, both time constants and the delay; eta follows from the
drawn steady rise at full heat without coolant. Requests, readings and switches
of condition are those of ``ProcessFleet``, so one model can run instruments and
thermal processes through the same events and actions.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from cdm.instrument import ProcessFleet, ProcessSpec

RANGES = ("coupling", "block_loss", "body_loss", "cooling", "rise", "heater_shape", "ambient", "block_time",
          "body_time")


@dataclass(frozen=True)
class ThermalSpec(ProcessSpec):
    coupling: tuple[float, float] = (0.1, 0.3)       # k12
    block_loss: tuple[float, float] = (0.02, 0.06)   # k1
    body_loss: tuple[float, float] = (0.06, 0.15)    # k2
    cooling: tuple[float, float] = (0.5, 2.0)        # kappa
    rise: tuple[float, float] = (0.9, 1.6)           # steady rise of the body at full heat without coolant
    heater_shape: tuple[float, float] = (0.5, 2.0)   # gamma
    ambient: tuple[float, float] = (-0.4, 0.2)       # Ta
    block_time: tuple[float, float] = (1.0, 3.0)     # C1 / (k12 + k1), ticks
    body_time: tuple[float, float] = (3.0, 8.0)      # C2 / (k12 + k2), ticks
    heater_delay: int = 2                            # largest transport delay, ticks


class ThermalFleet(ProcessFleet):
    """Parameters per condition: k12, k1, k2, kappa, rise, gamma, Ta, block time, body time, delay."""

    def __init__(self, n: int, spec: ThermalSpec = ThermalSpec(), seed: int = 0, device="cpu"):
        super().__init__(n, spec, seed, device)

    def sample(self, shape):
        s = self.spec
        u = self._rand(*shape, len(RANGES) + 1)
        p = torch.empty_like(u)
        for i, name in enumerate(RANGES):
            lo, hi = getattr(s, name)
            p[..., i] = lo + (hi - lo) * u[..., i]
        p[..., -1] = torch.floor(u[..., -1] * (s.heater_delay + 1))
        return p

    @staticmethod
    def _plant(p, c):
        """Loss of the body with coolant c, heater gain, and the two heat capacities."""
        k12, k1, k2, kappa = p[..., 0], p[..., 1], p[..., 2], p[..., 3]
        cooled = k2 * (1 + kappa * c)
        eta = p[..., 4] * (k2 + k1 + k1 * k2 / k12)
        return cooled, eta, p[..., 7] * (k12 + k1), p[..., 8] * (k12 + k2)

    def steady(self, p, values):
        drift = self.drift
        if values.dim() == 3:                                        # every action: [n, actions, dims]
            p, drift = p.unsqueeze(1), drift.unsqueeze(1)
        h, c = (values[..., 0] + 1) / 2, (values[..., 1] + 1) / 2
        k12, k1 = p[..., 0], p[..., 1]
        cooled, eta, _, _ = self._plant(p, c)
        return p[..., 6] + drift + eta * h ** p[..., 5] / (cooled + k1 + k1 * cooled / k12)

    def init_state(self):
        p = self.current()
        self.drift = torch.zeros(self.n, device=self.device)
        self.block, self.body = p[:, 6].clone(), p[:, 6].clone()
        self.heater = torch.zeros(self.n, self.spec.heater_delay + 1, device=self.device)   # newest command first
        self.output = self.body.clone()

    def reset_state(self, idx):
        ambient = self.current()[idx, 6]
        self.drift[idx] = 0.0
        self.block[idx], self.body[idx] = ambient, ambient
        self.heater[idx] = 0.0
        self.output[idx] = ambient

    def advance(self, u, p):
        s = self.spec
        command, c = (u[:, 0] + 1) / 2, (u[:, 1] + 1) / 2
        self.heater = torch.cat((command.unsqueeze(1), self.heater[:, :-1]), 1)
        h = self.heater.gather(1, p[:, 9].long().unsqueeze(1)).squeeze(1)
        k12, k1 = p[:, 0], p[:, 1]
        cooled, eta, c1, c2 = self._plant(p, c)
        heat = eta * h ** p[:, 5]
        self.drift = self.drift + s.drift * self._randn(self.n)
        ambient = p[:, 6] + self.drift
        for _ in range(4):                                           # four quarter-tick Euler steps
            exchange = k12 * (self.block - self.body)
            block = self.block + 0.25 * (heat - exchange - k1 * (self.block - ambient)) / c1
            self.body = self.body + 0.25 * (exchange - cooled * (self.body - ambient)) / c2
            self.block = block
        self.body = self.body + s.process_noise * self._randn(self.n)
        self.output = self.body.clone()

    def expected_next(self, values):
        p, k = self.current().unsqueeze(1), values.shape[1]
        command, c = (values[..., 0] + 1) / 2, (values[..., 1] + 1) / 2
        history = torch.cat((command.unsqueeze(-1), self.heater[:, None, :-1].expand(-1, k, -1)), -1)
        h = history.gather(-1, p[..., 9].long().unsqueeze(-1).expand(-1, k, 1)).squeeze(-1)
        k12, k1 = p[..., 0], p[..., 1]
        cooled, eta, c1, c2 = self._plant(p, c)
        heat = eta * h ** p[..., 5]
        ambient = p[..., 6] + self.drift.unsqueeze(1)
        block, body = self.block.unsqueeze(1).expand_as(c), self.body.unsqueeze(1).expand_as(c)
        for _ in range(4):
            exchange = k12 * (block - body)
            block, body = (block + 0.25 * (heat - exchange - k1 * (block - ambient)) / c1,
                           body + 0.25 * (exchange - cooled * (body - ambient)) / c2)
        return body

    def on_switch(self, switch):
        self.drift = torch.where(switch, torch.zeros_like(self.drift), self.drift)
