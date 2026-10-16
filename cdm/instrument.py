"""Nonlinear instruments with delayed measurements and changing operating conditions.

Each stream is one device over one lifetime. Control actions set actuators
that move with backlash; the output relaxes toward a response of the actuator
state, shifted by a slow drift. A cheap sensor reads the output with large
noise every tick. A precise measurement can be requested; it returns the
output at the request time after a delay. Setpoint requests arrive at random
times and stay active for a fixed duration; between them the device is idle.
A setpoint is the response at one of the actions under the current condition;
for a peaked response, at an action where the peak is visible.

Two families share this interface:

``dims=1``  one actuator, saturating response gain * tanh(slope * (v - centre)) + offset
``dims=2``  two actuators, a peaked response height * exp(-|v - c|^2 / (2 width^2)) + floor,
            as when aligning a beam onto a detector

A lifetime holds a few operating conditions. The active condition switches at
random times, earlier conditions return, and a switch occasionally brings a
condition never seen before. Nothing announces a switch. All state lives in
tensors on one device; every stream advances in parallel.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ProcessSpec:
    """What every process family shares: actions, conditions, requests and sensors."""
    dims: int = 2                # actuators
    levels: int = 9              # control levels per actuator over [-1, 1]
    conditions: int = 4          # operating conditions per lifetime
    dwell: float = 400.0         # mean ticks between condition switches
    novel: float = 0.2           # probability that a switch brings a new condition
    request_rate: float = 1 / 40 # probability per idle tick that a request starts
    request_ticks: int = 60      # ticks a request stays active
    cheap_noise: float = 0.25
    precise_noise: float = 0.02
    delay: int = 3               # ticks until a precise measurement returns
    process_noise: float = 0.01
    drift: float = 0.004         # random-walk step of the output offset

    @property
    def actions(self) -> int:
        return self.levels ** self.dims


@dataclass(frozen=True)
class InstrumentSpec(ProcessSpec):
    width: tuple[float, float] = (0.25, 0.6)        # peak width (dims=2)
    time_constant: tuple[float, float] = (2.0, 8.0)  # ticks
    backlash: float = 0.15       # largest actuator backlash


def action_values(spec: ProcessSpec, device=None) -> torch.Tensor:
    """Actuator targets [actions, dims] for every discrete action."""
    grid = torch.linspace(-1.0, 1.0, spec.levels, device=device)
    if spec.dims == 1:
        return grid.unsqueeze(1)
    return torch.cartesian_prod(grid, grid)


def sample_conditions(spec: InstrumentSpec, shape, generator, device) -> torch.Tensor:
    """Condition parameters [..., 8]; the meaning of the first five depends on ``dims``; 5: time constant, 6: backlash."""
    u = torch.rand(*shape, 8, generator=generator, device=device)
    p = torch.zeros(*shape, 8, device=device)
    if spec.dims == 1:
        p[..., 0] = (0.6 + 0.8 * u[..., 0]) * torch.where(u[..., 7] < 0.25, -1.0, 1.0)   # gain
        p[..., 1] = 0.8 + 2.2 * u[..., 1]                                                 # slope
        p[..., 2] = -0.5 + u[..., 2]                                                      # centre
        p[..., 3] = -0.3 + 0.6 * u[..., 3]                                                # offset
    else:
        p[..., 0] = 0.8 + 0.7 * u[..., 0]                                                 # height
        p[..., 1] = -0.7 + 1.4 * u[..., 1]                                                # peak position, axis 1
        p[..., 2] = -0.7 + 1.4 * u[..., 2]                                                # peak position, axis 2
        p[..., 3] = spec.width[0] + (spec.width[1] - spec.width[0]) * u[..., 3]            # width
        p[..., 4] = -0.3 + 0.4 * u[..., 4]                                                # floor
    tau = spec.time_constant
    p[..., 5] = tau[0] + (tau[1] - tau[0]) * u[..., 5]                                    # time constant (ticks)
    p[..., 6] = spec.backlash * u[..., 6]                                                 # backlash
    return p


def response(spec: InstrumentSpec, params: torch.Tensor, actuator: torch.Tensor, drift: torch.Tensor) -> torch.Tensor:
    if spec.dims == 1:
        return params[..., 0] * torch.tanh(params[..., 1] * (actuator[..., 0] - params[..., 2])) + params[..., 3] + drift
    d2 = (actuator[..., 0] - params[..., 1]) ** 2 + (actuator[..., 1] - params[..., 2]) ** 2
    return params[..., 0] * torch.exp(-d2 / (2 * params[..., 3] ** 2)) + params[..., 4] + drift


class ProcessFleet:
    """``n`` processes advancing in lockstep.

    The fleet keeps the operating conditions of every lifetime, switches between them, issues setpoint requests,
    and returns a cheap reading every tick and precise measurements after a delay. A family supplies its
    conditions (``sample``), its state and dynamics (``init_state``, ``reset_state``, ``advance``, ``on_switch``),
    and the steady response that defines reachable setpoints (``steady``, ``setpoint_weights``).
    """

    def __init__(self, n: int, spec: ProcessSpec, seed: int = 0, device="cpu"):
        self.n, self.spec, self.device = n, spec, torch.device(device)
        self.gen = torch.Generator(device=self.device).manual_seed(seed)
        self.table = action_values(spec, self.device)
        g, d = self.gen, self.device
        self.params = self.sample((n, spec.conditions))
        self.condition = torch.randint(0, spec.conditions, (n,), generator=g, device=d)
        self.seen = torch.zeros(n, spec.conditions, dtype=torch.bool, device=d)
        self.seen[torch.arange(n, device=d), self.condition] = True
        self.since_switch = torch.zeros(n, dtype=torch.long, device=d)
        self.returned = torch.zeros(n, dtype=torch.bool, device=d)   # current condition was active before
        self.init_state()
        self.setpoint = torch.zeros(n, device=d)
        self.active = torch.zeros(n, dtype=torch.bool, device=d)
        self.request_left = torch.zeros(n, dtype=torch.long, device=d)
        self.pending = torch.zeros(n, dtype=torch.long, device=d)    # ticks until the precise value returns
        self.pending_value = torch.zeros(n, device=d)
        self.tick = 0

    # ------------------------------------------------------------ family interface
    def sample(self, shape) -> torch.Tensor:
        raise NotImplementedError

    def init_state(self):
        raise NotImplementedError

    def reset_state(self, idx: torch.Tensor):
        raise NotImplementedError

    def advance(self, u: torch.Tensor, p: torch.Tensor):
        """One tick of the dynamics under actuator targets u [n, dims]; updates ``self.output``."""
        raise NotImplementedError

    def on_switch(self, switch: torch.Tensor):
        pass

    def steady(self, p: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        """Steady output for actuator values [n, dims], or [n, actions, dims], under conditions p [n, ...]."""
        raise NotImplementedError

    def setpoint_weights(self, p: torch.Tensor) -> torch.Tensor:
        return torch.ones(self.n, self.spec.actions, device=self.device)

    # ------------------------------------------------------------ shared machinery
    def current(self) -> torch.Tensor:
        return self.params[torch.arange(self.n, device=self.device), self.condition]

    def reset(self, mask: torch.Tensor):
        """Start new lifetimes (fresh conditions and process state) for the masked streams."""
        if not mask.any():
            return
        s, g, d = self.spec, self.gen, self.device
        idx = mask.nonzero().squeeze(1)
        self.params[idx] = self.sample((len(idx), s.conditions))
        self.condition[idx] = torch.randint(0, s.conditions, (len(idx),), generator=g, device=d)
        self.seen[idx] = False
        self.seen[idx, self.condition[idx]] = True
        self.since_switch[idx] = 0
        self.returned[idx] = False
        self.reset_state(idx)
        self.active[idx] = False
        self.request_left[idx] = 0
        self.pending[idx] = 0

    def _switch(self):
        s, g, d = self.spec, self.gen, self.device
        switch = torch.rand(self.n, generator=g, device=d) < 1.0 / s.dwell
        if not switch.any():
            self.since_switch += 1
            return
        target = (self.condition + torch.randint(1, s.conditions, (self.n,), generator=g, device=d)) % s.conditions
        novel = switch & (torch.rand(self.n, generator=g, device=d) < s.novel)
        fresh = self.sample((self.n,))
        idx = torch.arange(self.n, device=d)
        self.params[idx[novel], target[novel]] = fresh[novel]
        self.seen[idx[novel], target[novel]] = False
        self.condition = torch.where(switch, target, self.condition)
        self.returned = torch.where(switch, self.seen[idx, self.condition], self.returned)
        self.seen[idx[switch], self.condition[switch]] = True
        self.since_switch = torch.where(switch, torch.zeros_like(self.since_switch), self.since_switch + 1)
        self.on_switch(switch)

    def _requests(self):
        s, g, d = self.spec, self.gen, self.device
        self.request_left = (self.request_left - 1).clamp(min=0)
        ended = self.active & (self.request_left == 0)
        self.active &= ~ended
        start = ~self.active & (torch.rand(self.n, generator=g, device=d) < s.request_rate)
        p = self.current()
        reachable = torch.multinomial(self.setpoint_weights(p), 1, generator=g).squeeze(1)
        target = self.steady(p, self.table[reachable])
        self.setpoint = torch.where(start, target, self.setpoint)
        self.active |= start
        self.request_left = torch.where(start, torch.full_like(self.request_left, s.request_ticks), self.request_left)
        return start

    def step(self, action: torch.Tensor, measure: torch.Tensor) -> dict[str, torch.Tensor]:
        """Advance one tick; keep completed-transition metadata separate from the next observation."""
        s, g, d = self.spec, self.gen, self.device
        transition = {"transition_active": self.active.clone(),
                      "transition_setpoint": self.setpoint.clone(),
                      "transition_since_switch": self.since_switch.clone(),
                      "transition_returned": self.returned.clone()}
        u = self.table[action]                                       # [n, dims]
        self.advance(u, self.current())
        error = torch.where(self.active, (self.output - self.setpoint).abs(), torch.zeros_like(self.output))
        arrived = self.pending == 1
        arrived_value = self.pending_value + s.precise_noise * torch.randn(self.n, generator=g, device=d)
        self.pending = (self.pending - 1).clamp(min=0)
        accepted = measure & (self.pending == 0)
        self.pending = torch.where(accepted, torch.full_like(self.pending, s.delay), self.pending)
        self.pending_value = torch.where(accepted, self.output, self.pending_value)
        cheap = self.output + s.cheap_noise * torch.randn(self.n, generator=g, device=d)
        self._switch()
        started = self._requests()
        self.tick += 1
        return {"cheap": cheap, "precise": torch.where(arrived, arrived_value, torch.zeros_like(arrived_value)),
                "arrived": arrived, "accepted": accepted, "setpoint": self.setpoint.clone(),
                "active": self.active.clone(), "started": started, "error": error,
                "output": self.output.clone(), "since_switch": self.since_switch.clone(),
                "returned": self.returned.clone(), "control": u, **transition}


class InstrumentFleet(ProcessFleet):
    """Instruments whose actuators move with backlash and whose output relaxes toward ``response``."""

    def __init__(self, n: int, spec: InstrumentSpec = InstrumentSpec(), seed: int = 0, device="cpu"):
        super().__init__(n, spec, seed, device)

    def sample(self, shape):
        return sample_conditions(self.spec, shape, self.gen, self.device)

    def init_state(self):
        self.actuator = torch.zeros(self.n, self.spec.dims, device=self.device)
        self.drift = torch.zeros(self.n, device=self.device)
        self.output = response(self.spec, self.current(), self.actuator, self.drift)

    def reset_state(self, idx):
        self.actuator[idx] = 0.0
        self.drift[idx] = 0.0
        self.output[idx] = response(self.spec, self.current()[idx], self.actuator[idx], self.drift[idx])

    def advance(self, u, p):
        s, g, d = self.spec, self.gen, self.device
        backlash = p[:, 6:7]
        self.actuator = torch.where(u - self.actuator > backlash, u - backlash,
                                    torch.where(u - self.actuator < -backlash, u + backlash, self.actuator))
        self.drift = self.drift + s.drift * torch.randn(self.n, generator=g, device=d)
        target = response(s, p, self.actuator, self.drift)
        self.output = self.output + (target - self.output) / p[:, 5] + \
            s.process_noise * torch.randn(self.n, generator=g, device=d)

    def on_switch(self, switch):
        self.drift = torch.where(switch, torch.zeros_like(self.drift), self.drift)

    def steady(self, p, values):
        if values.dim() == 3:                                        # every action: [n, actions, dims]
            return response(self.spec, p.unsqueeze(1), values, self.drift.unsqueeze(1))
        return response(self.spec, p, values, self.drift)

    def setpoint_weights(self, p):
        """Setpoints come from actions where a peak is visible."""
        if self.spec.dims == 1:
            return super().setpoint_weights(p)
        d2 = (self.table[None, :, 0] - p[:, 1:2]) ** 2 + (self.table[None, :, 1] - p[:, 2:3]) ** 2
        return (torch.exp(-d2 / (2 * p[:, 3:4] ** 2)) > 0.1).float()


class MixedFleet:
    """Streams of several process families advancing together in fixed blocks; family membership is not
    observed through the events, only through how each process responds."""

    def __init__(self, fleets: list[ProcessFleet]):
        first = fleets[0].spec
        assert all((f.spec.dims, f.spec.levels, f.spec.request_ticks) == (first.dims, first.levels,
                                                                           first.request_ticks) for f in fleets)
        self.fleets, self.sizes = fleets, [f.n for f in fleets]
        self.n, self.spec, self.device, self.table = sum(self.sizes), first, fleets[0].device, fleets[0].table
        self.family = torch.cat([torch.full((f.n,), i, device=self.device) for i, f in enumerate(fleets)])

    @property
    def pending(self) -> torch.Tensor:
        return torch.cat([f.pending for f in self.fleets])

    @property
    def tick(self) -> int:
        return self.fleets[0].tick

    def reset(self, mask: torch.Tensor):
        for fleet, part in zip(self.fleets, mask.split(self.sizes)):
            fleet.reset(part)

    def step(self, action: torch.Tensor, measure: torch.Tensor) -> dict[str, torch.Tensor]:
        outs = [f.step(a, m) for f, a, m in zip(self.fleets, action.split(self.sizes), measure.split(self.sizes))]
        return {key: torch.cat([o[key] for o in outs]) for key in outs[0]}
