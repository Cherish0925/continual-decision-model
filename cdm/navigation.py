"""Navigation through a field behind walls, behind the same interface as the instruments.

A vehicle moves in the square [-1, 1]^2. An action names a point of the grid of
actuator values; the vehicle heads for it in a straight line at the speed of the
current condition and stops where a wall blocks the way, so reaching a point
behind a wall takes several actions, each a waypoint. The vehicle reports its
position as the state of its actuators. A beacon emits a field that falls with
the length of the shortest path around the walls; the sensor reads the field at
the vehicle, and a precise measurement returns it after a delay. A condition
draws the walls, the beacon, the field's height, width and floor, and the speed.
Requests give a field value to hold, the value at one of the grid points, or ask
for the largest field, which is at the beacon. Requests, readings and switches
of condition are those of ``ProcessFleet``, so one model can run instruments,
thermal processes and vehicles through the same events and actions.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from cdm.instrument import ProcessFleet, ProcessSpec


@dataclass(frozen=True)
class NavigationSpec(ProcessSpec):
    walls: int = 2                                   # wall segments per condition
    wall_length: tuple[float, float] = (0.8, 1.4)
    speed: tuple[float, float] = (0.06, 0.15)        # distance per tick
    height: tuple[float, float] = (0.8, 1.5)
    width: tuple[float, float] = (0.5, 1.0)          # path distance over which the field falls by e^-1/2
    floor: tuple[float, float] = (-0.3, 0.1)
    grid: int = 25                                   # cells per side of the path-distance grid
    jitter: float = 0.005                            # position noise per tick


FIELDS = 6           # height, floor, width, speed, beacon x, beacon y; then four numbers per wall


def wall_hits(start: torch.Tensor, end: torch.Tensor, walls: torch.Tensor) -> torch.Tensor:
    """Fraction of the way from start to end [..., 2] at which the first wall is met; 1 if none is.

    walls [..., W, 4]: orientation (0: the line x = c for y in [a, b]; 1: the line y = c for x in [a, b]), c, a, b.
    """
    vertical = walls[..., 0] < 0.5
    c, a, b = walls[..., 1], walls[..., 2], walls[..., 3]
    s, e = start.unsqueeze(-2), end.unsqueeze(-2)
    s_across, e_across = torch.where(vertical, s[..., 0], s[..., 1]), torch.where(vertical, e[..., 0], e[..., 1])
    s_along, e_along = torch.where(vertical, s[..., 1], s[..., 0]), torch.where(vertical, e[..., 1], e[..., 0])
    gap = e_across - s_across
    frac = (c - s_across) / torch.where(gap.abs() < 1e-9, torch.full_like(gap, 1e-9), gap)
    at = s_along + frac * (e_along - s_along)
    side = (s_across - c) * (e_across - c)
    hit = ((side < 0) | ((e_across == c) & (s_across != c))) & (at >= a) & (at <= b)   # crossing or landing on it
    return torch.where(hit, frac, torch.ones_like(frac)).min(-1).values


def path_distance(params: torch.Tensor, grid: int, walls: int) -> torch.Tensor:
    """Length of the shortest path around the walls from the beacon to every cell centre [m, grid, grid], for
    conditions params [m, P], through moves to the eight neighbouring cells; cells are indexed by x, then y."""
    m, device = params.shape[0], params.device
    centres = torch.linspace(-1.0, 1.0, grid, device=device)
    h = 2.0 / (grid - 1)
    w = params[:, FIELDS:].view(m, 1, walls, 4).expand(-1, grid * grid, -1, -1)
    xs, ys = torch.meshgrid(centres, centres, indexing="ij")
    point = torch.stack((xs, ys), -1).view(1, grid * grid, 2).expand(m, -1, -1)

    def span(d):                                            # cells a move of d leaves from, and arrives at
        return (slice(0, grid - d), slice(d, grid)) if d >= 0 else (slice(-d, grid), slice(0, grid + d))

    moves = []
    for dx, dy in ((1, 0), (0, 1), (1, 1), (1, -1)):
        step = point + torch.tensor([dx * h, dy * h], device=device)
        blocked = (wall_hits(point, step, w) < 1).view(m, grid, grid)
        (x0, x1), (y0, y1) = span(dx), span(dy)
        moves.append(((x0, y0), (x1, y1), blocked[:, x0, y0], h * (dx * dx + dy * dy) ** 0.5))
    beacon = params[:, 4:6]
    cell = ((beacon + 1) / h).round().long().clamp(0, grid - 1)
    D = torch.full((m, grid, grid), 1e3, device=device)
    D[torch.arange(m, device=device), cell[:, 0], cell[:, 1]] = (centres[cell] - beacon).norm(dim=-1)
    for _ in range(4 * grid):
        nxt = D.clone()
        for (a, b, blocked, cost) in moves:                 # each move both ways
            nxt[:, b[0], b[1]] = torch.minimum(nxt[:, b[0], b[1]], torch.where(blocked, 1e3, D[:, a[0], a[1]] + cost))
            nxt[:, a[0], a[1]] = torch.minimum(nxt[:, a[0], a[1]], torch.where(blocked, 1e3, D[:, b[0], b[1]] + cost))
        if torch.equal(nxt, D):
            break
        D = nxt
    return D


class NavigationFleet(ProcessFleet):
    """Vehicles that head for the named point, stop at walls, and read a field that bends around them."""

    def __init__(self, n: int, spec: NavigationSpec = NavigationSpec(), seed: int = 0, device="cpu"):
        super().__init__(n, spec, seed, device)

    def sample(self, shape):
        s = self.spec
        u = self._rand(*shape, FIELDS + 4 * s.walls)
        p = torch.empty_like(u)
        for i, (lo, hi) in enumerate((s.height, s.floor, s.width, s.speed)):
            p[..., i] = lo + (hi - lo) * u[..., i]
        p[..., 4:6] = -0.85 + 1.7 * u[..., 4:6]                                   # beacon
        w = u[..., FIELDS:].unflatten(-1, (s.walls, 4))
        length = s.wall_length[0] + (s.wall_length[1] - s.wall_length[0]) * w[..., 3]
        start = -1.0 + (2.0 - length) * w[..., 2]
        wall = torch.stack(((w[..., 0] < 0.5).float(), -0.6 + 1.2 * w[..., 1], start, start + length), -1)
        p[..., FIELDS:] = wall.flatten(-2)
        return p

    def _distances(self, params):
        s = self.spec
        return path_distance(params.reshape(-1, params.shape[-1]), s.grid, s.walls).view(
            *params.shape[:-1], s.grid, s.grid)

    def init_state(self):
        self.distance = self._distances(self.params)                             # [n, conditions, grid, grid]
        self.position = -0.9 + 1.8 * self._rand(self.n, 2)
        self.drift = torch.zeros(self.n, device=self.device)
        self.output = self.field(self.current(), self.position.unsqueeze(1), self.distance_now(), self.drift)[:, 0]

    def reset_state(self, idx):
        self.distance[idx] = self._distances(self.params[idx])
        self.position[idx] = -0.9 + 1.8 * torch.rand(len(idx), 2, generator=self.gen, device=self.device)
        self.drift[idx] = 0.0
        self.output[idx] = self.field(self.current()[idx], self.position[idx].unsqueeze(1),
                                      self.distance_now()[idx], self.drift[idx])[:, 0]

    def distance_now(self) -> torch.Tensor:
        return self.distance[torch.arange(self.n, device=self.device), self.condition]

    def field(self, p, position, distance, drift):
        """Field at positions [m, k, 2] for conditions p [m, P], path-distance grids [m, grid, grid] and drift [m]."""
        g = self.spec.grid
        x = ((position + 1) / 2 * (g - 1)).clamp(0, g - 1 - 1e-4)
        i, f = x.floor().long(), x - x.floor()
        rows = torch.arange(position.shape[0], device=position.device).unsqueeze(1)
        at = lambda di, dj: distance[rows, i[..., 0] + di, i[..., 1] + dj]
        fx, fy = f[..., 0], f[..., 1]
        d = at(0, 0) * (1 - fx) * (1 - fy) + at(1, 0) * fx * (1 - fy) + at(0, 1) * (1 - fx) * fy + at(1, 1) * fx * fy
        return p[:, :1] * torch.exp(-d ** 2 / (2 * p[:, 2:3] ** 2)) + p[:, 1:2] + drift.unsqueeze(1)

    def move(self, p, position, target, noise=None):
        """One tick from positions toward targets [m, k, 2], stopping just short of the first wall met."""
        way = target - position
        dist = way.norm(dim=-1, keepdim=True)
        end = position + way / dist.clamp_min(1e-9) * torch.minimum(dist, p[:, 3].view(-1, 1, 1))
        if noise is not None:
            end = end + noise
        walls = p[:, FIELDS:].view(p.shape[0], 1, -1, 4).expand(-1, position.shape[1], -1, -1)
        frac = wall_hits(position, end, walls).unsqueeze(-1)
        length = (end - position).norm(dim=-1, keepdim=True).clamp_min(1e-9)
        frac = torch.where(frac < 1, (frac - 1e-3 / length).clamp(min=0), frac)   # a millimetre short of the wall
        return (position + (end - position) * frac).clamp(-1, 1)

    def advance(self, u, p):
        s = self.spec
        self.drift = self.drift + s.drift * self._randn(self.n)
        jitter = s.jitter * self._randn(self.n, 2).unsqueeze(1)
        self.position = self.move(p, self.position.unsqueeze(1), u.unsqueeze(1), jitter)[:, 0]
        self.output = self.field(p, self.position.unsqueeze(1), self.distance_now(), self.drift)[:, 0] \
            + s.process_noise * self._randn(self.n)

    def on_switch(self, switch):
        self.drift = torch.where(switch, torch.zeros_like(self.drift), self.drift)
        fresh = switch & ~self.returned                     # the new condition may have new walls and beacon
        if fresh.any():
            idx = fresh.nonzero().squeeze(1)
            self.distance[idx, self.condition[idx]] = self._distances(self.params[idx, self.condition[idx]])

    def steady(self, p, values):
        """The field at the named points [n, dims] or [n, actions, dims]: what holding there would read."""
        points = values.unsqueeze(1) if values.dim() == 2 else values
        field = self.field(p, points, self.distance_now(), self.drift)
        return field[:, 0] if values.dim() == 2 else field

    def expected_next(self, values):
        p = self.current()
        reached = self.move(p, self.position.unsqueeze(1).expand_as(values), values)
        return self.field(p, reached, self.distance_now(), self.drift)

    def setpoint_weights(self, p):
        """Setpoints come from grid points where the field stands clear of its floor."""
        rise = self.field(p, self.table.expand(self.n, -1, -1), self.distance_now(), torch.zeros_like(self.drift))
        return (rise - p[:, 1:2] > 0.1 * p[:, 0:1]).float() + 1e-6
