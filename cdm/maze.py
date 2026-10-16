"""Dynamic room mazes for studying computation that continues while the world changes.

A maze is an ``n x n`` grid of 3x3 rooms separated by walls. A random spanning
tree of doors is always open, so every interior cell stays reachable. The
remaining doors are shortcuts that open and close. Single obstacles occupy room
corners or centres, which never block a doorway. Goals sit on the middle cells
of room edges, which obstacles never occupy.

Episodes are exogenous schedules of (blocked map, goal) pairs. The agent's own
position never changes the schedule, so every method sees the same world.
Labels are computed on the GPU: shortest-path distance to the goal for every
cell and the set of optimal actions for every cell.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

# stay, up, right, down, left
ACTION_DELTAS = ((0, 0), (-1, 0), (0, 1), (1, 0), (0, -1))
REGIMES = ("full_return", "partial_return", "continuous", "static")
ROOM = 3


@dataclass(frozen=True)
class MazeSpec:
    rooms: int = 5

    @property
    def side(self) -> int:
        return self.rooms * (ROOM + 1) + 1

    def room_origin(self, i: int, j: int) -> tuple[int, int]:
        return i * (ROOM + 1) + 1, j * (ROOM + 1) + 1


@dataclass(frozen=True)
class Layout:
    spec: MazeSpec
    walls: np.ndarray          # [side, side] walls with every door closed
    tree_doors: np.ndarray     # flat indices of doors that stay open
    extra_doors: np.ndarray    # flat indices of shortcut doors
    safe_cells: np.ndarray     # [rooms*rooms, 5] obstacle sites per room
    goal_cells: np.ndarray     # flat indices of admissible goals and starts


def make_layout(rng: np.random.Generator, spec: MazeSpec) -> Layout:
    n, side = spec.rooms, spec.side
    walls = np.zeros((side, side), dtype=bool)
    walls[::ROOM + 1, :] = True
    walls[:, ::ROOM + 1] = True
    doors = {}
    for i in range(n):
        for j in range(n):
            r0, c0 = spec.room_origin(i, j)
            if j + 1 < n:
                doors[((i, j), (i, j + 1))] = (r0 + ROOM // 2) * side + (j + 1) * (ROOM + 1)
            if i + 1 < n:
                doors[((i, j), (i + 1, j))] = (i + 1) * (ROOM + 1) * side + c0 + ROOM // 2
    # Random depth-first spanning tree over rooms keeps the maze connected.
    visited = np.zeros((n, n), dtype=bool)
    start = (int(rng.integers(n)), int(rng.integers(n)))
    stack, tree = [start], set()
    visited[start] = True
    while stack:
        i, j = stack[-1]
        options = [(i + di, j + dj) for di, dj in ((-1, 0), (0, 1), (1, 0), (0, -1))
                   if 0 <= i + di < n and 0 <= j + dj < n and not visited[i + di, j + dj]]
        if not options:
            stack.pop()
            continue
        nxt = options[int(rng.integers(len(options)))]
        visited[nxt] = True
        tree.add(tuple(sorted(((i, j), nxt))))
        stack.append(nxt)
    tree_doors = np.array(sorted(doors[edge] for edge in tree))
    extra_doors = np.array(sorted(cell for edge, cell in doors.items() if edge not in tree))
    safe, goals = [], []
    for i in range(n):
        for j in range(n):
            r0, c0 = spec.room_origin(i, j)
            corners_centre = [(0, 0), (0, ROOM - 1), (ROOM - 1, 0), (ROOM - 1, ROOM - 1), (ROOM // 2, ROOM // 2)]
            safe.append([(r0 + a) * side + c0 + b for a, b in corners_centre])
            goals.extend((r0 + a) * side + c0 + b for a, b in ((0, 1), (1, 0), (1, 2), (2, 1)))
    return Layout(spec, walls, tree_doors, extra_doors, np.array(safe), np.array(goals))


@dataclass(frozen=True)
class Config:
    doors_open: np.ndarray   # [len(extra_doors)] bool
    obstacles: np.ndarray    # flat obstacle cells, one per occupied room
    goal: int


def sample_config(rng: np.random.Generator, layout: Layout, obstacles: int | None = None) -> Config:
    rooms = layout.spec.rooms ** 2
    count = layout.spec.rooms if obstacles is None else obstacles
    occupied = rng.choice(rooms, size=count, replace=False)
    sites = layout.safe_cells[occupied, rng.integers(5, size=count)]
    return Config(rng.random(len(layout.extra_doors)) < 0.5, np.sort(sites),
                  int(rng.choice(layout.goal_cells)))


def render(layout: Layout, config: Config) -> np.ndarray:
    blocked = layout.walls.reshape(-1).copy()
    blocked[layout.tree_doors] = False
    blocked[layout.extra_doors[config.doors_open]] = False
    blocked[config.obstacles] = True
    return blocked.reshape(layout.walls.shape)


def _move_obstacle(rng, layout: Layout, obstacles: np.ndarray) -> np.ndarray:
    rooms_of = {int(cell): idx for idx, row in enumerate(layout.safe_cells) for cell in row}
    occupied = {rooms_of[int(c)] for c in obstacles}
    free_rooms = [r for r in range(layout.spec.rooms ** 2) if r not in occupied]
    moved = obstacles.copy()
    k = int(rng.integers(len(moved)))
    room = free_rooms[int(rng.integers(len(free_rooms)))] if free_rooms and rng.random() < .5 else rooms_of[int(moved[k])]
    moved[k] = layout.safe_cells[room, int(rng.integers(5))]
    return np.sort(moved)


def make_episode(rng: np.random.Generator, spec: MazeSpec, regime: str, ticks: int,
                 phase_range: tuple[int, int] = (6, 12), door_rate: float = 0.3,
                 obstacle_rate: float = 0.3) -> dict[str, np.ndarray]:
    """Exogenous schedule of maps and goals for ``ticks + 1`` observations.

    In the continuous regime a shortcut toggles with probability ``door_rate`` and an
    obstacle moves with probability ``obstacle_rate`` each tick.
    """
    if regime not in REGIMES:
        raise ValueError(regime)
    layout = make_layout(rng, spec)
    configs: list[Config] = []
    if regime == "static":
        configs = [sample_config(rng, layout)] * (ticks + 1)
    elif regime == "full_return":
        cycle = [sample_config(rng, layout) for _ in range(3)]
        index, left = 0, int(rng.integers(*phase_range, endpoint=True))
        for _ in range(ticks + 1):
            configs.append(cycle[index % 3])
            left -= 1
            if left == 0:
                index, left = index + 1, int(rng.integers(*phase_range, endpoint=True))
    elif regime == "partial_return":
        doors = [sample_config(rng, layout).doors_open for _ in range(3)]
        obstacles = [sample_config(rng, layout).obstacles for _ in range(3)]
        goals = [int(g) for g in rng.choice(layout.goal_cells, size=3, replace=False)]
        choice = tuple(int(x) for x in rng.integers(3, size=3))
        left = int(rng.integers(*phase_range, endpoint=True))
        for _ in range(ticks + 1):
            configs.append(Config(doors[choice[0]], obstacles[choice[1]], goals[choice[2]]))
            left -= 1
            if left == 0:
                new = choice
                while new == choice:  # recombine factors; at least one changes
                    new = tuple(int(x) for x in rng.integers(3, size=3))
                choice, left = new, int(rng.integers(*phase_range, endpoint=True))
    else:  # continuous: doors toggle, obstacles move, goals jump
        config = sample_config(rng, layout)
        goal_left = int(rng.integers(2 * phase_range[0], 2 * phase_range[1], endpoint=True))
        for _ in range(ticks + 1):
            configs.append(config)
            doors_open, obstacles, goal = config.doors_open, config.obstacles, config.goal
            if rng.random() < door_rate:
                doors_open = doors_open.copy()
                k = int(rng.integers(len(doors_open)))
                doors_open[k] = ~doors_open[k]
            if rng.random() < obstacle_rate:
                obstacles = _move_obstacle(rng, layout, obstacles)
            goal_left -= 1
            if goal_left == 0:
                goal = int(rng.choice(layout.goal_cells[layout.goal_cells != goal]))
                goal_left = int(rng.integers(2 * phase_range[0], 2 * phase_range[1], endpoint=True))
            config = Config(doors_open, obstacles, goal)
    blocked = np.stack([render(layout, c) for c in configs])
    goal = np.array([c.goal for c in configs])
    start_options = layout.goal_cells[layout.goal_cells != goal[0]]
    flat = blocked.reshape(len(configs), -1)
    changed = np.zeros(len(configs), dtype=bool)
    changed[1:] = (flat[1:] != flat[:-1]).any(1) | (goal[1:] != goal[:-1])
    return {"blocked": blocked, "goal": goal, "start": np.int64(rng.choice(start_options)),
            "changed": changed}


def make_batch(rng: np.random.Generator, spec: MazeSpec, regimes: list[str], ticks: int,
               phase_range: tuple[int, int] = (6, 12), door_rate: float = 0.3,
               obstacle_rate: float = 0.3) -> dict[str, np.ndarray]:
    episodes = [make_episode(rng, spec, r, ticks, phase_range, door_rate, obstacle_rate) for r in regimes]
    return {key: np.stack([e[key] for e in episodes]) for key in episodes[0]}


# ----------------------------------------------------------------- GPU labels

def _shift(x: torch.Tensor, dr: int, dc: int, fill: float) -> torch.Tensor:
    """y[r, c] = x[r + dr, c + dc] with out-of-range entries set to ``fill``."""
    padded = F.pad(x, (1, 1, 1, 1), value=fill)
    h, w = x.shape[-2:]
    return padded[..., 1 + dr:1 + dr + h, 1 + dc:1 + dc + w]


@torch.no_grad()
def distance_maps(blocked: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
    """Exact 4-neighbour shortest-path distance to the goal; inf where blocked or unreachable.

    blocked: [N, H, W] bool; goal: [N] flat index. Returns float32 [N, H, W].
    """
    n, h, w = blocked.shape
    inf = float("inf")
    dist = torch.full((n, h * w), inf, device=blocked.device)
    dist.scatter_(1, goal.view(-1, 1), 0.0)
    dist = dist.view(n, h, w)
    free = ~blocked
    for it in range(h * w):
        best = torch.minimum(torch.minimum(_shift(dist, -1, 0, inf), _shift(dist, 1, 0, inf)),
                             torch.minimum(_shift(dist, 0, -1, inf), _shift(dist, 0, 1, inf)))
        new = torch.where(free, torch.minimum(dist, best + 1), dist)
        if it % 8 == 7 and torch.equal(new, dist):
            break
        dist = new
    return dist


@torch.no_grad()
def optimal_actions(dist: torch.Tensor, blocked: torch.Tensor) -> torch.Tensor:
    """Multi-hot optimal actions for every cell: [N, 5, H, W] bool.

    A legal move enters an unblocked in-bounds cell; staying is always legal.
    The goal's only optimal action is to stay. On a cell that an obstacle has
    just covered, the agent may leave towards the best neighbour.
    """
    inf = float("inf")
    neighbour = torch.stack([dist] + [_shift(dist, dr, dc, inf) for dr, dc in ACTION_DELTAS[1:]], 1)
    entering = torch.stack([~blocked] + [~_shift(blocked.float(), dr, dc, 1.0).bool()
                                          for dr, dc in ACTION_DELTAS[1:]], 1)
    entering[:, 0] = True
    value = torch.where(entering, neighbour, torch.full_like(neighbour, inf))
    value[:, 0] = torch.where(dist == 0, torch.zeros_like(dist), torch.full_like(dist, inf))
    best = value.min(1, keepdim=True).values
    optimal = (value == best) & torch.isfinite(best)
    stuck = ~torch.isfinite(best).squeeze(1)  # unreachable: waiting is optimal
    optimal[:, 0] |= stuck
    return optimal


@torch.no_grad()
def move(blocked: torch.Tensor, position: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
    """Apply actions; illegal moves leave the agent in place."""
    n, h, w = blocked.shape
    deltas = torch.tensor(ACTION_DELTAS, device=blocked.device)
    r, c = position // w + deltas[action, 0], position % w + deltas[action, 1]
    inside = (r >= 0) & (r < h) & (c >= 0) & (c < w)
    target = r.clamp(0, h - 1) * w + c.clamp(0, w - 1)
    free = ~blocked.view(n, -1).gather(1, target.view(-1, 1)).squeeze(1)
    return torch.where(inside & free, target, position)
