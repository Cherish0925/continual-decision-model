"""Dynamic navigation with an exogenous, fully observed context stream."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import numpy as np

ACTION_DELTAS = np.array([(0, 0), (-1, 0), (0, 1), (1, 0), (0, -1)])


@dataclass(frozen=True)
class Context:
    blocked: np.ndarray
    goal: int
    position: int
    phase_id: int
    tick: int


@dataclass(frozen=True)
class Trace:
    maps: np.ndarray
    goals: np.ndarray
    phases: np.ndarray
    start: int
    side: int
    regime: str
    seed: int


def make_trace(seed: int, regime: str, ticks: int = 192, side: int = 9,
               phase_length: int = 24) -> Trace:
    """Future trace is retained by the environment, never passed to a planner."""
    if side != 9:
        raise ValueError("The initial maze uses side=9.")
    if regime not in {"full_return", "partial_return", "continuous"}:
        raise ValueError(regime)
    rng = np.random.default_rng(seed)
    base = np.zeros((side, side), dtype=bool)
    base[1:8, 3] = True
    base[1:8, 6] = True
    door_sets = [((2, 3), (6, 6)), ((6, 3), (2, 6)),
                 ((4, 3), (4, 6)), ((2, 3), (2, 6))]
    goals_pool = [1 * side + 8, 7 * side + 8, 7 * side + 1, 1 * side + 1]
    permutation = rng.permutation(4)
    movable = [(r, c) for r in (2, 4, 6) for c in (1, 4, 7)]
    obstacle_order = rng.permutation(len(movable))
    maps, goals, phases = [], [], []
    # Full return repeats complete map-goal pairs. Partial return repeats one
    # factor while recombining another. Continuous changes add a moving block.
    for tick in range(ticks + 1):
        phase = max(tick - 1, 0) // phase_length
        if regime == "full_return":
            key = int(permutation[phase % 4])
            door_key, goal_key = key, key
            obstacle_key = key % len(movable)
        elif regime == "partial_return":
            door_key = int(permutation[(phase + phase // 4) % 4])
            goal_key = int(permutation[phase % 4])
            obstacle_key = (phase // 3) % len(movable)
        else:
            door_key = int(permutation[(tick // 9) % 4])
            goal_key = int(permutation[phase % 4])
            obstacle_key = (tick // 3) % len(movable)
        blocked = base.copy()
        for door in door_sets[door_key]:
            blocked[door] = False
        obstacle = movable[int(obstacle_order[obstacle_key])]
        blocked[obstacle] = True
        maps.append(blocked.reshape(-1))
        goals.append(goals_pool[goal_key])
        phases.append(phase)
    return Trace(np.asarray(maps), np.asarray(goals), np.asarray(phases),
                 start=4 * side, side=side, regime=regime, seed=seed)


class GridWorld:
    def __init__(self, trace: Trace):
        self.trace = trace
        self.tick = 0
        self.position = trace.start
        self.completed_phase = -1

    def observe(self) -> Context:
        return Context(self.trace.maps[self.tick].copy(),
                       int(self.trace.goals[self.tick]), self.position,
                       int(self.trace.phases[self.tick]), self.tick)

    def advance_time(self) -> Context:
        """One bounded computation block consumes one environment tick."""
        if self.tick + 1 >= len(self.trace.maps):
            raise IndexError("Trace exhausted")
        self.tick += 1
        return self.observe()

    def move(self, action: int) -> tuple[Context, float, bool]:
        context = self.observe()
        nxt, legal = transition_table(context.blocked, self.trace.side)
        if not 0 <= action < 5:
            raise ValueError(action)
        self.position = int(nxt[self.position, action])
        reached = self.position == context.goal and self.completed_phase != context.phase_id
        if reached:
            self.completed_phase = context.phase_id
        # Only the first arrival in a goal phase is rewarded. There is no
        # artificial relocation when a moving obstacle occupies the agent cell;
        # the agent can leave but cannot enter a blocked destination.
        reward = 1.0 if reached else (-0.02 if self.completed_phase != context.phase_id else 0.0)
        return self.observe(), reward, reached

    def step(self, action: int) -> tuple[Context, float, bool]:
        self.advance_time()
        return self.move(action)


def transition_table(blocked: np.ndarray, side: int) -> tuple[np.ndarray, np.ndarray]:
    states = np.arange(side * side)
    row, col = states // side, states % side
    nr = row[:, None] + ACTION_DELTAS[None, :, 0]
    nc = col[:, None] + ACTION_DELTAS[None, :, 1]
    inside = (nr >= 0) & (nr < side) & (nc >= 0) & (nc < side)
    candidate = np.clip(nr, 0, side - 1) * side + np.clip(nc, 0, side - 1)
    legal = inside & ~blocked[candidate]
    legal[:, 0] = True
    return np.where(legal, candidate, states[:, None]), legal


def values_from_distance(distance: np.ndarray, gamma: float = .97,
                         step_cost: float = .02) -> np.ndarray:
    distance = np.asarray(distance, dtype=float)
    out = np.full(distance.shape, -step_cost / (1 - gamma))
    finite = np.isfinite(distance) & (distance > 0)
    power = gamma ** (distance[finite] - 1)
    out[finite] = power - step_cost * (1 - power) / (1 - gamma)
    out[distance == 0] = 0
    return out


def geometric_values(goal: int, side: int) -> np.ndarray:
    states = np.arange(side * side)
    distance = abs(states // side - goal // side) + abs(states % side - goal % side)
    return values_from_distance(distance)


def reference_values(context: Context, side: int) -> np.ndarray:
    """Exact stationary current-map shortest-path value, without future inputs."""
    distance = np.full(side * side, np.inf)
    distance[context.goal] = 0
    queue = deque([context.goal])
    while queue:
        state = queue.popleft()
        r, c = divmod(state, side)
        for dr, dc in ACTION_DELTAS[1:]:
            rr, cc = r + int(dr), c + int(dc)
            if 0 <= rr < side and 0 <= cc < side:
                adjacent = rr * side + cc
                if not context.blocked[adjacent] and np.isinf(distance[adjacent]):
                    distance[adjacent] = distance[state] + 1
                    queue.append(adjacent)
    # A dynamic obstacle can occupy the current agent cell. Its value is the
    # best outgoing transition, although no other state may enter it.
    values = values_from_distance(distance)
    nxt, legal = transition_table(context.blocked, side)
    q = action_values(values, context, side, nxt, legal)
    values[context.blocked] = q.max(axis=1)[context.blocked]
    return values


def action_values(values: np.ndarray, context: Context, side: int,
                  nxt=None, legal=None) -> np.ndarray:
    if nxt is None:
        nxt, legal = transition_table(context.blocked, side)
    q = np.where(nxt == context.goal, 1., -.02 + .97 * values[nxt])
    q[~legal] = -np.inf
    q[context.goal, :] = -np.inf
    q[context.goal, 0] = 0
    return q


def choose_action(values: np.ndarray, context: Context, side: int) -> int:
    """Every method uses fresh legal moves and the current goal in this head."""
    nxt, legal = transition_table(context.blocked, side)
    q = action_values(values, context, side, nxt, legal)[context.position]
    candidates = np.flatnonzero(np.isclose(q, q.max(), atol=1e-10, rtol=0))
    destinations = nxt[context.position, candidates]
    manhattan = abs(destinations // side - context.goal // side) + abs(destinations % side - context.goal % side)
    return int(candidates[np.argmin(manhattan)])


def optimal_action(context: Context, side: int) -> int:
    return choose_action(reference_values(context, side), context, side)
