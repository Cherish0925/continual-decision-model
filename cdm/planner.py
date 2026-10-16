"""Recurrent latent planner whose computation persists, is stored and is reused.

The workspace ``z`` is a spatial feature map refined by one shared gated
convolutional block. Each refinement propagates information about two cells,
so a plan for a large maze needs many refinements. When the map or goal
changes, a continuation mode decides which computation the next refinement
starts from:

``restart``        initialise from the new context
``carry``          continue the current workspace under the new context
                   (``goal_read`` adds the workspace at the goal cell to every
                   refinement and readout, without splitting the workspace;
                   ``change_gate`` adds a learned per-cell, per-channel choice
                   between keeping and rebuilding the workspace at each change)
``goal_restart``   initialise when the goal moves, continue when only the map changes
``snapshot``       load a stored workspace whose map and goal match exactly
``nearest``        load the stored workspace with the same goal and closest map
``memory_local``   learned per-cell attention over stored workspaces, then a
                   per-cell gate between carried, retrieved and fresh computation
``memory_global``  the same reader with one weight per stored workspace
``factorized``     split computation by what it depends on: a goal-independent
                   map workspace that survives goal changes, and a goal
                   workspace reseeded from the map workspace at the goal cell;
                   optionally keep map workspaces keyed by the map alone and
                   restore one when its map returns, whatever the goal

Every mode reads the current map and goal at every refinement and at the
action readout. Stored workspaces are kept with the map and goal that produced
them in a bounded ring that every memory mode fills identically.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import torch
import torch.nn.functional as F
from torch import Tensor, nn

MODES = ("restart", "carry", "goal_restart", "snapshot", "nearest", "memory_local", "memory_global", "factorized")
LEARNED_READERS = ("memory_local", "memory_global")


@dataclass(frozen=True)
class PlannerConfig:
    mode: str = "carry"
    channels: int = 96
    key_dim: int = 32
    memory_size: int = 8
    warmup: int = 16
    goal_broadcast: bool = True   # factorized: read the map workspace at the goal cell
    reseed: bool = True           # factorized: rebuild the goal workspace when the goal moves
    map_memory: int = 0           # factorized: stored map workspaces, keyed by the map alone
    goal_read: bool = False       # single workspace: read it at the goal cell in every refinement
    change_gate: bool = False     # single workspace: learned keep/rebuild gate applied when the input changes
    age_input: bool = False       # an extra input plane: how long ago each cell was last seen

    def __post_init__(self):
        if self.mode not in MODES:
            raise ValueError(f"unknown mode {self.mode}")


@dataclass(frozen=True)
class PlannerState:
    z: Tensor            # [B, C, H, W] workspace
    x: Tensor            # [B, C, H, W] encoded context the workspace was refined under
    blocked: Tensor      # [B, H, W]
    goal: Tensor         # [B]
    mem_z: Tensor        # [B, M, C, H, W]
    mem_x: Tensor        # [B, M, C, H, W]
    mem_blocked: Tensor  # [B, M, H, W]
    mem_goal: Tensor     # [B, M]
    mem_tick: Tensor     # [B, M] tick at which the record was written
    mem_valid: Tensor    # [B, M]
    cursor: Tensor       # [B]
    tick: int
    gates: Tensor | None = None    # [B, 3, H, W] keep / retrieve / initialise, or [B, C, H, W] rebuild share
    reads: Tensor | None = None    # [B, 1 + M] mean read weight per candidate

    def detach(self) -> "PlannerState":
        return replace(self, z=self.z.detach(), x=self.x.detach())


class LatentPlanner(nn.Module):
    def __init__(self, config: PlannerConfig):
        super().__init__()
        if config.mode == "factorized":
            raise ValueError("use FactorizedPlanner (build_planner) for the factorized mode")
        self.config = config
        c, k = config.channels, config.key_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(3 if config.age_input else 2, c, 3, padding=1), nn.GELU(), nn.Conv2d(c, c, 3, padding=1))
        inputs = (3 if config.goal_read else 2) * c
        self.initial = nn.Conv2d(c, c, 1)
        self.update_in = nn.Conv2d(inputs, c, 3, padding=1)
        self.update_out = nn.Conv2d(c, 2 * c, 3, padding=1)
        self.action_head = nn.Sequential(
            nn.Conv2d(inputs, c, 3, padding=1), nn.GELU(), nn.Conv2d(c, 5, 1))
        self.distance_head = nn.Sequential(nn.Conv2d(c, c // 2, 1), nn.GELU(), nn.Conv2d(c // 2, 1, 1))
        if config.change_gate:
            self.change_in = nn.Conv2d(5 * c, c, 1)
            self.change_out = nn.Conv2d(c, 2 * c, 1)
        if config.mode in LEARNED_READERS:
            self.query = nn.Conv2d(c, k, 1)
            self.key = nn.Conv2d(c, k, 1)
            self.goal_query = nn.Sequential(nn.Linear(2, k), nn.GELU(), nn.Linear(k, k))
            self.goal_key = nn.Sequential(nn.Linear(2, k), nn.GELU(), nn.Linear(k, k))
            self.gate = nn.Sequential(nn.Conv2d(4 * c, c, 1), nn.GELU(), nn.Conv2d(c, 3, 1))

    # ------------------------------------------------------------ core maps
    def encode(self, blocked: Tensor, goal: Tensor, age: Tensor | None = None) -> Tensor:
        b, h, w = blocked.shape
        planes = [blocked.float(), F.one_hot(goal, h * w).view(b, h, w).float()]
        if self.config.age_input:   # no age given: every cell is current
            planes.append(torch.zeros_like(planes[0]) if age is None else age.float())
        return self.encoder(torch.stack(planes, 1).to(self.initial.weight.dtype))

    def initialise(self, x: Tensor) -> Tensor:
        return torch.tanh(self.initial(x))

    def _inputs(self, z: Tensor, x: Tensor, goal: Tensor | None) -> Tensor:
        if not self.config.goal_read:
            return torch.cat((z, x), 1)
        b, c, h, w = z.shape
        at_goal = z.flatten(2).gather(2, goal.view(b, 1, 1).expand(b, c, 1)).view(b, c, 1, 1).expand(b, c, h, w)
        return torch.cat((z, x, at_goal), 1)

    def refine(self, z: Tensor, x: Tensor, goal: Tensor | None = None) -> Tensor:
        hidden = F.gelu(self.update_in(self._inputs(z, x, goal)))
        gate, candidate = self.update_out(hidden).chunk(2, 1)
        gate = torch.sigmoid(gate)
        return (1 - gate) * z + gate * torch.tanh(candidate)

    def readout(self, z: Tensor, x: Tensor, goal: Tensor | None = None) -> tuple[Tensor, Tensor]:
        """Action logits [B, 5, H, W] and normalised distance [B, H, W] for every cell."""
        return self.action_head(self._inputs(z, x, goal)), self.distance_head(z).squeeze(1)

    def readout_state(self, state: "PlannerState") -> tuple[Tensor, Tensor]:
        return self.readout(state.z, state.x, state.goal)

    # ------------------------------------------------------------ episode state
    def start(self, blocked: Tensor, goal: Tensor, age: Tensor | None = None) -> PlannerState:
        """Initial plan; every mode receives the same warm-up refinements."""
        x = self.encode(blocked, goal, age)
        z = self.initialise(x)
        for _ in range(self.config.warmup):
            z = self.refine(z, x, goal)
        b, c, h, w = z.shape
        m = self.config.memory_size
        dev = z.device
        return PlannerState(
            z=z, x=x, blocked=blocked, goal=goal,
            mem_z=z.new_zeros(b, m, c, h, w), mem_x=z.new_zeros(b, m, c, h, w),
            mem_blocked=torch.zeros(b, m, h, w, dtype=torch.bool, device=dev),
            mem_goal=torch.zeros(b, m, dtype=torch.long, device=dev),
            mem_tick=torch.full((b, m), -1, dtype=torch.long, device=dev),
            mem_valid=torch.zeros(b, m, dtype=torch.bool, device=dev),
            cursor=torch.zeros(b, dtype=torch.long, device=dev), tick=0)

    def observe(self, state: PlannerState, blocked: Tensor, goal: Tensor, age: Tensor | None = None) -> PlannerState:
        """Incorporate a new observation; continue, retrieve or restart computation."""
        x = self.encode(blocked, goal, age)
        changed = (blocked != state.blocked).flatten(1).any(1) | (goal != state.goal)
        mode = self.config.mode
        gates = reads = None
        if mode == "carry":
            z = state.z
            if self.config.change_gate:
                rebuilt, gates = self._change(state, x, goal)
                z = torch.where(changed[:, None, None, None], rebuilt, z)
        elif mode == "restart":
            z = torch.where(changed[:, None, None, None], self.initialise(x), state.z)
        elif mode == "goal_restart":
            z = torch.where((goal != state.goal)[:, None, None, None], self.initialise(x), state.z)
        elif mode in ("snapshot", "nearest"):
            z = torch.where(changed[:, None, None, None], self._lookup(state, blocked, goal), state.z)
        else:
            read, gates, reads = self._learned_read(state, x, goal)
            z = torch.where(changed[:, None, None, None], read, state.z)
        if mode not in ("carry", "restart", "goal_restart"):
            state = self._write(state, changed)
        return replace(state, z=z, x=x, blocked=blocked, goal=goal, tick=state.tick + 1,
                       gates=gates, reads=reads)

    def think(self, state: PlannerState, steps: int) -> tuple[PlannerState, list[tuple[Tensor, Tensor]]]:
        """Run ``steps`` shared refinements; return the readout after each one."""
        z, outputs = state.z, []
        for _ in range(steps):
            z = self.refine(z, state.x, state.goal)
            outputs.append(self.readout(z, state.x, state.goal))
        return replace(state, z=z), outputs

    def advance(self, state: PlannerState, steps: int) -> PlannerState:
        """Run ``steps`` shared refinements without intermediate readouts."""
        z = state.z
        for _ in range(steps):
            z = self.refine(z, state.x, state.goal)
        return replace(state, z=z)

    def _change(self, state: PlannerState, x: Tensor, goal: Tensor) -> tuple[Tensor, Tensor]:
        """Keep or rebuild every channel of every cell from the change of input and the goal reads."""
        z = state.z
        b, c, h, w = z.shape

        def read(cell):
            return z.flatten(2).gather(2, cell.view(b, 1, 1).expand(b, c, 1)).view(b, c, 1, 1).expand(b, c, h, w)

        hidden = F.gelu(self.change_in(torch.cat((z, x, x - state.x, read(goal), read(state.goal)), 1)))
        rebuild, candidate = self.change_out(hidden).chunk(2, 1)
        rebuild = torch.sigmoid(rebuild)
        return (1 - rebuild) * z + rebuild * torch.tanh(candidate), rebuild

    # ------------------------------------------------------------ memory
    def _write(self, state: PlannerState, changed: Tensor) -> PlannerState:
        """Store the computation of the context that just ended."""
        m = self.config.memory_size
        slot = F.one_hot(state.cursor, m).bool() & changed[:, None]   # [B, M]
        s5 = slot[:, :, None, None, None]
        return replace(
            state,
            mem_z=torch.where(s5, state.z.detach()[:, None], state.mem_z),
            mem_x=torch.where(s5, state.x.detach()[:, None], state.mem_x),
            mem_blocked=torch.where(slot[:, :, None, None], state.blocked[:, None], state.mem_blocked),
            mem_goal=torch.where(slot, state.goal[:, None], state.mem_goal),
            mem_tick=torch.where(slot, torch.full_like(state.mem_tick, state.tick), state.mem_tick),
            mem_valid=state.mem_valid | slot,
            cursor=(state.cursor + changed.long()) % m)

    def _lookup(self, state: PlannerState, blocked: Tensor, goal: Tensor) -> Tensor:
        """Exact (snapshot) or nearest-map (nearest) retrieval among same-goal records."""
        same_goal = (state.mem_goal == goal[:, None]) & state.mem_valid
        distance = (state.mem_blocked != blocked[:, None]).flatten(2).sum(-1)       # [B, M]
        recency = state.mem_tick.float() / (state.tick + 1)
        if self.config.mode == "snapshot":
            eligible = same_goal & (distance == 0)
            score = torch.where(eligible, recency, torch.full_like(recency, -1.0))
            best = score.argmax(1)
            found = eligible.any(1)
            stored = state.mem_z[torch.arange(len(best)), best]
            return torch.where(found[:, None, None, None], stored, state.z)
        # nearest: the carried workspace competes with stored ones
        carried_distance = (state.blocked != blocked).flatten(1).sum(-1)
        carried_ok = state.goal == goal
        big = blocked[0].numel() + 1.0
        score = torch.cat((torch.where(carried_ok, carried_distance.float() - 0.5, torch.full_like(recency[:, 0], big)
                                       )[:, None],
                           torch.where(same_goal, distance.float() - 0.1 * recency, torch.full_like(recency, big))), 1)
        best = score.argmin(1)
        candidates = torch.cat((state.z[:, None], state.mem_z), 1)
        chosen = candidates[torch.arange(len(best)), best]
        return torch.where((score.min(1).values < big)[:, None, None, None], chosen, state.z)

    def _learned_read(self, state: PlannerState, x: Tensor, goal: Tensor):
        b, c, h, w = x.shape
        k = self.config.key_dim
        cand_z = torch.cat((state.z[:, None], state.mem_z), 1)            # [B, 1+M, C, H, W]
        cand_x = torch.cat((state.x[:, None], state.mem_x), 1)
        cand_goal = torch.cat((state.goal[:, None], state.mem_goal), 1)  # [B, 1+M]
        valid = torch.cat((torch.ones_like(state.mem_valid[:, :1]), state.mem_valid), 1)
        n = cand_z.shape[1]

        def coords(cell):
            return torch.stack(((cell // w).float() / (h - 1), (cell % w).float() / (w - 1)), -1).to(x.dtype)

        goal_score = (self.goal_query(coords(goal))[:, None] * self.goal_key(coords(cand_goal))).sum(-1) / k ** .5
        keys = self.key(cand_x.flatten(0, 1)).view(b, n, k, h, w)
        query = self.query(x)
        if self.config.mode == "memory_local":
            score = (query[:, None] * keys).sum(2) / k ** .5 + goal_score[..., None, None]   # [B, N, H, W]
            score = score.masked_fill(~valid[..., None, None], float("-inf"))
            weight = torch.softmax(score.float(), 1).to(x.dtype)
            retrieved = (weight[:, :, None] * cand_z).sum(1)
            reads = weight.mean((-1, -2))
        else:
            score = (query.mean((-1, -2))[:, None] * keys.mean((-1, -2))).sum(-1) / k ** .5 + goal_score
            score = score.masked_fill(~valid, float("-inf"))
            weight = torch.softmax(score.float(), 1).to(x.dtype)
            retrieved = (weight[:, :, None, None, None] * cand_z).sum(1)
            reads = weight
        gate = torch.softmax(self.gate(torch.cat((state.z, retrieved, x, x - state.x), 1)).float(), 1).to(x.dtype)
        z = gate[:, :1] * state.z + gate[:, 1:2] * retrieved + gate[:, 2:] * self.initialise(x)
        return z, gate, reads


class FactorizedPlanner(nn.Module):
    """Computation organised by its premises.

    ``m`` depends only on the map and is refined from the map alone, so a goal
    change leaves it valid. ``g`` holds goal-specific computation; when the goal
    moves it is reseeded from ``m`` at every cell together with ``m`` at the new
    goal cell, then refined. With 60 channels per workspace one refinement costs
    within 2% of the 96-channel single-workspace planner.
    """

    def __init__(self, config: PlannerConfig):
        super().__init__()
        self.config = config
        c = config.channels
        self.encoder = nn.Sequential(nn.Conv2d(2 if config.age_input else 1, c, 3, padding=1), nn.GELU(),
                                     nn.Conv2d(c, c, 3, padding=1))
        self.map_initial = nn.Conv2d(c, c, 1)
        self.map_in = nn.Conv2d(2 * c, c, 3, padding=1)
        self.map_out = nn.Conv2d(c, 2 * c, 3, padding=1)
        self.seed = nn.Sequential(nn.Conv2d(3 * c + 1, c, 1), nn.GELU(), nn.Conv2d(c, c, 1))
        self.goal_in = nn.Conv2d(4 * c + 1, c, 3, padding=1)
        self.goal_out = nn.Conv2d(c, 2 * c, 3, padding=1)
        self.action_head = nn.Sequential(nn.Conv2d(3 * c, c, 3, padding=1), nn.GELU(), nn.Conv2d(c, 5, 1))
        self.distance_head = nn.Sequential(nn.Conv2d(c, c // 2, 1), nn.GELU(), nn.Conv2d(c // 2, 1, 1))

    @staticmethod
    def _gated(z, hidden, out):
        gate, candidate = out(F.gelu(hidden)).chunk(2, 1)
        gate = torch.sigmoid(gate)
        return (1 - gate) * z + gate * torch.tanh(candidate)

    def _goal_inputs(self, m, goal):
        b, c, h, w = m.shape
        goal_map = F.one_hot(goal, h * w).view(b, 1, h, w).to(m.dtype)
        if not self.config.goal_broadcast:  # goal information can only spread locally
            return goal_map, torch.zeros_like(m)
        at_goal = m.flatten(2).gather(2, goal.view(b, 1, 1).expand(b, c, 1)).view(b, c, 1, 1).expand(b, c, h, w)
        return goal_map, at_goal

    def encode(self, blocked: Tensor, age: Tensor | None = None) -> Tensor:
        planes = blocked.float()[:, None]
        if self.config.age_input:   # no age given: every cell is current
            planes = torch.cat((planes, torch.zeros_like(planes) if age is None else age.float()[:, None]), 1)
        return self.encoder(planes.to(self.map_initial.weight.dtype))

    def _split(self, z):
        c = self.config.channels
        return z[:, :c], z[:, c:]

    def _seed(self, m, x, goal):
        goal_map, at_goal = self._goal_inputs(m, goal)
        return torch.tanh(self.seed(torch.cat((m, at_goal, x, goal_map), 1)))

    def _refine(self, z, x, goal):
        m, g = self._split(z)
        m = self._gated(m, self.map_in(torch.cat((m, x), 1)), self.map_out)
        goal_map, at_goal = self._goal_inputs(m, goal)
        g = self._gated(g, self.goal_in(torch.cat((g, m, x, at_goal, goal_map), 1)), self.goal_out)
        return torch.cat((m, g), 1)

    def readout_state(self, state: PlannerState) -> tuple[Tensor, Tensor]:
        m, g = self._split(state.z)
        _, at_goal = self._goal_inputs(m, state.goal)
        return self.action_head(torch.cat((g, m, at_goal), 1)), self.distance_head(g).squeeze(1)

    def start(self, blocked: Tensor, goal: Tensor, age: Tensor | None = None) -> PlannerState:
        x = self.encode(blocked, age)
        m = torch.tanh(self.map_initial(x))
        z = torch.cat((m, self._seed(m, x, goal)), 1)
        for _ in range(self.config.warmup):
            z = self._refine(z, x, goal)
        b, _, h, w = z.shape
        c, n = self.config.channels, self.config.map_memory
        dev = z.device
        return PlannerState(z=z, x=x, blocked=blocked, goal=goal, mem_z=z.new_zeros(b, n, c, h, w),
                            mem_x=z.new_zeros(b, 0, 1, h, w),
                            mem_blocked=torch.zeros(b, n, h, w, dtype=torch.bool, device=dev),
                            mem_goal=torch.zeros(b, n, dtype=torch.long, device=dev),
                            mem_tick=torch.full((b, n), -1, dtype=torch.long, device=dev),
                            mem_valid=torch.zeros(b, n, dtype=torch.bool, device=dev),
                            cursor=torch.zeros(b, dtype=torch.long, device=dev), tick=0)

    def observe(self, state: PlannerState, blocked: Tensor, goal: Tensor, age: Tensor | None = None) -> PlannerState:
        x = self.encode(blocked, age)
        m, g = self._split(state.z)
        rebuild = goal != state.goal
        if self.config.map_memory:
            remapped = (blocked != state.blocked).flatten(1).any(1)
            same = (state.mem_blocked == blocked[:, None]).flatten(2).all(2) & state.mem_valid      # [B, N]
            best = torch.where(same, state.mem_tick, torch.full_like(state.mem_tick, -1)).argmax(1)
            found = same.any(1) & remapped
            stored = state.mem_z[torch.arange(len(best), device=best.device), best]
            state = self._write_map(state, m, remapped)
            m = torch.where(found[:, None, None, None], stored, m)
            rebuild = rebuild | found   # the goal plan was built on another map
        if self.config.reseed:
            g = torch.where(rebuild[:, None, None, None], self._seed(m, x, goal), g)
        return replace(state, z=torch.cat((m, g), 1), x=x, blocked=blocked, goal=goal, tick=state.tick + 1)

    def _write_map(self, state: PlannerState, m: Tensor, remapped: Tensor) -> PlannerState:
        """Store the map workspace of the map that just ended."""
        slot = F.one_hot(state.cursor, self.config.map_memory).bool() & remapped[:, None]
        return replace(
            state,
            mem_z=torch.where(slot[:, :, None, None, None], m.detach()[:, None], state.mem_z),
            mem_blocked=torch.where(slot[:, :, None, None], state.blocked[:, None], state.mem_blocked),
            mem_tick=torch.where(slot, torch.full_like(state.mem_tick, state.tick), state.mem_tick),
            mem_valid=state.mem_valid | slot,
            cursor=(state.cursor + remapped.long()) % self.config.map_memory)

    def think(self, state: PlannerState, steps: int):
        z, outputs = state.z, []
        for _ in range(steps):
            z = self._refine(z, state.x, state.goal)
            state = replace(state, z=z)
            outputs.append(self.readout_state(state))
        return state, outputs

    def advance(self, state: PlannerState, steps: int) -> PlannerState:
        z = state.z
        for _ in range(steps):
            z = self._refine(z, state.x, state.goal)
        return replace(state, z=z)


def build_planner(config: PlannerConfig) -> nn.Module:
    return FactorizedPlanner(config) if config.mode == "factorized" else LatentPlanner(config)
