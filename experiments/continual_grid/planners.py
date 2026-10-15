"""Equal-backup computational-memory baselines; no learned parameters."""
from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
import numpy as np

from grid import Context, action_values, geometric_values, transition_table

METHODS = ("restart", "carry", "snapshot", "nearest", "factorized")


@dataclass
class Record:
    blocked: np.ndarray
    goal: int
    values: np.ndarray
    backups: int
    tick: int


class Planner:
    def __init__(self, method: str, side: int = 9, budget_fraction: float = 1.,
                 capacity: int = 32, local_radius: int = 2):
        if method not in METHODS:
            raise ValueError(method)
        self.method, self.side, self.capacity = method, side, capacity
        self.cells_per_tick = max(1, int(round(side * side * budget_fraction)))
        self.values = None
        self.previous = None
        self.records = []
        self.context_backups = 0
        self.cursor = 0
        self.total_backups = 0
        self.read_seconds = self.write_seconds = self.backup_seconds = 0.
        self.context_changes = self.retrievals = 0
        self.exact_reads = self.local_reads = 0
        states = np.arange(side * side)
        distance = abs(states[:, None] // side - states[None, :] // side) + abs(states[:, None] % side - states[None, :] % side)
        self.local_scope = distance <= local_radius

    @staticmethod
    def same_context(a: Context, b: Context) -> bool:
        return a.goal == b.goal and np.array_equal(a.blocked, b.blocked)

    def _read(self, context: Context) -> np.ndarray:
        if self.values is None or self.previous.goal != context.goal or self.method == "restart":
            base = geometric_values(context.goal, self.side)
        else:
            base = self.values.copy()
        records = [record for record in self.records if record.goal == context.goal]
        if self.method in {"restart", "carry"} or not records:
            return base
        masks = np.stack([record.blocked for record in records])
        mismatch = masks != context.blocked[None, :]
        total_distance = mismatch.sum(axis=1)
        exact = np.flatnonzero(total_distance == 0)
        if len(exact):
            selected = max(exact, key=lambda i: records[i].tick)
            self.retrievals += 1
            self.exact_reads += 1
            return records[selected].values.copy()
        if self.method == "snapshot":
            return base
        if self.method == "nearest":
            selected = min(range(len(records)), key=lambda i: (total_distance[i], -records[i].tick))
            self.retrievals += 1
            return records[selected].values.copy()
        # Factorization is a testable warm-start heuristic: local transition
        # similarity does not imply equality of long-range planning values.
        local_distance = mismatch.astype(np.int16) @ self.local_scope.T.astype(np.int16)
        times = np.asarray([record.tick for record in records])
        for cell in range(self.side * self.side):
            matches = np.flatnonzero(local_distance[:, cell] == 0)
            if len(matches):
                best = min(matches, key=lambda i: (total_distance[i], -times[i]))
                base[cell] = records[best].values[cell]
                self.local_reads += 1
        self.retrievals += 1
        return base

    def _write(self, context: Context) -> None:
        if self.method in {"restart", "carry"}:
            return
        record = Record(context.blocked.copy(), context.goal, self.values.copy(),
                        self.context_backups, context.tick)
        for i, previous in enumerate(self.records):
            if previous.goal == context.goal and np.array_equal(previous.blocked, context.blocked):
                self.records[i] = record
                return
        self.records.append(record)
        if len(self.records) > self.capacity:
            self.records.pop(0)

    def compute(self, context: Context) -> np.ndarray:
        started = perf_counter()
        changed = self.previous is None or not self.same_context(self.previous, context)
        if changed:
            self.values = self._read(context)
            self.context_backups = 0
            self.context_changes += 1
        self.read_seconds += perf_counter() - started
        started = perf_counter()
        nxt, legal = transition_table(context.blocked, self.side)
        cells = (np.arange(self.cells_per_tick) + self.cursor) % (self.side * self.side)
        # Compute only the allocated rows: neither unselected states nor the
        # reference planner contribute a Bellman backup to the model workspace.
        destinations = nxt[cells]
        q = np.where(destinations == context.goal, 1., -.02 + .97 * self.values[destinations])
        q[~legal[cells]] = -np.inf
        updated = q.max(axis=1)
        updated[cells == context.goal] = 0.
        self.values[cells] = updated
        self.cursor = (self.cursor + self.cells_per_tick) % (self.side * self.side)
        self.context_backups += len(cells)
        self.total_backups += len(cells)
        self.backup_seconds += perf_counter() - started
        started = perf_counter()
        self._write(context)
        self.write_seconds += perf_counter() - started
        self.previous = context
        return self.values
