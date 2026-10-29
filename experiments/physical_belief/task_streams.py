"""Matched task streams combining observed settings and generated practice.

Native rows retain their recorded working settings. Generated rows use the
existing two-support calibration task. A whole lifetime uses one source type,
so an A→B→A sequence retains its meaning within that type. Neither source type
nor scientific-law identity is part of the actor's observation.
"""
from __future__ import annotations

import numpy as np
import torch

from experiments.discoveryworld.train_active_lifetime import ActiveLifetimeEnvironment, TaskStreamPlan


# Reactor Lab Normal's environment identity; used to schedule tasks only.
NATIVE_QUANTITIES = np.asarray((0, 1, 3, 2, 8), dtype=np.int64)


class MixedTaskStreamPlan(TaskStreamPlan):
    """Choose native or generated experience once per A→B→A lifetime.

    ``source_path`` is the same NPZ passed to CalibrationTaskPool. Its seed
    metadata identifies the native source pools; it never enters the actor.
    World selection and object permutations are fixed independently of actions.
    """

    def __init__(self, pool, *, source_path, streams, lifetimes, seed, device,
                 native_fraction=.5, native_quantities=(0, 1, 2, 3)):
        if not 0 <= native_fraction <= 1:
            raise ValueError("native_fraction must be between zero and one")
        super().__init__(pool, streams=streams, lifetimes=lifetimes, seed=seed, device=device)
        with np.load(source_path) as source:
            scenario_seed = source["scenario_seed"]
        if scenario_seed.shape != (pool.size,):
            raise ValueError("The native source metadata must match the calibration pool")
        families = NATIVE_QUANTITIES[scenario_seed % len(NATIVE_QUANTITIES)]
        self.native_quantities = tuple(int(q) for q in native_quantities)
        if len(set(self.native_quantities)) < 2:
            raise ValueError("Native changing-task streams need at least two relation families")
        native_pools = {q: np.flatnonzero(families == q) for q in self.native_quantities}
        if any(len(indices) < 11 for indices in native_pools.values()):
            raise ValueError("Each requested native relation needs eleven distinct source worlds")
        # A separate reproducible stream leaves the generated plans unchanged.
        rng = np.random.default_rng(np.random.SeedSequence([seed, 17431]))
        native = rng.random((streams, lifetimes)) < native_fraction
        laws = np.asarray(self.native_quantities)
        first_index = rng.integers(len(laws), size=native.shape)
        middle_index = (first_index + rng.integers(1, len(laws), size=native.shape)) % len(laws)
        phase = self.phase.cpu().numpy()
        native_laws = np.where(phase == 1, laws[middle_index][..., None], laws[first_index][..., None])
        quantities = self.quantities.cpu().numpy().copy()
        worlds = self.worlds.cpu().numpy().copy()
        quantities[native] = native_laws[native]
        for position in range(worlds.shape[-1]):
            for quantity, indices in native_pools.items():
                selected = native & (native_laws[..., position] == quantity)
                count = int(selected.sum())
                if not count:
                    continue
                proposed = rng.choice(indices, size=count)
                if position:
                    previous = worlds[..., :position][selected]
                    repeated = (proposed[:, None] == previous).any(-1)
                    while repeated.any():
                        proposed[repeated] = rng.choice(indices, size=int(repeated.sum()))
                        repeated = (proposed[:, None] == previous).any(-1)
                worlds[..., position][selected] = proposed
        self.worlds = torch.as_tensor(worlds, device=device)
        self.quantities = torch.as_tensor(quantities, device=device)
        self.native = torch.as_tensor(native, device=device)
        self.native_fraction = float(native_fraction)
        self.native_world_counts = {str(q): len(indices) for q, indices in native_pools.items()}

    def description(self):
        description = super().description()
        description.update(
            source_type="selected once per complete lifetime",
            native_fraction_requested=self.native_fraction,
            native_lifetimes=int(self.native.sum()),
            generated_lifetimes=int((~self.native).sum()),
            native_quantities=list(self.native_quantities),
            native_world_counts=self.native_world_counts,
            native_labels="recorded public working settings, unchanged",
            generated_labels="alternative affine queries through recorded public supports",
            relation_sampling="native source families or nine generated quantities; B differs from A",
        )
        return description


class PhysicalTaskEnvironment(ActiveLifetimeEnvironment):
    """Reuse the measured-action environment, replacing only native labels."""

    @torch.no_grad()
    def _begin_tasks(self, rows, initial=False):
        super()._begin_tasks(rows, initial=initial)
        selected = self.plan.native[rows, self.lifetime[rows]]
        native_rows = rows[selected]
        if not len(native_rows):
            return
        indices = self.table.indices[native_rows]
        permutation = self.table.permutation[native_rows]
        targets = self.pool.data["working_frequency_hz"][indices].gather(1, permutation)
        self.table.targets[native_rows] = targets
        self.table.known_hz[native_rows] = torch.where(self.table.known[native_rows], targets, 0.)
