"""Task streams made from recorded instrument readings and public settings.

This is an offline prediction protocol. All selected instrument returns are
revealed before a setting prediction. A target's working setting is revealed
only after its prediction has been scored. Native action cost is evaluated by
the separate simulator runner, not inferred from this task.
"""
from __future__ import annotations

import numpy as np
import torch


# Used only by the environment to arrange changes and returns of a relation.
# Neither these indices nor seeds are returned in a model observation.
NATIVE_QUANTITY = (0, 1, 3, 2, 8)


class RecordedWorlds:
    def __init__(self, path, device="cpu", radiation_readings=32):
        data = np.load(path)
        self.device = torch.device(device)
        self.values = torch.as_tensor(data["x"], device=device, dtype=torch.float64)
        self.seen = torch.as_tensor(data["reading_present"], device=device, dtype=torch.bool)
        # An explicit batch of actually recorded radiation returns. Other
        # quantities use their first real return; no hidden simulator value.
        radiation = data["sample_values"][:, :, 2, :radiation_readings, 2]
        present = data["sample_present"][:, :, 2, :radiation_readings, 2]
        minimum = np.where(present, radiation, np.inf).min(-1)
        self.values[:, :, 2] = torch.as_tensor(np.where(np.isfinite(minimum), minimum, 0.), device=device)
        self.seen[:, :, 2] = torch.as_tensor(np.isfinite(minimum), device=device)
        self.settings = torch.as_tensor(data["working_frequency_hz"], device=device, dtype=torch.float64)
        self.support = torch.as_tensor(data["initial_support_mask"], device=device, dtype=torch.bool)
        self.targets = torch.as_tensor(data["target_mask"] & data["label_present"], device=device, dtype=torch.bool)
        self.labels = torch.as_tensor(data["label_present"], device=device, dtype=torch.bool)
        self.seeds = torch.as_tensor(data["scenario_seed"], device=device)
        self.family = torch.as_tensor(NATIVE_QUANTITY, device=device)[self.seeds % 5]
        self.size, self.objects, self.quantities = self.values.shape
        self.native_pools = {int(q): (self.family == q).nonzero().flatten()
                             for q in self.family.unique().tolist()}
        self._prepare_rehearsal()
        self.radiation_readings = radiation_readings

    def _prepare_rehearsal(self):
        rows = torch.arange(self.size, device=self.device)
        support_indices = self.support.long().argsort(-1, descending=True)[:, :2]
        x0, x1 = (self.values[rows, support_indices[:, j]] for j in range(2))
        y0, y1 = (self.settings[rows, support_indices[:, j]] for j in range(2))
        delta = x1 - x0
        scale = torch.maximum(x0.abs(), x1.abs()).clamp_min(1.)
        good = delta.abs() > 1e-6 * scale
        slope = (y1 - y0)[:, None] / torch.where(good, delta, torch.ones_like(delta))
        intercept = y0[:, None] - slope * x0
        generated = slope[:, None] * self.values + intercept[:, None]
        present = (self.seen | ~self.labels[..., None]).all(1)
        in_range = (((generated >= 0) & (generated <= 10000)) | ~self.targets[..., None]).all(1)
        self.generated_valid = good & present & in_range
        self.generated_settings = torch.where(self.support[..., None], self.settings[..., None], generated)
        self.generated_pools = {q: self.generated_valid[:, q].nonzero().flatten()
                                for q in range(self.quantities)}

    def _indices(self, quantity, generated, generator):
        result = torch.empty_like(quantity)
        for is_generated, pools in ((False, self.native_pools), (True, self.generated_pools)):
            for q, pool in pools.items():
                selected = (quantity == q) & (generated == is_generated)
                count = int(selected.sum())
                if count:
                    result[selected] = pool[torch.randint(len(pool), (count,), device=self.device, generator=generator)]
        return result

    def sample_stream(self, batch, *, generator, segment_length=3, generated_fraction=.5,
                      kind="mixed", wide_scale=(.35, 2.0)):
        """Sample A→B→A; each episode has a newly calibrated physical instance.

        `rehearsal` keeps measured quantities and support settings, replacing
        target settings with a chosen affine alternative. `wide` additionally
        transforms all settings by an episode-specific scale/offset; it is a
        controlled stress test, not an additional native benchmark.
        """
        if kind not in ("mixed", "native", "rehearsal", "wide"):
            raise ValueError(kind)
        length = 3 * segment_length
        generated = torch.rand(batch, device=self.device, generator=generator) < generated_fraction
        if kind == "native": generated[:] = False
        if kind in ("rehearsal", "wide"): generated[:] = True
        native = torch.tensor(sorted(self.native_pools), device=self.device)
        a = torch.randint(self.quantities, (batch,), device=self.device, generator=generator)
        b = (a + torch.randint(1, self.quantities, (batch,), device=self.device, generator=generator)) % self.quantities
        ia = torch.randint(len(native), (batch,), device=self.device, generator=generator)
        ib = (ia + torch.randint(1, len(native), (batch,), device=self.device, generator=generator)) % len(native)
        a = torch.where(generated, a, native[ia])
        b = torch.where(generated, b, native[ib])
        quantities = torch.stack([a] * segment_length + [b] * segment_length + [a] * segment_length, 1)
        generated_mask = generated[:, None].expand(-1, length)
        indices = self._indices(quantities.flatten(), generated_mask.flatten(), generator).reshape(batch, length)
        values, seen = self.values[indices], self.seen[indices]
        settings = self.settings[indices]
        alt = self.generated_settings[indices].gather(-1, quantities[:, :, None, None].expand(-1, -1, self.objects, 1))[..., 0]
        settings = torch.where(generated_mask[..., None], alt, settings)
        if kind == "wide":
            # Frequency units stay Hz. The transformation changes local
            # coefficients while retaining the selected relation and readings.
            lo, hi = wide_scale
            scale = lo + (hi - lo) * torch.rand(batch, length, 1, device=self.device, generator=generator)
            offset = 2000 * (torch.rand(batch, length, 1, device=self.device, generator=generator) - .5)
            settings = settings * scale + offset
        return {"values": values, "seen": seen, "settings": settings,
                "support": self.support[indices], "targets": self.targets[indices],
                "labels": self.labels[indices], "indices": indices,
                "quantities": quantities, "generated": generated_mask,
                "seeds": self.seeds[indices], "segment_length": segment_length, "kind": kind}


def episode(stream, index):
    """Return public arrays only; identity and schedule metadata stay outside."""
    return {key: stream[key][:, index] for key in ("values", "seen", "settings", "support", "targets", "labels")}


def target_order(targets):
    # Objects keep their original source identity. No target value determines
    # this order, which is the same for all compared methods.
    positions = torch.arange(targets.shape[-1], device=targets.device)[None].expand_as(targets)
    return torch.where(targets, positions, targets.shape[-1]).argsort(-1)[:, :2]
