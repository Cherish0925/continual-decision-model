"""Draw examples across recorded streams, including unequal survey batches."""
from __future__ import annotations

import numpy as np


def sample_rows(sizes, batch_size, rng):
    """Return (record index, row indices) sampled uniformly without replacement.

    ``sizes`` gives the number of examples in each selected replay record.
    The caller applies each row selection to every field of that record so
    observations, ages, goals, and targets remain paired. The input pool must
    contain at least one example; an oversized batch returns the whole pool.
    """
    ends = np.cumsum(sizes, dtype=np.int64)
    flat = rng.choice(int(ends[-1]), size=min(batch_size, int(ends[-1])), replace=False)
    record = np.searchsorted(ends, flat, side="right")
    starts = np.r_[0, ends[:-1]]
    return [(int(i), flat[record == i] - starts[i]) for i in np.unique(record)]
