"""A direct recipe-reuse reference without learned context matching."""
import torch


def previous_success(state, candidates, action_mask):
    """Try a recent successful recipe that has not failed in the current task.

    Uses the same six public ledgers as the learned model. If no retained
    success is still a legal candidate, use the separately trained current-task
    policy. No response formula or hidden context identity is available.
    """
    if len(candidates) != 1:
        raise ValueError('Native retrieval executes one apparatus at a time')
    n = state['archive_mask'].shape[1]
    cursor = int(state['archive_cursor'][0])
    proportions = candidates[0, :, 8:16]
    components = candidates[0, 0, 16:24].bool()
    for age in range(min(cursor, n)):
        slot = (cursor - 1 - age) % n
        if not torch.equal(state['archive_components'][0, slot], components):
            continue
        history = state['archive_history'][0, slot]
        successful = state['archive_mask'][0, slot] & (history[:, 17] == 0)
        for row in successful.nonzero().flatten().flip(0):
            q = history[row, :8]
            q = q / q.sum().clamp_min(1e-8)
            matching = (proportions - q).abs().max(-1).values < 1e-6
            eligible = matching & action_mask[0]
            if eligible.any():
                return int(eligible.nonzero()[0])
    return None
