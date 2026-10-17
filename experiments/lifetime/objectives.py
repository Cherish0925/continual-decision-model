"""Training objectives that follow what the deployed agent did inside a training window.

``recompute_states``      fast state and workspace of a window recomputed by the current network; the workspace
                          is unrolled over segments of external events, so its gradients span them
``deployed_adapters``     the adapter each stream held at every tick, by replaying the learning steps it took,
                          each on the experience its buffer held when the step was chosen
``window_support``        that experience, rebuilt from the buffer at the start of the window and the window
``predictions_per_tick``  predictions for every action under the adapter in effect at each tick
``online_nll``            loss of the prediction for the action taken at every tick, under the adapter in effect
"""
from __future__ import annotations

from typing import Callable

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.func import functional_call
from torch.utils.checkpoint import checkpoint

from cdm.core import ContinualCore, adapter_update, reading_nll

NEVER = 10 ** 9                       # tick of a precise reading that has not returned


def recompute_states(model: ContinualCore, h_prev, z_prev, event, records, counts, dones, segment: int = 1):
    """Recompute a PPO chunk, resetting each row at its collected lifetime boundaries.

    Inputs have leading dimensions [time, streams]. A done at t resets the state read at t+1. Fast-state
    gradients span the chunk except across a reset. The workspace read at each tick is recomputed from the
    stored workspace at the start of its segment of ``segment`` ticks, one tick of refinements after another,
    so its gradients span up to ``segment`` external events; with ``segment=1`` they stop after one tick.
    """
    length, streams = event.shape[:2]
    reset_times = (dones[:-1].bool().any(1).nonzero().flatten() + 1).tolist()
    boundaries = [0, *reset_times, length]
    h, parts = h_prev[0], []
    for begin, end in zip(boundaries[:-1], boundaries[1:]):
        if begin:
            h = torch.where(dones[begin - 1].bool().unsqueeze(-1), h_prev[begin], h)
        hs = model.fast_sequence(h, event[begin:end])
        parts.append(hs)
        h = hs[-1]
    H = torch.cat(parts, 0)
    zin = z_prev.float()
    reset_before = torch.cat((torch.zeros(1, streams, dtype=torch.bool, device=dones.device),
                              dones[:-1].bool()), 0)
    if segment <= 1:
        zout = checkpoint(model.refine, zin.flatten(0, 1), H.flatten(0, 1), records.flatten(0, 1),
                          counts.flatten(0, 1), use_reentrant=False).view_as(zin)
        zread = torch.cat((zin[:1], zout[:-1]), 0)
        return H, torch.where(reset_before[..., None, None], zin, zread)
    k, pad = segment, (-length) % segment

    def split(x):                     # [length, n, ...] -> [k, segments * n, ...]
        if pad:
            x = torch.cat((x, x[-1:].expand(pad, *x.shape[1:])), 0)
        return x.reshape(-1, k, *x.shape[1:]).transpose(0, 1).flatten(1, 2)

    Z, Hs, R, C, Rb = split(zin), split(H), split(records), split(counts), split(reset_before)
    z, out = Z[0], [Z[0]]
    for j in range(1, k):
        z = checkpoint(model.refine, z, Hs[j - 1], R[j - 1], C[j - 1], use_reentrant=False)
        z = torch.where(Rb[j][..., None, None], Z[j], z)
        out.append(z)
    zread = torch.stack(out).unflatten(1, (-1, streams)).transpose(0, 1).flatten(0, 1)
    return H, zread[:length]


def window_support(model: ContinualCore, start, states: Tensor, actions: Tensor, cheap: Tensor,
                   precise_value: Tensor, precise_at: Tensor, arrived: Tensor, precise: Tensor,
                   measured_back: Tensor) -> Callable:
    """``support(rows, ticks)``: the learning buffer of each of ``rows`` as it stood after its tick in ``ticks``.

    ``start`` is the agent before the window's first tick. states [L, n, 2w], actions and cheap [L, n] are the
    window's transitions; precise_value and precise_at [L, n] give each transition's precise reading and the tick
    at which it returned (``NEVER`` if it did not). arrived, precise and measured_back [L, n] are the readings
    that returned at each tick and how far back they measured, which places those of transitions before the
    window in the starting buffer. An entry carries the precise reading if it had returned by the step's tick.
    """
    T, L = start.learn_s.shape[1], states.shape[0]
    seq_s = torch.cat((start.learn_s, states.transpose(0, 1).to(start.learn_s.dtype)), 1)     # [n, T + L, 2w]
    seq_a = torch.cat((start.learn_a, actions.transpose(0, 1)), 1)
    early_at = torch.full_like(start.learn_a, NEVER)              # returns, inside the window, for earlier entries
    early_value = torch.zeros_like(start.learn_y)
    tick = torch.arange(L, device=arrived.device).unsqueeze(1)
    j, i = (arrived & (measured_back > tick)).nonzero(as_tuple=True)
    q = T + j - measured_back[j, i]
    early_at[i, q], early_value[i, q] = j, precise[j, i]

    def support(rows: Tensor, ticks: Tensor):
        pos = ticks.unsqueeze(1) + 1 + torch.arange(T, device=ticks.device)                   # [k, T]
        r, now = rows.unsqueeze(1), ticks.unsqueeze(1)
        inside = pos >= T
        j, old = (pos - T).clamp(min=0), pos.clamp(max=T - 1)
        known = torch.where(inside, precise_at[j, r] <= now, early_at[r, old] <= now)
        y = torch.where(inside, torch.where(known, precise_value[j, r], cheap[j, r]),
                        torch.where(known, early_value[r, old], start.learn_y[r, old]))
        p = torch.where(inside, known, start.learn_p[r, old] | known)
        held = pos >= (T - start.filled[rows]).unsqueeze(1)
        s = seq_s[r, pos]                                         # stored precision; float for exact replays
        x = torch.cat((s, model.action_features(seq_a[r, pos]).to(s.dtype)), -1)
        return (x if torch.is_autocast_enabled() else x.float()), y, p, held.float()

    return support


def deployed_adapters(model: ContinualCore, start, learn: Tensor, support: Callable, learn_ticks: int,
                      learning_rate: float, differentiable: bool):
    """Replay the learning steps of a window.

    ``start`` is the agent before the window's first tick, including a step still running then; learn [L, n].
    Returns version [L, n] and the adapters As, Bs: at tick t stream i predicted with As[version[t, i]][i]. A step
    chosen at tick s is in effect from tick s + ``learn_ticks``, as in ``runtime.tick``. With ``differentiable``
    the adapters carry the graph of every step.
    """
    L, n = learn.shape
    t = torch.arange(L, device=learn.device).unsqueeze(1)
    pending = start.busy > 0
    base_A = torch.where(pending.view(-1, 1, 1), start.next_A, start.A)
    base_B = torch.where(pending.view(-1, 1, 1), start.next_B, start.B)
    taken = learn.long().cumsum(0)                                  # steps chosen up to and including each tick
    in_effect = torch.zeros_like(taken)
    in_effect[learn_ticks:] = taken[:L - learn_ticks]
    version = torch.where(pending & (t < start.busy), torch.zeros_like(taken), 1 + in_effect)
    As, Bs = [start.A, base_A], [start.B, base_B]
    total = taken[-1]
    for r in range(1, int(total.max()) + 1 if L else 1):
        rows = (total >= r).nonzero().squeeze(1)
        ticks = ((taken[:, rows] == r) & learn[:, rows]).float().argmax(0)
        x, y, p, held = support(rows, ticks)
        A, B = As[-1], Bs[-1]
        newA, newB, _ = adapter_update(model, A[rows], B[rows], x, y, p, held, learning_rate, differentiable)
        As.append(A.index_copy(0, rows, newA.to(A.dtype)))
        Bs.append(B.index_copy(0, rows, newB.to(B.dtype)))
    return version, As, Bs


def _gather_versions(stacked: Tensor, version: Tensor) -> Tensor:
    """stacked [V, n, ...] -> [L, n, ...]: the entry of each stream's version at each tick."""
    return stacked[version, torch.arange(version.shape[1], device=version.device)]


def predictions_per_tick(model: ContinualCore, state: Tensor, As, Bs, version: Tensor, chunk: int = 16):
    """Mean and log-variance [L, n, actions] for every action from state features [L, n, 2w], under the adapter in
    effect at each tick. Gradients reach the state features only: the predictor and the adapters are held fixed,
    so the loss that reads these predictions shapes the state, not the predictor."""
    s = state.shape[-1]
    weights = {k: v.detach() for k, v in model.pred_out.named_parameters()}
    W, b = model.pred_in.weight.detach(), model.pred_in.bias.detach()
    emb = model.action_features().detach()                                        # [K, w]
    A_all = torch.stack([a.detach() for a in As])                                 # [V, n, h, r]
    B_all = torch.stack([b_.detach() for b_ in Bs])                               # [V, n, r, 3w]
    low_act = torch.einsum("vnrd,kd->vnkr", B_all[..., s:], emb)                  # [V, n, K, r]
    act_pre = emb @ W[:, s:].T                                                    # [K, h]

    def part(st, ver):
        low = torch.einsum("vnrd,lnd->lvnr", B_all[..., :s], st)                  # [l, V, n, r]
        low = low.gather(1, ver.view(*ver.shape[:1], 1, -1, 1).expand(-1, 1, -1, low.shape[-1])).squeeze(1)
        low = low.unsqueeze(2) + _gather_versions(low_act, ver)                   # [l, n, K, r]
        gates = model.gates(st)
        if gates is not None:                                                     # adapter blocks of each state
            low = low * gates.detach().unsqueeze(2).to(low.dtype)
        pre = (st @ W[:, :s].T + b).unsqueeze(2) + act_pre                        # [l, n, K, h]
        pre = pre + torch.einsum("lnhr,lnkr->lnkh", _gather_versions(A_all, ver).to(pre.dtype), low.to(pre.dtype))
        out = functional_call(model.pred_out, weights, (pre,))
        return out[..., 0], out[..., 1].clamp(-8.0, 4.0)

    means, logvars = [], []
    for c0 in range(0, state.shape[0], chunk):
        m, lv = checkpoint(part, state[c0:c0 + chunk], version[c0:c0 + chunk], use_reentrant=False)
        means.append(m)
        logvars.append(lv)
    return torch.cat(means), torch.cat(logvars)


def online_nll(model: ContinualCore, state: Tensor, actions: Tensor, As, Bs, version: Tensor, targets: Tensor,
               precise: Tensor) -> Tensor:
    """[L, n] loss of the prediction for the action taken at every tick under the adapter in effect then,
    with gradients into the shared predictor, the state features and, through the adapters, every step."""
    x = torch.cat((state, model.action_features(actions)), -1)                   # [L, n, 3w]
    B_all, A_all = torch.stack(Bs), torch.stack(As)                               # [V, n, r, 3w], [V, n, h, r]
    current = F.one_hot(version, len(As)).unsqueeze(-1)                           # [L, n, V, 1]
    low = torch.einsum("vnrd,lnd->lnvr", B_all, x) * current                      # only the version in effect
    gates = model.gates(state)
    if gates is not None:                                                         # adapter blocks of each state
        low = low * gates.unsqueeze(2).to(low.dtype)
    pre = model.pred_in(x) + torch.einsum("vnhr,lnvr->lnh", A_all, low)
    mean, logvar = model._head(pre)
    return reading_nll(model, mean, logvar, targets, precise)
