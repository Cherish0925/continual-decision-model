"""A continual model that keeps one state and chooses how to measure, compute, act and learn.

Per tick the model receives an event vector (readings, setpoint, its previous
operations, elapsed-time features). Its parts:

``gru``         fast state update from every event
``workspace``   latent slots refined by one shared Transformer block that attends
                to the fast state and to the instance's recent experience records;
                every refinement ends with a normalisation of each slot, so the
                workspace keeps one scale over an unbounded number of refinements
``predictor``   action-conditioned prediction of the next reading for every action;
                its first layer carries an instance-owned low-rank adapter that a
                learning step updates from the instance's own experience
``control``     scores every action from the state, the action's features and the
                action's own predicted reading, with weights shared across actions,
                plus a learned preference for actions predicted to reach the goal
``activity``    heads for precise measurement and for the use of compute: a number
                of workspace refinements or one learning step
``value``       critic for the outer reinforcement learning

Shared parameters are trained across lifetimes; adapters belong to one instance
and change only through its learning steps.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

REFINEMENTS = (0, 1, 2, 4)     # compute options before the learning option


@dataclass(frozen=True)
class CoreConfig:
    event_dim: int
    record_dim: int
    actions: int               # discrete control actions
    action_dim: int = 1        # actuator values per action
    width: int = 512
    slots: int = 8
    heads: int = 8
    records: int = 32          # experience records the workspace reads
    rank: int = 8              # adapter rank
    hidden: int = 512          # predictor width
    action_width: int = 256    # width of the per-action control scores
    goal_term: bool = True     # control prefers actions predicted to reach the goal, with a learned weight
    bounded_workspace: bool = True   # normalise slots after every refinement
    scalar_dim: int = 8        # extra features for the activity and value heads

    @property
    def compute_options(self) -> int:
        return len(REFINEMENTS) + 1


class Block(nn.Module):
    """Pre-norm block: slot self-attention, attention to context tokens, MLP."""

    def __init__(self, width, heads):
        super().__init__()
        self.n1, self.n2, self.n3 = nn.LayerNorm(width), nn.LayerNorm(width), nn.LayerNorm(width)
        self.self_attn = nn.MultiheadAttention(width, heads, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(width, heads, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(width, 4 * width), nn.GELU(), nn.Linear(4 * width, width))

    def forward(self, z: Tensor, context: Tensor) -> Tensor:
        q = self.n1(z)
        z = z + self.self_attn(q, q, q, need_weights=False)[0]
        q = self.n2(z)
        z = z + self.cross_attn(q, context, context, need_weights=False)[0]
        return z + self.mlp(self.n3(z))


class ContinualCore(nn.Module):
    def __init__(self, config: CoreConfig, action_values: Tensor):
        super().__init__()
        self.config = c = config
        w = c.width
        self.register_buffer("action_values", action_values.float().reshape(c.actions, c.action_dim).clone())
        self.event = nn.Sequential(nn.Linear(c.event_dim, w), nn.GELU(), nn.Linear(w, w))
        self.gru = nn.GRU(w, w)
        self.slots = nn.Parameter(0.02 * torch.randn(c.slots, w))
        self.record = nn.Sequential(nn.Linear(c.record_dim + 1, w), nn.GELU(), nn.Linear(w, w))
        self.block = Block(w, c.heads)
        self.norm_slots = nn.LayerNorm(w) if c.bounded_workspace else nn.Identity()
        self.action = nn.Sequential(nn.Linear(c.action_dim, w), nn.GELU(), nn.Linear(w, w))
        self.pred_in = nn.Linear(3 * w, c.hidden)
        self.pred_out = nn.Sequential(nn.GELU(), nn.Linear(c.hidden, c.hidden), nn.GELU(), nn.Linear(c.hidden, 2))
        trunk = 2 * w + 2 * c.actions + c.scalar_dim
        self.norm_h, self.norm_z = nn.LayerNorm(w), nn.LayerNorm(w)
        a = c.action_width
        self.control_state, self.control_action = nn.Linear(trunk, a), nn.Linear(w, a, bias=False)
        self.control_pred = nn.Linear(2, a, bias=False)
        self.control_out = nn.Sequential(nn.GELU(), nn.Linear(a, a), nn.GELU(), nn.Linear(a, 1))
        self.goal_weight = nn.Parameter(torch.tensor(1.8)) if c.goal_term else None   # softplus(1.8) = 1.95
        self.activity = nn.Sequential(nn.Linear(trunk, w), nn.GELU(), nn.Linear(w, 2 + c.compute_options))
        self.value = nn.Sequential(nn.Linear(trunk, w), nn.GELU(), nn.Linear(w, 1))

    # ------------------------------------------------------------ state
    def initial(self, n: int, device) -> dict[str, Tensor]:
        c = self.config
        return {"h": torch.zeros(n, c.width, device=device),
                "z": self.slots.detach().unsqueeze(0).expand(n, -1, -1).clone(),
                "A": torch.zeros(n, c.hidden, c.rank, device=device),
                "B": 0.01 * torch.randn(n, c.rank, 3 * c.width, device=device)}

    def fast(self, h: Tensor, event: Tensor) -> Tensor:
        """Fast state after one event: h [n, w], event [n, e]."""
        return self.fast_sequence(h, event.unsqueeze(0))[0]

    def fast_sequence(self, h: Tensor, events: Tensor) -> Tensor:
        """Fast states [T, n, w] after each of the events [T, n, e], starting from h [n, w]."""
        x = self.event(events)
        with torch.autocast(x.device.type, enabled=False):
            return self.gru(x.float(), h.float().unsqueeze(0).contiguous())[0]

    def context(self, h: Tensor, records: Tensor) -> Tensor:
        """Tokens the workspace attends to: the fast state and the recent records [n, R, record_dim]."""
        n, r, _ = records.shape
        age = torch.linspace(1.0, 0.0, r, device=records.device).expand(n, r).unsqueeze(-1)
        return torch.cat((h.unsqueeze(1), self.record(torch.cat((records, age), -1))), 1)

    def refine(self, z: Tensor, h: Tensor, records: Tensor, counts: Tensor) -> Tensor:
        """Apply each instance's chosen number of shared refinements (counts [n] in 0..4)."""
        rows = (counts > 0).nonzero().squeeze(1)
        if len(rows) == 0:
            return z
        ctx, zr, cr = self.context(h[rows], records[rows]), z[rows], counts[rows]
        for step in range(int(cr.max())):
            sel = (cr > step).nonzero().squeeze(1)
            zr = zr.index_copy(0, sel, self.norm_slots(self.block(zr[sel], ctx[sel])).to(zr.dtype))
        return z.index_copy(0, rows, zr.to(z.dtype))

    # ------------------------------------------------------------ prediction with instance adapters
    def state_features(self, h: Tensor, z: Tensor) -> Tensor:
        """[n, 2w] state part of the predictor input."""
        return torch.cat((self.norm_h(h), self.norm_z(z.mean(1))), -1)

    def action_features(self, action: Tensor | None = None) -> Tensor:
        """[actions, w] for every action, or [..., w] for the given action indices."""
        values = self.action_values if action is None else self.action_values[action]
        return self.action(values)

    def predict(self, x: Tensor, A: Tensor | None = None, B: Tensor | None = None) -> tuple[Tensor, Tensor]:
        """Mean and log-variance of the next reading for predictor inputs x [n, k, 3w]."""
        pre = self.pred_in(x)
        if A is not None:
            pre = pre + torch.einsum("nhr,nkr->nkh", A, torch.einsum("nrd,nkd->nkr", B, x))
        return self._head(pre)

    def predict_actions(self, state: Tensor, A: Tensor | None = None, B: Tensor | None = None):
        """``predict`` for every action from state features [n, 2w], without forming the [n, actions, 3w] input."""
        s = state.shape[-1]
        emb = self.action_features()
        W, b = self.pred_in.weight, self.pred_in.bias
        pre = (state @ W[:, :s].T + b).unsqueeze(1) + (emb @ W[:, s:].T).unsqueeze(0)
        if A is not None:
            low = torch.einsum("nrd,nd->nr", B[..., :s], state).unsqueeze(1) + torch.einsum("nrd,kd->nkr", B[..., s:], emb)
            pre = pre + torch.einsum("nhr,nkr->nkh", A, low)
        return self._head(pre)

    def _head(self, pre: Tensor) -> tuple[Tensor, Tensor]:
        out = self.pred_out(pre)
        return out[..., 0], out[..., 1].clamp(-8.0, 4.0)

    # ------------------------------------------------------------ heads
    def trunk(self, h: Tensor, z: Tensor, pred_offset: Tensor, pred_logvar: Tensor, scalars: Tensor) -> Tensor:
        """Shared input of the heads; ``pred_offset`` [n, actions] is each action's predicted reading minus the goal."""
        return torch.cat((self.norm_h(h), self.norm_z(z.mean(1)), pred_offset, pred_logvar / 4, scalars), -1)

    def heads(self, trunk: Tensor, pred_offset: Tensor, pred_logvar: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Control logits [n, actions], activity logits [n, 2 + compute options], value [n]."""
        scores = self.control_state(trunk).unsqueeze(1) + self.control_action(self.action_features()).unsqueeze(0) \
            + self.control_pred(torch.stack((pred_offset, pred_logvar / 4), -1))
        control = self.control_out(scores).squeeze(-1)
        if self.goal_weight is not None:
            control = control - F.softplus(self.goal_weight) * pred_offset.abs()
        return control, self.activity(trunk), self.value(trunk).squeeze(-1)


def gaussian_nll(mean: Tensor, logvar: Tensor, target: Tensor) -> Tensor:
    return 0.5 * (logvar + (target - mean) ** 2 * torch.exp(-logvar))


def adapter_step(model: ContinualCore, A: Tensor, B: Tensor, inputs: Tensor, targets: Tensor,
                 learning_rate: float) -> tuple[Tensor, Tensor, Tensor]:
    """One update of instance adapters on their own experience.

    inputs [n, T, 3w] (predictor inputs of past ticks for the action taken), targets [n, T] (readings that
    followed). Each instance's step is normalised by its own gradient size.
    """
    with torch.enable_grad():
        A_ = A.detach().requires_grad_(True)
        B_ = B.detach().requires_grad_(True)
        mean, logvar = model.predict(inputs, A_, B_)
        loss = gaussian_nll(mean, logvar, targets).mean(1)                   # [n]
        gA, gB = torch.autograd.grad(loss.sum(), (A_, B_))
    scale = torch.sqrt(gA.pow(2).sum((1, 2)) + gB.pow(2).sum((1, 2)) + 1e-12).view(-1, 1, 1)
    return (A - learning_rate * gA / scale).detach(), (B - learning_rate * gB / scale).detach(), loss.detach()


def adaptation_loss(model: ContinualCore, A: Tensor, B: Tensor, support_x: Tensor, support_y: Tensor,
                    query_x: Tensor, query_y: Tensor, learning_rate: float,
                    differentiable: bool = True) -> tuple[Tensor, Tensor]:
    """Prediction loss on later ticks before and after one adapter step on earlier ticks, per instance.

    The step is the one a learning activity takes. With ``differentiable`` the loss after the step carries
    gradients through the step into the shared parameters, so that training on it makes an instance's own
    learning step improve its later predictions.
    """
    A_ = A.detach().requires_grad_(True)
    B_ = B.detach().requires_grad_(True)
    with torch.enable_grad():
        mean, logvar = model.predict(support_x, A_, B_)
        support = gaussian_nll(mean, logvar, support_y).mean(1).sum()
        gA, gB = torch.autograd.grad(support, (A_, B_), create_graph=differentiable)
    scale = torch.sqrt(gA.pow(2).sum((1, 2)) + gB.pow(2).sum((1, 2)) + 1e-12).view(-1, 1, 1)
    with torch.no_grad():
        before = gaussian_nll(*model.predict(query_x, A, B), query_y).mean(1)
    after = gaussian_nll(*model.predict(query_x, A_ - learning_rate * gA / scale, B_ - learning_rate * gB / scale),
                         query_y).mean(1)
    return before, after

