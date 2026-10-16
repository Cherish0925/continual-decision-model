"""One tick of a continual agent on instrument lifetimes: events, decisions, execution, adaptation.

Order inside a tick: the event updates the fast state; control and activities
are chosen from the fast state, the workspace completed so far and the adapted
predictions; the device advances under the control while a precise measurement
may be requested. The compute between this tick and the next goes to the
chosen option: a number of workspace refinements, read from the next tick on,
or a learning step. A learning step starts from the instance's experience at
the time it is chosen and occupies the compute for ``learn_ticks`` ticks; its
updated adapter is used once it completes. Control is produced every tick and
nothing waits for computation or learning.
"""
from __future__ import annotations

from dataclasses import dataclass, fields

import torch
from torch import Tensor

from cdm.core import REFINEMENTS, ContinualCore, adapter_step
from cdm.instrument import InstrumentFleet, InstrumentSpec

SCALAR_DIM = 8
MODES = ("full", "fixed_learn", "frozen", "world_model", "recurrent")
LEARN = len(REFINEMENTS)                  # index of the learning option among the compute options


def configure_backends():
    """TF32 matmuls and memory-efficient attention, which suits short slot sequences and varying batch sizes."""
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_cudnn_sdp(False)


def feature_dims(spec: InstrumentSpec) -> tuple[int, int]:
    """Sizes of the event vector and of one experience record."""
    return 13 + spec.dims, 5 + spec.dims


@dataclass(frozen=True)
class Costs:
    refinement: float = 0.002
    measurement: float = 0.01
    learning: float = 0.02


@dataclass
class Agent:
    h: Tensor
    z: Tensor
    A: Tensor
    B: Tensor
    next_A: Tensor         # adapter produced by the running learning step
    next_B: Tensor
    records: Tensor        # [n, R, record_dim] oldest to newest
    learn_s: Tensor        # [n, T, 2w] state features of recent ticks
    learn_a: Tensor        # [n, T] actions taken
    learn_y: Tensor        # [n, T] readings that followed
    filled: Tensor         # [n] valid entries of the learning buffer
    busy: Tensor           # [n] remaining ticks of a running learning step
    last: Tensor           # [n, dims + 3] previous control values, measurement, refinements/4, learning
    since_measure: Tensor
    since_learn: Tensor
    since_request: Tensor
    error_ema: Tensor
    predicted: Tensor      # [n] predicted next reading for the action taken
    age: Tensor            # [n] ticks since the lifetime began


def new_agent(model: ContinualCore, n: int, buffer: int, device) -> Agent:
    c = model.config
    s = model.initial(n, device)
    z = torch.zeros(n, device=device)
    return Agent(h=s["h"], z=s["z"], A=s["A"], B=s["B"], next_A=s["A"].clone(), next_B=s["B"].clone(),
                 records=torch.zeros(n, c.records, c.record_dim, device=device),
                 learn_s=torch.zeros(n, buffer, 2 * c.width, device=device, dtype=torch.bfloat16),
                 learn_a=torch.zeros(n, buffer, dtype=torch.long, device=device),
                 learn_y=torch.zeros(n, buffer, device=device), filled=z.long(), busy=z.long(),
                 last=torch.zeros(n, c.action_dim + 3, device=device), since_measure=z.clone(),
                 since_learn=z.clone(), since_request=z.clone(), error_ema=z.clone(), predicted=z.clone(),
                 age=z.long())


def reset_agents(model: ContinualCore, agent: Agent, mask: Tensor) -> Agent:
    if not mask.any():
        return agent
    fresh = new_agent(model, int(mask.sum()), agent.learn_s.shape[1], mask.device)
    idx = mask.nonzero().squeeze(1)
    out = {}
    for f in fields(agent):
        v = getattr(agent, f.name).clone()
        v[idx] = getattr(fresh, f.name).to(v.dtype)
        out[f.name] = v
    return Agent(**out)


def event_features(obs: dict, agent: Agent, pending: Tensor) -> Tensor:
    reading = torch.stack((obs["cheap"], obs["precise"], obs["arrived"].float(), obs["setpoint"],
                           obs["active"].float(), obs["started"].float()), 1)
    timing = torch.stack(((agent.busy > 0).float(), (pending > 0).float(), (agent.since_measure / 32).clamp(max=2),
                          (obs["cheap"] - agent.predicted).clamp(-2, 2)), 1)
    return torch.cat((reading, agent.last, timing), 1)


def scalar_features(obs: dict, agent: Agent, pending: Tensor) -> Tensor:
    return torch.stack(((agent.busy > 0).float(), (pending > 0).float(), (agent.since_measure / 32).clamp(max=2),
                        (agent.since_learn / 64).clamp(max=2), agent.error_ema.clamp(max=2), obs["active"].float(),
                        (agent.since_request / 64).clamp(max=2), agent.filled.float() / agent.learn_y.shape[1]), 1)


def masks(agent: Agent, pending: Tensor, mode: str, tick: int, every: int) -> Tensor:
    """Allowed options [n, 2 + compute]: measure (no, yes) | compute (0, 1, 2, 4 refinements, learn)."""
    n = agent.h.shape[0]
    k = 2 + LEARN                                         # column of the learning option
    m = torch.ones(n, k + 1, dtype=torch.bool, device=agent.h.device)
    m[:, 1] &= pending == 0                               # one precise measurement at a time
    busy = agent.busy > 0
    m[:, 3:] &= ~busy.unsqueeze(1)                        # a running learning step occupies the compute
    m[:, k] &= agent.filled >= 8
    if mode in ("frozen", "recurrent"):
        m[:, k] = False
    if mode == "recurrent":
        m[:, 3:k] = False
    if mode in ("fixed_learn", "world_model"):            # learn on a fixed schedule whenever possible
        due = m[:, k] & (tick % every == 0)
        m[:, 2:k] &= ~due.unsqueeze(1)
        m[:, k] = due
    if mode == "world_model":                             # otherwise one refinement; measure on a fixed schedule
        due = m[:, 1] & (tick % every == 0)
        m[:, 0], m[:, 1] = ~due, due
        learning = m[:, k].clone()
        m[:, 2:k] = False
        m[:, 2] = busy
        m[:, 3] = ~busy & ~learning
    return m


def distributions(control_logits: Tensor, activity_logits: Tensor, allowed: Tensor):
    """Control, measurement and compute distributions; an option forced by the mode has log-probability 0."""
    act = activity_logits.float().masked_fill(~allowed, -1e9)
    return (torch.distributions.Categorical(logits=control_logits.float()),
            torch.distributions.Categorical(logits=act[:, :2]),
            torch.distributions.Categorical(logits=act[:, 2:]))


@torch.no_grad()
def tick(model: ContinualCore, agent: Agent, env: InstrumentFleet, obs: dict, mode: str, costs: Costs,
         learning_rate: float, every: int, learn_ticks: int = 2, greedy: bool = False, sample_learn: int = 32):
    """Advance agents and devices one tick. Returns the new agent, next observation and a record of the tick."""
    n = agent.h.shape[0]
    device = agent.h.device
    rows = torch.arange(n, device=device)
    pending = env.pending.clone()
    event = event_features(obs, agent, pending)
    scalars = scalar_features(obs, agent, pending)
    allowed = masks(agent, pending, mode, env.tick, every)
    h_prev, z_prev = agent.h, agent.z
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        h = model.fast(h_prev, event)
        state = model.state_features(h, z_prev)
        if mode == "recurrent":
            mean = logvar = offset = torch.zeros(n, model.config.actions, device=device)
        else:
            mean, logvar = model.predict_actions(state, agent.A, agent.B)
            offset = mean.float() - obs["setpoint"].unsqueeze(1)      # predicted readings relative to the setpoint
        logvar = logvar.float()
        control_logits, activity_logits, value = model.heads(model.trunk(h, z_prev, offset, logvar, scalars),
                                                             offset, logvar)
    h = h.float()
    dists = distributions(control_logits, activity_logits, allowed)
    choice = [d.probs.argmax(-1) if greedy else d.sample() for d in dists]
    action, measure, compute = choice
    logp = sum(d.log_prob(a) for d, a in zip(dists, choice))
    nxt = env.step(action, measure.bool())
    counts = torch.tensor(REFINEMENTS + (0,), device=device)[compute]
    learn = compute == LEARN
    reward = -nxt["error"] - costs.refinement * counts - costs.measurement * nxt["accepted"].float() \
        - costs.learning * learn.float()
    record = torch.cat((nxt["control"], torch.stack((nxt["cheap"], nxt["precise"], nxt["arrived"].float(),
                                                     nxt["setpoint"], nxt["active"].float()), 1)), 1)
    records = torch.cat((agent.records[:, 1:], record.unsqueeze(1)), 1)
    learn_s = torch.cat((agent.learn_s[:, 1:], state.unsqueeze(1).to(agent.learn_s.dtype)), 1)
    learn_a = torch.cat((agent.learn_a[:, 1:], action.unsqueeze(1)), 1)
    learn_y = torch.cat((agent.learn_y[:, 1:], nxt["cheap"].unsqueeze(1)), 1)
    T = learn_y.shape[1]
    filled = (agent.filled + 1).clamp(max=T)
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        z = model.refine(z_prev, h, records, counts).float()
    complete = (agent.busy == 1).view(-1, 1, 1)                     # a learning step finishes with this tick
    A = torch.where(complete, agent.next_A, agent.A)
    B = torch.where(complete, agent.next_B, agent.B)
    next_A, next_B, busy = agent.next_A, agent.next_B, (agent.busy - 1).clamp(min=0)
    if learn.any():
        idx = learn.nonzero().squeeze(1)
        pick = torch.randint(0, T, (len(idx), sample_learn), device=device)
        pick = T - 1 - (pick % filled[idx].unsqueeze(1))             # only filled entries
        sel = (idx.unsqueeze(1), pick)
        xs = torch.cat((learn_s[sel].float(), model.action_features(learn_a[sel])), -1)
        newA, newB, _ = adapter_step(model, A[idx], B[idx], xs, learn_y[sel], learning_rate)
        if learn_ticks <= 1:
            A, B = A.clone(), B.clone()
            A[idx], B[idx] = newA, newB
        else:
            next_A, next_B = next_A.clone(), next_B.clone()
            next_A[idx], next_B[idx] = newA, newB
            busy[idx] = learn_ticks - 1
    pred_taken = mean.float()[rows, action]
    error = (nxt["cheap"] - pred_taken).abs()
    agent = Agent(h=h, z=z, A=A, B=B, next_A=next_A, next_B=next_B, records=records, learn_s=learn_s,
                  learn_a=learn_a, learn_y=learn_y, filled=filled, busy=busy,
                  last=torch.cat((nxt["control"], torch.stack((measure.float(), counts.float() / 4, learn.float()), 1)), 1),
                  since_measure=torch.where(nxt["accepted"], torch.zeros_like(agent.since_measure), agent.since_measure + 1),
                  since_learn=torch.where(learn, torch.zeros_like(agent.since_learn), agent.since_learn + 1),
                  since_request=torch.where(nxt["started"], torch.zeros_like(agent.since_request), agent.since_request + 1),
                  error_ema=0.95 * agent.error_ema + 0.05 * error, predicted=pred_taken, age=agent.age + 1)
    step = {"event": event, "scalars": scalars, "allowed": allowed, "records": records, "offset": offset,
            "logvar": logvar, "action": action, "measure": measure, "compute": compute, "counts": counts,
            "learn": learn, "logp": logp, "value": value.float(), "reward": reward, "target": nxt["cheap"],
            "h_prev": h_prev, "z_prev": z_prev.to(torch.bfloat16), "error": nxt["error"], "active": nxt["active"],
            "accepted": nxt["accepted"], "since_switch": nxt["since_switch"], "returned": nxt["returned"],
            "started": nxt["started"], "output": nxt["output"], "setpoint": nxt["setpoint"],
            "prediction_error": error}
    return agent, nxt, step
