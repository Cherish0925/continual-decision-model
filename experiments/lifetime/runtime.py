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

The experience a learning step reads is either 32 entries sampled from the
buffer (``support="sample"``) or every filled entry (``support="buffer"``).
With ``precise_targets`` a precise reading, when it returns, replaces the cheap
reading of the transition it measured, both as that entry's target and as the
target of the prediction loss.
"""
from __future__ import annotations

from dataclasses import dataclass, fields

import torch
from torch import Tensor

from cdm.core import REFINEMENTS, ContinualCore, adapter_update
from cdm.instrument import InstrumentFleet, MixedFleet, ProcessSpec
from cdm.navigation import NavigationFleet
from cdm.thermal import ThermalFleet

SCALAR_DIM = 8
MODES = ("full", "fixed_learn", "frozen", "world_model", "predict_only", "predict_hidden", "recurrent")
HIDDEN = ("recurrent", "predict_hidden")  # settings whose policy reads no prediction
LEARN = len(REFINEMENTS)                  # index of the learning option among the compute options


def configure_backends():
    """TF32 matmuls and memory-efficient attention, which suits short slot sequences and varying batch sizes."""
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_cudnn_sdp(False)


FLEETS = {"instrument": InstrumentFleet, "thermal": ThermalFleet, "navigation": NavigationFleet}
FAMILIES = (*FLEETS, "mixed", "all")


def make_fleet(family: str, n: int, seed: int, device, specs: dict[str, ProcessSpec], share: float = 0.5):
    """Processes of one family; ``mixed``: instruments and, for ``share`` of the streams, thermal processes;
    ``all``: instruments, thermal processes and vehicles, a third of the streams each. ``specs`` maps each family
    to its specification."""
    if family in FLEETS:
        return FLEETS[family](n, specs[family], seed, device)
    if family == "mixed":
        parts = (("instrument", n - int(round(n * share))), ("thermal", int(round(n * share))))
    else:
        parts = (("instrument", n - 2 * (n // 3)), ("thermal", n // 3), ("navigation", n // 3))
    return MixedFleet([FLEETS[k](m, specs[k], seed + i, device) for i, (k, m) in enumerate(parts)])


def goal_kinds(spec: ProcessSpec) -> int:
    """1 when every request gives a setpoint, 2 when some ask for the largest output."""
    return 2 if spec.search_share > 0 else 1


def feature_dims(spec: ProcessSpec, goals: int = 1) -> tuple[int, int]:
    """Sizes of the event vector and of one experience record; with two kinds of goal each carries a flag for
    requests that ask for the largest output, as do the scalar features (``SCALAR_DIM + goals - 1``)."""
    return 12 + spec.dims + goals, 4 + spec.dims + goals


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
    learn_p: Tensor        # [n, T] the reading is a precise one
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
                 learn_y=torch.zeros(n, buffer, device=device),
                 learn_p=torch.zeros(n, buffer, dtype=torch.bool, device=device), filled=z.long(), busy=z.long(),
                 last=torch.zeros(n, c.action_dim + 3, device=device), since_measure=z.clone(),
                 since_learn=z.clone(), since_request=z.clone(), error_ema=z.clone(), predicted=z.clone(),
                 age=z.long())


def select_agent(agent: Agent, rows: Tensor) -> Agent:
    """The agents of ``rows``, in that order; a row may repeat."""
    return Agent(**{f.name: getattr(agent, f.name)[rows] for f in fields(agent)})


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


def goal_flags(obs: dict, goals: int) -> list[Tensor]:
    return [obs["search"].float()] if goals > 1 else []


def event_features(obs: dict, agent: Agent, pending: Tensor, goals: int = 1) -> Tensor:
    reading = torch.stack((obs["cheap"], obs["precise"], obs["arrived"].float(), obs["setpoint"],
                           obs["active"].float(), obs["started"].float(), *goal_flags(obs, goals)), 1)
    timing = torch.stack(((agent.busy > 0).float(), (pending > 0).float(), (agent.since_measure / 32).clamp(max=2),
                          (obs["cheap"] - agent.predicted).clamp(-2, 2)), 1)
    return torch.cat((reading, agent.last, timing), 1)


def scalar_features(obs: dict, agent: Agent, pending: Tensor, goals: int = 1) -> Tensor:
    return torch.stack(((agent.busy > 0).float(), (pending > 0).float(), (agent.since_measure / 32).clamp(max=2),
                        (agent.since_learn / 64).clamp(max=2), agent.error_ema.clamp(max=2), obs["active"].float(),
                        (agent.since_request / 64).clamp(max=2), agent.filled.float() / agent.learn_y.shape[1],
                        *goal_flags(obs, goals)), 1)


def experience_record(obs: dict, goals: int = 1) -> Tensor:
    """[n, record_dim]: the control applied and what followed it."""
    return torch.cat((obs["control"], torch.stack((obs["cheap"], obs["precise"], obs["arrived"].float(),
                                                   obs["setpoint"], obs["active"].float(),
                                                   *goal_flags(obs, goals)), 1)), 1)


def goal_offsets(mean: Tensor, obs: dict, goals: int = 1) -> Tensor:
    """Predicted readings [n, actions] relative to the goal: the setpoint, or for a request for the largest
    output, the largest predicted reading."""
    goal = obs["setpoint"]
    if goals > 1:
        goal = torch.where(obs["search"], mean.detach().max(1).values, goal)
    return mean - goal.unsqueeze(1)


def precise_arrivals(agent: Agent, nxt: dict, learn_y: Tensor, learn_p: Tensor):
    """Put each precise reading that returned with ``nxt`` in place of the cheap reading of the transition it
    measured. Only one measurement is outstanding at a time, so that transition is ``since_measure + 1`` ticks
    before the newest entry."""
    T = learn_y.shape[1]
    position = T - 2 - agent.since_measure.long()
    hit = nxt["arrived"].unsqueeze(1) & (torch.arange(T, device=learn_y.device) == position.unsqueeze(1))
    return torch.where(hit, nxt["precise"].unsqueeze(1), learn_y), learn_p | hit


def masks(agent: Agent, pending: Tensor, mode: str, tick: int, every: int) -> Tensor:
    """Allowed options [n, 2 + compute]: measure (no, yes) | compute (0, 1, 2, 4 refinements, learn)."""
    n = agent.h.shape[0]
    k = 2 + LEARN                                         # column of the learning option
    m = torch.ones(n, k + 1, dtype=torch.bool, device=agent.h.device)
    m[:, 1] &= pending == 0                               # one precise measurement at a time
    busy = agent.busy > 0
    m[:, 3:] &= ~busy.unsqueeze(1)                        # a running learning step occupies the compute
    m[:, k] &= agent.filled >= 8
    if mode in ("frozen", "predict_only", "predict_hidden", "recurrent"):
        m[:, k] = False
    if mode in ("predict_only", "predict_hidden", "recurrent"):
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


def shared_sample(dist: torch.distributions.Categorical, generator: torch.Generator, copies: int) -> Tensor:
    """A sample per row by the Gumbel-max rule, with the same noise for the ``copies`` consecutive copies of a row."""
    n, k = dist.logits.shape
    u = torch.rand(n // copies, k, generator=generator, device=dist.logits.device).repeat_interleave(copies, 0)
    return (dist.logits - torch.log(-torch.log(u.clamp(1e-20, 1.0)))).argmax(-1)


@torch.no_grad()
def tick(model: ContinualCore, agent: Agent, env: InstrumentFleet, obs: dict, mode: str, costs: Costs,
         learning_rate: float, every: int, learn_ticks: int = 2, greedy: bool = False, sample_learn: int = 32,
         shared: bool = False, apply_updates: bool = True, support: str = "sample", precise_targets: bool = False,
         refinements: bool = True, idle_compute: bool = True, force: Tensor | None = None,
         generator: torch.Generator | None = None, copies: int = 1):
    """Advance agents and devices one tick. Returns the new agent, next observation and a record of the tick.

    With ``shared`` the record also holds the error of the shared predictor alone, without the instance adapter,
    for the action taken. With ``apply_updates`` false, chosen learning steps still occupy the compute and incur
    their cost, but their adapter updates are discarded: a matched-time control for the effect of the update.
    With ``refinements`` false the refinement options are removed; with ``idle_compute`` false, refinement and
    learning are removed while no request is active. ``force`` [n, 2 + compute options] restricts the
    activity options further. With a ``generator``, choices are sampled from it, with the same noise for each run
    of ``copies`` consecutive rows.
    """
    n = agent.h.shape[0]
    device = agent.h.device
    goals = model.config.goals
    rows = torch.arange(n, device=device)
    pending = env.pending.clone()
    event = event_features(obs, agent, pending, goals)
    scalars = scalar_features(obs, agent, pending, goals)
    allowed = masks(agent, pending, mode, env.tick, every)
    if not refinements:
        allowed[:, 3:2 + LEARN] = False
    if not idle_compute:
        allowed[:, 3:] &= obs["active"].unsqueeze(1)
    if force is not None:
        allowed &= force
    h_prev, z_prev = agent.h, agent.z
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        h = model.fast(h_prev, event)
        state = model.state_features(h, z_prev)
        if mode == "recurrent":
            mean = logvar = torch.zeros(n, model.config.actions, device=device)
        else:
            mean, logvar = model.predict_actions(state, agent.A, agent.B)
        logvar = logvar.float()
        offset = goal_offsets(mean.float(), obs, goals)               # predicted readings relative to the goal
        if mode in HIDDEN:                                            # the policy reads no prediction
            offset, logvar = torch.zeros_like(offset), torch.zeros_like(logvar)
        control_logits, activity_logits, value = model.heads(model.trunk(h, z_prev, offset, logvar, scalars),
                                                             offset, logvar)
    h = h.float()
    dists = distributions(control_logits, activity_logits, allowed)
    if greedy:
        choice = [d.probs.argmax(-1) for d in dists]
    elif generator is not None:
        choice = [shared_sample(d, generator, copies) for d in dists]
    else:
        choice = [d.sample() for d in dists]
    action, measure, compute = choice
    logp = sum(d.log_prob(a) for d, a in zip(dists, choice))
    nxt = env.step(action, measure.bool())
    counts = torch.tensor(REFINEMENTS + (0,), device=device)[compute]
    learn = compute == LEARN
    reward = -nxt["error"] - costs.refinement * counts - costs.measurement * nxt["accepted"].float() \
        - costs.learning * learn.float()
    records = torch.cat((agent.records[:, 1:], experience_record(nxt, goals).unsqueeze(1)), 1)
    learn_s = torch.cat((agent.learn_s[:, 1:], state.unsqueeze(1).to(agent.learn_s.dtype)), 1)
    learn_a = torch.cat((agent.learn_a[:, 1:], action.unsqueeze(1)), 1)
    learn_y = torch.cat((agent.learn_y[:, 1:], nxt["cheap"].unsqueeze(1)), 1)
    learn_p = torch.cat((agent.learn_p[:, 1:], torch.zeros_like(agent.learn_p[:, :1])), 1)
    if precise_targets:
        learn_y, learn_p = precise_arrivals(agent, nxt, learn_y, learn_p)
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
        if support == "buffer":                                       # every filled entry of the buffer
            xs = torch.cat((learn_s[idx].float(), model.action_features(learn_a[idx])), -1)
            held = torch.arange(T, device=device).unsqueeze(0) >= (T - filled[idx]).unsqueeze(1)
            newA, newB, _ = adapter_update(model, A[idx], B[idx], xs, learn_y[idx], learn_p[idx], held.float(),
                                           learning_rate)
        else:
            pick = torch.randint(0, T, (len(idx), sample_learn), device=device)
            pick = T - 1 - (pick % filled[idx].unsqueeze(1))         # only filled entries
            sel = (idx.unsqueeze(1), pick)
            xs = torch.cat((learn_s[sel].float(), model.action_features(learn_a[sel])), -1)
            newA, newB, _ = adapter_update(model, A[idx], B[idx], xs, learn_y[sel], learn_p[sel], None,
                                           learning_rate)
        if apply_updates:
            if learn_ticks <= 1:
                A, B = A.clone(), B.clone()
                A[idx], B[idx] = newA, newB
            else:
                next_A, next_B = next_A.clone(), next_B.clone()
                next_A[idx], next_B[idx] = newA, newB
        if learn_ticks > 1:
            busy[idx] = learn_ticks - 1
    measured_back = agent.since_measure.long() + 1                 # a returning precise reading measured this far back
    pred_taken = mean.float()[rows, action]
    error = (nxt["cheap"] - pred_taken).abs()
    seen = torch.zeros_like(pred_taken) if mode in HIDDEN else pred_taken   # what the policy's features may carry
    seen_error = (nxt["cheap"] - seen).abs()
    if shared and mode != "recurrent":
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            alone = model.predict(torch.cat((state, model.action_features(action)), -1).unsqueeze(1))[0]
        shared_error = (nxt["cheap"] - alone.float().squeeze(1)).abs()
    agent = Agent(h=h, z=z, A=A, B=B, next_A=next_A, next_B=next_B, records=records, learn_s=learn_s,
                  learn_a=learn_a, learn_y=learn_y, learn_p=learn_p, filled=filled, busy=busy,
                  last=torch.cat((nxt["control"], torch.stack((measure.float(), counts.float() / 4, learn.float()), 1)), 1),
                  since_measure=torch.where(nxt["accepted"], torch.zeros_like(agent.since_measure), agent.since_measure + 1),
                  since_learn=torch.where(learn, torch.zeros_like(agent.since_learn), agent.since_learn + 1),
                  since_request=torch.where(nxt["started"], torch.zeros_like(agent.since_request), agent.since_request + 1),
                  error_ema=0.95 * agent.error_ema + 0.05 * seen_error, predicted=seen, age=agent.age + 1)
    step = {"event": event, "scalars": scalars, "allowed": allowed, "records": records, "offset": offset,
            "logvar": logvar, "action": action, "measure": measure, "compute": compute, "counts": counts,
            "learn": learn, "logp": logp, "value": value.float(), "reward": reward, "target": nxt["cheap"],
            "h_prev": h_prev, "z_prev": z_prev.to(torch.bfloat16), "error": nxt["error"],
            "active": nxt["transition_active"], "accepted": nxt["accepted"],
            "since_switch": nxt["transition_since_switch"], "returned": nxt["transition_returned"],
            "started": obs["started"], "output": nxt["output"], "setpoint": nxt["transition_setpoint"],
            "next_active": nxt["active"], "next_started": nxt["started"],
            "goal": obs["setpoint"], "search": obs.get("search", torch.zeros_like(obs["active"])),
            "transition_search": nxt.get("transition_search", torch.zeros_like(obs["active"])),
            "arrived": nxt["arrived"], "precise": nxt["precise"], "measured_back": measured_back,
            "prediction_error": error}
    if shared and mode != "recurrent":
        step["shared_prediction_error"] = shared_error
    return agent, nxt, step


OPTIONS = 2 + LEARN + 1               # activity options: measure no / yes, then each compute option


@torch.no_grad()
def branch_values(model: ContinualCore, agent: Agent, env, obs: dict, rows: Tensor, mode: str, costs: Costs,
                  learning_rate: float, every: int, horizon: int, gamma: float, reward_scale: float, seed: int,
                  **options) -> tuple[Tensor, Tensor]:
    """What each activity option is worth to the given streams, from copies of their processes and agents.

    Each row of ``rows`` (increasing) gets one copy per activity option: not measuring, measuring, and each compute
    option. On the first tick the copy takes its option, with the other activity and the control chosen by the
    policy; then the policy runs for ``horizon`` ticks. The copies of a row share the processes' randomness and the
    policy's sampling noise, so they differ only through the option and its consequences. Returns the discounted
    scaled reward plus the discounted value after ``horizon`` ticks [rows, options], and which options were
    allowed; a copy whose option was not allowed runs unforced and its value is not used.
    """
    O, device = OPTIONS, agent.h.device
    fleet = env.branch(rows, O, seed)
    copies = rows.repeat_interleave(O)
    agents, obs = select_agent(agent, copies), {k: v[copies] for k, v in obs.items()}
    allowed = masks(agents, fleet.pending, mode, env.tick, every)
    option = torch.arange(O, device=device).repeat(len(rows))
    own = torch.arange(O, device=device) == option.unsqueeze(1)                 # the column of the copy's option
    measured = (option < 2).unsqueeze(1)
    force = torch.where(measured, torch.cat((own[:, :2], torch.ones_like(own[:, 2:])), 1),
                        torch.cat((torch.ones_like(own[:, :2]), own[:, 2:]), 1))
    valid = allowed[own]
    force[~valid] = True
    generator = torch.Generator(device=device).manual_seed(seed)
    total, discount = torch.zeros(len(copies), device=device), 1.0
    for h in range(horizon + 1):
        agents, obs, step = tick(model, agents, fleet, obs, mode, costs, learning_rate, every,
                                 force=force if h == 0 else None, generator=generator, copies=O, **options)
        if h < horizon:
            total += discount * reward_scale * step["reward"]
            discount *= gamma
        else:
            total += discount * step["value"]
    return total.view(len(rows), O), valid.view(len(rows), O)
