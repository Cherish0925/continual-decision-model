#!/usr/bin/env python3
"""Train the continual model over instrument lifetimes with recurrent PPO.

Streams run lifetimes of ``--lifetime`` ticks with staggered starts. Fast
state, workspace, experience records, instance adapters and learning buffers
persist across training windows; only the end of a lifetime resets them. Every
``--window`` ticks the shared parameters take PPO steps on the window, with the
fast state and workspace recomputed in chunks from stored states. Value targets
bootstrap from the next decision state; only a lifetime's end is terminal.

The predictor is trained on received readings, under one of three objectives:

``shared``    the shared predictor alone, without instance adapters;
``step``      as ``shared``, plus the loss on the second half of a window after
              one learning step on its first half, taken from the adapter the
              instance held when the window began (``--meta`` weighs it);
``deployed``  the prediction of every tick under the adapter the instance had
              in effect then: the window's learning steps are replayed at the
              ticks they were chosen, each on the buffer it read, and the loss
              trains the shared network and the step size through them.

``--segment`` unrolls the workspace over that many external events, so its
gradients span them. With ``--recompute-predictions`` the heads read
predictions recomputed by the current network under the adapter in effect at
each tick, so the policy loss reaches the state through the predictions.

With ``--branch-streams``, once per window that many streams are copied once
per activity option at a random tick; each copy takes its option and the policy
continues for ``--branch-horizon`` ticks under the same disturbances and
sampling noise as the other copies (``runtime.branch_values``), on its most
probable choices with ``--branch-greedy``; ``--branch-replicates`` averages that
many sets of copies with independent randomness. The returns of
the copies train an activity-value head (``--activity-values``) that estimates,
from the state, how much each option is worth against the others; at every tick
the activity heads follow the expected advantage of every option under these
estimates. The weight of that gradient is ``--branch-weight`` times the share
of the variance of fresh copies' returns the head predicts, a running estimate
from rows it has not yet trained on, so a head that predicts nothing has no
influence.
``--init`` starts from trained weights, with ``--fresh-activity`` a newly
initialised activity head; ``--train-only activity`` trains only the activity,
activity-value and value heads over a fixed model.
"""
from __future__ import annotations

import argparse
import json
import math
import platform
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

from types import SimpleNamespace

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from cdm.core import CoreConfig, ContinualCore, adaptation_loss, reading_nll  # noqa: E402
from cdm.instrument import InstrumentSpec, action_values  # noqa: E402
from cdm.navigation import NavigationSpec  # noqa: E402
from cdm.thermal import ThermalSpec  # noqa: E402
from experiments.lifetime.objectives import (NEVER, deployed_adapters, online_nll, predictions_per_tick,  # noqa: E402
                                             recompute_states, window_support)
from experiments.lifetime.runtime import (FAMILIES, HIDDEN, MODES, OPTIONS, SCALAR_DIM, Costs,  # noqa: E402
                                          branch_values, configure_backends, distributions, feature_dims,
                                          goal_kinds, make_fleet, new_agent, reset_agents, tick)

START = ("A", "B", "next_A", "next_B", "busy", "learn_s", "learn_a", "learn_y", "learn_p", "filled")


def instrument(args) -> InstrumentSpec:
    return InstrumentSpec(dims=args.dims, levels=args.levels, search_share=args.search_share)


def thermal(args) -> ThermalSpec:
    return ThermalSpec(dims=args.dims, levels=args.levels)


def navigation(args) -> NavigationSpec:
    return NavigationSpec(dims=args.dims, levels=args.levels, search_share=args.search_share)


def build(args, spec: InstrumentSpec, device) -> ContinualCore:
    goals = goal_kinds(spec)
    event_dim, record_dim = feature_dims(spec, goals)
    config = CoreConfig(event_dim=event_dim, record_dim=record_dim, actions=spec.actions, action_dim=spec.dims,
                        width=args.width, slots=args.slots, heads=args.heads, records=args.records, rank=args.rank,
                        hidden=args.hidden, action_width=args.action_width, scalar_dim=SCALAR_DIM + goals - 1,
                        learned_step=args.learned_step, goals=goals, sensor_noise=args.sensor_noise,
                        step_rule=args.step_rule, gated_workspace=args.gated_workspace,
                        activity_values=args.activity_values, adapter_blocks=args.adapter_blocks)
    return ContinualCore(config, action_values(spec)).to(device)


def first_observation(env):
    n = env.n
    return env.step(torch.full((n,), env.spec.actions // 2, dtype=torch.long, device=env.device),
                    torch.zeros(n, dtype=torch.bool, device=env.device))


def clear_observation(obs: dict, mask: torch.Tensor) -> dict:
    def clear(v):
        if v.dtype == torch.bool:
            return v & ~mask.view(-1, *([1] * (v.dim() - 1)))
        return torch.where(mask.view(-1, *([1] * (v.dim() - 1))), torch.zeros_like(v), v)
    return {k: clear(v) for k, v in obs.items()}


def attach_returns(steps: list, delay: int):
    """Record on each transition the precise reading that measured it and the step at which it returned.

    The newest step holds the readings that returned with it; each joins the step ``measured_back`` before it.
    """
    step, m = steps[-1], len(steps) - 1
    step["precise_at"] = torch.full_like(step["action"], NEVER)
    step["precise_value"] = torch.zeros_like(step["target"])
    for back in range(1, min(delay, m) + 1):
        sel, old = step["arrived"] & (step["measured_back"] == back), steps[m - back]
        old["precise_at"] = torch.where(sel, m, old["precise_at"])
        old["precise_value"] = torch.where(sel, step["precise"], old["precise_value"])


FACTORS = (slice(0, 2), slice(2, None))   # measurement options, compute options


def factor_probs(logits, allowed):
    """[N, OPTIONS] probabilities of each activity option within its factor."""
    x = logits.float().masked_fill(~allowed, -1e9)
    return torch.cat([x[:, f].softmax(-1) for f in FACTORS], 1)


def centred(values, weights):
    """Each factor's values minus their mean under ``weights`` [N, OPTIONS]."""
    return torch.cat([values[:, f] - (values[:, f] * weights[:, f]).sum(1, keepdim=True)
                      / weights[:, f].sum(1, keepdim=True).clamp_min(1e-8) for f in FACTORS], 1)


def expected_advantage(probs, values):
    """Mean over rows of each activity's expected value under the policy, measured from a baseline that is held
    fixed; its gradient moves probability toward options valued above that baseline."""
    return (probs * centred(values, probs.detach())).sum(1).mean()


def ppo_update(model, optimizer, steps, bootstrap, start, args, activity_weight=0.0):
    """PPO on one window. ``start`` is the agent before the window's first tick; ``activity_weight`` weighs the
    gradient of the activity heads from the activity-value estimates."""
    L = len(steps)
    S = lambda key: torch.stack([s[key] for s in steps])
    rewards, values, dones = args.reward_scale * S("reward"), S("value"), S("done").float()
    adv = torch.zeros_like(rewards)
    gae = torch.zeros_like(rewards[0])
    for t in reversed(range(L)):
        nv = bootstrap if t == L - 1 else values[t + 1]
        delta = rewards[t] + args.gamma * nv * (1 - dones[t]) - values[t]
        gae = delta + args.gamma * args.lam * (1 - dones[t]) * gae
        adv[t] = gae
    returns = adv + values
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    event, scalars, allowed, records = S("event"), S("scalars"), S("allowed"), S("records")
    offset, logvar, cheap = S("offset"), S("logvar"), S("target")
    action, measure, compute, counts, logp = S("action"), S("measure"), S("compute"), S("counts"), S("logp")
    h_prev, z_prev, learn = S("h_prev"), S("z_prev"), S("learn")
    goal, search = S("goal"), S("search")
    precise_value, precise_at = S("precise_value"), S("precise_at")
    arrived, returned, back = S("arrived") & args.precise_targets, S("precise"), S("measured_back")
    if not args.precise_targets:
        precise_at = torch.full_like(precise_at, NEVER)
    precise = precise_at < NEVER                          # a precise reading of this transition has returned
    target = torch.where(precise, precise_value, cheap)
    n = rewards.shape[1]
    device = rewards.device
    stats = dict.fromkeys(("policy", "value", "entropy", "prediction", "adapt_before", "adapt_after"), 0.0)
    branched = args.branch_streams > 0
    if branched:                                         # option values centred within each factor, in window units
        valid = S("branch_valid")
        q = centred(S("branch_q").flatten(0, 1), valid.flatten(0, 1).float()).view_as(valid)
        spread = q[valid].std() if int(valid.sum()) > 1 else q.new_tensor(1.0)
        q = q / spread.clamp_min(1e-6)
        stats.update(activity_value=0.0, branch_regret=0.0, branch_agree=0.0)
        branch_count, unseen = 0, [0.0, 0.0]            # head residual and total on rows it has not trained on
    whole = 1.0 - dones.amax(0)                          # streams whose lifetime runs through the whole window
    adapt_count = 0
    flat = lambda a: a.flatten(0, 1)
    predicts, reads = args.mode != "recurrent", args.mode not in HIDDEN
    deployed = predicts and args.objective == "deployed"
    recompute = reads and predicts and args.recompute_predictions
    amp = lambda: torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda")
    for epoch in range(args.epochs):
        perm = torch.randperm(n, device=device)
        for i in range(0, n, args.minibatch):
            mb = perm[i:i + args.minibatch]
            for c0 in range(0, L, args.chunk):
                c1 = min(c0 + args.chunk, L)
                T = c1 - c0
                part = lambda a: flat(a[c0:c1, mb])
                keep = whole[mb] > 0
                with amp():
                    H, zread = recompute_states(model, h_prev[c0:c1, mb], z_prev[c0:c1, mb], event[c0:c1, mb],
                                               records[c0:c1, mb], counts[c0:c1, mb], dones[c0:c1, mb],
                                               args.segment)
                    state = model.state_features(flat(H), flat(zread)).view(T, len(mb), -1)   # [T, m, 2w]
                    if deployed or recompute:                     # the adapter in effect at every tick
                        first = SimpleNamespace(**{k: getattr(start, k)[mb] for k in START})
                        support = window_support(model, first, state.detach(), action[:, mb], cheap[:, mb],
                                                 precise_value[:, mb], precise_at[:, mb], arrived[:, mb],
                                                 returned[:, mb], back[:, mb])
                        version, As, Bs = deployed_adapters(model, first, learn[:, mb] & keep, support,
                                                            args.learn_ticks, args.adapter_lr, deployed)
                    if recompute:
                        mean_all, lv_all = predictions_per_tick(model, state, As, Bs, version)
                        g = torch.where(search[:, mb], mean_all.detach().max(-1).values, goal[:, mb])
                        o = torch.where(keep[None, :, None], mean_all.float() - g.unsqueeze(-1), offset[:, mb])
                        lv = torch.where(keep[None, :, None], lv_all.float(), logvar[:, mb])
                        o, lv = flat(o), flat(lv)
                    else:
                        o, lv = part(offset), part(logvar)
                    Hf, zf = flat(H), flat(zread)
                    trunk = model.trunk(Hf, zf, o, lv, part(scalars))
                    cl, al, v = model.heads(trunk, o, lv)
                    if branched and model.activity_value is not None:
                        estimate = model.activity_value(trunk).float()
                    if deployed:
                        nll = online_nll(model, state, action[:, mb], As, Bs, version, target[:, mb], precise[:, mb])
                        prediction = (nll * keep).sum() / (keep.sum() * T).clamp_min(1)
                    elif predicts:
                        x = torch.cat((flat(state), model.action_features(part(action))), -1)    # [T * m, 3w]
                        pm, plv = model.predict(x.unsqueeze(1))
                        prediction = reading_nll(model, pm.squeeze(1), plv.squeeze(1), part(target),
                                                 part(precise)).mean()
                dists = distributions(cl, al, part(allowed))
                taken = (part(action), part(measure), part(compute))
                new_logp = sum(d.log_prob(a) for d, a in zip(dists, taken))
                entropy = sum(d.entropy() for d in dists)
                ratio = torch.exp(new_logp - part(logp))
                a_t = part(adv)
                policy = -torch.min(ratio * a_t, ratio.clamp(1 - args.clip, 1 + args.clip) * a_t).mean()
                value_loss = F.mse_loss(v.float(), part(returns))
                loss = policy + 0.5 * value_loss - args.entropy * entropy.mean()
                if branched:
                    ok, bq, mask = part(allowed), part(q), part(valid)
                    probs = factor_probs(al, ok)
                    at = mask.any(1)
                    if bool(at.any()):                   # the policy and the head against the copies' returns
                        best = bq.masked_fill(~mask, -1e9)
                        regret = [((probs[:, f] * best[:, f] * mask[:, f]).sum(1) - best[:, f].max(1).values)[at]
                                  for f in FACTORS]
                        stats["branch_regret"] += float(sum(r.mean().detach() for r in regret)) * T
                        agree = [(probs[:, f].argmax(1) == best[:, f].argmax(1))[at].float().mean() for f in FACTORS]
                        stats["branch_agree"] += float(sum(agree) / 2) * T
                        branch_count += T
                        if model.activity_value is not None:
                            residual = (((estimate - bq) ** 2) * mask)[at]
                            fit = residual.sum() / mask[at].sum()
                            loss = loss + 0.5 * fit
                            stats["activity_value"] += float(fit.detach()) * T
                            if epoch == 0:
                                unseen[0] += float(residual.sum().detach())
                                unseen[1] += float(((bq ** 2) * mask)[at].sum())
                    if model.activity_value is not None and activity_weight > 0:   # every tick, from the estimates
                        loss = loss - activity_weight * expected_advantage(probs, estimate.detach())
                if predicts:
                    loss = loss + args.prediction * prediction
                    stats["prediction"] += float(prediction.detach()) * T
                if deployed:                                     # what the window's own steps did to its predictions
                    with torch.no_grad(), amp():
                        unchanged = online_nll(model, state, action[:, mb], As[:2], Bs[:2], version.clamp(max=1),
                                               target[:, mb], precise[:, mb])
                    w = keep.sum().clamp_min(1) * T
                    stats["adapt_before"] += float((unchanged * keep).sum() / w)
                    stats["adapt_after"] += float((nll.detach() * keep).sum() / w)
                    adapt_count += 1
                elif predicts and c0 == 0 and T >= 2:
                    with amp(), torch.set_grad_enabled(args.objective == "step"):
                        xs, ys, k = x.view(T, -1, x.shape[-1]).transpose(0, 1), target[c0:c1, mb].T, T // 2
                        ps = precise[c0:c1, mb].T
                        before, after = adaptation_loss(model, start.A[mb], start.B[mb], xs[:, :k], ys[:, :k],
                                                        xs[:, k:], ys[:, k:], args.adapter_lr,
                                                        differentiable=args.objective == "step",
                                                        support_p=ps[:, :k], query_p=ps[:, k:])
                    w = whole[mb]
                    after = (after.float() * w).sum() / w.sum().clamp_min(1)
                    if args.objective == "step":
                        loss = loss + args.meta * after
                    stats["adapt_before"] += float(((before.float() * w).sum() / w.sum().clamp_min(1)))
                    stats["adapt_after"] += float(after.detach())
                    adapt_count += 1
                (loss * T / L).backward()
                stats["policy"] += float(policy.detach()) * T
                stats["value"] += float(value_loss.detach()) * T
                stats["entropy"] += float(entropy.mean().detach()) * T
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
    count = args.epochs * math.ceil(n / args.minibatch) * L
    out = {k: v / count for k, v in stats.items()}
    for key in ("adapt_before", "adapt_after"):
        out[key] = stats[key] / max(adapt_count, 1)
    if branched:
        for key in ("activity_value", "branch_regret", "branch_agree"):
            out[key] = stats[key] / max(branch_count, 1)
        out["branch_spread"] = float(spread)
        if model.activity_value is not None:
            out["activity_value_r2"] = 1 - unseen[0] / max(unseen[1], 1e-12)
    return out


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=MODES, default="full")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dims", type=int, default=2, help="actuators of the instrument")
    p.add_argument("--levels", type=int, default=9, help="control levels per actuator")
    p.add_argument("--family", choices=FAMILIES, default="instrument", help="process family of the training lifetimes")
    p.add_argument("--thermal-share", type=float, default=0.5, help="share of thermal streams in a mixed fleet")
    p.add_argument("--streams", type=int, default=2048)
    p.add_argument("--window", type=int, default=128)
    p.add_argument("--chunk", type=int, default=128, help="backpropagation length through the fast state")
    p.add_argument("--lifetime", type=int, default=8192)
    p.add_argument("--interactions", type=float, default=3e7, help="external transitions over all streams")
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--slots", type=int, default=8)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--records", type=int, default=32)
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--adapter-blocks", type=int, default=1, help="adapter blocks weighed by a selector of the state")
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--action-width", type=int, default=256)
    p.add_argument("--buffer", type=int, default=64)
    p.add_argument("--adapter-lr", type=float, default=0.05)
    p.add_argument("--learn-ticks", type=int, default=2, help="ticks a learning step occupies the compute")
    p.add_argument("--every", type=int, default=8, help="fixed-schedule period")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--minibatch", type=int, default=64, help="streams per optimizer step")
    p.add_argument("--gamma", type=float, default=0.995)
    p.add_argument("--lam", type=float, default=0.95)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--entropy", type=float, default=0.003)
    p.add_argument("--prediction", type=float, default=0.5)
    p.add_argument("--meta", type=float, default=0.0, help="weight of the loss after one step, objective step")
    p.add_argument("--objective", choices=("shared", "step", "deployed"),
                   help="prediction objective; by default step when --meta is set, otherwise shared")
    p.add_argument("--learned-step", action="store_true", help="learn the size of a learning step")
    p.add_argument("--step-rule", choices=("normalized", "evidence"), default="normalized",
                   help="normalized: steps of one size; evidence: -eta g / (|g| + kappa) with eta, kappa learned")
    p.add_argument("--support", choices=("sample", "buffer"), default="sample",
                   help="a learning step reads 32 sampled entries or every filled entry of the buffer")
    p.add_argument("--precise-targets", action="store_true",
                   help="a returning precise reading replaces the cheap reading of the transition it measured")
    p.add_argument("--sensor-noise", action="store_true", help="likelihoods add a learned noise for each sensor")
    p.add_argument("--segment", type=int, default=1, help="external events the workspace is unrolled over in PPO")
    p.add_argument("--gated-workspace", action="store_true", help="refinements keep, revise or reset each slot")
    p.add_argument("--recompute-predictions", action="store_true",
                   help="PPO heads read predictions recomputed under the adapter in effect at each tick")
    p.add_argument("--search-share", type=float, default=0.0, help="share of requests for the largest output")
    p.add_argument("--cost-refinement", type=float, default=Costs.refinement)
    p.add_argument("--cost-measurement", type=float, default=Costs.measurement)
    p.add_argument("--cost-learning", type=float, default=Costs.learning)
    p.add_argument("--branch-streams", type=int, default=0, help="streams copied per window to value activities")
    p.add_argument("--branch-horizon", type=int, default=32, help="ticks each copy runs before its value bootstrap")
    p.add_argument("--branch-replicates", type=int, default=1, help="independent sets of copies averaged")
    p.add_argument("--branch-greedy", action="store_true", help="copies take the policy's most probable choices")
    p.add_argument("--branch-weight", type=float, default=1.0,
                   help="weight of the activity gradient from the activity-value estimates")
    p.add_argument("--activity-values", action="store_true", help="a head that values every activity option")
    p.add_argument("--init", type=Path, help="start from the weights of a trained model")
    p.add_argument("--fresh-activity", action="store_true", help="with --init, a newly initialised activity head")
    p.add_argument("--train-only", choices=("activity",), help="train only the activity and value heads")
    p.add_argument("--reward-scale", type=float, default=0.1, help="rewards are multiplied by this for value learning")
    p.add_argument("--save-every", type=int, default=50, help="windows between checkpoints")
    p.add_argument("--device", default="cuda")
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args(argv)
    args.objective = args.objective or ("step" if args.meta > 0 else "shared")
    if args.objective == "deployed" or args.recompute_predictions:
        if args.support != "buffer" or args.chunk < args.window:
            p.error("replaying learning steps needs --support buffer and one chunk per window")
    return args


def main(argv=None):
    args = parse_args(argv)
    torch.manual_seed(args.seed)
    configure_backends()
    device = torch.device(args.device)
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    specs = {"instrument": instrument(args), "thermal": thermal(args), "navigation": navigation(args)}
    spec = specs["instrument"]
    model = build(args, spec, device)
    if args.init:
        missing, unexpected = model.load_state_dict(torch.load(args.init, map_location=device)["model"], strict=False)
        assert not unexpected and all(k.startswith("activity_value.") for k in missing), (missing, unexpected)
        if args.fresh_activity:
            for layer in model.activity:
                if hasattr(layer, "reset_parameters"):
                    layer.reset_parameters()
    if args.train_only == "activity":
        for name, q in model.named_parameters():
            q.requires_grad_(name.startswith(("activity.", "activity_value.", "value.")))
    optimizer = torch.optim.Adam([q for q in model.parameters() if q.requires_grad], lr=args.lr, eps=1e-5)
    env = make_fleet(args.family, args.streams, args.seed + 1000, device, specs, args.thermal_share)
    agent = new_agent(model, args.streams, args.buffer, device)
    gen = torch.Generator(device=device).manual_seed(args.seed + 2000)
    agent.age = torch.randint(0, args.lifetime, (args.streams,), generator=gen, device=device)
    obs = first_observation(env)
    costs = Costs(args.cost_refinement, args.cost_measurement, args.cost_learning)
    meta = {"args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
            "config": asdict(model.config), **{k: asdict(v) for k, v in specs.items()},
            "parameters": sum(q.numel() for q in model.parameters()),
            "revision": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                                       text=True).stdout.strip() or None,
            "torch": torch.__version__, "python": platform.python_version(),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"}
    (out / "config.json").write_text(json.dumps(meta, indent=2))
    log = (out / "train_log.jsonl").open("a")
    ticks = int(args.interactions // args.streams)
    steps, window, start = [], 0, time.perf_counter()
    keys = ("reward", "error", "active", "refine", "measure", "learn", "prediction_error")
    sums = dict.fromkeys(keys, 0.0)

    def save():
        torch.save({"model": model.state_dict(), "config": asdict(model.config),
                    **{k: asdict(v) for k, v in specs.items()}, "args": meta["args"]}, out / "model.pt")

    window_start = previous = agent                     # the agent when the window began / before the last tick
    collection = update_time = 0.0
    branch_gen = torch.Generator(device=device).manual_seed(args.seed + 3000)
    branch_at = int(torch.randint(0, args.window, (1,), generator=branch_gen, device=device))
    branch, branch_transitions = None, 0
    explained = 0.0                                      # running share of fresh option values the head predicts
    mark = time.perf_counter()
    for t in range(ticks):
        model.eval()
        if args.branch_streams and len(steps) == branch_at:      # value each activity option on copies
            eligible = (agent.age + args.branch_horizon < args.lifetime).nonzero().squeeze(1)
            pick = torch.randperm(len(eligible), generator=branch_gen, device=device)[:args.branch_streams]
            rows = eligible[pick].sort().values
            values = [branch_values(model, agent, env, obs, rows, args.mode, costs, args.adapter_lr, args.every,
                                    args.branch_horizon, args.gamma, args.reward_scale,
                                    seed=(args.seed * 100003 + t) * 16 + k, greedy=args.branch_greedy,
                                    learn_ticks=args.learn_ticks, support=args.support,
                                    precise_targets=args.precise_targets) for k in range(args.branch_replicates)]
            branch = (rows, torch.stack([q for q, _ in values]).mean(0), values[0][1])
            branch_transitions += args.branch_replicates * len(rows) * OPTIONS * (args.branch_horizon + 1)
        previous = agent                                 # tick() replaces the agent's tensors rather than editing them
        agent, obs, step = tick(model, agent, env, obs, args.mode, costs, args.adapter_lr, args.every,
                                learn_ticks=args.learn_ticks, support=args.support,
                                precise_targets=args.precise_targets)
        done = agent.age >= args.lifetime
        step["done"] = done
        if args.branch_streams:
            step["branch_q"] = torch.zeros(args.streams, OPTIONS, device=device)
            step["branch_valid"] = torch.zeros(args.streams, OPTIONS, dtype=torch.bool, device=device)
            if branch is not None:
                rows, q, valid = branch
                step["branch_q"][rows], step["branch_valid"][rows] = q, valid
                branch = None
        sums["reward"] += float(step["reward"].mean())
        sums["error"] += float(step["error"].sum())
        sums["active"] += float(step["active"].float().sum())
        sums["refine"] += float(step["counts"].float().mean())
        sums["measure"] += float(step["accepted"].float().mean())
        sums["learn"] += float(step["learn"].float().mean())
        sums["prediction_error"] += float(step["prediction_error"].mean())
        if done.any():
            env.reset(done)
            agent = reset_agents(model, agent, done)
            obs = clear_observation(obs, done)
        steps.append(step)
        attach_returns(steps, env.spec.delay)
        if len(steps) == args.window + 1:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            collection += time.perf_counter() - mark
            mark = time.perf_counter()
            model.train()
            weight = args.branch_weight * min(max(explained, 0.0), 1.0)
            stats = ppo_update(model, optimizer, steps[:-1], steps[-1]["value"], window_start, args, weight)
            if "activity_value_r2" in stats:
                explained = 0.9 * explained + 0.1 * stats["activity_value_r2"]
                stats["activity_weight"] = weight
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            update_time += time.perf_counter() - mark
            mark = time.perf_counter()
            steps, window_start = steps[-1:], previous
            branch_at = int(torch.randint(1, args.window + 1, (1,), generator=branch_gen, device=device))
            window += 1
            k = args.window
            row = {"window": window, "transitions": (t + 1) * args.streams,
                   **({"branch_transitions": branch_transitions} if args.branch_streams else {}),
                   "reward_per_100_ticks": 100 * sums["reward"] / k,
                   "error_while_active": sums["error"] / max(sums["active"], 1),
                   "refinements_per_tick": sums["refine"] / k, "measurements_per_100_ticks": 100 * sums["measure"] / k,
                   "learning_per_100_ticks": 100 * sums["learn"] / k,
                   "prediction_error": sums["prediction_error"] / k,
                   "workspace_norm": float(agent.z.norm(dim=-1).mean()), "adapter_norm": float(agent.A.norm(dim=(1, 2)).mean()),
                   **({"step_eta": float(model.log_step.exp())} if model.log_step is not None else {}),
                   **({"step_kappa": float(model.log_kappa.exp())} if model.log_kappa is not None else {}),
                   **({"noise_cheap": float(model.log_noise[0].exp()), "noise_precise": float(model.log_noise[1].exp())}
                      if model.log_noise is not None else {}),
                   **{f"loss_{a}": round(b, 5) for a, b in stats.items()},
                   "seconds": round(time.perf_counter() - start, 1), "collection_seconds": round(collection, 1),
                   "update_seconds": round(update_time, 1)}
            if device.type == "cuda":
                row["peak_memory_gb"] = round(torch.cuda.max_memory_allocated(device) / 1e9, 2)
            sums = dict.fromkeys(keys, 0.0)
            log.write(json.dumps(row) + "\n")
            log.flush()
            if window % 10 == 1:
                print(json.dumps({a: (round(b, 4) if isinstance(b, float) else b) for a, b in row.items()}), flush=True)
            if window % args.save_every == 0:
                save()
    save()
    meta["train_seconds"] = round(time.perf_counter() - start, 1)
    (out / "config.json").write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()

