#!/usr/bin/env python3
"""Train the continual model over instrument lifetimes with recurrent PPO.

Streams run lifetimes of ``--lifetime`` ticks with staggered starts. Fast
state, workspace, experience records, instance adapters and learning buffers
persist across training windows; only the end of a lifetime resets them. Every
``--window`` ticks the shared parameters take PPO steps on the window, with the
fast state and workspace recomputed in chunks from stored states. Value targets
bootstrap from the next decision state; only a lifetime's end is terminal. The
shared predictor is also trained on received readings. With ``--meta`` it is
also trained so that one learning step on the first half of a window, taken
from the adapter the instance held when the window began, improves its
predictions on the second half.
"""
from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import time
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cdm.core import CoreConfig, ContinualCore, adaptation_loss, gaussian_nll  # noqa: E402
from cdm.instrument import InstrumentFleet, InstrumentSpec, action_values  # noqa: E402
from runtime import (MODES, SCALAR_DIM, Costs, configure_backends, distributions, feature_dims,  # noqa: E402
                     new_agent, reset_agents, tick)


def instrument(args) -> InstrumentSpec:
    return InstrumentSpec(dims=args.dims, levels=args.levels)


def build(args, spec: InstrumentSpec, device) -> ContinualCore:
    event_dim, record_dim = feature_dims(spec)
    config = CoreConfig(event_dim=event_dim, record_dim=record_dim, actions=spec.actions, action_dim=spec.dims,
                        width=args.width, slots=args.slots, heads=args.heads, records=args.records, rank=args.rank,
                        hidden=args.hidden, action_width=args.action_width, scalar_dim=SCALAR_DIM)
    return ContinualCore(config, action_values(spec)).to(device)


def first_observation(env: InstrumentFleet):
    n = env.n
    return env.step(torch.full((n,), env.spec.actions // 2, dtype=torch.long, device=env.device),
                    torch.zeros(n, dtype=torch.bool, device=env.device))


def clear_observation(obs: dict, mask: torch.Tensor) -> dict:
    def clear(v):
        if v.dtype == torch.bool:
            return v & ~mask.view(-1, *([1] * (v.dim() - 1)))
        return torch.where(mask.view(-1, *([1] * (v.dim() - 1))), torch.zeros_like(v), v)
    return {k: clear(v) for k, v in obs.items()}


def ppo_update(model, optimizer, steps, bootstrap, start, args):
    """PPO on one window. ``start`` holds the adapters each stream had when the window began."""
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
    offset, logvar, target = S("offset"), S("logvar"), S("target")
    action, measure, compute, counts, logp = S("action"), S("measure"), S("compute"), S("counts"), S("logp")
    h_prev, z_prev = S("h_prev"), S("z_prev")
    n = rewards.shape[1]
    device = rewards.device
    stats = {"policy": 0.0, "value": 0.0, "entropy": 0.0, "prediction": 0.0, "adapt_before": 0.0, "adapt_after": 0.0}
    whole = 1.0 - dones.amax(0)                          # streams whose lifetime runs through the whole window
    adapt_count = 0
    flat = lambda a: a.flatten(0, 1)
    for _ in range(args.epochs):
        perm = torch.randperm(n, device=device)
        for i in range(0, n, args.minibatch):
            mb = perm[i:i + args.minibatch]
            for c0 in range(0, L, args.chunk):
                c1 = min(c0 + args.chunk, L)
                T = c1 - c0
                part = lambda a: flat(a[c0:c1, mb])
                with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    H = flat(model.fast_sequence(h_prev[c0, mb], event[c0:c1, mb]))   # backpropagation through the chunk
                    zin = z_prev[c0:c1, mb].float()              # workspace before each tick's refinement
                    zout = checkpoint(model.refine, flat(zin), H, part(records), part(counts),
                                      use_reentrant=False).view_as(zin)
                    # tick t reads the workspace completed at t-1: recomputed (with gradient) inside the chunk
                    zread = flat(torch.cat((zin[:1], zout[:-1]), 0))
                    o, lv = part(offset), part(logvar)
                    cl, al, v = model.heads(model.trunk(H, zread, o, lv, part(scalars)), o, lv)
                    if args.mode != "recurrent":
                        x = torch.cat((model.state_features(H, zread), model.action_features(part(action))), -1)
                        pm, plv = model.predict(x.unsqueeze(1))
                dists = distributions(cl, al, part(allowed))
                taken = (part(action), part(measure), part(compute))
                new_logp = sum(d.log_prob(a) for d, a in zip(dists, taken))
                entropy = sum(d.entropy() for d in dists)
                ratio = torch.exp(new_logp - part(logp))
                a_t = part(adv)
                policy = -torch.min(ratio * a_t, ratio.clamp(1 - args.clip, 1 + args.clip) * a_t).mean()
                value_loss = F.mse_loss(v.float(), part(returns))
                loss = policy + 0.5 * value_loss - args.entropy * entropy.mean()
                if args.mode != "recurrent":
                    prediction = gaussian_nll(pm.float().squeeze(1), plv.float().squeeze(1), part(target)).mean()
                    loss = loss + args.prediction * prediction
                    stats["prediction"] += float(prediction.detach()) * T
                if args.mode != "recurrent" and c0 == 0 and T >= 2:
                    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"), \
                            torch.set_grad_enabled(args.meta > 0):
                        xs, ys, k = x.view(T, -1, x.shape[-1]).transpose(0, 1), target[c0:c1, mb].T, T // 2
                        before, after = adaptation_loss(model, start[0][mb], start[1][mb], xs[:, :k], ys[:, :k],
                                                        xs[:, k:], ys[:, k:], args.adapter_lr,
                                                        differentiable=args.meta > 0)
                    w = whole[mb]
                    after = (after.float() * w).sum() / w.sum().clamp_min(1)
                    if args.meta > 0:
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
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=MODES, default="full")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dims", type=int, default=2, help="actuators of the instrument")
    p.add_argument("--levels", type=int, default=9, help="control levels per actuator")
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
    p.add_argument("--meta", type=float, default=0.0, help="weight of the prediction loss after a learning step")
    p.add_argument("--reward-scale", type=float, default=0.1, help="rewards are multiplied by this for value learning")
    p.add_argument("--save-every", type=int, default=50, help="windows between checkpoints")
    p.add_argument("--device", default="cuda")
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args(argv)
    torch.manual_seed(args.seed)
    configure_backends()
    device = torch.device(args.device)
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    spec = instrument(args)
    model = build(args, spec, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, eps=1e-5)
    env = InstrumentFleet(args.streams, spec, seed=args.seed + 1000, device=device)
    agent = new_agent(model, args.streams, args.buffer, device)
    gen = torch.Generator(device=device).manual_seed(args.seed + 2000)
    agent.age = torch.randint(0, args.lifetime, (args.streams,), generator=gen, device=device)
    obs = first_observation(env)
    costs = Costs()
    meta = {"args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
            "config": asdict(model.config), "instrument": asdict(spec),
            "parameters": sum(q.numel() for q in model.parameters()),
            "torch": torch.__version__, "python": platform.python_version(),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"}
    (out / "config.json").write_text(json.dumps(meta, indent=2))
    log = (out / "train_log.jsonl").open("a")
    ticks = int(args.interactions // args.streams)
    steps, window, start = [], 0, time.perf_counter()
    keys = ("reward", "error", "active", "refine", "measure", "learn", "prediction_error")
    sums = dict.fromkeys(keys, 0.0)

    def save():
        torch.save({"model": model.state_dict(), "config": asdict(model.config), "instrument": asdict(spec),
                    "args": meta["args"]}, out / "model.pt")

    window_adapters = previous_adapters = (agent.A, agent.B)   # when the window began / before the last tick
    for t in range(ticks):
        model.eval()
        previous_adapters = (agent.A, agent.B)           # tick() replaces these tensors rather than editing them
        agent, obs, step = tick(model, agent, env, obs, args.mode, costs, args.adapter_lr, args.every,
                                learn_ticks=args.learn_ticks)
        done = agent.age >= args.lifetime
        step["done"] = done
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
        if len(steps) == args.window + 1:
            model.train()
            stats = ppo_update(model, optimizer, steps[:-1], steps[-1]["value"], window_adapters, args)
            steps, window_adapters = steps[-1:], previous_adapters
            window += 1
            k = args.window
            row = {"window": window, "transitions": (t + 1) * args.streams,
                   "reward_per_100_ticks": 100 * sums["reward"] / k,
                   "error_while_active": sums["error"] / max(sums["active"], 1),
                   "refinements_per_tick": sums["refine"] / k, "measurements_per_100_ticks": 100 * sums["measure"] / k,
                   "learning_per_100_ticks": 100 * sums["learn"] / k,
                   "prediction_error": sums["prediction_error"] / k,
                   "workspace_norm": float(agent.z.norm(dim=-1).mean()), "adapter_norm": float(agent.A.norm(dim=(1, 2)).mean()),
                   **{f"loss_{a}": round(b, 5) for a, b in stats.items()},
                   "seconds": round(time.perf_counter() - start, 1)}
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
