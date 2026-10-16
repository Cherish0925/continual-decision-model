#!/usr/bin/env python3
"""Train a planner that sees the map only around a moving agent.

A scripted agent walks each training episode: it steps along a shortest path
on the map it currently believes (with probability ``--explore`` a random
step), sees the true map in a window around itself, and receives the whole map
with probability ``--survey`` per tick. The planner reads that belief, the goal
and, with ``--age-input``, how many ticks ago each cell was last seen. Targets
are the optimal actions and distances on the true map, so the planner learns
which routes are still likely to be open where its map is old.
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

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cdm.maze import MazeSpec, distance_maps, move, optimal_actions  # noqa: E402
from cdm.planner import MODES, PlannerConfig, build_planner  # noqa: E402
from train import Batches, labels, step_losses  # noqa: E402

AGE_SCALE = math.log1p(64.0)


def age_plane(age: torch.Tensor) -> torch.Tensor:
    return (torch.log1p(age.float()) / AGE_SCALE).clamp(max=1.0)


def window(pos, h, w, radius):
    r, c = pos // w, pos % w
    rows = torch.arange(h, device=pos.device)
    cols = torch.arange(w, device=pos.device)
    return (((rows[None, :] - r[:, None]).abs() <= radius)[:, :, None]
            & ((cols[None, :] - c[:, None]).abs() <= radius)[:, None, :])


@torch.no_grad()
def beliefs(blocked, goal, start, radius, survey, explore, generator):
    """Belief maps [B, T+1, H, W] and ages [B, T+1, H, W] along a scripted agent's walk."""
    b, t1, h, w = blocked.shape
    belief = blocked[:, 0].clone()
    age = torch.zeros(b, h, w, device=blocked.device)
    pos = start.clone()
    out_b, out_a = [belief], [age]
    for t in range(1, t1):
        seen = window(pos, h, w, radius) | (torch.rand(b, 1, 1, device=blocked.device, generator=generator) < survey)
        belief = torch.where(seen, blocked[:, t], belief)
        age = torch.where(seen, torch.zeros_like(age), age + 1)
        out_b.append(belief)
        out_a.append(age)
        dist = distance_maps(belief, goal[:, t])
        best = optimal_actions(dist, belief).float().view(b, 5, -1)
        here = best.gather(2, pos.view(b, 1, 1).expand(b, 5, 1)).squeeze(2)
        action = torch.multinomial(here + 1e-6, 1, generator=generator).squeeze(1)
        random = torch.randint(0, 5, (b,), device=blocked.device, generator=generator)
        action = torch.where(torch.rand(b, device=blocked.device, generator=generator) < explore, random, action)
        pos = move(blocked[:, t], pos, action)
    return torch.stack(out_b, 1), torch.stack(out_a, 1)


def episode_loss(model, belief, age, true, goal, think, distance_weight, use_age):
    dist, opt = labels(true, goal)
    free = ~true
    a = (lambda t: age_plane(age[:, t])) if use_age else (lambda t: None)
    state = model.start(belief[:, 0], goal[:, 0], a(0))
    total, count, accuracy = torch.zeros((), device=true.device), 0, 0.0
    logits, distance = model.readout_state(state)
    la, ld, _ = step_losses(logits, distance, opt[:, 0], dist[:, 0], free[:, 0])
    total, count = total + la + distance_weight * ld, 1
    for t in range(1, true.shape[1]):
        state = model.observe(state, belief[:, t], goal[:, t], a(t))
        state, outputs = model.think(state, think)
        for logits, distance in outputs:
            la, ld, c = step_losses(logits, distance, opt[:, t], dist[:, t], free[:, t])
            total, count = total + la + distance_weight * ld, count + 1
        accuracy += float(c)
    return total / count, accuracy / (true.shape[1] - 1)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=MODES, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch", type=int, default=48)
    p.add_argument("--ticks", type=int, default=32)
    p.add_argument("--rooms", type=int, default=5)
    p.add_argument("--channels", type=int, default=96)
    p.add_argument("--goal-read", action="store_true")
    p.add_argument("--age-input", action="store_true")
    p.add_argument("--radius", type=int, default=2)
    p.add_argument("--survey", type=float, default=1 / 12)
    p.add_argument("--explore", type=float, default=0.1)
    p.add_argument("--think", type=int, nargs="+", default=[1, 2, 4])
    p.add_argument("--regimes", nargs="+", default=["full_return", "partial_return", "continuous"])
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--distance-weight", type=float, default=0.5)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--device", default="cuda")
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args(argv)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device(args.device)
    config = PlannerConfig(mode=args.mode, channels=args.channels, memory_size=0, goal_read=args.goal_read,
                           age_input=args.age_input)
    model = build_planner(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    warm = max(1, args.steps // 20)
    schedule = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / args.steps))))
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    meta = {"args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
            "config": asdict(config), "side": MazeSpec(args.rooms).side,
            "parameters": sum(q.numel() for q in model.parameters()), "torch": torch.__version__,
            "python": platform.python_version(), "gpu": torch.cuda.get_device_name(device)}
    (out / "config.json").write_text(json.dumps(meta, indent=2))
    data = Batches(args.seed + 10_000, MazeSpec(args.rooms), args.regimes, args.batch, args.ticks, with_start=True)
    think_rng = np.random.default_rng(args.seed + 20_000)
    generator = torch.Generator(device=device).manual_seed(args.seed + 30_000)
    log = (out / "train_log.jsonl").open("w")
    start = time.perf_counter()
    running = []
    for step in range(args.steps):
        true, goal, first = data.next(device)
        belief, age = beliefs(true, goal, first, args.radius, args.survey, args.explore, generator)
        think = int(think_rng.choice(args.think))
        with torch.autocast(device.type, dtype=torch.bfloat16):
            loss, accuracy = episode_loss(model, belief, age, true, goal, think, args.distance_weight, args.age_input)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        schedule.step()
        running.append((float(loss.detach()), accuracy))
        if (step + 1) % args.log_every == 0 or step == 0:
            l, a = np.mean(running, 0)
            running = []
            row = {"step": step + 1, "loss": round(float(l), 5), "map_accuracy": round(float(a), 5), "think": think,
                   "grad_norm": round(float(norm), 4), "seconds": round(time.perf_counter() - start, 1)}
            log.write(json.dumps(row) + "\n")
            log.flush()
            print(json.dumps(row), flush=True)
    torch.save({"config": asdict(config), "rooms": args.rooms, "state_dict": model.state_dict(),
                "partial": {"radius": args.radius, "age_input": args.age_input}}, out / "model.pt")
    meta["train_seconds"] = round(time.perf_counter() - start, 1)
    (out / "config.json").write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
