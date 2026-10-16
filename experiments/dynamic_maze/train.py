#!/usr/bin/env python3
"""Train one latent planner on changing room mazes.

Every tick the world may change; the planner observes the new map and goal,
continues its computation according to its mode, refines ``k`` times, and is
supervised after every refinement on the optimal action and distance of every
free cell. Gradients run through the whole episode, so the shared block learns
to repair computation after changes.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
import threading
import queue
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from cdm.maze import MazeSpec, distance_maps, make_batch, optimal_actions  # noqa: E402
from cdm.planner import MODES, PlannerConfig, build_planner  # noqa: E402

DISTANCE_SCALE = 32.0


def labels(blocked: torch.Tensor, goal: torch.Tensor):
    """Distances and optimal-action targets for [B, T, H, W] maps."""
    b, t, h, w = blocked.shape
    dist = distance_maps(blocked.flatten(0, 1), goal.flatten(0, 1))
    opt = optimal_actions(dist, blocked.flatten(0, 1))
    return dist.view(b, t, h, w), opt.view(b, t, 5, h, w)


def step_losses(logits, distance, opt, dist, free):
    """Soft cross-entropy over optimal actions and distance regression on free cells."""
    target = opt.float() / opt.sum(1, keepdim=True).clamp_min(1)
    action = -(target * F.log_softmax(logits.float(), 1)).sum(1)
    count = free.sum().clamp_min(1)
    action = (action * free).sum() / count
    d_target = torch.where(free, dist / DISTANCE_SCALE, torch.zeros_like(dist))
    d_loss = (F.smooth_l1_loss(distance.float(), d_target, reduction="none") * free).sum() / count
    correct = (opt.gather(1, logits.argmax(1, keepdim=True)).squeeze(1) & free).sum() / count
    return action, d_loss, correct


def episode_loss(model, blocked, goal, think, distance_weight):
    dist, opt = labels(blocked, goal)
    free = ~blocked
    state = model.start(blocked[:, 0], goal[:, 0])
    total = torch.zeros((), device=blocked.device)
    accuracy, count = 0.0, 0
    logits, distance = model.readout_state(state)
    a, d, c = step_losses(logits, distance, opt[:, 0], dist[:, 0], free[:, 0])
    total = total + a + distance_weight * d
    count += 1
    for t in range(1, blocked.shape[1]):
        state = model.observe(state, blocked[:, t], goal[:, t])
        state, outputs = model.think(state, think)
        for logits, distance in outputs:
            a, d, c = step_losses(logits, distance, opt[:, t], dist[:, t], free[:, t])
            total = total + a + distance_weight * d
            count += 1
        accuracy += float(c)
    return total / count, accuracy / (blocked.shape[1] - 1)


class Batches:
    """Background producer of numpy episode batches."""

    def __init__(self, seed, spec, regimes, batch, ticks, depth=6, with_start=False):
        self.rng = np.random.default_rng(seed)
        self.spec, self.regimes, self.batch, self.ticks = spec, regimes, batch, ticks
        self.with_start = with_start
        self.queue = queue.Queue(depth)
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        while True:
            chosen = [self.regimes[i] for i in self.rng.integers(len(self.regimes), size=self.batch)]
            self.queue.put(make_batch(self.rng, self.spec, chosen, self.ticks))

    def next(self, device):
        b = self.queue.get()
        out = (torch.from_numpy(b["blocked"]).to(device, non_blocking=True),
               torch.from_numpy(b["goal"]).to(device, non_blocking=True))
        return out + (torch.from_numpy(b["start"]).to(device),) if self.with_start else out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=MODES, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--batch", type=int, default=48)
    p.add_argument("--ticks", type=int, default=32)
    p.add_argument("--rooms", type=int, default=5)
    p.add_argument("--channels", type=int, default=96)
    p.add_argument("--memory-size", type=int, default=8)
    p.add_argument("--warmup", type=int, default=16)
    p.add_argument("--no-goal-broadcast", action="store_true")
    p.add_argument("--no-reseed", action="store_true")
    p.add_argument("--map-memory", type=int, default=0)
    p.add_argument("--goal-read", action="store_true")
    p.add_argument("--change-gate", action="store_true")
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
    spec = MazeSpec(args.rooms)
    config = PlannerConfig(mode=args.mode, channels=args.channels, memory_size=args.memory_size,
                           warmup=args.warmup, goal_broadcast=not args.no_goal_broadcast,
                           reseed=not args.no_reseed, map_memory=args.map_memory, goal_read=args.goal_read,
                           change_gate=args.change_gate)
    model = build_planner(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    warm = max(1, args.steps // 20)
    schedule = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / args.steps))))
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    meta = {"args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
            "config": asdict(config), "side": spec.side,
            "parameters": sum(p.numel() for p in model.parameters()),
            "torch": torch.__version__, "python": platform.python_version(),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"}
    (out / "config.json").write_text(json.dumps(meta, indent=2))

    data = Batches(args.seed + 10_000, spec, args.regimes, args.batch, args.ticks)
    think_rng = np.random.default_rng(args.seed + 20_000)
    log = (out / "train_log.jsonl").open("w")
    start = time.perf_counter()
    model.train()
    running = []
    for step in range(args.steps):
        blocked, goal = data.next(device)
        think = int(think_rng.choice(args.think))
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            loss, accuracy = episode_loss(model, blocked, goal, think, args.distance_weight)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        schedule.step()
        running.append((float(loss.detach()), accuracy))
        if (step + 1) % args.log_every == 0 or step == 0:
            l, a = np.mean(running, 0)
            running = []
            row = {"step": step + 1, "loss": round(float(l), 5), "map_accuracy": round(float(a), 5),
                   "think": think, "grad_norm": round(float(norm), 4), "lr": schedule.get_last_lr()[0],
                   "seconds": round(time.perf_counter() - start, 1)}
            log.write(json.dumps(row) + "\n")
            log.flush()
            print(json.dumps(row), flush=True)
    torch.save({"config": asdict(config), "rooms": args.rooms, "state_dict": model.state_dict()},
               out / "model.pt")
    meta["train_seconds"] = round(time.perf_counter() - start, 1)
    (out / "config.json").write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
