#!/usr/bin/env python3
"""Keep training a deployed planner on the mazes it meets.

A planner trained on 5x5 rooms is deployed on 9x9 rooms. Each deployment batch
is a set of parallel episodes the agent runs with one refinement per tick;
after it the learner takes gradient steps on that batch. Targets come from the
exact solver run on the observed maps (``bfs``), from the learner's own
readout 16 refinements deeper than its current state (``self``), or from a
frozen copy of the deployed network that runs the same episode and reads out
16 refinements deeper (``frozen``). Fixed held-out episodes measure adaptation
on 9x9 and retention on 5x5.

Conditions: ``frozen`` (no updates), ``online-new`` (updates on 9x9 batches),
``online-replay`` (each update mixes half 9x9 and half 5x5 training episodes).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cdm.maze import MazeSpec  # noqa: E402
from evaluate import episodes, load, prepare, rollout  # noqa: E402
from train import Batches, episode_loss  # noqa: E402

REGIMES = ("full_return", "partial_return", "continuous")
TEACHERS = ("bfs", "self", "frozen")
CONDITIONS = ("frozen", "online-new", "online-replay")


def deep_episode_loss(model, blocked, goal, think, distance_weight, depth=16, teacher=None):
    """Match every refinement's readout to a readout ``depth`` refinements deeper.

    Without ``teacher`` the deeper readout continues the learner's own state with its current weights;
    with a frozen ``teacher`` it continues the teacher's own state over the same episode.
    """
    free = ~blocked
    state = model.start(blocked[:, 0], goal[:, 0])
    total = torch.zeros((), device=blocked.device)
    count = 0
    t_state = None

    def deeper(state, t):
        nonlocal t_state
        if teacher is None:
            return model.readout_state(model.advance(state.detach(), depth))
        t_state = teacher.start(blocked[:, 0], goal[:, 0]) if t == 0 else \
            teacher.advance(teacher.observe(t_state, blocked[:, t], goal[:, t]), think)
        return teacher.readout_state(teacher.advance(t_state, depth))

    def score(outputs, state, t):
        nonlocal total, count
        mask = free[:, t]
        with torch.no_grad():
            logits_t, distance_t = deeper(state, t)
            target = torch.softmax(logits_t.float(), 1)
        n = mask.sum().clamp_min(1)
        for logits, distance in outputs:
            action = (-(target * F.log_softmax(logits.float(), 1)).sum(1) * mask).sum() / n
            d = (F.smooth_l1_loss(distance.float(), distance_t.float(), reduction="none") * mask).sum() / n
            total = total + action + distance_weight * d
            count += 1

    score([model.readout_state(state)], state, 0)
    for t in range(1, blocked.shape[1]):
        state = model.observe(state, blocked[:, t], goal[:, t])
        state, outputs = model.think(state, think)
        score(outputs, state, t)
    return total / count


def update_loss(model, batches, teacher, think, distance_weight, frozen=None, depth=16):
    losses = []
    for blocked, goal in batches:
        if teacher == "bfs":
            losses.append(episode_loss(model, blocked, goal, think, distance_weight)[0])
        else:
            losses.append(deep_episode_loss(model, blocked, goal, think, distance_weight, depth,
                                            frozen if teacher == "frozen" else None))
    return sum(losses) / len(losses)


def measure(model, worlds, device):
    out = {}
    for (rooms, regime), world in worlds.items():
        r = rollout(model, world, 1, device)
        out[f"{rooms}:{regime}"] = {k: float(np.mean(r[k])) for k in ("agent_optimal", "return", "success",
                                                                       "map_accuracy")}
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--condition", choices=CONDITIONS, required=True)
    p.add_argument("--teacher", choices=TEACHERS, default="bfs")
    p.add_argument("--updates", type=int, default=400)
    p.add_argument("--steps-per-batch", type=int, default=1)
    p.add_argument("--micro-batch", type=int, default=32, help="episodes per backward pass; gradients accumulate")
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--ticks", type=int, default=32)
    p.add_argument("--rooms", type=int, default=9)
    p.add_argument("--old-rooms", type=int, default=5)
    p.add_argument("--think", type=int, nargs="+", default=[1])
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--distance-weight", type=float, default=0.5)
    p.add_argument("--depth", type=int, default=16, help="self teacher: extra refinements")
    p.add_argument("--eval-every", type=int, default=20)
    p.add_argument("--eval-episodes", type=int, default=512)
    p.add_argument("--eval-ticks", type=int, default=96)
    p.add_argument("--eval-regimes", nargs="+", default=["partial_return", "continuous"])
    p.add_argument("--stream-seed", type=int, default=7000)
    p.add_argument("--eval-seed", type=int, default=909090)
    p.add_argument("--device", default="cuda")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args(argv)

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device(args.device)
    torch.manual_seed(args.stream_seed)
    model = load(args.model, device)
    frozen = load(args.model, device).requires_grad_(False) if args.teacher == "frozen" else None
    worlds = {(rooms, regime): prepare(episodes(rooms, regime, args.eval_episodes, args.eval_ticks,
                                                args.eval_seed + 1000 * rooms + 100 * i), device)
              for rooms in (args.rooms, args.old_rooms) for i, regime in enumerate(args.eval_regimes)}
    record = {"args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
              "curve": [{"updates": 0, **measure(model, worlds, device)}], "stream": []}
    if args.condition != "frozen":
        new = Batches(args.stream_seed, MazeSpec(args.rooms), list(REGIMES), args.batch, args.ticks)
        old = Batches(args.stream_seed + 1, MazeSpec(args.old_rooms), list(REGIMES), args.batch // 2, args.ticks) \
            if args.condition == "online-replay" else None
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        think_rng = np.random.default_rng(args.stream_seed + 1)
        amp = dict(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda")
        seconds = 0.0
        for update in range(1, args.updates + 1):
            blocked, goal = new.next(device)
            batches = [(blocked, goal)]
            if old is not None:
                half = args.batch // 2
                batches = [(blocked[:half], goal[:half]), old.next(device)]
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            begin = time.perf_counter()
            chunks = [(b[i:i + args.micro_batch], g[i:i + args.micro_batch])
                      for b, g in batches for i in range(0, len(b), args.micro_batch)]
            model.train()
            for _ in range(args.steps_per_batch):
                think = int(think_rng.choice(args.think))
                optimizer.zero_grad(set_to_none=True)
                loss = 0.0
                for chunk in chunks:
                    with torch.autocast(**amp):
                        part = update_loss(model, [chunk], args.teacher, think, args.distance_weight, frozen,
                                           args.depth) / len(chunks)
                    part.backward()
                    loss += float(part.detach())
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            model.eval()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            seconds += time.perf_counter() - begin
            record["stream"].append({"updates": update, "loss": round(loss, 5)})
            if update % args.eval_every == 0 or update == args.updates:
                row = {"updates": update, "seconds_per_update": seconds / update, **measure(model, worlds, device)}
                record["curve"].append(row)
                print(json.dumps({"updates": update, "loss": round(loss, 4),
                                  **{k: round(v["agent_optimal"], 4) for k, v in row.items() if isinstance(v, dict)}}),
                      flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=1))


if __name__ == "__main__":
    main()
