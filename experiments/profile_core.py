"""Measure complete neural-core steps with fixed-size persistent state."""

import argparse
from dataclasses import asdict, fields
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cdm import CDMConfig, ContinualDecisionModel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--hidden-dim', type=int, default=64)
    parser.add_argument('--slots', type=int, default=4)
    parser.add_argument('--memory-size', type=int, default=8)
    parser.add_argument('--refinements', type=int, default=2)
    parser.add_argument('--ticks', type=int, default=200)
    parser.add_argument('--warmup', type=int, default=20)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--output', type=Path, default=Path('runs/core-profile.json'))
    args = parser.parse_args()
    if args.ticks < 1 or args.warmup < 0:
        parser.error('ticks must be positive and warmup nonnegative')
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    device = torch.device(args.device)
    config = CDMConfig(observation_dim=32, action_dim=5, goal_dim=4,
                       environment_dim=8, hidden_dim=args.hidden_dim,
                       slots=args.slots, memory_size=args.memory_size)
    model = ContinualDecisionModel(config).to(device).eval()
    total = args.warmup + args.ticks
    inputs = [torch.randn(total, args.batch_size, dim, device=device)
              for dim in (32, 4, 8)]

    def synchronize():
        if device.type == 'cuda':
            torch.cuda.synchronize(device)

    rows = []
    with torch.inference_mode():
        for mode in ('restart', 'persistent', 'memory'):
            state = model.initialize(args.batch_size)
            durations = []
            if device.type == 'cuda':
                torch.cuda.reset_peak_memory_stats(device)
            for tick in range(total):
                synchronize()
                started = time.perf_counter_ns()
                state = model.assimilate(state, *(value[tick] for value in inputs))
                if mode == 'restart':
                    state = model.reset_workspace(state)
                if mode == 'memory':
                    state, _ = model.read_memory(state)
                state = model.refine(state, args.refinements)
                output = model.readout(state)
                if mode == 'memory':
                    state = model.write_memory(state)
                synchronize()
                elapsed_ms = (time.perf_counter_ns() - started) / 1e6
                if tick >= args.warmup:
                    durations.append(elapsed_ms)
            state_bytes = sum(getattr(state, f.name).numel() * getattr(state, f.name).element_size()
                              for f in fields(state))
            rows.append(dict(mode=mode, p50_ms=float(np.percentile(durations, 50)),
                             p95_ms=float(np.percentile(durations, 95)),
                             p99_ms=float(np.percentile(durations, 99)),
                             mean_ms=float(np.mean(durations)), state_bytes=state_bytes,
                             peak_cuda_allocated_bytes=(torch.cuda.max_memory_allocated(device)
                                                        if device.type == 'cuda' else None),
                             duration_ms=durations))
    report = dict(config=asdict(config), arguments={k: str(v) if isinstance(v, Path) else v
                                                   for k, v in vars(args).items()},
                  parameter_count=sum(p.numel() for p in model.parameters()),
                  python=platform.python_version(), torch=torch.__version__,
                  numpy=np.__version__, device=str(device),
                  hardware=(torch.cuda.get_device_name(device) if device.type == 'cuda'
                            else platform.processor() or platform.machine()),
                  measurement='Untrained weights; complete batch step including memory access. '
                              'Inputs already reside on device. No sensor, environment, or actuator latency.',
                  results=rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    for row in rows:
        print(f"{row['mode']:12s} p50={row['p50_ms']:.3f} ms p95={row['p95_ms']:.3f} ms "
              f"state={row['state_bytes']} bytes")
    print(args.output)


if __name__ == '__main__':
    main()
