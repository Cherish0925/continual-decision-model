"""Run independent training seeds across devices; keep final-family scoring separate."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys

from .study import LEARNED, REFERENCES


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=('plan', 'train-dev', 'test'))
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--devices', default='cuda:0,cuda:1')
    p.add_argument('--seeds', default='101,102,103,104,105,106,107,108')
    p.add_argument('--modes', default=','.join(LEARNED))
    p.add_argument('--width', type=int, default=128)
    p.add_argument('--slots', type=int, default=8)
    p.add_argument('--batch', type=int, default=64)
    p.add_argument('--budget', type=int, default=8, choices=range(2, 13))
    p.add_argument('--updates', type=int, default=1000)
    p.add_argument('--starts', type=int, default=4)
    p.add_argument('--init-native', type=Path)
    p.add_argument('--skip-references', action='store_true', help='Useful on secondary machines')
    a = p.parse_args()
    seeds = [int(s) for s in a.seeds.split(',')]
    modes, devices = a.modes.split(','), a.devices.split(',')
    gpu_devices = [device for device in devices if device != 'cpu']
    if (set(modes) - set(LEARNED) or len(set(modes)) != len(modes) or len(set(seeds)) != len(seeds)
            or len(set(gpu_devices)) != len(gpu_devices) or min(a.width, a.slots, a.batch, a.starts, a.updates) < 1
            or a.width % 4):
        p.error('Use unique modes, seeds and GPU devices and positive sizes (width divisible by four)')
    jobs = [(m, s) for s in seeds for m in modes]
    plan = dict(modes=modes, seeds=seeds, devices=devices, learned_runs=len(jobs), width=a.width,
                slots=a.slots, batch=a.batch, updates=a.updates, budget=a.budget,
                total_training_lookups=len(jobs) * a.updates * a.batch * a.budget)
    print(json.dumps(plan), flush=True)
    if a.stage == 'plan':
        return
    if not a.skip_references:
        jobs.extend((m, None) for m in REFERENCES)

    def execute(device, lane):
        for mode, seed in lane:
            run = a.out / (mode if seed is None else f'{mode}{seed}')
            run.mkdir(parents=True, exist_ok=True)
            opts = ['--mode', mode, '--data', str(a.data), '--run', str(run), '--device', device,
                    '--budget', str(a.budget), '--batch', str(a.batch), '--starts', str(a.starts)]
            def call(*args):
                with (run / 'campaign.log').open('a') as f:
                    subprocess.run([sys.executable, '-m', 'experiments.measured_reactions.study',
                                    args[0], *opts, *args[1:]], stdout=f, stderr=subprocess.STDOUT, check=True)
            if seed is not None:
                spec = dict(seed=seed, mode=mode, width=a.width, slots=a.slots, batch=a.batch,
                            updates=a.updates, budget=a.budget,
                            init_native=str(a.init_native) if a.init_native else None)
                record = run / 'train.json'
                if not record.exists() and a.stage == 'train-dev':
                    args = ['--seed', str(seed), '--updates', str(a.updates), '--width', str(a.width),
                            '--slots', str(a.slots)]
                    if a.init_native:
                        args.extend(('--init-native', str(a.init_native)))
                    call('train', *args)
                if not record.exists() or not (run / 'model.pt').is_file():
                    raise RuntimeError(f'Missing finished checkpoint in {run}')
                saved = json.loads(record.read_text())
                if any(saved[k] != v for k, v in spec.items()):
                    raise ValueError(f'Existing run has different training settings: {run}')
            split = 'test' if a.stage == 'test' else 'dev'
            call('eval', '--split', split)
            if mode in ('conditional', 'cdm'):
                call('eval', '--split', split, '--without-readings')
            print(f'{device}: {run.name} {split} finished', flush=True)

    with ThreadPoolExecutor(max_workers=len(devices)) as workers:
        futures = [workers.submit(execute, device, jobs[i::len(devices)]) for i, device in enumerate(devices)]
        for f in futures:
            f.result()


if __name__ == '__main__':
    main()
