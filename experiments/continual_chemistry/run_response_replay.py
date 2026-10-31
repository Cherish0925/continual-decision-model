"""Run the two-parent, four-update comparison on two GPUs."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys


def run(command, log, device):
    with log.open('w') as output:
        subprocess.run([sys.executable, '-m', *command], check=True, stdout=output,
                       stderr=subprocess.STDOUT, env={**os.environ, 'CUDA_VISIBLE_DEVICES': str(device)})


def lane(seed, device, root, epochs, active):
    parent = Path(f'runs/continual_physics/active_feedback_joint_20261030/learned_seed{seed}/last.pt')
    cache = root / f'prefixes{seed}.pt'
    module = 'cache_active_response' if active else 'cache_response_replay'
    run(['experiments.continual_chemistry.' + module, '--checkpoint', str(parent),
         '--output', str(cache)], root / f'cache{seed}.log', device)
    for mode in ('calibration', 'stored', 'refreshed', 'relation'):
        run(['experiments.continual_chemistry.train_response_replay', '--cache', str(cache),
             '--mode', mode, '--seed', str(seed + 120), '--epochs', str(epochs),
             '--output', str(root / f'{mode}{seed}')], root / f'{mode}{seed}.log', device)
    return seed


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--active', action='store_true')
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'protocol.json').write_text(json.dumps(dict(
        parents=[911, 912], modes=['calibration', 'stored', 'refreshed', 'relation'], epochs=args.epochs,
        training='selected experiments on training split, 20 percent random legal exploration' if args.active else
                 '320 exploration episodes; fixed inherited representation and action histories',
        development='selected experiments on 30 validation instances' if args.active else '30 recorded validation episodes',
        inference='four censored-likelihood functional gradient updates; no policy or core retraining',
        comparison='global calibration controls temperature and bias; stored/refreshed isolate representation',
        selection='fixed final epoch for every condition; native confirmation follows separately'), indent=2) + '\n')
    with ThreadPoolExecutor(2) as pool:
        jobs = [pool.submit(lane, seed, gpu, args.output, args.epochs, args.active) for gpu, seed in enumerate((911, 912))]
        for job in jobs:
            print(json.dumps({'finished_parent': job.result()}), flush=True)
    (args.output / 'completed.json').write_text(json.dumps({'trainings': 8}) + '\n')


if __name__ == '__main__':
    main()
