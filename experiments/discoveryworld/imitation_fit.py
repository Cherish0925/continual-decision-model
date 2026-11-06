"""Measure frozen-checkpoint request accuracy on recorded demonstration states."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
from time import perf_counter

import torch

from .features import FeatureCache, FeatureConfig
from .train import load_checkpoint, load_episodes, prepare_episodes, run_epoch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoints', type=Path, nargs='+', required=True)
    parser.add_argument('--train', type=Path, required=True)
    parser.add_argument('--development', type=Path, required=True)
    parser.add_argument('--feature-cache', type=Path, required=True)
    parser.add_argument('--encoder-path', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.backends.cuda.enable_cudnn_sdp(False)
    episodes = {name: load_episodes([path]) for name, path in
                [('training', args.train), ('development', args.development)]}
    previous, prepared, rows = None, None, []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for checkpoint in args.checkpoints:
        started = perf_counter()
        model, source = load_checkpoint(checkpoint, args.device)
        config = FeatureConfig(**source['features'])
        training = source['training_args']
        preparation = dict(allow_failed_actions=training.get('allow_failed_actions', False),
                           imitate_successful_episodes=training.get('imitate_successful_episodes', False))
        identity = (asdict(config), preparation)
        if identity != previous:
            cache = FeatureCache(args.feature_cache, config, args.device, args.encoder_path)
            prepared = {name: prepare_episodes(items, cache, **preparation)
                        for name, items in episodes.items()}
            cache.close()
            previous = identity
        results = {name: run_epoch(model, items, batch_size=4, sequence_length=8,
                    device=args.device, mode=training['mode'],
                    persistent=not training.get('restart_state', False),
                    restart_workspace=training.get('restart_workspace', False),
                    immediate_weight=0., precision=training.get('precision', 'float32'))
                   for name, items in prepared.items()}
        rows.append(dict(model=checkpoint.parent.parent.name, epoch=source['epoch'],
                         **results, seconds=perf_counter()-started))
        args.output.write_text(json.dumps(dict(protocol='frozen_weights_recorded_states',
                                               results=rows), indent=2)+'\n')
        print(json.dumps(rows[-1]), flush=True)
        del model


if __name__ == '__main__':
    main()
