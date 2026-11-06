"""Time one observed request window without changing its causal decision loop."""
import argparse
import copy
import json
from pathlib import Path
import random
from time import perf_counter

import torch

from .features import FeatureCache, batch_features, load_features
from .train import parse_args, initialize_training, load_episodes, prepare_episodes, create_model, run_epoch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True, help='Contains train/ and development/')
    parser.add_argument('--encoder-path', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--features', choices=('tokens', 'lexical'), default='tokens')
    parser.add_argument('--text-batch-size', type=int, default=256)
    parser.add_argument('--text-checkpointing', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--window-text', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--sequence-length', type=int, default=8)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--profile', action='store_true')
    a = parser.parse_args()
    torch.set_num_threads(2)
    torch.backends.cuda.enable_cudnn_sdp(False)
    torch.manual_seed(501)
    argv = ['--train-trajectories', str(a.root/'train'), '--validation-trajectories', str(a.root/'development'),
            '--output', str(a.output), '--feature-cache', str(a.root/'features'), '--features', a.features,
            '--encoder-path', str(a.encoder_path), '--action-interface', 'object_requests_v1',
            '--measurement-memory', '--object-relations', '--action-history', '8', '--candidate-read',
            '--width', '512', '--heads', '4', '--slots', '8', '--batch-size', str(a.batch_size),
            '--sequence-length', str(a.sequence_length), '--allow-failed-actions', '--immediate-weight', '0',
            '--outcome-weight', '.1', '--device', a.device, '--text-batch-size', str(a.text_batch_size),
            '--text-checkpointing' if a.text_checkpointing else '--no-text-checkpointing']
    if a.features == 'tokens':
        argv += ['--train-text-backbone']
    args = parse_args(argv)
    _, _, config = initialize_training(args)
    episodes = load_episodes(args.train_trajectories)
    random.Random(501).shuffle(episodes)
    episodes = episodes[:a.batch_size]
    for episode in episodes:
        episode.transitions = episode.transitions[:a.sequence_length + 16]
    cache = FeatureCache(a.root/'features', config, 'cpu', a.encoder_path)
    prepared = prepare_episodes(episodes, cache, allow_failed_actions=True)
    for episode in prepared:
        episode.steps = episode.steps[-a.sequence_length:]
    backbone = cache.encoder_config.to_dict() if a.features == 'tokens' else None
    model = create_model(args, cache.text_dim, cache.image_dim, backbone).to(a.device)
    cache.close()
    statistics = []
    for t in range(max(len(e.steps) for e in prepared)):
        rows = [e.steps[min(t, len(e.steps) - 1)] for e in prepared]
        obs, actions = batch_features([load_features(s.feature_path) for s in rows], 'cpu')
        statistics.append(dict(observation_shape=list(obs['text'].shape),
                               candidate_shape=list(actions['text'].shape)))
    kwargs = dict(batch_size=a.batch_size, sequence_length=a.sequence_length, device=a.device,
                  mode='cdm', outcome_weight=.1, immediate_weight=0., precision='bfloat16',
                  window_text=a.window_text)
    base = copy.deepcopy(model.state_dict())
    elapsed = []
    for repeat in range(4):
        model.load_state_dict(base)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
        torch.manual_seed(100)
        torch.cuda.synchronize(a.device)
        torch.cuda.reset_peak_memory_stats(a.device)
        start = perf_counter()
        row = run_epoch(model, prepared, optimizer=optimizer, **kwargs)
        torch.cuda.synchronize(a.device)
        if repeat:
            elapsed.append(perf_counter() - start)
    report = dict(seconds=elapsed, loss=row['loss'], transitions=row['transitions'],
                  parameters=sum(p.numel() for p in model.parameters()), batch=statistics,
                  episode_lengths=[len(e.steps) for e in prepared],
                  episode_names=[e.episode.path.parent.name for e in prepared],
                  peak_allocated_bytes=torch.cuda.max_memory_allocated(a.device),
                  gpu=torch.cuda.get_device_name(a.device), torch=torch.__version__,
                  text_batch_size=a.text_batch_size, text_checkpointing=a.text_checkpointing,
                  window_text=a.window_text, sequence_length=a.sequence_length,
                  model_initialization='random seed 501; dropout seed 100 before each timed repetition',
                  timing='Three repetitions after one warmup; preparation excluded; one optimizer update each')
    a.output.mkdir(parents=True, exist_ok=True)
    (a.output/'results.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'batch'}))
    if a.profile:
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                               torch.profiler.ProfilerActivity.CUDA]) as prof:
            run_epoch(model, prepared, optimizer=optimizer, **kwargs)
        (a.output/'operators.txt').write_text(prof.key_averages().table(
            sort_by='self_cuda_time_total', row_limit=30)+'\n')


if __name__ == '__main__':
    main()
