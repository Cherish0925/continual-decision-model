"""Compare language understanding before attributing failure to persistent state.

All conditions use the same observations, demonstration trajectories, request
interface, final epochs and interaction schedule. No task score selects a model.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys


CONDITIONS = ('lexical', 'frozen_text', 'adapted_text', 'random_text')
SEEDS = (501, 502, 503, 504)
THEMES = ('Combinatorial Chemistry', 'Archaeology Dating', 'Plant Nutrients', 'Reactor Lab', 'Proteomics')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--condition', choices=CONDITIONS, required=True)
    p.add_argument('--mode', choices=('cdm', 'gru'), required=True)
    p.add_argument('--seed', type=int, choices=SEEDS, required=True)
    p.add_argument('--encoder-path', type=Path, required=True)
    p.add_argument('--output', type=Path, help='Job directory; defaults to root/models/condition_mode_seed')
    p.add_argument('--epochs', type=int, default=32)
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--sequence-length', type=int, default=8)
    p.add_argument('--state-gradient-length', type=int)
    p.add_argument('--window-text', action=argparse.BooleanOptionalAction, default=False)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--prepare-workers', type=int, default=1)
    p.add_argument('--development-seeds', nargs='+', type=int, default=(6300,6301))
    p.add_argument('--evaluation-seeds', nargs='+', type=int, default=(6800,6801))
    p.add_argument('--text-batch-size',type=int,default=64,
                   help='Text chunks per encoder call; independent of the four-episode training batch')
    p.add_argument('--text-checkpointing',action=argparse.BooleanOptionalAction,default=True)
    args = p.parse_args()
    root = args.root.resolve()
    job = args.output.resolve() if args.output else root/'models'/f'{args.condition}_{args.mode}_s{args.seed}'
    job.mkdir(parents=True, exist_ok=True)
    cache = root/'features'/args.condition

    def run(stage, module, arguments):
        marker = job/(stage+'.done.json')
        if marker.exists():
            return
        command = [sys.executable, '-u', '-m', module, *map(str, arguments)]
        (job/(stage+'.command.json')).write_text(json.dumps(command, indent=2)+'\n')
        with (job/(stage+'.log')).open('a') as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        marker.write_text(json.dumps({'stage':stage,'completed':True})+'\n')

    options = ['--features', 'lexical' if args.condition == 'lexical' else 'tokens']
    if args.window_text:
        options += ['--window-text']
    if args.state_gradient_length is not None:
        options += ['--state-gradient-length', args.state_gradient_length]
    if args.condition != 'lexical':
        options += ['--encoder-path', args.encoder_path,
                    '--text-initialization', 'random' if args.condition == 'random_text' else 'pretrained',
                    '--text-batch-size',args.text_batch_size,
                    '--text-checkpointing' if args.text_checkpointing else '--no-text-checkpointing']
        if args.condition != 'frozen_text':
            options += ['--train-text-backbone']
    run('imitation', 'experiments.discoveryworld.train', [
        '--train-trajectories', root/'train', '--validation-trajectories', root/'development',
        '--output', job/'imitation', '--feature-cache', cache, *options,
        '--action-interface','object_requests_v1','--measurement-memory','--object-relations',
        '--action-history',8,'--candidate-read','--width',512,'--heads',4,'--slots',8,
        '--mode',args.mode,'--epochs',args.epochs,'--batch-size',args.batch_size,'--sequence-length',args.sequence_length,
        '--prepare-workers',args.prepare_workers,
        '--allow-failed-actions','--immediate-weight',0,'--outcome-weight',.1,
        '--lr',args.lr,'--text-lr',2e-5,'--precision','bfloat16','--disable-cudnn-attention',
        '--device','cuda:0','--threads',2,'--seed',args.seed])

    def evaluate(stage, checkpoint, seeds, temperature=0., reset=False):
        config = dict(checkpoint=str(checkpoint),feature_cache=str(cache),encoder_path=str(args.encoder_path),
                      device='cuda:0',temperature=temperature,sampling_seed=args.seed,restart_state=reset)
        path=job/(stage+'.json')
        path.write_text(json.dumps(config,indent=2)+'\n')
        run(stage,'experiments.discoveryworld.full_suite',[
            'run','--output',job/stage,'--object-requests',
            '--policy','experiments.discoveryworld.evaluate:create_evaluation_policy','--policy-config',path,
            '--seeds',*seeds,'--difficulties','Easy','Normal','--themes',*THEMES,
            '--max-steps',300,'--max-request-actions',12,'--workers',1,'--threads',2,
            '--object-rng-seed',73106])

    weight=job/'imitation/last.pt'
    evaluate('imitation_development',weight,args.development_seeds)
    evaluate('imitation_greedy',weight,args.evaluation_seeds)
    run('interaction','experiments.discoveryworld.finetune',[
        '--checkpoint',weight,'--output',job/'interaction','--feature-cache',cache,
        '--encoder-path',args.encoder_path,'--native-suite','all','--themes',*THEMES,
        '--seeds',6600,6601,6602,6603,6604,6605,6606,6607,
        '--difficulties','Easy','Normal','Challenge','--max-steps',300,'--max-request-actions',12,
        '--sequence-length',8,'--workers',4,'--device','cuda:0','--threads',2,
        '--mode',args.mode,'--random-seed',args.seed,'--gamma',.999,'--native-action-cost',.001,
        '--outcome-weight',.1,'--text-lr',1e-5])
    for stage,temperature,reset in [('interaction_greedy',0.,False),('interaction_sampled',1.,False),('interaction_reset',0.,True)]:
        evaluate(stage,job/'interaction/last.pt',args.evaluation_seeds,temperature,reset)
    (job/'complete.json').write_text(json.dumps(dict(condition=args.condition,mode=args.mode,seed=args.seed))+'\n')


if __name__=='__main__':
    main()
