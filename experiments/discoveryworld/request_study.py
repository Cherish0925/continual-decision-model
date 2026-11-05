"""Run a fixed paired planner study on a selected device.

The self-learning arm starts immediately. The demonstration arm waits for a
complete, shared collection and starts from its paired arm's saved initial
weights. Every completed stage and its command remain in the run directory.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from time import sleep


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--mode',choices=('cdm','gru'),required=True)
    p.add_argument('--arm',choices=('self','demonstrations'),required=True)
    p.add_argument('--seed',type=int,required=True)
    args=p.parse_args()
    plan=json.loads(Path(__file__).with_name('common_execution.json').read_text())
    root=args.root.resolve();job=root/'models'/f'{args.arm}_{args.mode}_s{args.seed}'
    job.mkdir(parents=True,exist_ok=True)
    cache=root/'features'/job.name

    def run(name,module,options):
        command=[sys.executable,'-u','-m',module,*map(str,options)]
        done=job/f'{name}.done.json'
        if done.exists():
            return
        (job/f'{name}.command.json').write_text(json.dumps(command,indent=2)+'\n')
        with (job/f'{name}.log').open('a') as log:
            subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
        done.write_text(json.dumps({'stage':name,'completed':True})+'\n')

    initial=root/'models'/f'self_{args.mode}_s{args.seed}'/'interaction'/'initial.pt'
    checkpoint=None
    if args.arm=='demonstrations':
        (job/'waiting.json').write_text(json.dumps({'data':'initial-data-ready.json','initial':str(initial.relative_to(root))})+'\n')
        while not (root/'initial-data-ready.json').exists() or not initial.exists():
            sleep(20)
        (job/'waiting.json').unlink()
        run('initial_imitation','experiments.discoveryworld.train',[
            '--train-trajectories',root/'initial_data/train','--validation-trajectories',root/'initial_data/development',
            '--resume-checkpoint',initial,'--output',job/'initial_imitation',
            '--feature-cache',cache,'--device','cuda:0','--threads',2,
            '--epochs',plan['initial_imitation_epochs'],'--batch-size',plan['imitation_batch_size'],
            '--sequence-length',plan['sequence_length'],'--allow-failed-actions',
            '--immediate-weight',0,'--precision','bfloat16','--disable-cudnn-attention','--seed',args.seed])
        (job/'waiting.json').write_text(json.dumps({'data':'data-ready.json'})+'\n')
        while not (root/'data-ready.json').exists():
            sleep(20)
        (job/'waiting.json').unlink()
        run('imitation','experiments.discoveryworld.train',[
            '--train-trajectories',root/'train','--validation-trajectories',root/'development',
            '--resume-checkpoint',job/'initial_imitation/last.pt','--output',job/'imitation',
            '--feature-cache',cache,'--device','cuda:0','--threads',2,
            '--epochs',plan['imitation_epochs'],'--batch-size',plan['imitation_batch_size'],
            '--sequence-length',plan['sequence_length'],'--allow-failed-actions',
            '--immediate-weight',0,'--precision','bfloat16','--disable-cudnn-attention','--seed',args.seed])
        checkpoint=job/'imitation'/'last.pt'

    options=['--output',job/'interaction','--feature-cache',cache,
        '--native-suite','all','--themes',*plan['themes'],'--seeds',*plan['interaction_seeds'],
        '--max-steps',plan['native_action_limit'],'--max-request-actions',plan['request_native_action_limit'],
        '--sequence-length',plan['sequence_length'],'--workers',plan['interaction_workers'],
        '--device','cuda:0','--threads',2,'--mode',args.mode,'--random-seed',args.seed,
        '--gamma',plan['gamma_per_native_action'],'--native-action-cost',plan['native_action_cost'],
        '--outcome-weight',.1]
    options += ['--checkpoint',checkpoint] if checkpoint else ['--random-native-width',plan['width'],'--object-requests']
    run('interaction','experiments.discoveryworld.finetune',options)

    endpoints=[('interaction_greedy',job/'interaction'/'last.pt',0.,False),
               ('interaction_sampled',job/'interaction'/'last.pt',1.,False),
               ('interaction_reset',job/'interaction'/'last.pt',0.,True)]
    if checkpoint:
        endpoints.insert(0,('imitation_greedy',checkpoint,0.,False))
    for name,weight,temperature,reset in endpoints:
        config=job/f'{name}.json'
        config.write_text(json.dumps(dict(checkpoint=str(weight),feature_cache=str(cache),
            device='cuda:0',temperature=temperature,sampling_seed=args.seed,restart_state=reset),indent=2)+'\n')
        run(name,'experiments.discoveryworld.full_suite',['run','--output',job/name,
            '--object-requests','--policy','experiments.discoveryworld.evaluate:create_evaluation_policy',
            '--policy-config',config,'--seeds',*plan['evaluation_seeds'],'--difficulties',*plan['difficulties'],
            '--themes',*plan['themes'],'--max-steps',plan['native_action_limit'],
            '--max-request-actions',plan['request_native_action_limit'],'--workers',1,'--threads',2])
    (job/'complete.json').write_text(json.dumps({'mode':args.mode,'arm':args.arm,'seed':args.seed})+'\n')


if __name__=='__main__':
    main()
