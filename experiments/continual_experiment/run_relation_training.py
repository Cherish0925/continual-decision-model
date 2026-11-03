"""Two-device, two-initialization comparison of joint and marginal predictors."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--updates',type=int,default=6000)
    p.add_argument('--phase',choices=('initial','refinement'),default='initial')
    args=p.parse_args()
    target=args.root/'train'; target.mkdir(exist_ok=True)
    if args.phase == 'initial':
        queues={0:[('functions',31),('conditional',32),('attentive',31)],
                1:[('conditional',31),('functions',32),('attentive',32)]}
    else:
        queues={0:[('long',31),('modulated',32),('attentive_long',31)],
                1:[('modulated',31),('long',32),('attentive_long',32)]}
    (target/(args.phase+'_plan.json')).write_text(json.dumps(dict(updates=args.updates,queues=queues,
        endpoint='Final update fixed before evaluation',data='One common corpus of returned measurements',
        scope='Predictor training; inherited CDM policy is preserved for subsequent integration'),indent=2)+'\n')
    def run(device):
        results=[]
        for label,seed in queues[device]:
            name=f'{label}{seed}'
            kind=('functions' if label in ('long','modulated') else
                  'attentive' if label == 'attentive_long' else label)
            command=[sys.executable,'-m','experiments.continual_experiment.train_relation_belief',
                '--data',str(args.root/'data'),'--output',str(target/name),'--kind',kind,'--seed',str(seed),
                '--device',f'cuda:{device}','--updates',str(args.updates)]
            if label == 'modulated':
                command += ['--decoder','modulated']
            with (target/(name+'.log')).open('w') as log:
                result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
            results.append(dict(name=name,exit_code=result.returncode))
            print(json.dumps(results[-1]),flush=True)
            if result.returncode: break
        return results
    with ThreadPoolExecutor(2) as pool:
        results=[r for rows in pool.map(run,(0,1)) for r in rows]
    (target/(args.phase+'_completed.json')).write_text(json.dumps(results,indent=2)+'\n')
    if any(r['exit_code'] for r in results): raise SystemExit(1)


if __name__=='__main__': main()
