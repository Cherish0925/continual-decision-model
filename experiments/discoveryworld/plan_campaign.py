"""Train experimental choices and compare five policies on paired new instances."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT=Path("runs/discoveryworld/experimental_plans")


def start(module,args,name):
    with (ROOT/f"{name}.log").open("w") as stream:
        return subprocess.Popen([sys.executable,"-m",module,*map(str,args)],stdout=stream,stderr=subprocess.STDOUT,
                                env={**os.environ,"TOKENIZERS_PARALLELISM":"false","RAYON_NUM_THREADS":"2",
                                     "OMP_NUM_THREADS":"2","OPENBLAS_NUM_THREADS":"2","MKL_NUM_THREADS":"2",
                                     "PYTHONFAULTHANDLER":"1"})


def done(p,name):
    if p.wait()!=0:raise RuntimeError(f"{name} failed; see its run log")
    print(json.dumps({"completed":name}),flush=True)


def evaluate(arm,kind,plan=None):
    design=json.loads((ROOT/'design.json').read_text())
    args=["--policy-kind",kind,"--checkpoint","runs/discoveryworld/predictive_actions/feedback/last.pt",
          "--feature-cache","runs/discoveryworld/features","--encoder-path","data/cache/siglip2",
          "--output",ROOT/'native'/arm,"--seeds",*design['native_evaluation_instances'],
          "--max-steps","400","--devices","cuda:0","cuda:1","--workers","4","--threads","2",
          "--local-probes","--initial-rng-root",ROOT/'native/reference']
    if plan:args += ["--plan-checkpoint",plan]
    return start('experiments.discoveryworld.evaluate',args,'native_'+arm)


def collection_complete(split):
    path=ROOT/split/'comparison.json'
    if not path.exists():return False
    try:
        rows=json.loads(path.read_text())
    except json.JSONDecodeError:return False
    selected=json.loads((ROOT/split/'selection.json').read_text())
    return len(rows)==sum(len(s['hypotheses']) for s in selected if s)


def transfer_reader():
    area = ROOT/'transfer'
    area.mkdir(parents=True, exist_ok=True)
    seeds = [s for s in range(3000,3020) if s%5 != 4] + list(range(3004,3040,5))
    design = {'instances':seeds,'action_budget':400,
              'source_reader':'runs/discoveryworld/belief_actions/full_seed61/last.pt',
              'heads':['numeric','numeric_seed172'],
              'selection':'fixed heads from the first comparison; no further training',
              'question':'Can outcome-based experimental choice improve the original reader on fresh instances?'}
    (area/'design.json').write_text(json.dumps(design,indent=2)+'\n')

    def run(arm, kind, head=None, workers=4):
        args=['--policy-kind',kind,'--checkpoint',design['source_reader'],
              '--feature-cache','runs/discoveryworld/features','--encoder-path','data/cache/siglip2',
              '--output',area/arm,'--seeds',*seeds,'--max-steps','400',
              '--devices','cuda:0','cuda:1','--workers',workers,'--threads','2','--local-probes']
        if arm!='reference':args += ['--initial-rng-root',area/'reference']
        if head:args += ['--plan-checkpoint',ROOT/head/'last.pt']
        return start('experiments.discoveryworld.evaluate',args,'transfer_'+arm)

    done(run('reference','belief',workers=8),'transfer reference')
    jobs=[('posterior',run('posterior','experimental_belief')),
          *[(arm,run(arm,'experimental_belief',arm)) for arm in design['heads']]]
    for arm,p in jobs:done(p,'transfer '+arm)
    rows=[]
    for arm in ['reference','posterior',*design['heads']]:
        for seed in seeds:
            s=json.loads((area/arm/f'seed{seed}/summary.json').read_text())
            rows.append({'arm':arm,'seed':seed,'cohort':'reserved_family' if seed%5==4 else 'new_instance',
                         'completed':int(s['completed_successfully']),'progress':s['progress'],'actions':s['steps']})
    with (area/'comparison.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--replicate', type=int, help='Repeat both learned heads with a new training seed')
    parser.add_argument('--transfer-reader', action='store_true', help='Evaluate fixed numerical heads with the original reader on fresh instances')
    args = parser.parse_args()
    if args.transfer_reader:
        transfer_reader()
        return
    while not all(collection_complete(s) for s in ('train','validation')):
        time.sleep(5)
    print(json.dumps({'completed':'plan outcome collection'}),flush=True)
    trainers=[]
    seed = args.replicate or 171
    arms = [('state', 'state' if args.replicate is None else f'state_seed{seed}'),
            ('numeric', 'numeric' if args.replicate is None else f'numeric_seed{seed}')]
    for index,(kind,arm) in enumerate(arms):
        training_args=['--train',ROOT/'train','--validation',ROOT/'validation','--output',ROOT/arm,
                       '--device',f'cuda:{index}','--epochs','100','--seed',seed]
        if kind=='numeric':training_args.append('--no-state')
        trainers.append((arm,start('experiments.discoveryworld.train_plans',training_args,'train_'+arm)))
    for arm,p in trainers:done(p,arm+' training')
    while not (ROOT/'native/reference/summary.json').exists():time.sleep(5)
    if args.replicate is None:
        baselines=[('parent',evaluate('parent','predictive')),
                   ('posterior',evaluate('posterior','experimental'))]
    else:
        # Let the first comparison release its environment workers. Both
        # replications use the same already fixed native instances and RNGs.
        while not all((ROOT/'native'/arm/'summary.json').exists() for arm in ('parent','posterior')):
            time.sleep(5)
        baselines=[]
    learned=[(arm,evaluate(arm,'experimental',ROOT/arm/'last.pt')) for _,arm in arms]
    for arm,p in baselines+learned:done(p,arm+' native evaluation')
    rows=[]
    design=json.loads((ROOT/'design.json').read_text())
    selected_arms = ['reference','parent','posterior',*[arm for _,arm in arms]]
    for arm in selected_arms:
        for seed in design['native_evaluation_instances']:
            s=json.loads((ROOT/'native'/arm/f'seed{seed}/summary.json').read_text())
            rows.append({'arm':arm,'seed':seed,'cohort':'reserved_family' if seed%5==4 else 'new_instance',
                         'completed':int(s['completed_successfully']),'progress':s['progress'],'actions':s['steps']})
        group=[r for r in rows if r['arm']==arm]
        print(json.dumps({'arm':arm,'completed':sum(r['completed'] for r in group),
                          'instances':len(group),'actions':sum(r['actions'] for r in group)/len(group)}),flush=True)
    destination = 'comparison.csv' if args.replicate is None else f'comparison_seed{seed}.csv'
    with (ROOT/destination).open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


if __name__=='__main__':main()
