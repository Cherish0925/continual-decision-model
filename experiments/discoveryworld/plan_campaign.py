"""Train experimental choices and compare five policies on paired new instances."""
from __future__ import annotations

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
                                env={**os.environ,"TOKENIZERS_PARALLELISM":"false","RAYON_NUM_THREADS":"2"})


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


def main():
    baselines=[]
    while not all(collection_complete(s) for s in ('train','validation')):
        if not baselines and (ROOT/'native/reference/summary.json').exists():
            baselines=[('parent',evaluate('parent','predictive')),
                       ('posterior',evaluate('posterior','experimental'))]
        time.sleep(5)
    print(json.dumps({'completed':'plan outcome collection'}),flush=True)
    trainers=[]
    for index,arm in enumerate(('state','numeric')):
        args=['--train',ROOT/'train','--validation',ROOT/'validation','--output',ROOT/arm,
              '--device',f'cuda:{index}','--epochs','100','--seed','171']
        if arm=='numeric':args.append('--no-state')
        trainers.append((arm,start('experiments.discoveryworld.train_plans',args,'train_'+arm)))
    for arm,p in trainers:done(p,arm+' training')
    while not (ROOT/'native/reference/summary.json').exists():time.sleep(5)
    if not baselines:
        baselines=[('parent',evaluate('parent','predictive')),
                   ('posterior',evaluate('posterior','experimental'))]
    learned=[(arm,evaluate(arm,'experimental',ROOT/arm/'last.pt')) for arm in ('state','numeric')]
    for arm,p in baselines+learned:done(p,arm+' native evaluation')
    rows=[]
    design=json.loads((ROOT/'design.json').read_text())
    for arm in ('reference','parent','posterior','state','numeric'):
        for seed in design['native_evaluation_instances']:
            s=json.loads((ROOT/'native'/arm/f'seed{seed}/summary.json').read_text())
            rows.append({'arm':arm,'seed':seed,'cohort':'reserved_family' if seed%5==4 else 'new_instance',
                         'completed':int(s['completed_successfully']),'progress':s['progress'],'actions':s['steps']})
        group=[r for r in rows if r['arm']==arm]
        print(json.dumps({'arm':arm,'completed':sum(r['completed'] for r in group),
                          'instances':len(group),'actions':sum(r['actions'] for r in group)/len(group)}),flush=True)
    with (ROOT/'comparison.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


if __name__=='__main__':main()
