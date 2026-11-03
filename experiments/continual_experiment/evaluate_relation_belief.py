"""Evaluate learned response beliefs on complete, changing task streams."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from experiments.continual_chemistry.learned_belief import LearnedExperiments, load_model
from experiments.continual_chemistry.response_families import ResponseFamilyEnvironment


def worker(job):
    stream, options=job
    torch.set_num_threads(1)
    model,saved=load_model(options['checkpoint'])
    env=ResponseFamilyEnvironment(None,1,17,'cpu',manifest=[stream],
        max_parts=stream.get('max_parts',6),
        max_components=max(t['law']['components'] for t in stream['tasks']),
        rename_components=stream.get('rename_components',False))
    initial_observation=env.observation()
    components=int(initial_observation['candidates'][0,0,16:24].sum())
    belief=LearnedExperiments(model,components,
        capacity=options['capacity'],selection=options['selection'],budget=env.budget,
        max_parts=stream.get('max_parts',6),
        permutation=env.public_names[0] if stream.get('rename_components',False) else None)
    records, forecasts=[],[]
    started=perf_counter(); seconds=0.
    while len(records)<len(stream['tasks']):
        boundary,_=(env.new_task,env.new_lifetime) if env.dispatched==0 else env.reset_finished()
        if boundary[0]: belief.start()
        obs=env.observation()
        tick=perf_counter()
        current=int(env.rust[0])
        if current==0:
            action=env.leave_index; probabilities=None
        else:
            action=belief.choose(current,obs['action_mask'][0,:len(belief.recipes)].numpy())
            probabilities=belief.probabilities(current)[action]
        seconds+=perf_counter()-tick
        outcome=env.step(torch.tensor([action]))
        if outcome['returned'][0] and not outcome['leave'][0]:
            after=int(outcome['after'][0])
            forecasts.append(dict(stream=stream['name'],task_index=env.task_index[0]-1,
                experiment=belief.returns,action=action,before=current,after=after,
                probabilities=probabilities.tolist(),nll=-float(np.log(max(probabilities[after],1e-30)))))
            belief.receive(action,current,after)
        if env.task_rows:
            records.extend({**r,'stream':stream['name'],'group':stream['group'],
                'inference_seconds':seconds} for r in env.task_rows)
            env.task_rows.clear(); seconds=0.
    return records,forecasts,perf_counter()-started


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--selection',choices=('greedy','information','two_step','marginal_plan',
                                        'finite_plan','finite_marginal'),default='information')
    p.add_argument('--capacity',type=int,default=12)
    p.add_argument('--workers',type=int,default=8)
    args=p.parse_args()
    model, saved=load_model(args.checkpoint)
    if saved['update'] != saved['configuration']['updates']:
        p.error('Evaluation requires the fixed final checkpoint')
    args.output.mkdir(parents=True,exist_ok=False)
    manifest=json.loads(args.manifest.read_text())
    options={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    records=[]; forecasts=[]; start=perf_counter()
    with ProcessPoolExecutor(args.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        for f in as_completed([pool.submit(worker,(s,options)) for s in manifest['streams']]):
            rows,pred,elapsed=f.result(); records.extend(rows); forecasts.extend(pred)
            print(json.dumps(dict(stream=rows[0]['stream'],completed=sum(r['completed'] for r in rows),
                                  operations=sum(r['native_actions'] for r in rows)/len(rows),seconds=elapsed)),flush=True)
    records.sort(key=lambda r:(r['stream'],r['task_index']))
    result=dict(configuration=options,checkpoint_update=saved['update'],records=records,tasks=len(records),
        completed=sum(r['completed'] for r in records),mean_operations=np.mean([r['native_actions'] for r in records]),
        seconds=perf_counter()-start,prediction_nll=np.mean([r['nll'] for r in forecasts]),
        scope='Learned predictors; only returned experiments used at execution; no generating formulas or task identities')
    (args.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    with (args.output/'predictions.jsonl').open('w') as f:
        for r in forecasts: f.write(json.dumps(r)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='records'}),flush=True)


if __name__=='__main__': main()
