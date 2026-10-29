"""Measure conditional predictions on frozen development histories."""
from __future__ import annotations
import argparse,json,math
from pathlib import Path
import torch
from torch.nn import functional as F
from .dataset import load_episode
from .observations import feedback,load_chemistry_policy


def evaluate(checkpoint,data_root,output):
    torch.set_num_threads(1)
    model,_=load_chemistry_policy(checkpoint);model.eval()
    manifest=json.loads((data_root/'manifest.json').read_text());rows=[]
    with torch.no_grad():
        for episode in manifest['episodes']:
            if episode['split']!='validation':continue
            state=model.initial(1,'cpu');records=[]
            for obs,target,record in load_episode(data_root/episode['directory']/'experiments.jsonl'):
                state=model.receive(state,obs);out=model.readout(state,obs['candidates'],obs['action_mask'])
                index=int(target['action'][0]);pred=out['outcomes'][:,index];returned=feedback(record)
                probability=pred.softmax(-1)[0];prediction=out['predictions'][0,index]
                records.append({'rust_nll':float(F.cross_entropy(pred,target['rust_after'])),
                    'rust_correct':int(pred.argmax(-1)==target['rust_after']),
                    'clean_brier':float((probability[0]-int(record['rust_after']==0))**2),
                    'clean_target':int(record['rust_after']==0),'clean_probability':float(probability[0]),
                    'partial_progress_target':int(0<record['rust_after']<record['rust_before']),
                    'progress_absolute_error':float((prediction[0]-returned[0,0]).abs()),
                    'log_cost_absolute_error':float((prediction[1]-returned[0,1]).abs()),
                    'cost_absolute_error':abs(math.expm1(float(prediction[1].clamp(0,math.log(1001))))-record['native_actions'])})
                state=model.begin_action(state,out['features'],target['action']);state=model.feedback(state,returned)
            rows.append({'directory':episode['directory'],'regime':episode['evaluation_regime'],
                         'recipe_group':episode['recipe_group'],'n':len(records),'transitions':records,
                         **{key:sum(row[key] for row in records)/len(records) for key in records[0]}})
    summaries=[]
    for regime in ['all','untrained_recipe','seen_recipe_new_instance']:
        group=[row for row in rows if regime=='all' or row['regime']==regime]
        keys=['rust_nll','rust_correct','clean_brier','progress_absolute_error','log_cost_absolute_error','cost_absolute_error']
        summaries.append({'regime':regime,'episodes':len(group),'transitions':sum(row['n'] for row in group),
            'episode_mean':{key:sum(row[key] for row in group)/len(group) for key in keys},
            'transition_mean':{key:sum(row[key]*row['n'] for row in group)/sum(row['n'] for row in group) for key in keys}})
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps({'checkpoint':str(checkpoint),'scope':'public reference histories, no model-generated policy shift',
                                 'summary':summaries,'episodes':rows},indent=2)+'\n')
    print(json.dumps({'checkpoint':str(checkpoint),'summary':summaries}),flush=True)



def frequency_reference(data_root,output):
    """A train-only conditional frequency table, not an acting policy."""
    from collections import defaultdict
    manifest=json.loads((data_root/'manifest.json').read_text())
    counts=defaultdict(lambda:[1.,1.,1.,1.])
    for episode in manifest['episodes']:
        if episode['split']!='train':continue
        for line in (data_root/episode['directory']/'experiments.jsonl').read_text().splitlines():
            row=json.loads(line)
            counts[len(row['component_counts']),row['rust_before']][row['rust_after']]+=1
    rows=[]
    for episode in manifest['episodes']:
        if episode['split']!='validation':continue
        records=[json.loads(line) for line in (data_root/episode['directory']/'experiments.jsonl').read_text().splitlines()]
        nll=correct=persistent=0.
        for row in records:
            values=counts[len(row['component_counts']),row['rust_before']]
            probabilities=[value/sum(values) for value in values]
            nll-=math.log(probabilities[row['rust_after']])
            correct+=probabilities.index(max(probabilities))==row['rust_after']
            persistent+=row['rust_before']==row['rust_after']
        rows.append({'directory':episode['directory'],'regime':episode['evaluation_regime'],'n':len(records),
                     'nll':nll/len(records),'accuracy':correct/len(records),'rust_persistence_accuracy':persistent/len(records)})
    summary={regime:{key:sum(row[key] for row in rows if regime=='all' or row['regime']==regime)/
                   sum(regime=='all' or row['regime']==regime for row in rows)
                   for key in ('nll','accuracy','rust_persistence_accuracy')}
             for regime in ('all','untrained_recipe','seen_recipe_new_instance')}
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps({'summary':summary,'episodes':rows,
        'training_cells':{str(key):value for key,value in counts.items()},
        'scope':'Train-only frequency conditional on component count and pre-treatment rust; no native acting result'},indent=2)+'\n')
    print(json.dumps(summary,indent=2))

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoint',type=Path);p.add_argument('--frequency-reference',action='store_true')
    p.add_argument('--data-root',type=Path,default=Path('runs/continual_physics/chemistry_data'))
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    if args.frequency_reference:frequency_reference(args.data_root,args.output)
    elif args.checkpoint:evaluate(args.checkpoint,args.data_root,args.output)
    else:p.error('Supply --checkpoint or --frequency-reference')

if __name__=='__main__':main()
