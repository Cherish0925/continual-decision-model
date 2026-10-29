"""Reuse recorded native chemical trajectories without future-observation input."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import torch
from .observations import observe


def load_episode(path,*,max_parts=6,max_actions=1000,device='cpu'):
    records=[json.loads(line) for line in Path(path).read_text().splitlines() if line]
    sequence=[]
    for index,record in enumerate(records):
        state=record['state']
        if len(state['history'])!=index:raise ValueError('Each input must contain exactly the already returned prefix')
        if state['rust_level']!=record['rust_before']:raise ValueError('Pre-treatment state does not match transition')
        obs,choices=observe(state,max_parts=max_parts,max_actions=max_actions,
                            elapsed_actions=record['transition_start'],device=device)
        action=choices.index(tuple(record['component_counts']))
        targets={'action':torch.tensor([action],device=device),'rust_after':torch.tensor([record['rust_after']],device=device),
                 'progress':torch.tensor([(record['rust_before']-record['rust_after'])/3],device=device),
                 'log_cost':torch.tensor([__import__('math').log1p(record['native_actions'])],device=device)}
        sequence.append((obs,targets,record))
    return sequence


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path('runs/discoveryworld/task_transfer/chemistry'))
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    manifest=json.loads((args.root/'manifest.json').read_text());rows=[]
    for episode in manifest['episodes']:
        path=args.root/episode['directory']/'experiments.jsonl';sequence=load_episode(path)
        rows.append({**episode,'transitions':len(sequence),'input_keys':sorted(sequence[0][0]),
                     'components':sequence[0][2]['state']['components'],
                     'candidate_count':sequence[0][0]['candidates'].shape[1],
                     'partial_improvements':sum(r['rust_after']<r['rust_before'] and r['rust_after']>0 for _,_,r in sequence),
                     'source':str(path.relative_to(args.root))})
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps({'episodes':rows,'total_transitions':sum(r['transitions'] for r in rows),
                        'input_scope':'past returned recipes, public rust state, visible components and own action budget',
                        'excluded_inputs':['seed','hidden recipe','task score','future rust outcome'],
                        'labels':'next rust state, rust change, actual native cost; reference action is an exploratory behavior label'},indent=2)+'\n')
    print(json.dumps({'episodes':len(rows),'transitions':sum(r['transitions'] for r in rows)}))

if __name__=='__main__':main()
