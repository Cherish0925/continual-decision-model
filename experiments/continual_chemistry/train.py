"""Train shared recurrent computation to predict returned chemical outcomes.

The exploratory reference action indexes the observed outcome. It is not an
optimal-action label: this entry does not imitate exhaustive recipe ordering.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
from time import perf_counter
import torch
from torch.nn import functional as F
from cdm.continual_experiment import detach_state
from .dataset import load_episode
from .observations import model_from_core_checkpoint,feedback


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',default='runs/physical_belief/joint_20261029/carry_seed221/policy.pt')
    p.add_argument('--data-root',type=Path,default=Path('runs/discoveryworld/task_transfer/chemistry'))
    p.add_argument('--output',type=Path,required=True);p.add_argument('--updates',type=int,default=100)
    p.add_argument('--sequence-length',type=int,default=8);p.add_argument('--device',default='cpu');p.add_argument('--threads',type=int,default=1)
    p.add_argument('--lr',type=float,default=3e-5);p.add_argument('--seed',type=int,default=730)
    args=p.parse_args();torch.set_num_threads(args.threads);torch.manual_seed(args.seed)
    model=model_from_core_checkpoint(args.checkpoint,device=args.device,initialization_seed=args.seed)
    model.train();optimizer=torch.optim.AdamW([x for x in model.parameters() if x.requires_grad],lr=args.lr)
    manifest=json.loads((args.data_root/'manifest.json').read_text())
    training=[row for row in manifest['episodes'] if row['split']=='train']
    sequences=[load_episode(args.data_root/row['directory']/'experiments.jsonl',device=args.device) for row in training]
    args.output.mkdir(parents=True,exist_ok=True);started=perf_counter();rows=[]
    for update in range(args.updates):
        sequence=sequences[update%len(sequences)];state=model.initial(1,args.device)
        optimizer.zero_grad(set_to_none=True);losses=[]
        for obs,target,record in sequence[:args.sequence_length]:
            state=model.receive(state,obs);out=model.readout(state,obs['candidates'],obs['action_mask'])
            selected=target['action'];index=int(selected[0])
            outcome=F.cross_entropy(out['outcomes'][:,index],target['rust_after'])
            returned=feedback(record,args.device)
            prediction=F.smooth_l1_loss(out['predictions'][:,index],returned)
            losses.append(outcome+prediction)
            state=model.begin_action(state,out['features'],selected)
            state=model.feedback(state,returned)
        loss=torch.stack(losses).mean();loss.backward()
        gradients={name:float(sum(p.grad.square().sum() for p in module.parameters() if p.grad is not None).sqrt())
                   for name,module in [('gru',model.core.gru),('workspace',model.core.block),('outcome',model.base.outcome)]}
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
        row={'update':update+1,'transitions':len(losses),'loss':float(loss.detach()),'gradient_norms':gradients,
             'seconds':perf_counter()-started};rows.append(row)
        with (args.output/'training.jsonl').open('a') as file:file.write(json.dumps(row)+'\n')
    torch.save({'kind':'chemistry_continual_v1',**model.configuration(),'model':model.state_dict(),
                'training':{'source_core':'physical-belief parent 221','objective':'conditional rust class and received progress/cost',
                  'updates':args.updates,'sequence_length':args.sequence_length,'seed':args.seed,
                  'training_episode_directories':[r['directory'] for r in training],
                  'is_task_performance_result':False}},args.output/'last.pt')
    (args.output/'completed.json').write_text(json.dumps({'updates':args.updates,'seconds':perf_counter()-started,
           'records':rows,'scope':'training-path verification, not a learned task-performance comparison'},indent=2)+'\n')
    print(json.dumps(rows[-1]))

if __name__=='__main__':main()
