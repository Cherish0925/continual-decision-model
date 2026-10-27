"""Learn complete experiment outcomes from matched native hypothesis trials."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

import torch
import torch.nn.functional as F

from .experimental_plans import PlanValue


def load(root):
    comparison = json.loads((root / "comparison.json").read_text())
    grouped = {}
    for row in comparison:
        if not row["finished"]:
            continue  # A time limit is censored, not an observed failed experiment.
        trial, = json.loads((Path(row["output"])/"experimental_trials.json").read_text())
        key = (row["source"],row["prefix"])
        if key not in grouped:
            n = len(trial["candidates"]["features"])
            grouped[key] = {"source":key[0], "prefix":key[1], "state":trial["state"],
                            "plans":trial["candidates"]["features"],
                            "outcomes":[[0.,0.] for _ in range(n)], "observed":[False]*n}
        record = grouped[key]
        if trial["state"] != record["state"] or trial["candidates"]["features"] != record["plans"]:
            raise ValueError("Plan alternatives must share the actual incoming policy state")
        i = row["hypothesis"]
        record["outcomes"][i] = [float(trial["success"]),trial["actions"]/64]
        record["observed"][i] = True
    return list(grouped.values())


def tensors(rows,device):
    return (torch.tensor([r[k] for r in rows],device=device,dtype=dtype)
            for k,dtype in (("state",torch.float32),("plans",torch.float32),
                            ("outcomes",torch.float32),("observed",torch.bool)))


def loss(model,rows,device):
    state,plans,targets,observed=tensors(rows,device)
    out=model(state,plans)
    per=F.binary_cross_entropy_with_logits(out[...,0],targets[...,0],reduction="none")
    per=per+F.smooth_l1_loss(out[...,1],targets[...,1],reduction="none")
    # Within-state measured preference supplements the calibrated outcome loss.
    value=out[...,0].sigmoid()-.064*out[...,1].clamp_min(0)
    utility=targets[...,0]-.064*targets[...,1]
    differences=utility[:,:,None]-utility[:,None,:]
    pairs=observed[:,:,None]&observed[:,None,:]&(differences>0)
    preference=F.softplus(value[:,None,:]-value[:,:,None])*differences.clamp_min(0)
    return per[observed].mean()+preference[pairs].mean() if pairs.any() else per[observed].mean()


@torch.no_grad()
def measure(model,rows,device):
    state,plans,targets,observed=tensors(rows,device)
    model.eval();out=model(state,plans)
    value=out[...,0].sigmoid()-.064*out[...,1].clamp_min(0)
    choice=value.masked_fill(~observed,-torch.inf).argmax(-1)
    prior=plans[...,2].masked_fill(~observed,-torch.inf).argmax(-1)
    oracle=(targets[...,0]-.064*targets[...,1]).masked_fill(~observed,-torch.inf).argmax(-1)
    def selected(index):
        ys=targets[torch.arange(len(rows),device=device),index]
        return {"success":float(ys[:,0].mean()),"actions":float(ys[:,1].mean()*64)}
    return {"loss":float(loss(model,rows,device)),"learned":selected(choice),
            "prior":selected(prior),"observed_oracle":selected(oracle),
            "success_brier":float(((out[...,0].sigmoid()-targets[...,0])**2)[observed].mean()),
            "choice_indices":choice.tolist()}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train",type=Path,required=True)
    p.add_argument("--validation",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--device",default="cuda:0")
    p.add_argument("--seed",type=int,default=171)
    p.add_argument("--epochs",type=int,default=100)
    p.add_argument("--lr",type=float,default=1e-3)
    p.add_argument("--no-state",action="store_true")
    args=p.parse_args()
    if args.output.exists():p.error("Choose a fresh output")
    args.output.mkdir(parents=True)
    train,validation=load(args.train),load(args.validation)
    assert not {r['source'] for r in train}&{r['source'] for r in validation}
    torch.set_num_threads(2);torch.manual_seed(args.seed)
    config={"state_dim":len(train[0]["state"]),"use_state":not args.no_state}
    model=PlanValue(**config).to(args.device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=.01)
    settings={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    settings.update(train_contexts=len(train),validation_contexts=len(validation),
                    train_trials=sum(sum(r['observed']) for r in train),
                    objective="observed plan success, cost, and within-history preference",
                    context="fixed continuing CDM before the experimental choice")
    history=[]
    for epoch in range(args.epochs):
        model.train();order=list(range(len(train)));random.Random(args.seed+epoch).shuffle(order)
        for start in range(0,len(order),8):
            rows=[train[i] for i in order[start:start+8]]
            value=loss(model,rows,args.device);optimizer.zero_grad(set_to_none=True)
            value.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
        if epoch%10==9 or epoch+1==args.epochs:
            row={"epoch":epoch+1,"train":measure(model,train,args.device),
                 "validation":measure(model,validation,args.device)}
            history.append(row);print(json.dumps(row),flush=True)
    torch.save({"config":config,"model":model.state_dict(),"training":settings},args.output/"last.pt")
    (args.output/"training.json").write_text(json.dumps({**settings,"history":history},indent=2)+"\n")


if __name__=="__main__":main()
