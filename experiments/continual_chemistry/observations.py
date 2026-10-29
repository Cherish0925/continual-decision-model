"""Public chemical experiment tensors for the shared continuing core."""
from __future__ import annotations
import math
from pathlib import Path
import torch
from torch import nn
from cdm.active_experiment import ActiveExperimentConfig,ActiveExperimentPolicy,load_active_checkpoint
from cdm.continual_experiment import ContinualExperimentConfig,ContinualExperimentPolicy,load_continual_checkpoint
from experiments.discoveryworld.collect_chemistry import integer_recipes

MAX_COMPONENTS=8
EVENT_DIM,EVIDENCE_DIM,CANDIDATE_DIM=36,20,30


def recipes(component_count,max_parts=6):
    if not 1<=component_count<=MAX_COMPONENTS:
        raise ValueError(f'This tensor interface supports 1–{MAX_COMPONENTS} observed components')
    return list(integer_recipes(component_count,max_parts))+[(0,)*component_count]


def observe(state,*,max_parts=6,max_actions=1000,elapsed_actions=0,device='cpu',candidate_recipes=None):
    """Encode only the current public state and already returned experiments.

    A zero-count candidate means using the cleaned key to leave. It is available
    only after rust is visibly absent; all nonempty recipes are then disabled.
    """
    n=len(state['components']);choices=recipes(n,max_parts) if candidate_recipes is None else list(candidate_recipes)
    history=state['history'];rust=int(state['rust_level'])
    event=torch.zeros(1,EVENT_DIM,device=device);event[0,8:8+n]=1
    event[0,20]=len(history)/(len(history)+10);event[0,22]=rust==0
    event[0,23]=elapsed_actions/max_actions;event[0,35]=1
    if history:
        last=history[-1];event[0,:n]=torch.tensor(last['component_counts'],device=device)/max_parts
        event[0,16:20]=torch.tensor([last['rust_before']/3,last['rust_after']/3,
                         (last['rust_before']-last['rust_after'])/3,math.log1p(last['native_actions'])/math.log(101)],device=device)
        event[0,21]=1
    else:event[0,16:18]=rust/3
    evidence=torch.zeros(1,len(history)+1,EVIDENCE_DIM,device=device)
    evidence[0,:,8:8+n]=1
    evidence[0,0,16:18]=rust/3;evidence[0,0,18]=elapsed_actions/max_actions;evidence[0,0,19]=1
    for index,row in enumerate(history):
        evidence[0,index+1,:n]=torch.tensor(row['component_counts'],device=device)/max_parts
        evidence[0,index+1,16:20]=torch.tensor([row['rust_before']/3,row['rust_after']/3,
                           math.log1p(row['native_actions'])/math.log(101),(index+1)/(len(history)+1)],device=device)
    candidate=torch.zeros(1,len(choices),CANDIDATE_DIM,device=device)
    mask=torch.ones(1,len(choices),dtype=torch.bool,device=device)
    for index,counts in enumerate(choices):
        total=sum(counts);candidate[0,index,:n]=torch.tensor(counts,device=device)/max_parts
        candidate[0,index,8:8+n]=torch.tensor(counts,device=device)/max(total,1)
        candidate[0,index,16:16+n]=1
        candidate[0,index,24]=total/max_parts;candidate[0,index,25]=rust/3
        seen=[row for row in history if tuple(row['component_counts'])==counts]
        candidate[0,index,26]=len(seen)/(len(seen)+1)
        if seen:
            candidate[0,index,27]=seen[-1]['rust_after']/3;candidate[0,index,28]=1
        candidate[0,index,29]=math.log1p(total)/math.log1p(max_parts)
        mask[0,index]=(total==0) if rust==0 else (total>0)
        # The public chemical response is deterministic and rust only decreases.
        # Repeating a returned recipe cannot improve the current rust if its
        # previously observed endpoint is at least the present rust level.
        if total and any(row['rust_after']>=rust for row in seen):
            mask[0,index]=False
    return {'event':event,'evidence':evidence,'evidence_mask':torch.ones(1,len(history)+1,dtype=torch.bool,device=device),
            'candidates':candidate,'action_mask':mask},choices


def model_from_core_checkpoint(path,*,device='cpu',initialization_seed=730,internal_steps=4):
    """Reuse the trained GRU/workspace; chemical input/output heads are new."""
    torch.manual_seed(initialization_seed)
    source,_=load_active_checkpoint(path,'cpu')
    base=ActiveExperimentPolicy(ActiveExperimentConfig(EVENT_DIM,EVIDENCE_DIM,CANDIDATE_DIM,outcome_dim=4),source.core)
    model=ContinualExperimentPolicy(base,ContinualExperimentConfig(max_internal_steps=internal_steps,
                                    activity_mode='fixed',fixed_refinements=0))
    return model.to(device)


def feedback(record,device='cpu'):
    """Returned physical progress and actual operation cost, not task scores."""
    return torch.tensor([[(record['rust_before']-record['rust_after'])/3,
                          math.log1p(record['native_actions'])]],device=device)


def load_chemistry_policy(path,*,device='cpu'):
    """Load a trained chemical checkpoint or initialize its heads from a core."""
    saved=torch.load(path,map_location='cpu',weights_only=True)
    if saved.get('kind')=='chemistry_reaction_v1':
        from .reaction import load_reaction_checkpoint
        return load_reaction_checkpoint(path,device)
    if saved.get('kind')=='chemistry_continual_v1':
        model,record=load_continual_checkpoint(path,device)
        return model,record
    return model_from_core_checkpoint(path,device=device),{'kind':'untrained_chemistry_heads'}


def outcome_action_scores(out,observation):
    """One-step expected rust reduction per predicted native operation.

    Both terms are trained from actual conditional experiment returns. The
    untrained base action head is not used to select the chemical experiment.
    """
    probabilities=out['outcomes'].softmax(-1)
    current_rust=observation['candidates'][...,25]*3
    levels=torch.arange(4,device=probabilities.device,dtype=probabilities.dtype)
    reductions=(current_rust[...,None]-levels).clamp_min(0)
    expected_reduction=(probabilities*reductions).sum(-1)
    expected_cost=out['predictions'][...,1].clamp(0,math.log(1001)).expm1().clamp_min(1)
    return (expected_reduction/expected_cost).masked_fill(~observation['action_mask'],-torch.inf)
