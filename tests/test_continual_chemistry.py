from copy import deepcopy
import torch
from experiments.continual_chemistry.observations import observe


def public_state():
    return {'components':['A','B','C'],'rust_level':2,'history':[
        {'component_counts':[1,0,0],'rust_before':3,'rust_after':2,'native_actions':8}]}


def test_future_outcomes_and_task_scores_do_not_enter_the_observation():
    state=public_state();first,_=observe(state)
    forged=deepcopy(state);forged.update(hidden_recipe=[2,1,0],task_score=.9,future_rust=0,seed=99)
    second,_=observe(forged)
    for name in first:assert torch.equal(first[name],second[name])


def test_same_recipe_is_conditioned_on_the_returned_persistent_state():
    first=public_state();second=deepcopy(first);second['rust_level']=1
    first_obs,choices=observe(first);second_obs,_=observe(second)
    index=choices.index((1,0,0))
    assert first_obs['candidates'][0,index,25]!=second_obs['candidates'][0,index,25]
    assert first_obs['evidence'][0,1,16]==1
    assert first_obs['evidence'][0,1,17]==torch.tensor(2/3)


def test_door_action_requires_publicly_observed_clean_key():
    state=public_state();obs,choices=observe(state)
    assert choices[-1]==(0,0,0) and not obs['action_mask'][0,-1]
    state['rust_level']=0;clean,_=observe(state)
    assert clean['action_mask'].sum()==1 and clean['action_mask'][0,-1]


def test_returned_deterministic_recipe_is_not_repeated():
    obs,choices=observe(public_state())
    assert not obs['action_mask'][0,choices.index((1,0,0))]
    assert obs['action_mask'][0,choices.index((0,1,0))]
