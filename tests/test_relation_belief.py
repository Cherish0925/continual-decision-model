import torch
import numpy as np

from cdm.conditional_response import observable_log_probabilities
from cdm.relation_belief import condition, predict, information


def test_equal_marginals_have_different_updates_after_an_experiment():
    levels = torch.tensor([[0,0],[1,1]])
    opposite = torch.tensor([[0,1],[1,0]])
    p = torch.nn.functional.one_hot(levels,2).float().log()
    q = torch.nn.functional.one_hot(opposite,2).float().log()
    prior = torch.full((2,),.5).log()
    torch.testing.assert_close(predict(prior,p,torch.tensor(1)),predict(prior,q,torch.tensor(1)))
    after_p = condition(prior,p[:,0,1])
    after_q = condition(prior,q[:,0,1])
    assert predict(after_p,p,torch.tensor(1)).exp()[1,0] == 0
    assert predict(after_q,q,torch.tensor(1)).exp()[1,0] == 1


def test_censored_non_improvement_keeps_both_higher_levels_possible():
    p = torch.tensor([[.1,.2,.3,.4],[.1,.2,.6,.1]]).log()
    observed = observable_log_probabilities(p,torch.tensor(1))
    torch.testing.assert_close(observed[:,1].exp(),torch.tensor([.9,.9]))
    prior=torch.full((2,),.5).log()
    torch.testing.assert_close(condition(prior,observed[:,1]),prior)


def test_information_excludes_noise_shared_by_identical_hypotheses():
    p=torch.tensor([[[.4,.6]],[[.4,.6]]]).log()
    prior=torch.full((2,),.5).log()
    torch.testing.assert_close(information(prior,p,torch.tensor(1)),torch.zeros(1),atol=1e-7,rtol=0)


def test_finite_completion_order_respects_shared_laws_and_budget():
    from experiments.continual_chemistry.learned_belief import completion_cost
    # Half the laws succeed at the first trial; the rest at the second.
    p=np.array([[[1.,0.],[0.,1.]],[[0.,1.],[1.,0.]]])
    w=np.array([.5,.5]);cost=np.array([10.,20.]);valid=np.array([True,True])
    assert completion_cost(p,w,cost,valid,100)==24 # .5*(10+4)+.5*(10+20+4)
    assert completion_cost(p,w,cost,valid,25)==119.5 # second trial cannot finish
