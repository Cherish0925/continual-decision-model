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


def test_batched_joint_conditioning_preserves_likelihood_and_query_order():
    from cdm.relation_belief import RelationBelief, RelationBeliefConfig
    torch.manual_seed(71)
    model = RelationBelief(RelationBeliefConfig(particles=7, width=16, latent=5))
    queries = torch.rand(2, 4, 8)
    components = torch.ones(2, 8, dtype=torch.bool)
    support = queries[:, :3].clone()
    before = torch.tensor([[3, 2, 1], [3, 1, 0]])
    after = torch.tensor([[2, 1, 1], [1, 1, 0]])
    valid = torch.tensor([[True, True, True], [True, True, False]])
    expected, likelihood = model.conditioned(queries, components, support, before, after, valid)
    joint = model.joint_conditioned(queries, components, support, before, after, valid)
    torch.testing.assert_close(joint['marginal'], expected)
    torch.testing.assert_close(joint['evidence'], likelihood)
    order = torch.tensor([2, 0, 3, 1])
    permuted = model.joint_conditioned(queries[:, order], components, support, before, after, valid)
    torch.testing.assert_close(permuted['marginal'], expected[:, order])


def test_action_return_can_train_hypotheses_only_in_joint_condition():
    from cdm.relation_belief import RelationBelief, RelationBeliefConfig
    from cdm.belief_decision import BeliefDecision, BeliefDecisionConfig
    torch.manual_seed(72)
    model = RelationBelief(RelationBeliefConfig(particles=7, width=16, latent=5))
    queries = torch.rand(2, 3, 8)
    components = torch.ones(2, 8, dtype=torch.bool)
    support = queries[:, :1]
    before = torch.full((2, 1), 3, dtype=torch.long)
    after = torch.ones_like(before)
    valid = torch.ones_like(before, dtype=torch.bool)
    for learning in ('joint', 'prediction_only'):
        model.zero_grad(set_to_none=True)
        head = BeliefDecision(4, 5, BeliefDecisionConfig(width=16, learning=learning))
        with torch.no_grad():
            head.output[-1].weight.fill_(.1)
        result = model.joint_conditioned(queries, components, support, before, after, valid)
        features = torch.randn(2, 3, 4, requires_grad=True)
        logits = head(features, result['functions'], result['weights'], model.particles)
        (-logits.log_softmax(-1)[:, 0]).mean().backward()
        assert features.grad is not None and torch.isfinite(features.grad).all()
        gradient = sum(float(p.grad.abs().sum()) for p in model.parameters() if p.grad is not None)
        assert (gradient > 0) == (learning == 'joint')


def test_public_renaming_does_not_change_simulator_laws_or_returns():
    from experiments.continual_chemistry.response_families import ResponseFamilyEnvironment, EXPANDED_FAMILIES
    plain = ResponseFamilyEnvironment(None, 3, 75, 'cpu', families=EXPANDED_FAMILIES)
    renamed = ResponseFamilyEnvironment(None, 3, 75, 'cpu', families=EXPANDED_FAMILIES, rename_components=True)
    assert plain.current_laws == renamed.current_laws
    assert not torch.equal(plain.observation()['candidates'], renamed.observation()['candidates'])
    action = torch.tensor([0, 1, 2])
    a, b = plain.step(action), renamed.step(action)
    torch.testing.assert_close(a['after'], b['after'])
    torch.testing.assert_close(a['cost'], b['cost'])
