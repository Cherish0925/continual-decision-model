# Experimental decisions across changing response laws

This study follows the [integrated research direction](../../research/CONTINUAL_AGENT.md).
It first separates the value of experimental records from the value of training
their reader, then compares action representations on changing response laws.
The official DiscoveryWorld protocol remains unchanged. The new generators are
a digital extension with the same public observations and operation costs.

## Starting model: memory without extra training

The previously missing comparison is complete. Both original response parents
were evaluated with twelve experimental ledgers and fixed evidence weighting.
Predictors, within-task recurrence, tools and saved environment initializations
were preserved. This reuses the primary twenty-stream cohort; it is not another
independent test cohort.

| Mean operations over 240 chemical tasks | Continuation parent 911 | Continuation parent 912 |
|---|---:|---:|
| Original model without archived records | 78.22 | 84.30 |
| Original model with twelve ledgers | **66.84** | **69.34** |
| Earlier memory continuation with twelve ledgers | 69.77 | 72.43 |

Every model in this table completes 240/240 tasks. Adding memory to the original
parents saves 11.38 [4.71, 18.05] and 14.96 [9.85, 20.40] operations, using a
paired bootstrap of complete streams. The earlier continuation costs 2.93
[−3.09, 9.86] and 3.09 [−2.20, 8.91] more than these simpler starting models.
Its extra training therefore has no established benefit in this comparison.

New tasks still cost more with memory: 89.17 versus 79.17 and 93.17 versus 84.80.
Returns cost less: 44.51 versus 77.28 and 45.51 versus 83.79. These observations
motivate using the original parents for the next training, and examining action
selection when several old and new explanations remain possible.

## The representation being compared

The inherited actor receives a mixture prediction for each possible experiment.
The new selector can additionally read each archived predictor's responses
across the available experiments. A shared encoder represents each response
function; a candidate then queries that set. Mixture weights normalize the set,
and the computation is invariant to its ordering. Its final residual starts at
zero, preserving the parent's action scores before training.

Why preserve the functions? Two experiments can have the same marginal outcome
distribution while partitioning the possible relationships differently. Those
partitions can leave different future actions useful. Marginal uncertainty alone
does not encode that distinction. This is a motivation from belief-based decision
making, not a new theorem or a claim that the proposed neural encoder has already
learned an optimal policy.

The declared comparison has three independently continued conditions per parent:

| Condition | Action input and training |
|---|---|
| Inherited | Existing progress/cost term and task-return residual |
| Marginal | New selector of the same parameter size, receiving only mixture forecasts |
| Hypotheses | New selector receiving the individual forecasts and their weights |

All conditions update the shared recurrent core, workspace, response model and
task interfaces. The two new selectors receive the same longitudinal actor–critic
objective. Expert forecasts are detached on this action-gradient path; actual
chosen outcomes train the predictor. The new head outputs action-score residuals,
not calibrated remaining-cost estimates. It does not yet introduce latent
function particles inside the fresh component or autonomous archive writing.

## Response laws, task streams and budgets

Five response mechanisms are present during training:

- cosine response, retaining the official transition;
- anisotropic distance, with different weights on component dimensions;
- pair interactions, with nonlinear dependence on combinations of ingredients;
- inhibition above a component threshold;
- multiple successful compositions.

Each has an attainable solution among the public candidate actions. The physical
state retains `min(current rust, intrinsic response)`, so an unchanged result is
censored. Experiment and exit operations are charged, including unsuccessful
attempts and budget exhaustion. Generator parameters and task identities remain
inside the simulator. The model observes proportions, past measurements, current
rust and operation costs.

Twelve-task training streams contain new relationships, exact returns and changed
familiar targets. Ingredient counts remain fixed within a stream; parameters and
return order vary. A changed multiple-solution relationship may retain a useful
old solution, so recognizing a change is not required when the old action still
achieves the objective.

Six joint trainings are declared: three conditions, two parent continuations,
256 windows per run, batch 24, sixteen chemical and eight physical requests per
window. This is 98,304 chemical and 49,152 physical requests per condition.
The final window is the endpoint; test outcomes do not select checkpoints.
The two continuations reuse existing parents rather than pretraining new cores.

The frozen development set has 192 tasks. The final set has 576 tasks in 48
streams: independently sampled parameters from trained mechanisms, wider parameter ranges, and
the withheld combination of interaction and inhibition. The combination uses
operators seen separately during training. It is not a wholly unspecified
physical law. Discrete targets, including exact cosine laws, can recur from
training; the first group establishes new-stream performance, not guaranteed
novelty of every law. The wider group expands the parameter support and still
overlaps the training range. Completion, operations including failures, and paired stream
intervals are reported by group and by new/changed/return exposure.

The observed state has four response levels. If a composition has intrinsic
level $f(q)$ and the device is currently at $r$, its returned level is
$\min(r,f(q))$. An unchanged reading therefore only bounds the intrinsic
response. The actions are 55 or 168 primitive compositions for three or four
ingredients, followed by a completion operation. Each composition costs
$6+\sum_i q_i+|\{i:q_i>0\}|$ operations, plus two after the first experiment;
preparation costs fourteen and completion costs four. Every condition receives
the same public history, candidates, remaining budget and response interface.

The joint model contains approximately 47.84 million parameters. These runs use
structured scientific observations; no additional vision or language backbone
is invoked. The added selector contains 16,849 parameters and is trained
together with the existing core and prediction interfaces.

## Model-based references

Alongside frozen parents and the trained alternatives, a finite-grid belief
reference knows the five generating formulas. It does not receive the current
parameters, task identities or test answers. Its discretized prior has 1,155
functions for three ingredients and 3,864 for four. A likelihood floor of 0.005
accommodates grid mismatch; it is an approximation to the deterministic task.

Three fixed selection rules are compared on development data: immediate
completion per cost, completion plus information per cost, and a two-step
approximation with a geometric continuation estimate. The information rule is
selected by completion first, then operations. Its coefficient is fixed at 0.5.
This reference has generator knowledge unavailable to the neural model; it is
not a matched-information learned baseline or an optimality bound.

Cost-ordered enumeration and a privileged solver that knows the actual current
law measure two further reference points. The latter confirms achievable success
and cost with perfect information; it does not measure attainable performance
for a learner that must discover the law.

## Running the study

```bash
python -m experiments.continual_experiment.compare_parent_memory \
  --previous runs/continual_physics/experimental_memory_20261101 \
  --output runs/continual_physics/continual_belief_20261102/parent_memory

python -m experiments.continual_experiment.train_response_families \
  --previous runs/continual_physics/experimental_memory_20261101 \
  --output runs/continual_physics/continual_belief_20261102/train \
  --updates 256 --batch 24 --conditions hypotheses marginal inherited \
  --concurrency-per-gpu 2
```

The original parents and training inputs are available in
[v0.14.0](https://github.com/Cherish0925/continual-decision-model/releases/tag/v0.14.0).
The same inherited checkpoint can be evaluated with the new digital evaluator.
The selector and response families are still being trained in this record;
their final model results will be added when the fixed endpoints complete.

This experiment addresses relationship-conditioned action and transfer. A
learned latent-context baseline, independent-time operation, revision of ongoing
inference and lasting consolidation remain separate parts of the integrated
research direction. The known-family reference and internal marginal comparison
do not replace RL²-style and VariBAD-style learned competitors.
