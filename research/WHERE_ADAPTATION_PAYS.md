# Where to test sustained adaptation

Updated 6 November 2026. The main comparison holds a capable
executor fixed while learning which experiment to request and how its result
changes subsequent decisions. This separates scientific decisions from
navigation and instrument operation. Native operation learning remains an
application goal; its current low completion rate has not established the
limits of CDM or the cause of every failure.

The [shared object-request implementation and paired training](../experiments/discoveryworld/COMMON_EXECUTION.md)
have completed their first study. After demonstration and interaction learning,
CDM completes 6/160 repeated evaluations and GRU 2/160; resetting CDM neural
state gives 8/160. Shared execution alone has not established reliable scientific
decisions or a persistent-state benefit. The next comparison trains the text
representation jointly with the decision model, alongside frozen, random-text
and lexical controls. Recovery demonstrations start from states reached by the
students. These changes address learning the recorded operations and responding
to student-generated histories. Deployment-time parameter adaptation remains a
separate uncompleted comparison.

## What the completed studies establish

In [process lifetimes](../experiments/lifetime/RESULTS.md#second-round-training-through-deployment),
training through deployed updates produces instrument-control error
0.153 ± 0.003 across three training seeds. Discarding the same updates while
still computing them increases error by 0.154 ± 0.013. The benefit also appears
under slower/narrower responses, noisier readings and faster changes. These
are three held-out settings of one process family. The separate three-family
and real-time studies have their own results. A separately trained
prediction-only model also has rounded mean 0.307; it is a different control.

The [native input study](../experiments/discoveryworld/FULL_SUITE.md#input-state-and-decoding-completed-comparison)
uses 150 training trajectories, 16,015 actions and 3,812 imitation labels.
Policies contain 53,787,710 or 55,360,574 parameters. Each final checkpoint runs
the same sixteen Easy development instances. These totals describe sampled
execution, not the official 120-instance benchmark:

| Matched factor comparison | Success totals | First setting wins / ties / loses across 16 matched model pairs |
| --- | --- | --- |
| CDM / GRU | 24 / 39 | 3 / 5 / 8 |
| Lexical / frozen SigLIP text | 47 / 16 | 13 / 3 / 0 |
| Eight-step / no explicit history | 31 / 32 | 6 / 3 / 7 |
| Inherited / random core | 31 / 32 | 5 / 5 / 6 |

Frozen text features lose or tie in every matched pair. History and inheritance
have small aggregate effects with different signs across pairs. CDM has lower
aggregate completion. These observations do not isolate a common cause. Pairs
share instances and data; they are not sixteen independent replications.

The encoder arm contains no images and does not test SigLIP's visual role in
[DESIGN.md](../DESIGN.md). Inheritance transfers a core learned under a different
action schema, with new input and action interfaces. Its aggregate result does
not show that inherited weights contain no useful information or establish a
minimum amount of training needed by a pretrained representation.

Clearing incoming neural state changes 119 of 1,156 sampled post-failure
decisions. Clearing explicit history changes none of 593 tested decisions,
while leaving previous state intact. State can affect choices without
producing successful tasks; this does not yet show useful sustained reasoning.

## The missing causal comparison

Procedural competence is a plausible constraint. Sparse successful examples,
representation, action scoring, exploration and training objectives remain
plausible contributors. The factor study did not vary competent execution while
holding experimental reasoning fixed. It cannot establish procedural knowledge
as the sole bottleneck or an architecture-independent performance ceiling.

The [DiscoveryWorld paper](https://arxiv.org/html/2406.06769v2) reports GPT-4o
ReAct at 29/120, Plan+Execute at 20/120 and Hypothesizer at 23/120, with all three
at 0/15 on Reactor Lab. Zero-shot excludes prior task training; these agents
still use observations within a task. ReAct receives recent thought/action/
observation history, and Hypothesizer updates its measurement and hypothesis
notebook after actions. Their scores cannot be attributed to pretraining alone
or treated as a control without adaptation.

The native input study evaluates fixed network weights with changing recurrent
state. It does not test the deployed parameter-update mechanism used in the
lifetime study. The two studies therefore do not establish that online learning
contributes nothing when procedural knowledge is insufficient. That proposed
boundary remains a hypothesis requiring a direct intervention.

## A common executor, three decision policies

Start with Plant Nutrients, Archaeology Dating, Proteomics, Combinatorial
Chemistry and Reactor Lab. These provide varied experimental outcomes. They
are an initial subset, not the only themes requiring inference. Retain
eight-theme native evaluation as the broader application target.

Use the same executor implementation, documented tools and action limits in three arms:

| Decision policy | Execution | Question |
| --- | --- | --- |
| Frozen language-model planner | Common executor | How well does the capable planner perform with this interface? |
| Trained recurrent planner | Same executor | What does a simpler learned state achieve on the same experience? |
| CDM planner | Same executor | Does CDM improve experiment selection and use of feedback? |

The executor receives one declared request, such as measuring an identified
sample with an identified instrument. It may navigate, obtain the tool and
perform that operation. It returns actual readings, unsuccessful operations,
elapsed time and action cost. Choosing another sample, another measurement or
a new scientific hypothesis belongs to the planner. An underspecified or
unachievable request returns to the planner. Demonstrate request execution
before treating this component as competent in the comparison.

All planners receive the same public experiment descriptions, available object
bindings and measurement record. CDM and GRU receive identical training
experience and interaction budgets. The executor retains the same public object
address catalogue for all planners; scientific beliefs and decisions stay with
the planners. Record native
operations to establish what the executor actually contributed. The
language-model planner's inference cost is measured separately.

For CDM, also compare the same weights with planner state reset between
experiments and with past numerical readings withheld. These interventions
separate state and feedback effects from changing the planner. They complement
the independently trained recurrent comparison.

## Train the consequence of information

A measurement can be useful when one result supports action A and another
supports action B, even when its average predicted state change is small.
Preserve these alternatives through prediction, state update and subsequent
action choice. Train on the actual return after the request, including native
execution costs; prediction accuracy alone is insufficient.

Use fresh interactions, unsuccessful requests and continuations from student
mistakes. Fix development and final instance sets before choosing a model.
Separate new instances from unseen response rules: different seeds can yield
the same rule. A returning-rule stream tests experience retention through
interference. Report completion, native actions, measurement requests and
elapsed time, including every executor call. Hybrid-model latency includes
the executor; it is not the latency of the small planner alone.

Existing function-belief and measured-reaction studies supply mechanisms and
controls for this work. The interface alone does not implement or prove
cross-task learning, adaptive computation or recursive self-improvement.
Those remain part of the full CDM objective and require their corresponding
training and comparisons.

[EvoBelief](https://arxiv.org/abs/2610.12789) is a relevant external reference
for experiment-driven belief revision. Its published scores become a direct
comparison only when difficulty, instances, tools, observation access and
action/call budgets are aligned. A shared task name does not establish that match.

## Execution decision

The [specified action-state comparison](../experiments/discoveryworld/FULL_SUITE.md#completed-autonomous-interaction-continuations)
has now completed both stages. Across eight matched models per variant,
action-conditioned prediction/correction improves sampled imitation-endpoint
completion from 16/128 to 22/128. After 140,366 total autonomous training
actions across all 32 models, its completion is 16/128 against the unchanged
architecture's 18/128. The other two variants also reach 18/128. The local gain
does not survive the interaction curriculum. Scaling this same curriculum
again is not the next main experiment.

These runs change candidate memory access and executed-action prediction;
they are not repeats of the input-factor study. Their results establish the
outcomes of those interventions at this budget, not the full CDM claim.

Prioritize the common-executor comparison for scientific experiment choice,
alongside measured reactions and process tasks where observations already
influence decisions. Keep native operation learning as a measured capability
gap. Do not infer an impossibility result from that gap or narrow the project
permanently to a high-level experiment selector.
