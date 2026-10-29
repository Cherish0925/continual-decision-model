# Research progress

## Continuing physical learning and activity selection

The [unified study](experiments/continual_physics/README.md) is now training
from the two released physical-memory policies. One persistent model selects
experiments, internal computation and online learning. Physical updates change
the settings actually executed; neural state, physical memory and learned
corrections continue across task changes. Received outcomes enter the model
after an explicit delivery delay.

All eight fixed-budget trainings have started: two parent repetitions of the
full model, discarded physical online corrections, task-reset state and a
fixed activity schedule. Each trains on 204,800 external requests with
32-request gradient windows. The original simulator supplies exact training
transitions; only public measurements and actual setting feedback enter the
policy. Twelve native conditions are queued on a new 120-instance cohort.
The new model's performance is not yet known. Code, commands, comparison
conditions and the measured integration results are in the study directory.

Two additional fixed-computation recurrent policies use the same physical tools,
parent weights and external training budget. They retain neural state across
tasks and keep neural parameters fixed during deployment. Their training and
native evaluation queues are separate from the primary eight trainings.

An independent [Bayesian active-calibration reference](experiments/classical_calibration/RESULTS.md)
has completed both new cohorts. On the 120 confirmation instances, its
relationship-memory version completes 118 tasks with 98.41 mean native actions;
its task-reset version completes 117 with 133.34. The frozen neural parents
complete 117 and 119 with 106.68 and 102.84 actions. This reference uses the same
public measurement returns and numerical execution tools, with its own
uncertainty model and measurement-planning rule. It provides a strong comparison
for the joint models currently training.

## Physical prediction and experimental decisions

Updated 29 October 2026. The [physical-learning study](experiments/physical_belief/README.md)
connects acquired relationship knowledge directly to physical prediction,
measurement choice and setting control. Four predictive pretrainings and six
joint neural-policy trainings are complete. The shared experimental core and
physical beliefs learn together; Bayesian state updates during deployment can
change the settings proposed for later tasks.

On 120 new native instances, the complete models finish 117 and 116 tasks,
against 115 for each independent parent. Mean actions fall from 117.18 to 105.63
and from 125.91 to 106.93. Matched ordinary continuation uses 122.40 and 137.80.
These action reductions hold in both repetitions and their whole-stream
uncertainty intervals exclude zero. Completion differences are small and their
intervals include zero.

Retaining the same model's physical memory saves 7.94 and 0.72 actions relative
to discarding it, but loses one completion in each repetition. It does not
improve the first task when an earlier relationship returns. Thus the complete
learning method advances native experimental efficiency, while reliable old
knowledge recovery remains the next substantive problem. The comparison fixes
one internal refinement, resets neural state between tasks, and uses structured
observations; it does not establish general autonomous computation or learning
across different scientific domains.

The [release guide](experiments/discoveryworld/RELEASE.md#physical-prediction-and-action-v0110)
contains the six policies, four physical pretrainings, source data dependencies,
all native executions and reproduction commands. The [result table](experiments/physical_belief/results/native_instances.csv)
keeps every instance and intervention. Earlier research and its unresolved
questions remain below.

## Preceding joint activity and learning study

The [integrated model](experiments/continual_experiment/README.md) now trains the
scientific-task core, internal activity selection, bounded experience retrieval
and fast/slow prediction updates together. All eight trainings and twenty native
conditions have completed. The [results](experiments/continual_experiment/results/README.md)
include 184 common instances in twenty changing-relation streams, three reference
policies, same-weight interventions and separately identified exploratory checks.

The two full models complete 179/184 and 174/184, using 148.58 and 193.76 native
actions per instance. Calibration rehearsal completes 179/184 with 116.32.
Learned activities use fewer actions than the fixed complete activity schedule
in both repetitions, but the new models do not outperform that stronger
reference. The second model uses slow prediction updates during execution;
the first does not under the primary greedy protocol. Sampling internal
activities activates learning in both, and retained updates improve later cost
forecasts. They have not produced a clear completion or action-cost benefit over
computing and discarding the same updates.

Continuing state helps later instances while a relation remains unchanged.
It does not provide rapid recovery when an earlier relation returns: the first
returning instance costs 23.20 and 32.60 more native actions than resetting the
same weights. This motivated the physical-prediction study above.

The subsequent [state interventions](experiments/continual_experiment/results/state_controls.md)
locate that return cost in the retained neural state. Across 736 additional
native executions, carrying only the GRU and workspace states reproduces both
first-return penalties exactly. Carrying replay and online adapters while
resetting those states produces zero first-return difference from a full reset.
The same neural states substantially reduce actions during consecutive tasks
with an unchanged relationship. These results identify a task-transition effect.
This version recomputes its physical posterior from each task's fresh ledger,
so inherited physical covariance cannot explain the measured return penalty.

A further [736 native executions](experiments/continual_experiment/results/neural_state_controls.md)
separate GRU history from workspace carry. Carrying only the GRU reproduces
every measured per-instance outcome of carrying both states; carrying only the
workspace matches a full reset. The effect is therefore localized to recurrent
history on these tasks. Its contents and the way they influence later decisions
remain to be explained. The next candidate lets the recurrent update read
current evidence and the preceding workspace result, preserving the parent
policy at initialization. It is a training proposal, with no performance claim.

The study fixes final checkpoints after 1,200 updates, with 128 parallel streams
and 32 model ticks per update. Both training repetitions start from the same
v0.9 parent. Optional multimodal tokens and event-driven operation are implemented;
this experiment trains structured observations and evaluates synchronous native
tools. The [release guide](experiments/discoveryworld/RELEASE.md) supplies models,
prepared data and full execution records.

The 28 October active-measurement study trains the continuing core to select
measurements and setting experiments from incomplete evidence. Its
[method](experiments/discoveryworld/ACTIVE_LEARNING.md),
[paired results](experiments/discoveryworld/results/active_experiments.md), and
[calibration-rehearsal comparison](experiments/discoveryworld/results/calibration_rehearsal.md)
provide the working entry. Code and compact result tables are in the repository;
[release downloads](experiments/discoveryworld/RELEASE.md) provide weights,
measured datasets and complete native trajectories.

The active policy updates 26.65 million parameters, including the inherited GRU
and Transformer workspace. Two training seeds both complete all 52 new instances
from the original training relation families, using 100.15 and 90.69 actions on
average against 242.13 for full measurement. Independently trained ledger
controls and state-reset interventions distinguish continuing computation from
explicit measurement records. Transfer to the relation absent from original
native training varies between seeds.

Calibration rehearsal expands that training experience: measured quantities and
public support settings remain fixed while alternative affine hypotheses supply
generated task outcomes. Two matched continuations from independently trained parents both reach
62/64 native completions, against 54/64 and 60/64 for ordinary continued
training. Both complete exactly the same instances as full measurement, with
53.1% and 54.6% fewer actions. This is transfer
from generated practice within an explicit hypothesis family. The detailed
results retain exploration costs and every failure.

The completed [task-stream study](experiments/discoveryworld/results/active_stream.md)
then trains the same core across fresh calibrations and changing relations.
Carrying state reduces native actions by 32.9% and 31.7% relative to resetting
identical weights, with 179/184 and 176/184 completions. The effect is recent
relation reuse. The first return to an earlier relation shows no clear efficiency
gain, and the new trainings do not consistently outperform their frozen
parents. All four trainings and ten native evaluations are complete.

Earlier scientific studies, including successful task execution, prediction
training, unsuccessful computation allocation and learned complete-experiment
selection, remain in [RESULTS.md](experiments/discoveryworld/RESULTS.md). The
separate process-lifetime studies below establish useful online parameter
learning. The joint-activity study connects those mechanisms to the active scientific policy;
the subsequent physical-prediction study improves the complete method while
leaving activity scheduling and reliable returning-context recovery unresolved.

## 1. The question

A model that keeps perceiving, reasoning, acting and learning while its world changes: it keeps state across interactions, chooses its next observation, computation or learning step itself, and stays able to act while slower work runs ([RESEARCH.md](RESEARCH.md)).

The question for the next publication is narrower:

> When the environment keeps changing and computation is limited, how can a model keep and revise inference that is still in progress, turn what it experiences while acting into ability it can reuse, and keep its decisions timely?

Choosing to observe, act, keep reasoning or learn serves this question. A capability counts only when a measured behaviour shows it on tasks where other methods can also do well. The five conditions of success, and which are met, are in [research/PHASE_REVIEW.md §a3](research/PHASE_REVIEW.md).

## 2. Where to read what

| Document | Owns |
| --- | --- |
| [research/PHASE_REVIEW.md](research/PHASE_REVIEW.md) | Where the project stands after two trained rounds: goals, what the experiments show, failures and their candidate causes, falsifiable hypotheses, route |
| [experiments/lifetime/RESULTS.md](experiments/lifetime/RESULTS.md) | Every measured number of both rounds, with the controls that identify each effect |
| [experiments/lifetime/README.md](experiments/lifetime/README.md) | Processes, model options, training through deployment, evaluation protocol, exact commands |
| [experiments/discoveryworld/README.md](experiments/discoveryworld/README.md) | Executable scientific-task data collection, image/text features, training, on-policy updates and native evaluation |
| [experiments/discoveryworld/RESULTS.md](experiments/discoveryworld/RESULTS.md) | Learned scientific-task results, state intervention and the changes motivated by actual execution |
| [DESIGN.md](DESIGN.md) | The selected first multimodal architecture, the state/event contract, and a component-by-component code boundary |
| [TRAINING.md](TRAINING.md) | The training route for that architecture: phases, trajectory adapter, starting optimization settings |
| [configs/cdm_multimodal_v1.json](configs/cdm_multimodal_v1.json) | The machine-readable dimensions and settings of that architecture |
| [ref/PRETRAINED_COMPONENTS.md](ref/PRETRAINED_COMPONENTS.md) | Which pretrained models are reused, their pinned revisions and their roles |
| [STATUS.md](STATUS.md) | Capability-by-capability coverage against the objective |
| [EXPERIMENTS.md](EXPERIMENTS.md), [RUNNING.md](RUNNING.md) | The campaign design across all studies; executable commands |
| [ref/](ref/README.md) | 123 source records and the closest related methods |

Earlier studies — neural pilot, dynamic grid, numerical transport, changing maze, continual agent — keep their own READMEs and results under `experiments/`. They are completed comparisons, not prerequisites.

## 3. What the experiments established

Two trained rounds on process lifetimes. Both evaluate one uninterrupted lifetime per process on newly sampled processes, with every setting of a model on the same random stream, so differences are paired.

**First round** (eight settings, five seeds, 2·10⁷ interactions each):

| Setting | Error |
| --- | ---: |
| Recurrent policy | 0.1461 ± 0.0084 |
| Recurrent policy trained to predict, reading no prediction | 0.1088 ± 0.0024 |
| Predictions only | 0.1058 ± 0.0018 |
| All activities learned | 0.1078 ± 0.0009 |
| Trained through its learning step | 0.0971 ± 0.0038 |

The gain over a recurrent policy comes from prediction supervision shaping the state, not from the policy reading the predictions. Workspace refinement, online learning and the choice of when to learn added nothing on top. The instance's own updates slightly hurt: +0.0015 ± 0.0003 against discarding them.

**Second round** (training through deployment; harder task mix, half the requests ask for an unknown largest output, so errors are not comparable with the first round):

| Variant | Seeds | Error |
| --- | ---: | ---: |
| Recurrent policy | 2 | 0.3980 ± 0.0066 |
| Recurrent policy trained to predict, reading no prediction | 2 | 0.2968 ± 0.0369 |
| Predictions only | 3 | 0.3069 ± 0.0371 |
| Same model, no learning | 3 | 0.3260 ± 0.0047 |
| Deployed learning, cheap targets only | 2 | 0.2223 ± 0.0227 |
| One meta step per window | 3 | 0.1864 ± 0.0066 |
| Deployed learning, workspace unrolled with gate | 3 | 0.1766 ± 0.0038 |
| Deployed learning, one-tick workspace gradients | 3 | **0.1534 ± 0.0030** |
| Same, four adapter blocks | 2 | 0.1739 ± 0.0049 |
| New activity head over the fixed model | 2 | 0.1475 ± 0.0011 |

The result that matters: **the instance's own learning now carries half the performance.** Keeping its updates instead of discarding them at matched compute — every learning step still computed — lowers error by 0.154 ± 0.013 (one-tick workspace) and 0.105 ± 0.009 (unrolled). The gain holds on every held-out family (0.06–0.14) and against a real clock (0.156 against 0.311 at 64 instances, 10 ms period, 3.6 ms median response). After a condition change, error falls from 0.40 to under 0.09 with learning and stays at 0.29 without.

What made the difference is the training objective, not a new module: training replays the learning steps the agent actually took, each on the buffer it read, and lets the online loss of every later tick train the shared network and the step size through them. Precise readings, when they return, become the target of the transition they measured. Trained instead through one step per window, the same model relies on its updates far less (0.047 ± 0.013 when discarded) and its learned step grows to η = 0.12.

## 4. What failed, and what is known about why

| Attempt | Outcome | Cause |
| --- | --- | --- |
| Workspace unrolled over 16 events with a keep/revise/restart gate | 0.177 against 0.153, ~2× training time | Not established. The trained gates apply one option to every slot, whatever the request or condition; insufficient benefit from conditional revision is a candidate explanation to test |
| Bank of four adapter blocks with a state selector | Collapsed onto one block, shared by every condition (purity 0.80–0.91) | Not established. The selector is trained only through predictions within a 128-tick window, where one block suffices; retention pays hundreds of ticks later |
| Following activity values measured on copies of a process | Untested | The policy term had a zero gradient — its baseline was computed from the same probabilities and left in the graph. Corrected and regression-tested. Its weight also stayed near zero because the value head did not predict fresh states |
| Control by imagined rollouts of the model itself | Worse than the control head | The fast state was never trained on events it imagines itself |
| One model on three process families | Works, behind single-family models on each | Each specialist saw three times as much of its family within the same interactions |

Persistent workspace computation does pay once the model is trained: computing refinements but leaving the workspace unchanged costs 0.046 ± 0.010 in the best model. It was the unrolled *training* of it that added nothing.

Values from copies repeat only over short horizons and when averaged: one sampled set over 32 ticks repeats with correlation ≈ 0, four averaged greedy 4-tick sets with 0.78 (measurement) and 0.69 (compute).

## 5. Rules these failures imply

They are errors of method, not of compute. Two H100s ran 35 training runs in 66 run-hours.

1. A capability is acquired only if the task makes it pay **and** training carries that payoff to the parameters producing it. Check both before scaling.
2. Verify at small scale that a new training term changes the behaviour it is meant to train. A zero-gradient term reached a full campaign because this was not done.
3. Measure first, with simple reference methods, how much room a capability has to pay in a task; then train for it.
4. When a route repeatedly fails, reconsider the decision being learned and the
   feedback reaching it before extending the same training. Complete experimental
   choices produced informative outcomes after isolated action pairs did not.
5. Removing an option at evaluation also changes the rest of the policy. Prefer a control that holds the activity fixed — as `discard`, `discard_refine` and `noidle` do.

## 6. Code: what is implemented and what is specified

[DESIGN.md §10](DESIGN.md) is the authoritative component-by-component boundary. In summary:

**Implemented and tested**

| Area | Files |
| --- | --- |
| Continual core: GRU fast state, persistent workspace, action-conditioned predictor with instance adapters, control/activity/value heads | [cdm/core.py](cdm/core.py) |
| Process families: instruments, thermal, vehicles behind walls; shared request/measurement/condition machinery; process copies with shared randomness | [cdm/instrument.py](cdm/instrument.py), [cdm/thermal.py](cdm/thermal.py), [cdm/navigation.py](cdm/navigation.py) |
| Runtime: one tick of events, decisions, execution, adaptation; forced options; branch values | [experiments/lifetime/runtime.py](experiments/lifetime/runtime.py) |
| Training through deployment: replayed learning steps, window support reconstruction, per-tick predictions, online loss | [experiments/lifetime/objectives.py](experiments/lifetime/objectives.py) |
| Trainer, launcher, evaluation, reports, probes, branch repeatability, imagined-rollout control, real-time and latency measurement | [experiments/lifetime/](experiments/lifetime/) |
| Frozen SigLIP 2 towers, trainable projections, modality masks, feature-cache interface, multimodal assimilation and direct workspace attention | [cdm/multimodal.py](cdm/multimodal.py) |
| Exact loading of released CDM core configurations and weights | `MultimodalCore.from_lifetime_checkpoint` |
| Frozen semantic-model interface and state-conditioned reader; separate native-task generation with optional model-authored notes and a public-candidate decoder | [cdm/semantic.py](cdm/semantic.py), [semantic_policy.py](experiments/discoveryworld/semantic_policy.py), [native_decoding.py](experiments/discoveryworld/native_decoding.py) |
| Public observations, dynamic object actions, learned outcome prediction, continued h/z and candidate-conditioned reading of object-bound measurements | [cdm/digital.py](cdm/digital.py) |
| Native scientific-task interaction, exact public measurement ledger, complete trajectories, frozen feature cache, sequence imitation, on-policy actor–critic and strict prefix replay/evaluation | [experiments/discoveryworld/](experiments/discoveryworld/) |
| Persistent experimental plans, observed success/cost learning, and transfer between predictive and original action readers | [experimental_plans.py](experiments/discoveryworld/experimental_plans.py), [collect_plans.py](experiments/discoveryworld/collect_plans.py), [train_plans.py](experiments/discoveryworld/train_plans.py) |
| Joint scientific-task core, learned internal activities, bounded experience retrieval, fast/slow prediction updates and event-driven execution | [cdm/continual_experiment.py](cdm/continual_experiment.py), [training and comparison](experiments/continual_experiment/README.md) |

**Specified, not implemented**

Asynchronous semantic-model execution, a learned semantic-call policy, general
long-term episode retrieval, skill consolidation and teacher distillation from
Qwen3.6-35B-A3B remain proposed. The implemented bounded retrieval and slow
prediction updates are narrower mechanisms, now jointly trained with the
scientific-task core. Neither the frozen semantic baseline nor the binding
reader supplies joint semantic–CDM learning. Physical prediction and precise
setting still use numerical tools. Cross-instance state reuse, deployed
prediction updates and shared-parameter training have separate measured results.

The first application is a **digital experimental operator**, with [DiscoveryWorld](https://github.com/allenai/discoveryworld) selected as the external task suite ([DESIGN.md §9](DESIGN.md)). Reactor Lab is the first trained task. Handwritten reference completion establishes a usable task/data path; learned policies are evaluated separately for actual completion.

## 7. Released materials and environment

The [scientific-task release guide](experiments/discoveryworld/RELEASE.md)
provides checkpoints, records, observations and executable loading instructions.
Archives restore the relative `runs/` paths used in the commands.
Regenerable feature caches and third-party pretrained weights are excluded;
perception is downloaded from its pinned upstream model.

Earlier process-model releases remain available: v0.2.0 and v0.2.1 contain maze
and agent studies; v0.3.0 contains the first lifetime campaign; v0.4.0 contains
the second lifetime campaign. Their results remain distinct from the visual
scientific-task studies.

The current scientific-task environment uses Python 3.12, Transformers 5.4.0
and the pinned DiscoveryWorld simulator. The earlier Transformers import
mismatch is resolved. See [RUNNING.md](RUNNING.md) for installation and the task
README for collection, training and evaluation. All experiments described in
this overview have finished; planned capabilities are listed below.

## 8. Next research unit

Physical knowledge now reaches the action core and improves the complete native
policy in both repetitions. The next problem is using that knowledge reliably
under partial and noisy measurements. The completed evidence separates two
failure mechanisms that need different experiments.

In the wider-calibration prediction study, wrong relationship selection causes
more than 99.99% of first-target absolute error. Two calibration pairs can fit
all nine candidate affine relationships. Distinguishing these hypotheses needs
additional evidence whose value depends on their conflicting predictions.

The [examined native first-return failures](experiments/physical_belief/results/native_memory_cases.csv)
have a different cause structure.
The model already places 0.98–1.00 probability on the correct radiation
relationship. Some calibration points are nearly coincident, while the new
object requires distant extrapolation. In one instance, support readings near
43.96 and 44.60 are used to predict at 23.30. Subsequent support-reading changes
of only 0.02–0.04 produce setting predictions separated by tens of Hz. The
physical posterior treats these measured inputs as exact; the tool's successful
setting also represents an interval of acceptable values, not the hidden exact
resonance. Same-ledger reconstruction shows that removing memory shifts some
failed predictions by much less than the difference between the actually
executed measurement trajectories. These records do not identify memory drift
as the sole cause of failure.

The next model should therefore maintain uncertainty about measured quantities
as well as relation coefficients, and condition physical inference on the
interval evidence supplied by successful and unsuccessful settings. The action
core should compare the consequences of obtaining a more precise reading,
measuring a distinguishing quantity, and testing a setting. These choices need
training on later completion and actual interaction cost, including adverse
calibration geometry and returning contexts. The concrete goal is fewer real
experiments at matched completion, especially on the first returning task.

The failed-setting replay already implements one part of this inference. It
does not improve likelihood over the existing cumulative failure mask, and all
29 relationship choices agree. Its remaining use is as an inference component
in the joint uncertain-measurement problem; it is not an additional demonstrated
capability gain. Strong numerical decision references and same-weight physical
memory interventions remain necessary comparisons.

Extend this same belief, activity and feedback interface beyond affine
calibration. The chemistry executor already has 15 successful reference
trajectories and 262 recipe experiments, but no established learned policy.
The instrument, thermal and vehicle lifetimes supply independently timed
changing environments. Autonomous calculation scheduling, multimodal inference
and cross-domain skill accumulation remain part of the complete CDM objective.
