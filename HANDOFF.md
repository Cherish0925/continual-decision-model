# Research progress

Updated 28 October 2026. The current study trains the continuing core to select
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

Earlier scientific studies, including successful task execution, prediction
training, unsuccessful computation allocation and learned complete-experiment
selection, remain in [RESULTS.md](experiments/discoveryworld/RESULTS.md). The
separate process-lifetime studies below establish useful online parameter
learning. Connecting that capability and adaptive computation to this active
scientific policy remains the next research step.

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

**Specified, not implemented**

Asynchronous semantic execution and a learned semantic-call policy; long-term episode retrieval; consolidation; teacher distillation from Qwen3.6-35B-A3B. The native decoder constrains syntax and public argument domains while the frozen model chooses the action; it supplies no scientific solution. Neither that baseline nor the new binding reader connects semantic reasoning to CDM through joint training. The scientific-task path includes deployed A/B updates and a causal future-query training objective. Its numerical prediction-to-action connection currently uses an explicit scalar controller; the action head has not learned this control. A learned computation schedule and cross-task retention also remain unimplemented. Shared-parameter actor–critic updates are distinct from those capabilities.

The first application is a **digital experimental operator**, with [DiscoveryWorld](https://github.com/allenai/discoveryworld) selected as the external task suite ([DESIGN.md §9](DESIGN.md)). Reactor Lab is the first trained task. Handwritten reference completion establishes a usable task/data path; learned policies are evaluated separately for actual completion.

## 7. Released materials and environment

The [scientific-task release guide](experiments/discoveryworld/RELEASE.md)
provides checkpoints, records, observations and executable loading instructions.
Archives restore the relative `runs/discoveryworld/` paths used in the commands.
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

## 8. Next steps

The active-measurement and rehearsal studies now provide the starting policy.
The next experiment keeps its GRU and workspace across fresh experimental
instances. Each instance has its own measurements and two calibrations; a
relation can persist, change without a label, and later return. The explicit
measurement ledger and numerical fit restart at each instance. Training and
paired native execution will compare retained state with a model trained to
reset at those same boundaries. The question is whether previous experiments
reduce the observations needed in later ones while preserving completion.

This measures learning carried by state. Deployment-time parameter updates and
idle-period consolidation require their own training and subsequent-task
comparison. The successful process-lifetime objective supplies an existing
implementation to connect once the scientific task stream is working.

Adaptive computation remains another substantive part of the goal. Earlier
fixed-depth comparisons found little decision value in additional refinements.
A useful continuation must train an identifiable prediction or plan refinement,
then learn when its subsequent payoff warrants its cost. The present
one-refinement scientific policy is the reference, not an adaptive-compute result.

Chemistry and nutrient tasks can broaden the same process beyond numerical
calibration. Their observed quantities and interventions need task interfaces;
the continuing core, activity choices and future-task learning objective should
remain shared. The instrument, thermal and vehicle studies continue to provide
independently timed environments for response and online-learning comparisons.
