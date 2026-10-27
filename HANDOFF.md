# Research progress

Updated 28 October 2026 after native
scientific-task training and evaluation. Linked documents contain the detailed
protocols and results. The scientific-task code and results are integrated into
`main`; [release downloads](experiments/discoveryworld/RELEASE.md) supply the run artifacts.

**Current study — complete experimental choices.** The new collection executes
302 complete hypothesis tests from 40 matched histories; every history has
successful and unsuccessful alternatives. Two 317,314-parameter heads learn
success and cost with or without the persistent neural state. On 24 fresh
from-reset instances, the predictive parent completes 14/24, posterior-ordered
experiments 16/24 and both learned selectors 17/24. Mean actions fall from
288.92 to 269.58/269.04. The added state input does not improve aggregate
completion. A second initialization and transfer of the fixed numerical heads
to the original successful reader are running. [Results and native records](experiments/discoveryworld/RESULTS.md#learning-complete-experimental-choices--28-october)
and the [protocol](experiments/discoveryworld/PREDICTION.md#learning-choices-between-complete-experiments)
describe this study. Outputs are in `runs/discoveryworld/experimental_plans/`.

**Latest completed study — action consequences.** Two matched joint trainings
use 141 training trajectories (32,578 actions) and 24 validation trajectories
(5,000 actions), with a 53,970,907-parameter decision network. On 24 paired native
instances, the original reader completes 18/24 and both prediction-training
arms complete 16/24. Removing extra workspace cycles from the feedback model
reduces completion to 9/24; removing forecast scoring and feedback retains 16/24
and identical action sequences in 21/24 cases. Extra computation affects the
trained model, but reading the forecasts has not improved task completion.
[Results](experiments/discoveryworld/RESULTS.md#action-consequence-prediction--27-october)
include all five methods and the six selected same-prefix computation branches.

**Completed follow-up — learning alternative actions.** Forty-eight ambiguous
training states supply 96 paired native continuations. No pair changes final
completion or progress; 34 differ by one action. Two matched updates learn from
these consequences or replay alone. On 24 fresh instances the reference
completes 17/24, and the predictive parent, replay and paired updates complete
the same 16/24. Return prediction improves without improving task completion.
The
[protocol](experiments/discoveryworld/PREDICTION.md#learning-from-alternative-actions)
and [results](experiments/discoveryworld/RESULTS.md#learning-from-paired-action-consequences--27-october)
describe the full comparison. Outputs are under
`runs/discoveryworld/action_consequences/` and
`runs/discoveryworld/consequence_learning/`. The next learning data need
alternatives that change the experiment or evidence acquired; low action-score
margins alone did not select such alternatives here.

**Latest continuation — joint training.** Observation reads now use the
in-memory PNG implementation from PR #3; the measured read time is 78.78 ms
versus 216.97 ms for the native API. A new sequence trainer updates the GRU,
Transformer workspace, projections and action reader together, and optionally
learns 0/1/2/4 internal refinements. Two three-epoch trainings have finished.
On eight new paired native instances the original reader completes 3/8,
fixed-compute joint training 4/8, and multi-depth joint training 2/8. The latter
always selects one refinement; forcing one with the same weights reproduces
its trajectories. These results do not establish adaptive-computation gains.
The original successful reader remains the reference. Full conditions and
per-instance data are at the top of [task results](experiments/discoveryworld/RESULTS.md).
The actual-return comparison has also finished: 32 continuations from eight
training prefixes change only the first decision's refinement count. All
prefixes match, and all four depths produce identical executed action
sequences within each group. Those states supply no task-return preference
for additional computation. Local plans and outputs are under
`runs/discoveryworld/joint_compute/compute_branches/`; the committed CSV is
linked from the task results. Those joint-computation runs have finished.

**The task policy now completes experiments from reset.** Across two cohorts of sixteen Reactor
Lab Normal instances, two independently trained action readers complete **14/16
and 13/16** tasks. On the first eight-instance cohort, they complete 8/8 and 7/8. Each chooses its own measurements and calibrations and uses the
same learned numerical prior and explicit scalar/probe tool. Matched readers
with the explicit experimental-evidence input zeroed complete **0/8**. Clearing
the trained policy's workspace while preserving recurrent state and measurement
records also yields **0/8**. A reader of received facts without derived numerical
predictions completes **11/16** across both cohorts: access to experimental facts is effective, while
the extra decision value of exposing posterior predictions remains unresolved.

Only the 442,753-parameter action reader is trained in this comparison; source
perception, recurrence and the numerical prior stay frozen. Both evaluation cohorts
include a held-out relation family within Reactor Lab, not new scientific
domains. The model uses nine explicit affine hypotheses and an explicit scalar
primitive. These are autonomous whole-task results under that defined method,
not purely neural equation discovery or cross-task skill retention.

The [task results](experiments/discoveryworld/RESULTS.md) retain per-instance native
evaluations and the earlier failed routes. Supplied-prefix takeovers previously
completed 14/16 in each of two training seeds; the new from-reset study closes
the evidence-acquisition gap in that comparison. Successful trajectories still
contain about thirty repeated measurements of some objects. The first actual-return
update reduces second-cohort completion from 6/8 to 5/8; a training-state
KL step-length rule restores 6/8 without saving actions. That round has not
improved the method. A separate conditional-mixture
controller comparison repairs one failure but introduces another; on a fresh
cohort it completes 7/8 against the original tool’s 8/8. It has not replaced the
reference. A second actual-return round
uses eight fresh training instances and the learned value estimator; it remains
at 6/8 with the original mean action count. Paired native continuations then
show that earlier departure saves 29 and 24 actions in two successful training
instances; two other pairs remain failed. Learning from the two informative
pairs reduces development completion to 5/8, versus 6/8 for matched extra
reference replay. All runs have finished. The results, losses and actual
trajectories are retained under `runs/discoveryworld/belief_actions/` and in
the linked result tables. The release guide maps these records to downloadable archives.

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
4. Leave a branch as soon as results show no gain.
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

## 8. Next steps, in order

Joint prediction training makes the policy dependent on extra workspace updates,
but reading the forecasts has not improved completion. The next paired-action
study improves return prediction while preserving the same sixteen completions;
its 48 low-margin action pairs produce no completion or progress differences.
The useful next unit of comparison is an experimental choice with consequences
for later evidence or operation, followed through to its outcome. The numerical
tool remains useful; its calculations are not learned workspace operations.

1. **Learn the trade-off between more evidence and acting now.** Two reward
   rounds and the two-pair preference update have finished without improving
   completion. The direct action interventions do save work in two histories,
   but teaching only those departures does not transfer reliably. Collect
   complete experimental choices covering both useful and unnecessary measurements,
   including uncertain predictions and variable measurement noise. Train the
   choice on later intervention outcomes; evaluate it from reset on new
   instances. The original reader and its numerical tool remain the reference.

2. **Make uncertainty useful to action and computation.** The current
   received-facts control already completes eleven of sixteen tasks. Construct
   ambiguous and variable-noise decisions where acquiring a discriminating
   observation can change a useful intervention. Train and measure when more
   measurement or internal computation pays; a prediction feature alone is not
   evidence of experiment design.
3. **Train learning across task return.** Preserve experience across condition
   changes and A→B→A returns, then measure saved interactions and recovered
   performance. The current scientific-task comparisons reset each instance.
4. **Extend the same learned process to other scientific tasks.** Chemistry and
   nutrients require their own observed quantities and interventions, while
   keeping the continuing core and action/compute interfaces shared. Compare
   against recurrent and pretrained-agent baselines with the same observations.

Keep the instrument/thermal/vehicle families as the controlled setting for mechanism analysis. They answer questions about timing and adaptation that a visual suite does not.
