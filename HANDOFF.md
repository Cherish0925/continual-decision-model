# Research progress

## 4 November: native actions across eight scientific themes

The new [teacher and architecture comparison](experiments/discoveryworld/FULL_SUITE.md#separating-teacher-and-architecture-effects)
keeps autonomous learning, basic-operation instruction and complete-solution
distillation separate. Each has CDM and GRU conditions from the same random
initialization. Bailian Flash is collecting demonstrations through the public
native interface. The first eight Easy teacher cases give 7/8 completions;
the basic-operation training/development collections give 65/80 and 15/20.
[The records](experiments/discoveryworld/results/flash_demonstrations.json)
include unsuccessful attempts and operation counts.

Both basic-operation students have completed eight supervised epochs on the
same 2,457 training operations. Autonomous and post-imitation actor–critic
learning now run against eight simulator processes per policy. The two
complete-solution pipelines wait for their separate scientific collections,
then train and evaluate both their imitation and reinforcement endpoints.
Independent execution after basic-operation imitation completed 0/8 Easy
development cases for each student. All evaluation runs without a teacher.
These new policies use fixed
lexical/numeric features; the earlier SigLIP comparison below is separate.

Three first interaction-learning attempts exited with native-library
segmentation faults. Their trajectories and checkpoints remain available;
replacement runs start from the same initial or completed imitation weights
with one CPU thread and native stack reporting. The cause is still being
investigated. This interruption is not a task failure or a completed model
comparison. Final reinforcement and official-case scores are pending.

Continue under `runs/discoveryworld/full_suite_20261104/native/`:
`flash_skills` and `flash_skills_development` are complete;
`flash_science_corrected` and `flash_development_corrected` record the new
collections with public format-error feedback. `autonomous`,
`basic_operations` and `complete_solutions` hold the three student routes.
The retry folders preserve the failed first attempts separately.
`evaluation_initializations` holds common official-case initial random streams.
Use `updates.jsonl` for actual student interactions and optimizer updates,
and native case records for task results. No final score has selected a model.
The final endpoints are queued for independent native evaluation after their
training finishes. The [learning-procedure comparison](RESEARCH.md#does-experience-improve-how-the-model-learns)
is the next distinct research question: fixed starting task weights and empty
task memory must isolate what transfers with an acquired experience-selection
rule. The present actor–critic uses a fixed curriculum and update rule; it has
not implemented that experiment or demonstrated recursive self-improvement.

### Earlier inherited-core comparison

The [full-suite study](experiments/discoveryworld/FULL_SUITE.md#native-policy-training)
now has mixed-theme trajectory collection, a dynamic native-action policy,
source-core loading and a complete training queue. The
[protocol](experiments/discoveryworld/native_learning.json) separates eight-theme
coverage from a whole-theme holdout of Plant Nutrients and Lost in Translation.
Both compare CDM with a separately trained recurrent policy; two initial cores
give eight declared trainings. No new official-case CDM score is available.

The earlier [random-initialization integration run](experiments/discoveryworld/results/native_actor_cold_start.json) had only one progress-changing action out of 256; 32 policy updates produced no completed task. The parallel comparison above now studies learning beyond that short run.

Initial local teacher collection reached all eight themes but yielded many
interrupted or ineffective trajectories. These are data-collection attempts,
not model results. Actual feedback and failed operations remain in the records;
failed operations receive no imitation label. Both use the native public API;
out-of-reach teacher proposals remain recorded failures. There is no
task-specific solver or hidden answer. The declared comparison imitates valid
actions from successful episodes and learns received outcomes from all episodes.
Collection is text/structured-only. The native policy has 55,360,574 parameters,
excluding its frozen feature encoder.

An initial sequence-training run completed on fixed copies of 741 training and
342 development transitions, covering all eight themes. The
[run record](experiments/discoveryworld/results/native_pipeline.json) records
source initialization, features and losses. That first model completed **0/8**
Easy development tasks at seed 2001. These incomplete teacher episodes
have not taught scientific task competence. The full comparison requires completed data collection, trajectory
selection, declared training endpoints and native execution. Source and runtime
records are under `runs/discoveryworld/full_suite_20261104/`; this work remains
in progress. Current collection uses `native/train_a`,
`native/train_b` and `native/development`. Earlier collection
attempts remain separate; the local Gemma checkpoint was pretrained rather than
instruction-tuned and its attempted demonstrations are excluded. The older
v0.17.0 result below is unchanged.

The local teachers are GPT-OSS-20B instruction models. Forty environments run
concurrently across two inference servers. Their shared public interface now
defines observed object handles once, preserves intended action arguments and
retains received numerical measurements separately from authored notes. The
training queue is waiting for these collections and will then fit all eight endpoints
and execute their full suites; commands are in the study document. Neither
the teacher's successes nor the initial imitation accuracy is a CDM task score.

## 4 November: completed joint learning and broader task transfer

The [joint continuation](experiments/continual_experiment/FUNCTION_BELIEF.md#joint-prediction-and-action-learning)
connects action-return gradients to learned response hypotheses while retaining
the inherited policy at initialization. Four fresh predictors, twelve full
continuations and all digital/native comparisons are complete. The
[results](experiments/continual_experiment/FUNCTION_BELIEF.md#completed-joint-learning-results)
and [v0.17.0](https://github.com/Cherish0925/continual-decision-model/releases/tag/v0.17.0)
contain every declared final endpoint and the material needed to replay it.

The three conditions compare joint action/response gradients, prediction-only
gradients with the same forward architecture, and inherited continuation.
There are four continuation seeds from two original cores. Training includes
seven chemical response mechanisms and consistent ingredient renaming.
The [fixed test](experiments/continual_experiment/results/joint_belief_test_cohort.json)
contains 1,008 instances in 84 new streams, plus 144 paired naming replays.
It separates wider parameters, untrained response combinations, untrained
threshold laws, new target ratios and five-ingredient inputs. Original native
physics and chemistry are retention comparisons on saved initial states:
5,760 executions across twelve models on the original 480 instances.
Eighteen digital conditions produce 20,736 task executions, not 20,736
independent test instances.

Broader continuation improves transfer. Joint models complete 952, 959, 961
and 959 of the 1,008 primary tasks, reducing mean cost by 23–26% against their
original parents. Inherited continuation completes 956, 963, 952 and 966 at
similar cost. Only run 203 establishes a cost benefit over its matched
response-only gradient control; the complete four-run comparison does not
establish a stable contribution from the added gradient path.

The learned response forecasts have lower NLL than the inherited forecasts
on the same selected experiments. Native chemical costs nevertheless increase
for all joint endpoints, and physical outcomes vary. Better prediction alone
still does not establish a better complete policy. Preserve both the expanded
training continuations and original native parents as references; changing
the evaluation cohort or selecting one favorable run would not resolve this
method comparison.

Code: `cdm/belief_decision.py`, `cdm/relation_belief.py`, the shared trainer and
`experiments/continual_experiment/joint_belief.py`. Run outputs are under
`runs/continual_physics/joint_belief_20261103/`; `plan.json` fixes the conditions,
budget and endpoint. Every declared model completed 256 windows. Interrupted
resource/CPU-library attempts remain separate from complete evaluations.
No training or evaluation from this round remains running. Independent-time
operation and parameter consolidation remain open. The next work should use
these results to improve how predicted consequences affect decisions and
preserve previous capability; repeating the same gradient experiment on
these final test streams would not supply independent evidence.

## 3 November: learned response functions and finite-cost planning

The [function-belief study](experiments/continual_experiment/FUNCTION_BELIEF.md)
implements coherent alternatives for unfamiliar relationships. One learned
latent function predicts several experiments; actual measurements update its
posterior. Twelve predictor trainings, 21 comparisons on one independent
576-task cohort, and 480 native executions are complete. The shared CDM core
was not retrained in this study.

The selected 106,500-parameter response models complete 570/576 and 574/576 at
148.16 and 147.53 operations. Matched attentive predictors need 205.56 and
203.76; the original CDM with twelve ledgers needs 144.48 and 146.64. The new
representation improves on that conditional-prediction reference and reaches
the strong parent's range; it has not established superiority over the parent.

Finite-budget lookahead supplies a useful, measured use of joint hypotheses.
Allowing a hypothetical observation to update the next experimental choice
saves 118.86 [80.03, 159.32] and 124.85 [86.91, 164.06] operations relative to
the same finite planner with that update disabled. Both still update from real
observations. The cheaper information rule remains competitive. A repeated-trial
continuation approximation reverses the benefit, showing why the actual
non-repeating, budgeted execution needs to enter planning.

The direct CDM evidence-mixture integration is not a recommended replacement:
native cost rises to 91.93 / 91.35 from 66.84 / 69.34, with one lost completion
in the first repetition. Its original weights and archive scoring are intact;
new forecasts change the inherited ranking. The next integrated training must
learn subsequent action cost and preserve native competence. Predictive
likelihood by itself has not achieved that connection.

Code, model endpoints, data, complete tables, native records and the comparison
figure are supplied in [v0.16.0](https://github.com/Cherish0925/continual-decision-model/releases/tag/v0.16.0).
[Download and run](experiments/discoveryworld/RELEASE.md#learned-response-functions-v0160).
All outputs are under `runs/continual_physics/function_belief_20261103/`.
No training or evaluation from this comparison remains running. The architecture
document records implemented function inference and keeps independent-time
operation, learned computation and consolidation as the next connected work.

## 2 November: response-family results and independent confirmation

The [response-family study](experiments/continual_experiment/RESPONSE_FAMILIES.md)
implements the first two changes from the integrated direction and completes
six joint trainings on five response mechanisms. The frozen-parent comparison is complete:
twelve ledgers reduce the original parents' costs to 66.84 / 69.34 on the primary
cohort, with 240/240 completions. The earlier memory continuation does not
establish an additional gain. New training therefore starts from the originals.

The fixed-budget joint trainings compare the inherited actor, a marginal
selector, and a selector of individual response hypotheses. The new selectors
preserve inherited action scores at initialization. Training updates the shared
physical/chemical core and task interfaces. The digital extension retains the
public response/cost interface; hidden laws never enter policy inputs.

All 256-window endpoints, the 576-task primary comparison, a separate 576-task
confirmation and 3,840 native executions are complete. The hypothesis-set head
does not establish an advantage over its same-size marginal control. Removing
it from the trained models restores native chemistry to 240/240 at 64.06 / 67.70
operations. Its exported `core_only.pt` preserves all other trained weights.

On independent digital streams these simpler models use 155.18 / 156.93
operations against 161.47 / 160.93 for the frozen parents. Both paired intervals
include zero; the stronger primary-cohort gain is not a confirmed overall
advantage. The known-formula reference uses 121.12 operations and completes
574/576. The [study](experiments/continual_experiment/RESPONSE_FAMILIES.md) contains
every condition, native retention, training curves, cohorts and comparison code.

Run outputs are under `runs/continual_physics/continual_belief_20261102/`:
`parent_memory/`, `train/`, `cohorts/`, `development/`, `test/`, `confirmation/`
and `native/` are complete. No training or evaluation job from this study remains.
Models and records are supplied in v0.15.0, with repository-relative paths.

Continue with a learned distribution over unfamiliar response functions.
The current fresh component remains one marginal predictor; the new set head
only exposes distinctions among available archived predictors. Keep the
inherited actor as the action reference, and compare recurrent and learned
latent-context policies before claiming a general model advantage. Do not
continue tuning the unsuccessful extra action head on these test cohorts.
The original research objective and remaining model capabilities are preserved
in the integrated direction below.

The [current architecture](research/CONTINUAL_AGENT.md#observation-selection-buffers-and-state-estimation)
now distinguishes observation history, input buffering, state filtering,
experimental memory and pending results. Several already exist; the physical
delivery wrapper alone does not advance the world during inference. Connect
the current model to the independent-clock path while learning coherent
unfamiliar response alternatives. The implementation table marks the parent
and selector comparisons complete, so subsequent work starts from their results.

## 2 November: the plan preceding this study

Start with **[CONTINUAL_AGENT.md](research/CONTINUAL_AGENT.md)** for the current
architecture, training objectives, applications, implementation order and
compute allocation. [MEMORY_AND_EXPERIMENTS.md](research/MEMORY_AND_EXPERIMENTS.md)
contains the result calculations and corrected interpretations;
[the focused method comparison](ref/BELIEF_AND_COMPUTATION.md) specifies the
nearest learned and model-based alternatives.

The initial integration recomputed published CSV summaries and a finite
experimental-design example. The new training and evaluations are reported
above; this section preserves their motivation.

The twelve-ledger configurations save 10.8% and 14.1% against the frozen initial
models on the primary cohort, but new tasks remain more expensive. The original
parent with the same twelve-ledger memory was then the missing comparison;
the response-law study above now supplies it.
The full trained actor already has a task-return residual; a failure of its
bare progress-per-cost term is not automatically a failure of that actor.

The plan proceeds from that initialization comparison to learned relationship
beliefs and future-cost action selection on changing response families. Integrate
pending events and reusable inference in the timed process through the same
model interfaces. Retain autonomous computation, multimodal perception and
quiet-period consolidation as explicit parts of the model, with subsequent
actions and later-task improvement as their training outcomes.

The execution plan above supersedes the next-step suggestions in older dated
entries. Their measurements and implementation history are preserved below.

## Experimental memory: joint learning, action credit and retention

[v0.14.0](https://github.com/Cherish0925/continual-decision-model/releases/tag/v0.14.0)
provides the models, training inputs and complete experimental records.
[Download and reproduction instructions](experiments/discoveryworld/RELEASE.md#experimental-memory-v0140).

The [memory study](experiments/continual_experiment/EXPERIMENTAL_MEMORY.md)
completed four 200-window joint trainings and 5,280 native executions. Both
continuations recover old chemical relationships more efficiently: 59.33 and
64.23 operations, versus 88.67 and 83.07 for separately trained no-memory
references. Both memory and no-memory models complete all 240 chemical tasks. New-task costs
increase, and only the first continuation establishes an overall gain. Fixed
evidence weighting is competitive; resetting h/z does not remove the archive
benefit. Physics completes 232/240 and 234/240, versus initial 233/240 and
229/240. All conditions, predictions and paired intervals are in the study.

Two development checks are complete. Retaining exact observed responses adds
no task gain. Updating memory evidence inside a two-experiment forecast helps
that planner's mean but still does not surpass the trained action policy.

Four additional 100-window continuations and all twelve native conditions are
complete. Direct action-return credit into applicability changes chemical cost
by −3.63 [−8.45, +0.25] and +0.57 [−4.37, +5.93] relative to prediction-only
continuation. Neither gain is established; both action-trained models retain
119/120 physical and 120/120 chemical completions. New-task reuse costs remain.
The reader receives the new gradient while response experts remain detached
on that path; its actual task effect has now been tested.

Six archived tasks cannot cover a return after six intervening tasks. The
same-weight twelve-ledger intervention saves 7.88 and 9.56 operations overall
on the primary cohort, mostly through those long returns. Short returns become
slightly more expensive. On the second cohort, the same capacity change reduces
overall operations from 78.02 / 85.38 to 69.97 / 75.02, with 120/120 completed
in every condition. Both paired intervals exclude zero. This comparison uses
the same saved initialization as the other native references.
This improves retention without adding a learned write mechanism or more
parameter training. Full interval and prediction results are in the study.

Outputs: `runs/continual_physics/experimental_memory_20261101`. The completed
primary study is in `train/`, `native/`, `references/` and `cohort/`; the completed
follow-up uses `decision_train/`, `decision_native/` and `decision_cohort/`.
Capacity outputs are `capacity_native/` and `capacity_confirmation_matched/`.
An earlier capacity run used a separately saved initialization in
`capacity_confirmation/`; it is not used for cross-method comparisons.
Repeating its four conditions with the common initialization gave the same
task costs and completions. All planned trainings and native comparisons
are complete.
All primary endpoints remain fixed; development comparisons use their separate
recorded intermediate checkpoints.

The evidence supports cross-task reuse of experimental records and a concrete
retention limit. Learned applicability, persistent h/z, and extra planning have
not shown a repeatable additional advantage in these comparisons. The next
scientific problem is how a policy acquires evidence that distinguishes an old
relationship from a new one, while avoiding expensive attempts based on the
wrong archive. The action reader currently receives mixed outcome predictions;
the individual archived hypotheses and their disagreement are not explicit
action features. The two-experiment planner does propagate hypothetical
evidence, but has not surpassed the learned policy. A useful next comparison
would train experiment choice from the consequences of distinguishing those
hypotheses, preserving the current execution behavior at initialization.
This representation issue is a candidate explanation, not an established cause
of the new-task penalty. General multimodal operation and autonomous consolidation
remain broader project goals.

## Response learning: completed update and joint-training comparisons

The [response-learning study](experiments/continual_experiment/RESPONSE_LEARNING.md)
completes 16 update trainings and 1,040 native executions. Re-fitting old hidden
features or refreshing them offers little beyond probability calibration.
Training a relation-conditioned correction on currently selected experiments
produces a useful same-weight update: subsequent NLL drops from 0.7441 to
0.7241 and from 0.7689 to 0.7602. On 40 new native tasks per condition, keeping
that correction reduces operations from 117.38 to 109.95 and from 119.93 to
111.75; all tasks complete and both paired operation intervals exclude zero.
The predictive gains include 2,167 and 2,098 queries whose mixtures differ
from every earlier support experiment, with negative recipe-group intervals.
There is still no established advantage over independently trained calibration.

All four full physical/chemical joint trainings and ten native conditions are
complete: 2,400 executions on the fixed interleaved task cohort. The two joint
relation models finish all 120 chemical tasks at 75.22 and 78.99 operations.
Disabling response fitting in the same weights raises cost to 82.18 and 87.54;
both paired intervals exclude zero. Independently trained calibration uses
81.18 and 80.41 operations, and superiority over it remains unestablished.
Physical completion is 117/120 in both repetitions, versus initial 119/120
and 117/120. Returning-task gains from carrying chemical state remain unclear.

The run root is `runs/continual_physics/response_replay_20261031`; `joint/`
contains training and `joint_native/` the finished native evaluation. Training
finished at 18:20 and evaluation at 18:26 on 31 October (local time). No further
training is queued by these launchers. Both existing parents, all controls and
fixed final endpoints remain part of the comparison. The next research target
is useful knowledge retained across task changes and its advantage over strong
separately trained references. Fixed response fitting is established here;
autonomous response-compute allocation and consolidation remain open.

## Completed execution-aware study and shared-core training

The four execution-aware trainings and all eleven native comparisons are
complete. On their new 120-instance cohort, full models finish 115/120 and
117/120 at 103.61 and 101.58 mean actions; parents finish 116/120 and 117/120
at 105.02 and 97.98. Matched recurrent models use 103.69 and 100.72 actions.
The complete models have no established advantage over those references.
Retaining all task state saves 25.42 and 27.33 actions over resetting it;
both whole-stream intervals exclude zero. Numerical memory, neural state and
adapters change together in that comparison. Physical online updates alone
still have no stable benefit. The [full table](experiments/continual_physics/EXECUTION.md#completed-native-comparison)
keeps all 1,320 executions, return-task results and timing conditions.

Four [shared-core trainings](experiments/continual_experiment/SHARED_TRAINING.md)
and all eight mixed-task comparisons have completed: 1,920 native executions.
Each training uses 102,400 physical requests and 25,600 recorded chemical
returns. One continuing h/z has no consistent advantage over separate task
states with shared weights. Repetition 831 favors separate state for physics;
repetition 832 favors continuing state for chemistry. All endpoints and paired
stream intervals are preserved in the linked results. Chemistry in that study
learns from recorded actions. The completed active-feedback study below
replaces them with the model's own selected experiments.

## Completed active-feedback training

All four [active-feedback trainings](experiments/continual_experiment/ACTIVE_TRAINING.md#completed-native-comparison)
and ten native conditions are complete. The full models finish 118/120 and
119/120 physical tasks at 113.55 and 105.03 operations, and both finish every
chemical task at 97.83 and 96.56 operations. The fixed-computation references
use 97.41 and 101.93 chemical operations. No stable overall advantage is
established. Discarding response updates uses fewer operations in both
repetitions (96.92 and 93.10); the paired intervals include zero.

The deployed models execute 1,588 and 1,616 chemical learning updates, but no
consolidation or retrieval. Retaining chemical task state has no stable
first-return benefit. Current evidence supports a working joint training path
and actual response updates, while their desired task advantages remain unmet.
A completed same-history replay finds that retaining online response
corrections worsens next-outcome NLL from 0.7105 to 0.7488 and from 0.7036 to
0.7483. Both whole-stream difference intervals exclude zero. One additional
update worsens prediction further. The study links replay code and all 1,623
paired predictions. This identifies an ineffective predictive update under
the tested histories. The response-learning study above follows this result
with explicit update comparisons and new native execution results.

The full results and all 2,400 executions are linked from the study. Local raw
outputs remain `runs/continual_physics/active_feedback_joint_20261030` and
`runs/continual_physics/active_feedback_confirmation_20261030`. The completed
shared-state study remains the basis for separate task states.

## Continuing physical learning and activity selection

The [unified study](experiments/continual_physics/README.md) is now training
from the two released physical-memory policies. One persistent model selects
experiments, internal computation and online learning. Physical updates change
the settings actually executed; neural state, physical memory and learned
corrections continue across task changes. Received outcomes enter the model
after an explicit delivery delay.

All eight fixed-budget trainings have completed: two parent repetitions of the
full model, discarded physical online corrections, task-reset state and a
fixed activity schedule. Each trains on 204,800 external requests with
32-request gradient windows. The original simulator supplies exact training
transitions; only public measurements and actual setting feedback enter the
policy. Twelve native conditions have completed on a new 120-instance cohort.
Both full models fall below their frozen parents. Full seed 231
completes 94/120 at 159.29 native actions, compared with its parent's 117/120 at
106.68. Its same-weight discarded-update control also completes 94/120. Full
seed 232 completes 71/120 at 215.43 actions, against 119/120 at 102.84 for its
parent. No intermediate checkpoint replaces the predeclared endpoints.

Two concrete problems now guide the next training. Training counts each scalar
test as one request, while actual navigation, device adjustment and probing can
consume tens of native actions. Separately, failed setting intervals affect
proposals and online losses without updating the decision's relation-confidence
feature. Correcting the latter improves one fixed model on a new independent
cohort from 83/120 at 185.29 actions to 105/120 at 135.64; the strong parent still
reaches 119/120 at 106.93. The [complete record](experiments/continual_physics/README.md)
includes the matched intervention and every instance. The unstarted recurrent-
feedback candidate queue was deferred while execution-aware training and its
matching reference were prepared. The new execution path reproduces all 243
recorded requests in twelve native episodes, including intermediate settings,
activation and operation costs. It trains with a 400-operation task budget and
charges the actual operations in each experimental request. Formal training
exposed a boundary case where consistent threshold observations identify an
exact frequency. The corrected likelihood preserves that point, and both full
models restarted from their original initialization and completed all 200
training windows, as did both matched recurrent references. The completed
native comparison appears at the top of this record.
[Execution details](experiments/continual_physics/EXECUTION.md) record the first
failed attempts, corrected commands and the exact four checkpoint locations.

Two additional fixed-computation recurrent policies used the same physical tools,
parent weights and external training budget. They retain neural state across
tasks and keep neural parameters fixed during deployment. Their training and
native evaluation queues are separate from the primary eight trainings. Both
are complete: 92/120 at 166.07 actions and 84/120 at 176.55. The first full
model has no clear advantage over its recurrent reference; the second full
model is worse. These are the original scalar-training repetitions 231/232,
distinct from the four completed execution-aware trainings 241/242.

The separate initial-checkpoint comparison also finished. Before continuation,
the new runtime/module combinations complete only 63/120 and 74/120. Training
then raises the first to 94/120 and leaves the second at 71/120. The initial
combination already changes the parent's behavior substantially; the final
parent gap cannot be attributed wholly to continued training. These post-hoc
results are [reported separately](experiments/continual_physics/results/initial_native.json)
from the fourteen fixed conditions.

Six [matched public prefixes](experiments/continual_physics/INITIALIZATION.md)
now identify two local causes: tied initial activity scores force extra workspace
refinement while results are pending, and cleared failure flags/masks change the
action contract learned by the parent. A separately selectable inherited-interface
candidate preserves the parent's logits and masks on all 27 replayed decisions,
while keeping task-boundary state reset separate. It changes the available retry
actions as well as initialization, and has no new task-performance result. The
four execution-aware trainings retained their frozen source and protocol.

An independent [Bayesian active-calibration reference](experiments/classical_calibration/RESULTS.md)
has completed both new cohorts. On the 120 confirmation instances, its
relationship-memory version completes 118 tasks with 98.41 mean native actions;
its task-reset version completes 117 with 133.34. The frozen neural parents
complete 117 and 119 with 106.68 and 102.84 actions. This reference uses the same
public measurement returns and numerical execution tools, with its own
uncertainty model and measurement-planning rule. It provides a strong comparison
for the completed joint models.

## Learning from chemical experiments

The [second scientific task](experiments/continual_chemistry/README.md) reuses the
same recurrent architecture with a candidate-conditioned response model. It
learns from actual mixtures, partial cleaning, unsuccessful experiments and
their native operation costs. A new collection contains 320 tasks and 10,610
returned experiments; no hidden solution supplies a training response.

The frozen 60-instance confirmation is complete: 360 native executions across
six conditions, each completing 60/60 with no interface errors. Mean native
actions are 367.38 for primitive-recipe enumeration, 162.10 after response
training with the core frozen, and 143.35 / 278.37 for the two joint trainings.
Removing direct experiment history from the first joint model raises cost to
343.75; removing its recurrent-state input to the response head gives 178.95.
The direct-history effect is supported by recipe-group intervals. The additional
core-readout effect and the second training's improvement over enumeration have
intervals that include zero. Both training repetitions are retained.

The confirmation comprises 22 named recipe groups. Thirty instances use six
ingredient–ratio groups absent from original native sampling, but their ratio
structures occur in training and ingredient-renaming augmentation can generate
isomorphic combinations. This measures composition reuse within the chemical
task, not discovery of unseen chemical laws. The [full table and interventions](experiments/continual_chemistry/README.md#native-confirmation-of-history-conditioned-decisions)
define these conditions. The released training experience and four fixed
checkpoints are in [v0.12.0](https://github.com/Cherish0925/continual-decision-model/releases/tag/v0.12.0).

An additional [Bayesian comparison](experiments/continual_chemistry/BAYESIAN_REFERENCE.md)
uses the public reaction-function family and only the actual observed history.
Its one-step and two-step policies both complete the same 60 instances at 98.60
and 101.40 mean operations. This known-family reference makes its extra prior
knowledge explicit and is stronger than enumeration.

The learned response model has also completed a separate two-experiment
planning comparison on the original 30 development instances. Same-objective
one-step and two-step policies both finish 30/30 at 192.60 and 143.10 operations;
the existing rust-reduction policy uses 143.73. Mean inference rises from 0.666
to 1.634 seconds per episode. The paired recipe-group interval for the
two-step operation gain includes zero, and it does not improve on the existing
policy. [The complete comparison](experiments/continual_chemistry/results/planning_development.json)
retains both objectives and actual computation time. Imagined observations
change only copied state, leaving actual experience untouched. A stable benefit
from extra computation remains to be established.

## One core across both scientific tasks

The [shared-core entry](experiments/continual_experiment/README.md#a-single-core-for-physical-and-chemical-experiments)
now attaches both task interfaces to one 1,024-wide recurrent/Transformer core
with eight workspace slots and one continuing h/z state. The physical interface
and chemical warmup start from identical core weights. Sharing the core preserves
their outputs relative to running those same interfaces separately; the physical
interface uses the new runtime whose initialization is examined above. One update using an actual physical
measurement and a recorded chemical experiment sends gradients from both tasks
into that same core. Checkpoint reconstruction preserves every task head and
stores the core once. This established the joint training entry; the four
longer mixed-task trainings described above are complete. A shared learned
activity policy across both domains remains to be developed.

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
