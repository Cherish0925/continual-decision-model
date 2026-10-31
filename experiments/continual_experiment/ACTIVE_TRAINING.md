# Learning from selected experiments

The preceding [shared-core comparison](SHARED_TRAINING.md#completed-comparison)
found no consistent benefit from passing one neural state between physical and
chemical tasks. It also left a more consequential gap: chemical predictions
learned from recorded behavior, while deployment selected experiments using
those predictions. Earlier high-confidence chemical predictions often failed
on the experiments the planner actually selected.

This study trains physical and chemical decisions through one shared core.
Both interfaces select experiments and receive environment feedback. Each
keeps its own continuing state. Chemical online updates now change the outcome
distribution used to select the next experiment, and later received outcomes
train the update rates through those updates.

## Model and learning

The inherited chemical model predicts the distribution of a reaction's rust
endpoint from the proposed mixture, prior returns and recurrent state. A small
per-stream correction acts on its final response features. If the current
rust level is `c`, the observed result is `min(c, R)`: unchanged rust identifies
a range of possible outcomes, not one exact outcome. The update therefore uses
the gradient of the observed likelihood:

\[
 W^+ = W - \eta\,N^{-1}\sum_i
 \phi_i\left(p_i(R)-p_i(R\mid\text{received result})\right)^\top.
\]

Fast and consolidated corrections have separately trained rates. ACT, REFINE,
RETRIEVE, LEARN and CONSOLIDATE use the existing continuing activity interface.
ACT dispatches a mixture. LEARN and CONSOLIDATE update response predictions
and the existing progress/cost predictor; REFINE and RETRIEVE change the
workspace. The activity policy reads the continuing state, available feedback,
internal work already performed and progress/cost replay error. The resulting
response probabilities enter the experiment policy. A zero-initialized action
residual starts with the inherited progress-per-operation ranking and learns
from subsequent task return.

Only selected experiments receive response labels. The complete candidate set
is scored to choose an experiment; no unexecuted alternative supplies a training
outcome. The selected response's proper likelihood trains the core and response
model. Actor–critic losses train experimental actions and internal choices.
Gradients through online updates reach their features and step sizes through
later predictions in the training window. Future return is bootstrapped at the
window boundary. The inherited physical objective and parent-policy anchor
remain active in the same optimizer.

One chemical lifetime contains two tasks using relationship A, two using B,
then two using A again. Current-apparatus evidence and fast corrections reset
at a task boundary; h/z and consolidated corrections continue until the
lifetime ends. This supplies later uses of acquired information. Whether the
model actually learns useful retention is measured by the return-task results.

The generic ideas of online gradient adaptation, recurrent learning and
actor–critic training are established methods. The present experimental
question is whether connecting them to **action-relevant, received-outcome
prediction** improves this continuing model over its inherited policy and a
jointly trained fixed-computation reference.

## Environment and budget

Chemical training uses the official simulator's cosine-threshold response and
persistent rust rule, evaluated in batches. The hidden training recipes belong
to the existing 320-instance training split and remain inside the environment.
Policy inputs contain only public component identities, current rust, elapsed
operations and prior returned experiments. Native execution is used for the
final comparisons.

The batched environment reproduced all **10,610** recorded returns and costs,
and the public input tensors at **949** recorded prefixes. A truncated
experiment consumes the remaining budget without supplying an unreceived
response label. Setup consumes the native task budget and is included in task
totals; the training reward charges chosen experiments and final exit, with a
completion reward and a separate internal-computation charge. These are
operation costs, not measured wall-clock time.

Each run has 200 optimizer windows, 32 streams, 16 physical requests and eight
chemical requests per window: **102,400 physical and 51,200 chemical requests**.
Chemical requests include exit actions and budget-ending attempts; the number
of actual returned experiments is logged separately. Two continuation seeds
start from the same physical parent and chemical warmup 753. This warmup is the
stronger previously published initialization; the two continuations do not
constitute independent pretrained models.

| Training | Chemical computation | Other training conditions |
|---|---|---|
| Learned activity | Up to two selected internal activities per received event | Shared core, active experiments, response and return losses |
| Fixed computation | One received-observation update, then an action | Same initial weights, objectives and external-request budgets |

Each mode has two continuations. Their internal work and primitive operation
totals can differ. Parameters are owned by one optimizer, with the inherited
shared-core rate 3e-6 and chemical-interface rate 1e-4. Final checkpoints are
fixed at update 200.

```bash
python -m experiments.continual_experiment.launch_active_tasks \
  --output runs/continual_physics/active_feedback_joint \
  --updates 200 --batch 32
```

## Fixed native comparison

The [manifest](active_confirmation.json) contains twenty streams. Each combines
six physical and six chemical tasks with A–A–B–B–A–A relationships and distinct
native instances. Chemical identities are used by the evaluator to select the
return structure; they are never model inputs. The response families are known,
so this evaluates new instances and relationship recovery, not unseen laws.

For each continuation, evaluate the initial model, both trained conditions,
and the learned model with either response corrections discarded or chemical
state reset at each new apparatus. Discarding computes the same response
update but does not retain its result. Resetting clears h/z and consolidated
corrections together. Physical state is unchanged by these chemical controls.
All ten conditions use identical simulator randomness and native budgets:
400 operations per physical task and 1,000 per chemical task.

The summary reports completion, operations including failures, and the first
task after returning to A. Paired intervals resample the complete twelve-task
stream. Chemical traces retain predictions, actual returns, internal activity
counts and response-correction norms. This makes it possible to distinguish
better outcome prediction, different actions and useful retained learning.

```bash
PYTHONHASHSEED=0 python -m experiments.continual_experiment.shared_confirmation \
  --study active --manifest experiments/continual_experiment/active_confirmation.json \
  --training-root runs/continual_physics/active_feedback_joint \
  --output runs/continual_physics/active_feedback_confirmation
```

## Completed native comparison

All four trainings completed their fixed 200-window budget. Each dispatched
102,400 physical and 51,200 chemical requests; 47,521–47,611 chemical experiments
per run returned a response. The ten native conditions completed all 2,400
executions without execution errors. They reuse the same 120 physical and
120 chemical instances, rather than representing 2,400 distinct instances.

Mean operation counts include failed tasks. All chemical conditions completed
120/120 tasks.

| Condition | Physics completed | Physics operations | Chemistry operations |
|---|---:|---:|---:|
| Initial model, either continuation | 119/120 | 112.11 | 100.28 |
| Learned activity, 911 | 118/120 | 113.55 | 97.83 |
| Fixed computation, 911 | 114/120 | 109.65 | 97.41 |
| Discard response updates, 911 | 118/120 | 113.55 | 96.92 |
| Reset chemical state, 911 | 118/120 | 113.55 | 102.62 |
| Learned activity, 912 | 119/120 | 105.03 | 96.56 |
| Fixed computation, 912 | 118/120 | 109.68 | 101.93 |
| Discard response updates, 912 | 119/120 | 105.03 | 93.10 |
| Reset chemical state, 912 | 119/120 | 105.03 | 104.63 |

The complete method has no established overall advantage. Its chemical
operation differences from the fixed-computation reference are +0.43
(whole-stream 95% interval −7.41 to +8.46) and −5.38 (−14.75 to +2.99).
Relative to the initial model, all full-model operation intervals also include
zero. Physical completion and cost must be read together: the first full
model finishes four more tasks than its fixed reference while using more
operations; the second finishes one more with lower average cost.

Retaining response updates costs +0.92 and +3.46 chemical operations compared
with discarding them at the same weights. Both intervals include zero, so
neither a reliable benefit nor a reliable harm is established. Clearing task
state costs +4.78 and +8.08 operations on average; those intervals also include
zero. At the first chemical return to relationship A, full versus reset costs
+2.20 and −13.40 operations respectively, again without a stable effect.

### What the model actually chose

The two learned policies execute **1,588 and 1,616 LEARN activities**, and
**zero CONSOLIDATE activities**. The first performs no extra REFINE; the second
performs 200. Neither uses RETRIEVE. The response corrections are nonzero and
alter subsequent predictions and some actions. Nevertheless, the acting
comparison does not establish a benefit from retaining them. In particular,
this run does not demonstrate cross-task consolidation of acquired knowledge.

Prediction loss on each model's own selected chemical experiments changes
from 1.050 nats initially to 0.749 and 0.748 for the learned models. The fixed
references reach 0.713 and 0.699. These use different selected histories;
they are descriptive losses, not a same-prefix estimate of predictive gains.
All 54 initial predictions assigning more than 90% probability to complete
rust removal include 25 actual clean outcomes; the four trained policies make
no predictions above that threshold. Removing extreme confidence alone does
not establish useful experimental planning or calibrated uncertainty.

### Predictions at identical histories

A completed replay compares response corrections at the same neural state,
received history and selected experiment. Each prediction is scored against
the same next actual outcome. Removing corrections changes only the response
readout; an additional learning update uses only earlier returned outcomes.
The retained branch alone advances the replay. Its log probabilities reproduce
the original native records to within 1.1e-6.

| Continuation | Returned experiments | Zero correction NLL | Retained correction NLL | One additional update NLL |
|---|---:|---:|---:|---:|
| 911 | 815 | 0.7105 | 0.7488 | 0.7555 |
| 912 | 808 | 0.7036 | 0.7483 | 0.7563 |

Lower negative log likelihood (NLL) means a better prediction of the received
outcome. Retaining the correction increases NLL by 0.0382 nats (whole-stream
95% interval 0.0241–0.0506) and 0.0448 (0.0340–0.0549). An additional update
increases it again by 0.0067 (0.0026–0.0119) and 0.0080 (0.0034–0.0136).
Thus the current online correction worsens subsequent response prediction on
these histories. This narrows the unresolved problem to the update itself
before attributing it to planning. It does not isolate the cause within the
update rule or establish a task-performance effect: these are post-result
readout interventions on fixed trajectories, not new acting policies.

[Replay code](../continual_chemistry/compare_response_updates.py) and the
complete predictions for [911](results/active_feedback_predictions_911.json)
and [912](results/active_feedback_predictions_912.json) preserve the comparison.

[Complete paired results](results/active_feedback.json),
[all instance outcomes](results/active_feedback_instances.csv), and
[activity and prediction records](results/active_feedback_activity.json) retain
both repetitions and every control. Physics pauses during chemical segments,
and this study uses structured observations. Independently timed multimodal
operation remains a separate extension.
