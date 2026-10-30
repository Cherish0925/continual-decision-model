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

The training path and a four-task native execution check passed. Formal paired
training and the fixed native comparison are running; final task benefits are
not yet available. Two early implementation checks caught mutable masks and
returned-target tensors being reused by autograd. Their outputs remain local;
the formal runs use separate immutable returned targets. Physics still pauses
while chemical segments execute, and this study uses structured observations.
The independently timed, multimodal application remains a separate extension.
