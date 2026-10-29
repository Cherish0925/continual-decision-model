# Continuous scientific learning and action

This study trains one model to choose external experiments and internal
activities over a sequence of changing scientific tasks. Its state retains the
current experimental history, unfinished computation, uncertain physical
relationships, and learned corrections. Results returned by an experiment can
change both the next action and the physical prediction used to set a device.

The starting points are the two trained physical-memory policies in
[the preceding study](../physical_belief/README.md). Those policies improved
native task efficiency, but reset neural state between tasks and used a fixed
amount of internal computation. Their online physical memory did not reliably
improve recovery after an intervening task. This study addresses those specific
limitations together.

## A single continuing process

A received observation updates the recurrent state once. The model then chooses
whether to act, refine its workspace, retrieve an earlier experience, update a
fast physical correction, consolidate a slower correction, or wait. Internal
activities update the same state used to choose subsequent experiments.
New measurements and activation feedback arrive through a pending-result queue.
They are unavailable until delivery, and each delivered record enters the
learning history once.

The physical prediction has two cooperating parts. Bayesian calibration uses
the current public measurement ledger and memories of previous relationships.
Online residual learning uses the model's prediction before an experiment and
the activation interval returned afterward. Its output changes the predicted
setting and uncertainty used for execution. The recurrent core, workspace,
action and activity policies, predictive parameters, and learning features are
trained jointly from subsequent outcomes. The numerical calibration remains a
tool within the model; physical learning is not reduced to predicting the cost
of an otherwise unchanged setting.

Within a task stream, neural state, physical memory and online corrections
continue across A–B–A task changes. Independent streams reset all state.
The task-reset comparison removes this continuation explicitly. Repeated
optimization of received records does not add new observations to the physical
ledger.

## Training tasks and feedback

Training uses the existing 1,663 recorded worlds and generated affine tasks
over their measured quantities. A whole task stream uses one source type.
Related tasks recur with different calibration parameters and different
objects. The generating relationship and task schedule are unavailable to
the policy.

For recorded worlds, [prepare_environment.py](prepare_environment.py) reconstructs
the original official simulator. All 6,652 crystals match the recorded objects
through their deterministic measurements. Transitions use the actual activation
condition, an absolute setting error strictly below 2 Hz. Previously observed
working settings remain public calibration observations; they are not replaced
with exact resonances. A newly successful experiment adds the setting actually
executed to the ledger. The hidden resonance belongs only to the environment
and outer training supervision.

Recorded measurement returns are consumed sequentially. Feedback is delivered
after one to four internal ticks. Pending results cannot be read by the policy
or online learner. This exercises continuing computation and result handling;
the simulator itself still advances through explicit environment actions.

## Joint training and comparison

Each optimizer window contains the same number of external requests per stream.
Internal activity and waiting add their own costs instead of reducing the
external training budget. Task return, elapsed waiting, internal computation,
and physical predictive likelihood train the shared model. Final-update
checkpoints are used without selection on the confirmation instances.

Two parent models each have four continuations:

| Training | Physical online correction | State between tasks | Internal activities |
| --- | --- | --- | --- |
| Full | Retained | Retained | Learned |
| No physical online correction | Computed and discarded | Retained | Learned |
| Task reset | Retained within task | Reset | Learned |
| Fixed activity schedule | Retained | Retained | Fixed |

The no-correction condition retains Bayesian physical memory and the existing
operation-cost learner. It isolates the new physical correction. Evaluation
also discards that correction in the full trained weights. A fixed activity
schedule tests learned allocation against performing the available activities
in a predetermined order.

## Native evaluation

The [native protocol](NATIVE.md) fixes new development and confirmation cohorts.
The confirmation set contains 120 instances in 20 A–B–A streams. Every method
starts from the same public world state with the same execution tools and
external action allowance. Evaluation records task completion, actual actions
including failures, elapsed time, internal activity, prediction changes, and
performance when an earlier relationship returns. Comparisons are paired by
whole stream.

The principal questions are whether online physical learning improves future
execution, whether retained state helps after interference, and whether learned
activity choices improve the quality–cost tradeoff. Implementing all three
paths makes these questions testable; their answers come from the matched
training and native execution results.

The current task interface is structured scientific observation. It does not
train a new multimodal encoder or establish transfer to unrelated scientific
domains. The shared state and activity mechanisms are the objects being
investigated; Reactor Lab supplies a concrete environment in which their
consequences can be measured.
