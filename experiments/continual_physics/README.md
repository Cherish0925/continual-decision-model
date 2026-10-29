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

Prediction and execution remain distinct. A failed setting rules out a range
of possible resonances; the model then proposes a reachable setting with high
remaining activation probability. It can retry the same physical relationship
at a different setting. The proposal uses the public dial range and resolution.
Its probability distribution is also trained by the received outcome, so the
online update affects both later predictions and later choices.

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

The fixed budget is 200 updates, 32 parallel streams and 32 external requests
per stream per update: 204,800 requests per training, or 1,638,400 across the
eight trainings. Gradients propagate through each 32-request window. State
continues across windows, and value estimates bootstrap subsequent returns.
This separates the duration of retained experience from the length of direct
gradient propagation. The logs record how often a window crosses task and
complete-stream boundaries.

A small action-distribution term preserves the frozen parent's demonstrated
behavior on the same received observations. Activity entropy is averaged over
the active ticks, then annealed to zero in the first half of training. Extra
internal steps therefore cannot accumulate an unbounded entropy bonus.
Operation feedback trains operation predictions; delivery latency contributes
its own cost.

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

Two additional trainings compare a recurrent policy with shared physical tools.
They use the same parent weights, observations, task streams, objective and
204,800-request budget. The model updates its GRU and workspace once per received
observation, retains their state across tasks, and waits while results are
pending. Neural parameter updates are disabled at deployment; the common
Bayesian calibration and setting proposal remain available. This tests the
additional value of learned internal activity and neural online correction
against a simpler trained policy. It is a recurrent meta-learning comparator,
not a reproduction of a particular published RL² implementation.

`launch_reference.py` queues these additional trainings after the three learned
activity conditions finish on each device. They remain separate from the
original eight-training comparison.

The complete comparison can be launched with:

```bash
python -m experiments.continual_physics.prepare_environment \
  --data runs/discoveryworld/active_experiments/data_large/train.npz \
  --output runs/continual_physics/environment_train --workers 16

python -m experiments.continual_physics.launch \
  --environment-truth runs/continual_physics/environment_train/environment.npz \
  --output runs/continual_physics/joint_20261029 \
  --updates 200 --batch 32 --external-per-stream 32 --jobs-per-gpu 4

python -m experiments.continual_physics.native_campaign \
  --training-root runs/continual_physics/joint_20261029 \
  --output runs/continual_physics/native_confirmation \
  --feedback-delay-ticks 2 --parallel 8 --workers 6
```

The evaluation command waits for each final training checkpoint. Dependencies
are the released parent policies, beliefs and prepared measurement data listed
in the [physical-learning release guide](../discoveryworld/RELEASE.md#physical-prediction-and-action-v0110).

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

Training and the native comparison are in progress. The frozen parent models
complete 117/120 and 119/120 new confirmation instances, with 106.68 and 102.84
mean native actions. These new instances differ from the preceding study's
cohort. New-model results will be reported together after the fixed comparison
finishes.

The current task interface is structured scientific observation. It does not
train a new multimodal encoder or establish transfer to unrelated scientific
domains. The shared state and activity mechanisms are the objects being
investigated; Reactor Lab supplies a concrete environment in which their
consequences can be measured.

## Recurrent feedback candidate

The [task-boundary controls](../continual_experiment/results/neural_state_controls.md)
locate the earlier cross-task benefits and transition costs in the GRU state.
The original GRU reads operation events; the workspace reads that state and the
public ledger, but its computation does not feed directly into the next GRU
update. A candidate connection adds a projection of the preceding workspace
summary to the received-event embedding. A separate evidence-only condition
tests direct access to the current visible ledger. Both projections start at
zero, preserving the parent policy and its random-generator state.

The `workspace`, `evidence` and `both` options select these inputs independently.
The first comparison should use `workspace` versus an equally trained parent;
`evidence` distinguishes computational feedback from more direct observation
access. Internal REFINE still changes z alone. Its result can change h on the
next real receive, without incrementing the observation count during internal
computation. Immediate REFINE-to-h writeback is a separate, unimplemented
timing comparison. No additional memory bank or activity selector is introduced.

The existing training entry accepts the candidate directly:

```bash
python -m experiments.continual_physics.train \
  --data runs/discoveryworld/active_experiments/data_large \
  --checkpoint runs/physical_belief/joint_20261029/carry_seed221/policy.pt \
  --initial-state runs/continual_physics/joint_20261029/full_seed231/initial.pt \
  --belief-checkpoint runs/physical_belief/joint_20261029/carry_seed221/belief.pt \
  --mixture-checkpoint runs/discoveryworld/calibration/mixture/shared_train_only_seed31/last.pt \
  --environment-truth runs/continual_physics/environment_train/environment.npz \
  --recurrent-feedback workspace --seed 231 --updates 200 \
  --batch 32 --external-per-stream 32 \
  --output runs/continual_physics/recurrent_feedback_candidate
```

When starting from an unmodified parent, omit `--recurrent-feedback` for its
unchanged architecture, or use `evidence` / `both` for the other inputs. Loading
a candidate checkpoint retains its saved connection. The connection uses the existing
new-parameter learning rate, and its source and weights are saved in the model
checkpoint. CPU checks establish initialization equivalence, future-state
gradients, masked observation behavior, checkpoint restoration and two short
cross-task training windows. They do not establish task-performance gains;
the candidate has not entered the original running comparison.

The initial follow-up plan specified four trainings: workspace-only and evidence-only for each
original full-model repetition, 231 and 232. Each retains its original parent,
seed, objective, task schedule and 204,800-request budget; the four candidates
total 819,200 new requests. All 135 shared state tensors are restored from the
original full condition's update-zero checkpoint. Each candidate adds one
1,048,576-parameter zero projection. The two original full models supply the
no-connection comparison without retraining. CPU initialization checks also
compare every receive and readout in an initial 1,024-request window for each
repetition, with exact equality.

[Development](feedback_development.json) reserves 30 new instances and
[confirmation](feedback_confirmation.json) reserves 120. These do not overlap
the training worlds or the previous confirmation cohort. Both candidates are
also executed with the learned connection computed and then discarded; all
other weights stay fixed. Ten conditions across the two sets produce 1,500
native executions. Every final checkpoint is evaluated without selection by
development outcomes. Results separate same-relation continuation, first
switch and first return, including completion, actual action cost and elapsed
inference/feedback time.

```bash
python -m experiments.continual_physics.verify_feedback_start \
  --primary runs/continual_physics/joint_20261029 \
  --output runs/continual_physics/recurrent_feedback_20261030/initial_equivalence.json
python -m experiments.continual_physics.feedback_campaign \
  --output runs/continual_physics/recurrent_feedback_20261030
```

The queue starts the two candidates on a GPU only after that GPU's original
four conditions and recurrent reference have exited successfully at the full
budget. A failed dependency ends the affected candidate job. The queue then
runs the fixed native comparisons on CPU and writes phase summaries, paired
whole-stream intervals and per-instance outcomes in its output directory.

This follow-up was deferred before any candidate training began. The first
completed primary comparisons expose a substantial execution-cost mismatch:
training permits 160 individual setting requests, while the native 400-action
budget also pays for navigation, device adjustment and local probing within
each request. The task-reset seed-232 model completes about 99% of its ended
training tasks but 81/120 native tasks. Its native failures repeatedly test the
same relationship while changing the proposed frequency. The ongoing primary
comparisons remain unchanged; the feedback candidates will be compared after
the training execution model and its matching reference are revised.

## Failure evidence in experimental decisions

The original implementation uses failed-setting intervals when proposing the
next setting and when learning physical corrections. Its candidate relation
probabilities, however, still come from the successful calibration ledger.
Consequently, excluding a disproved frequency does not directly reduce the
candidate's relation-confidence feature.

The optional `--condition-failure-weights` intervention makes that feature
consistent with the observed failures. Starting from the current original
Gaussian mixture, it multiplies each relation weight by the probability of
remaining outside the union of failed intervals and normalizes across
relations. It recomputes this single event from the original mixture on each
read, so duplicate failures and repeated inference do not count twice.
This is Bayesian conditioning applied to the existing decision interface,
not a new probabilistic inference rule.

The online learning buffer still stores the original pre-return forecast;
its likelihood target and the scalar proposal rule are unchanged. Saved
`physical_decision_weights` identify the conditional weights used for candidate
features separately from `physical_weights`, the original weights used for
prediction training. The intervention has no new learned parameters.

On the fixed 30-instance development cohort, the same task-reset seed-232
weights produce:

| Decision confidence | Completed | Mean native actions, including failures |
| --- | ---: | ---: |
| Calibration ledger only | 17/30 | 220.30 |
| Calibration ledger and failed-setting evidence | 25/30 | 159.87 |

Eight failures become completions and no completed instance becomes a failure.
All sixty executions finish without interface errors. These development results
support the importance of propagating failure evidence into the actual decision;
they do not establish an advantage over the strongest parent or independent
transfer. A new matched cohort compares both versions and the frozen parent.
The [paired instances](results/failure_weights_development/instances.csv) and
[summary](results/failure_weights_development/summary.json) are generated by
`summarize_failure_weights.py` from the complete public execution records.
