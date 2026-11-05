# Why the native policies repeat, and what separates the candidate causes

Written 5 November 2026 against `cc4f4f7`, reading the completed
[native policy training](../experiments/discoveryworld/FULL_SUITE.md#native-policy-training)
records. Three independent routes — autonomous interaction, basic-operation
instruction, and fixed-snapshot imitation with and without action history —
all produced policies that repeat a failed command indefinitely. The records
already establish that this is neither a teacher defect nor a device fault.
This note adds what the existing measurements cannot yet separate, and the
measurements that would separate it.

## The repetition statistic measures two things at once

Training samples from the action distribution
([`parallel_learning.py:102`](../experiments/discoveryworld/parallel_learning.py)).
Independent execution takes the highest-scoring candidate
([`predictive_actions.py:159`](../experiments/discoveryworld/predictive_actions.py),
`temperature = 0`). The scores are a deterministic function of the observation,
the candidate set, the belief and the recurrent state
([`predictive_actions.py:63`](../experiments/discoveryworld/predictive_actions.py)).

A failed command leaves the world unchanged. Between two such steps the only
input that changes is the recurrent state, and — with `--action-history` — the
history record. Under the greedy rule the action therefore changes only if that
one changing input moves the argmax. Under sampling it changes whenever the
distribution has mass elsewhere.

So "repeats after a failed command" conflates two different facts:

1. the policy has not learned that a failed command should not be retried;
2. the state update is too weak to move the argmax, whatever the policy knows.

Every repetition count in the current records is greedy-only: 72,900/72,900 for
the autonomous CDM, 978/978 and 1,114/1,114 in the action-history comparison.
The eight-case sampled comparison shows the repetition breaks under sampling
without any task being solved, which is consistent with both facts and
distinguishes neither.

This matters now because the repetition count is the primary diagnostic in the
32-training input and initialization protocol currently running. Without
separating the two facts, that protocol returns 32 numbers that cannot be
attributed.

**Report repetition conditioned on the input having changed**, and report both
decoding rules for the same weights. The weights are already final; this is a
re-analysis of recorded traces plus one extra evaluation pass, not new training.

## Why explicit action feedback changed nothing

The history input is visible and separately encoded
([`cdm/digital.py:135`](../cdm/digital.py)), yet imitation accuracy fell
slightly (CDM 29.44% → 28.73%, GRU 28.99% → 27.95%) and repetition was total.
An input that carries information but has no learned effect is the expected
outcome when the training distribution never rewards using it.

Of 16,015 teacher actions, 3,812 carry imitation labels, and labels exist only
for valid actions in **successful** episodes. A successful teacher episode
contains almost no failed commands. The supervised distribution therefore
contains close to zero after-failure states, so the history feature is
uninformative under that loss, and a near-zero learned weight is the correct
fit. The policy is not ignoring feedback it was taught to use; it was never
shown a state where the feedback predicts anything.

This yields a falsifiable prediction for the recovery collection, which places
the student in exactly these missing states: if after-failure coverage is the
binding constraint, the history input acquires a measurable effect once
recovery transitions enter training, and the history-enabled and history-free
arms separate. If they still do not separate, the constraint is elsewhere and
the input representation should be dropped rather than extended.

## The measurement that distinguishes the remaining causes

Counting repeats cannot say whether an input is unused or used and overridden.
The score function is differentiable and the checkpoints are final, so measure
it directly:

- **Sensitivity.** On recorded after-failure observations, compare the candidate
  scores with the history record present and with it cleared. Report the change
  in score margin between the repeated action and the runner-up. A margin that
  does not move means the input carries no learned weight; a margin that moves
  but does not cross means the policy prefers the failed action for other
  reasons. The same comparison on the recurrent state — current state against
  the state carried in from the previous step — gives the state's contribution
  on the same scale.
- **Steps to change.** Hold the observation fixed and step the policy
  repeatedly, recording when the argmax first changes. Report the distribution
  over cases. The autonomous CDM's traces say the answer is "never" within 700
  steps; measuring it on the demonstration and recovery endpoints gives a
  single comparable number for every arm of the protocol.

Both run on existing weights in minutes and produce one scalar per condition.
They belong next to the repetition counts in
[`compare_native_feedback`](../experiments/discoveryworld/compare_native_feedback.py),
which already reads the traces.

## The claim the project has not yet tested

Everything measured so far concerns the action interface: which command is
emitted, whether it is accepted, whether it repeats. CDM's claim is about
state — that inference carried across events changes the next decision. The
native records contain the strongest available evidence on that claim, and it
is negative: with `persistent = True`
([`parallel_learning.py:34`](../experiments/discoveryworld/parallel_learning.py);
`restart_state` defaults to false), after dozens of identical failures the
evolving state never moved the argmax.

The "steps to change" measurement above is the direct test. It is cheaper than
any further training and its result governs what to do next. If the carried
state does not move the decision on this task, then the native benchmark is
measuring an action-decoding problem and should not be read as evidence about
persistent inference either way; the
[process lifetimes](../experiments/lifetime/README.md), where carried state and
deployed updates demonstrably change behaviour, remain the setting where the
claim holds.

## What the evidence does not support doing

- **Adding conditions to the protocol before the decoding confound is removed.**
  More arms multiply unattributable numbers.
- **Adding inputs.** Action history was the natural candidate and had no effect;
  the constraint is label coverage, not observability.
- **Reading 0/120 as a capability regression.** The earlier 30/30 is a
  single-theme chemistry development set with a specialized execution
  interface, recorded as such in
  [`FULL_SUITE.md`](../experiments/discoveryworld/FULL_SUITE.md) and
  [`continual_chemistry`](../experiments/continual_chemistry/README.md). The two
  numbers share no task, interface or protocol.

## One result worth keeping in view

In the handoff experiment the autonomous CDM completes 6/12 and the
basic-operation GRU 5/12, against 0/12 for uniform selection
([`native_action_behavior.json`](../experiments/discoveryworld/results/native_action_behavior.json)).
Handoff is one action before specialist completion with nine candidates, so this
establishes only that terminal action selection is better than chance. It does
not show that a task can be completed, and it should not be summarized as such.
It does show the learned scoring is not uniformly degenerate, which is the
reason to localize the failure rather than restart the route.
