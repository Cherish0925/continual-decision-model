# What repeated native actions do and do not establish

The four released native policies completed 0/120 official instances. Their
traces show repeated failed commands and unproductive accepted movements.
These are behavior measurements; several causes can produce the same behavior.
The [execution record](../experiments/discoveryworld/FULL_SUITE.md#what-the-four-frozen-policies-actually-did)
separates these results from earlier specialized controllers and assisted handoffs.

## Action choice and score changes

Interaction training samples an action distribution. The default independent
[policy](../experiments/discoveryworld/evaluate.py) takes its maximum. A policy
can change its probabilities after feedback while retaining the same highest
scoring action. Conversely, sampling can change actions without learning to
recover. The earlier sampled 0/8 result demonstrates that breaking a loop alone
does not establish task competence.

A failed operation also does not imply an unchanged observation. The native
adapter advances the world clock; messages, elapsed processes, object access and
public records may change. The new analysis therefore compares the actual
encoded observations and candidates, both including and excluding explicit
clock fields, before counting repeated commands.

The 32-condition input/initialization study measures independent completion and
progress as well as repetition. Those comparisons remain useful. Repetition
alone cannot attribute their outcomes to semantic understanding, neural state
or feedback processing.

## Supervision after a failed command

Re-running the actual imitation-label selection gives:

| Training set | Executed actions | Imitation labels | States following a failed command | Labels in those states |
| --- | ---: | ---: | ---: | ---: |
| Original fixed demonstrations | 16,015 | 3,812 | 2,237 | 230 |
| Demonstrations plus completed recovery collection | 22,758 | 4,450 | 3,102 | 295 |

In both sets every labeled action immediately after a failed command differs
from that failed command. These examples constitute 6.03% and 6.63% of the
imitation labels. Thus, the earlier explanation that the model was never shown
recovery from failure is incorrect. Coverage can still be insufficient, but its
causal role has not been established. The new collection adds 65 such labels;
it is not thousands of distinct demonstrations of recovery.

Labels come from valid teacher actions in successful episodes. Student-prefix
actions remain available for outcome prediction without becoming imitation
answers. Failed teacher episodes also supply outcome targets. The counts concern
an immediately preceding failed command, not every longer-term recovery need.

The [coverage calculation](../experiments/discoveryworld/compare_native_feedback.py)
uses the training code's label selector. The two data snapshots fix which
trajectories are counted; they are not selected by student evaluation scores.

## Fixed-weight comparisons

The following analyses use each final checkpoint without further training:

- Replay its recorded execution with the actual previous actions. Verify that
  reconstructed greedy decisions match the recorded decisions.
- At sampled states after failure, remove action-history records while keeping
  the observation, candidate set and incoming neural state fixed. Separately
  reset neural state, or skip assimilation of the current observation.
- Report the failed action's probability, its score margin over the best
  alternative, changes in the full action distribution, and whether the
  selected action changes.
- Hold the encoded observation, time and candidate set fixed and repeatedly
  assimilate that input, recording the first different greedy choice up to
  32 updates. A missing change is reported as censored at that limit.
- Execute the same checkpoint with sampling temperature 1 on the same 16
  development instances and initial environment RNG states used by greedy
  execution. Report both completion and operation behavior.

An unchanged margin at these sampled states does not prove zero learned weights
or global irrelevance of an input. Clearing history or resetting state may also
create inputs outside the training distribution. These interventions measure
local reliance. Reassimilating one frozen observation tests stability of that
computation; it is not an environment rollout, useful planning, or a proof about
all persistent-state mechanisms. The existing 700-action traces cannot substitute
for it because their observations were not held fixed.

## Completed measurements

The [32-endpoint comparison](../experiments/discoveryworld/FULL_SUITE.md#input-state-and-decoding-completed-comparison)
reproduces all 49,304 recorded choices. Resetting state changes 119/1,156 sampled
choices, and skipping the current state update changes 152/1,156. Removing
explicit history changes none of 593 choices; its mean distribution change is
0.00064 in total variation. Thus state use and history use are measurably
different in these trained models. No single explanation covers all endpoints.

Freezing the input and clock changes the first choice within 32 further updates
in 23/301 sampled states. Sampling at temperature 1 yields 63 successful
executions across the 32 models, versus one with greedy decoding. Each model
runs the same 16 development instances; the count is not a single model's
score. Uniform random actions complete 0/16 under each of two seeds. These
results support further learning from the existing weights, without establishing
that sampled execution provides a competent general scientific policy.

The [full label counts](../experiments/discoveryworld/results/native_supervision_coverage.json)
and [per-state measurements](../experiments/discoveryworld/results/native_input_comparison.json)
allow these calculations to be reproduced.

## Implications for the next implementation

The result predictor currently receives supervision but is separate from the
native action readout. The next optional model connects candidate-specific
memory access, action-conditioned state prediction, observation correction and
prediction-dependent action scores. It keeps the loaded policy unchanged at
initialization and requires independent task execution after training.

Outcome prediction and latent dynamics for control have established antecedents,
including [PlaNet](https://arxiv.org/abs/1811.04551) and
[DreamerV3](https://arxiv.org/abs/2301.04104). Connecting these computations in CDM
is a concrete implementation hypothesis, not a novelty claim or an established
performance improvement. Fixed-weight measurements and matched training serve
different questions and should inform the same implementation.

The earlier handoff successes—6/12 and 5/12, versus 0/12 in the recorded uniform
comparison—came from fixed adjustment directions near completion. They do not
show a complete experimental procedure or state-dependent reasoning. Likewise,
the old 30/30 chemistry result used a different task set and a specialized
execution interface; it cannot be compared directly with native 0/120.
