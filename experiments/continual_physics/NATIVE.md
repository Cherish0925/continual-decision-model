# Native execution and comparison

The model controls the existing public DiscoveryWorld operator. Its physical
prediction supplies the actual frequency and uncertainty used for device
settings. The operator and its local scalar probing budget remain identical
across conditions.

## New task streams

`native_development.json` contains 30 new instances in five A–B–A streams.
`native_confirmation.json` contains 120 new instances in 20 streams, covering
all ordered pairs of distinct native relation families. Each segment contains
two instances. These are new instances of known relation families, not an
evaluation of unseen scientific laws. Schedule labels are for analysis and
never enter the model.

Compare both frozen v0.11 parent models and two independent trainings of each
new condition: full model, no physical online correction, task-boundary reset,
and fixed internal activity schedule. The full checkpoints also run with their computed
physical online corrections discarded. This last intervention preserves the
trained weights, calculation of updates, numerical posterior memory and cost
adapter. All conditions share initial native RNG snapshots, 400 actual
environment actions and at most 160 external
decisions per instance. Internal activity does not consume or replace the
external-decision allowance; its count, latency and cost are reported separately.

Report task completion, actual environment actions including failures, internal
activity counts, physical prediction changes, and first/second return-instance
performance. Pair comparisons by complete six-instance stream, with the stream
as the bootstrap unit. Do not combine two training repetitions into an enlarged
set of independent test instances.

## Public feedback

Every tool request has an identity. Its returned record contains the received
instrument readings, actual device settings and activation feedback. Only
these returned observations update the model. Unknown target frequencies and
the simulator's relation-family label are unavailable to it. Each successful
or unsuccessful setting is associated with the object tested and the native
activation tolerance (2 Hz), independently of the dial resolution; an
unsuccessful setting is not an exact target label.

Current-task measurement records, cross-task physical memory, persistent
neural state and online parameter corrections remain distinct model state.
Their reset or discard interventions must match the named comparison. Native
execution uses the corrected physical predictions from the model, rather than
recording an update while sending an unchanged numerical fit to the operator.
The model reports its predictive mean separately from its proposed device
setting. The latter accounts for previously failed setting intervals and the
current public dial grid. Failure does not permanently disable a whole relation:
the same relation can propose a new setting after further evidence or learning.
Prediction corrections, proposed settings and actual returned settings are
recorded separately.

## Delayed feedback

The runtime may hold a completed tool result for a configured number of
internal ticks. While it is pending, the model retains its state and can run
internal activities using the last delivered observation. It cannot see the
new measurement ledger or dispatch another dependent action before delivery.
Delivery consumes that request result once; a repeated result must not update
the state or learning record twice.

This tests the event interface and continuity of internal computation.
DiscoveryWorld advances through explicit actions and ticks; delayed delivery
alone does not demonstrate adaptation to an independently changing world.

## Reproduction

`native_campaign.py` writes the complete twelve-condition plan before execution
and waits for each training's `finished.json`. It can first run only the two
frozen parents with `--only parent_seed221 parent_seed222`; a subsequent call
uses the existing completion record and runs the remaining conditions.
`summarize_native.py` checks that every condition completed the declared cohort,
then reports paired task results, phase timing, internal activity, physical
prediction corrections, and actual trajectory changes under discarded updates.
Parent records do not separate model and operator time, so those fields remain
unavailable for parents instead of being estimated from their total wall time.
