# Training with the deployed experimental tools

The earlier table environment tested one scalar setting per request. The native
operator can make many additive adjustments and local probes, paying for every
movement, tool use and dialogue action. The corrected execution path uses that
operator's deterministic control semantics inside batched training tasks.

A task starts with the native 15-action bootstrap and has a 400-action budget.
Instrument acquisition, facing and position, crystal placement, radiation
isolation and restoration remain in the environment's persistent state. A test
uses the actual ±1/10/100/1000 Hz controls, stops at observed activation, and
otherwise performs at most 16 further local adjustments. Intermediate settings
are observations. Dialogue entry and exit count as actions. A task that ends
at activation does not receive a fictitious exit action.

The reactor applies its strict `< 2 Hz` activation rule and its public integer
and exact-frequency return behavior. Private resonance values remain inside
the environment and the outer training target. The model receives only the
returned settings, activation flags and requested measurements. Failed settings
are retained as intervals, including failed intermediate adjustments.

The returned cost target is `log1p(actual_actions)`. Task reward is one per
activated target minus `0.005 × actual_actions`. The task budget includes the
bootstrap. The existing fixed number of external requests remains the training
exposure budget; total primitive actions and bootstrap actions are reported
separately. Return discount is 0.99 per external request and 1 for internal
activities. This is not a claim of equal primitive-action training exposure.

## Reconstruct and run

```bash
python -m experiments.continual_physics.prepare_execution \
  --data runs/discoveryworld/active_experiments/data_large/train.npz \
  --output runs/continual_physics/environment_train/execution_geometry.npz
```

Add these flags to the existing training command:

```bash
--execution-geometry runs/continual_physics/environment_train/execution_geometry.npz \
--native-action-budget 400 --native-action-cost 0.005 \
--condition-failure-weights --cache-observations
```

Geometry is reconstructed only for the recorded training worlds. Measurements
still replay the recorded instrument sample sequence; they do not regenerate
the native simulator's measurement noise. Generated practice tasks use the same
execution controls and their declared generated physical targets.

The public-observation graph can be reused while a stream waits or computes.
It is rebuilt after returned evidence, a new task, a physical-memory commit,
and the next optimizer window. Neural state and parameter updates remain live;
the cache contains the unchanged public ledger and its physical posterior.

## Execution checks

Fixed public requests from 12 recorded native success/failure episodes were
replayed without policy training: all 243 request costs and success flags matched,
and every available intermediate frequency/activation sequence matched. These
are execution checks, not new policy results. Three additional scalar checks
cover activation en route, an exhausted primitive-action budget and agreement
with the deployed breadth-first probe selection.

A two-window GPU pilot executed 64 external requests with delayed receipts and
finite gradients. With identical initialization and sampling, cached and
uncached observations produced identical actions and losses; the largest final
parameter difference was 5.3e-13. The two windows took 12.49 seconds uncached and
10.30 seconds cached in this small concurrent pilot. This timing is a local
throughput observation, not a standalone hardware benchmark.

The longer 8-stream, 32-request pilot exposed a zero-prior-weight derivative in
the failure-weight normalization. The correction avoids evaluating `log(0)`
and preserves exactly zero prior mass. The original failing two-window rollout
then completed with finite gradients, task transitions and online updates.

The two formal reserved task plans contain 161,953 generated query targets.
All lie within the executable 0–10,000 Hz range and within activation tolerance
of an integer setting; no generated target was rejected in these plans.

The unchanged recurrent initialization completed the same five of eight
training-pool worlds under both execution paths. Its three failures required
150–151 scalar tests in the previous table; the native-cost path exhausted 400
primitive actions after 17–20 test requests. This comparison identifies the
training incentive mismatch; it does not select a model or measure held-out
performance.
