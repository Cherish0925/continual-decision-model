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

## Fixed-endpoint training

Four matched runs are in progress. Each trains for 200 windows of 32 streams ×
32 external requests: 204,800 requests per run. The two repetitions start from
parent policies 221 and 222, using initialization seeds 241 and 242. All four
use the same task-plan construction, failure-conditioned decision weights,
400-action task limit and actual operation costs. The full model learns its
activities and physical updates; the recurrent reference performs one state
update per received observation and no online neural parameter updates.

| Condition | Final checkpoint location after update 200 |
|---|---|
| Full, seed 241 | `runs/continual_physics/native_execution_joint_20261030_point/full_seed241/last.pt` |
| Full, seed 242 | `runs/continual_physics/native_execution_joint_20261030_point/full_seed242/last.pt` |
| Recurrent, seed 241 | `runs/continual_physics/native_execution_joint_20261030/recurrent_reference_seed241/last.pt` |
| Recurrent, seed 242 | `runs/continual_physics/native_execution_joint_20261030/recurrent_reference_seed242/last.pt` |

The first full-model attempts remain in `native_execution_joint_20261030`:
seed 241 stopped before its first update; seed 242 completed seven updates.
A valid sequence of strict-threshold outcomes could identify an exact frequency
—for example, failures at 2390 and 2394 Hz followed by success at 2393 Hz identify
2392 Hz. The continuous-interval implementation incorrectly removed that point.
The correction uses the conditional-density limit only for an identified point;
ordinary interval events retain their probability measure. It adds no noise or
probability floor. Both full models restarted from update zero in the `_point`
directory and have passed their initial complete training windows with finite
gradients. The original full, recurrent and restarted full checkpoints have
identical values for all 135 initial parameter tensors within each repetition.
The unaffected recurrent runs continue in their original locations.

A fresh reproduction uses the corrected source and one new output directory:

```bash
python -m experiments.continual_physics.launch \
  --output runs/continual_physics/native_execution \
  --environment-truth runs/continual_physics/environment_train/environment.npz \
  --execution-geometry runs/continual_physics/environment_train/execution_geometry.npz \
  --condition-failure-weights --cache-observations --seed-offset 10 \
  --modes full recurrent_reference --updates 200 --batch 32 \
  --external-per-stream 32 --jobs-per-gpu 2
```

The actual full-model restart used this same command with
`--output runs/continual_physics/native_execution_joint_20261030_point`,
`--modes full` and `--jobs-per-gpu 1`; the continuing recurrent runs were started
with the first four-condition command and its original output directory.
Evaluation waits for `finished.json`, which is written only after update 200;
intermediate checkpoints are not selected by task performance. Final native
results are pending.

The [independent native comparison](execution_confirmation.json) fixes 120 new
instances in twenty six-task streams. It compares both parents, both trained
full models and recurrent references, each full model with physical updates
discarded or all state reset at task boundaries, and the Bayesian memory
reference. The [automatic queue](execution_confirmation.py) waits for each
fixed final checkpoint and writes all 1,320 executions with whole-stream
comparisons. Parent and classical execution is synchronous; native operation
costs match, while the internal synthetic feedback delay applies to the new
policies. That distinction remains explicit in timing results.

```bash
python -m experiments.continual_physics.execution_confirmation
```
