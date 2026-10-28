# Experience across native task instances

The experiment asks whether a continuing neural state can improve experimental
choices after solving related tasks, through an intervening change of relation,
and when the earlier relation returns. Every instance is solved in the actual
DiscoveryWorld simulator from its initial observation.

## State and task boundaries

`evaluate_active_stream.py` loads one model for an ordered stream of instances.
The `carry` condition retains its GRU state and Transformer workspace across
instances. The `reset` condition initializes those tensors at each new instance;
it still retains state between actions within that instance. Weights remain
fixed during native evaluation.

Both conditions create a new measurement ledger, failure record and numerical
posterior for every instance. They read the new devices' publicly available
working settings, then choose measurements and tests using the same executor.
An earlier device's coefficient estimates or measurement values are never
copied into the next instance's numerical tools.

The final executed action can end an instance before the next policy decision.
Its returned observation is therefore consumed once through `model.advance`
before moving to the next instance. This updates the continuing state without
selecting an action or advancing the simulator. At the new instance, the event
starts at zero with only the ordinary current-query indicator set. The model
receives no relation, switch, return, seed or lifetime-position marker.

## Fixed native sequence

[`active_lifetime_comparison.json`](active_lifetime_comparison.json) contains
20 streams, one for each ordered pair of distinct native relation families.
Each stream follows A–B–A, with two to four fresh instances in each segment.
The fixed manifest contains 184 distinct instances, using seeds 7000–7196.
Segment lengths vary; they are recorded for analysis and never enter policy
inputs. Each instance allows 400 native actions and 160 policy decisions.

The official [Reactor Lab source](https://github.com/allenai/discoveryworld/blob/fd591323920be0d3786ef350955de1945aa571e5/discoveryworld/scenarios/reactor_lab.py)
selects the Normal task relation using `randomSeed % 5`:

| Remainder | Measured quantity determining frequency |
|---:|---|
| 0 | Density |
| 1 | Temperature |
| 2 | Quantum gap |
| 3 | Radiation |
| 4 | Fifth returned spectrum channel |

Each instance separately generates its affine slope and offset. Returning to
A therefore restores the kind of relation, not the previous calibration.
The simulator shuffles crystals before naming and placing them. The executor's
four slots are the two publicly observed working-device supports, followed by
the two uncalibrated targets. Public names, handles and this slot order are
recorded for every instance; handles are not numerical model features.

Four conditions use the same per-instance initial simulator RNG snapshots:

1. The model trained with cross-instance state, evaluated with state carried.
2. The same weights, with state reset between instances.
3. The separately trained task-reset control, evaluated with task resets.
4. The frozen calibration-rehearsal parent, with task resets.

Results retain completion, native action cost, measurement/test requests and
full trajectories. The stream summary marks the first and subsequent instances
of each segment, the switch to B, and the return to A. This permits comparisons
of initial exploration, within-family reuse, interference and return. The
structured public observation interface is the same in all four conditions.

## Training through a task stream

`train_active_lifetime.py` constructs A–B–A streams from recorded instrument
returns. A relation is a choice of measured quantity; each fresh world fits
its own affine coefficients through its original public calibrations.
Precomputed world order and object permutations keep each paired comparison's
latent sequence identical even when policies spend different numbers of actions
on a task. No relation or segment marker enters the policy.

The ordinary measurement ledger and numerical fit restart between tasks.
The carry arm retains the GRU and workspace; the separately trained control
resets those tensors at task boundaries. Both consume the last received outcome
before leaving a task. Value targets bootstrap across task boundaries and stop
at the end of the lifetime. Gradients pass across task boundaries within each
32-decision unroll. Shared neural weights are trained offline and remain fixed
in native evaluation; this experiment measures adaptation through continuing
state.

Each arm runs 1,200 updates with batch 128 and unroll 32. Two repetitions start
from the corresponding final calibration-rehearsal models: training seed 191
from rehearsal 183, and seed 192 from rehearsal 184. Within each repetition,
both arms share initial weights, optimizer, task stream and training budget.
The final update is selected in advance. The fixed native streams above are
not used for training or checkpoint selection.

```bash
python -m experiments.discoveryworld.train_active_lifetime \
  --data runs/discoveryworld/active_experiments/data_large \
  --checkpoint runs/discoveryworld/active_experiments/rehearsal/counterfactual_seed183/update1200.pt \
  --state-mode carry --seed 191 --batch 128 --unroll 32 --updates 1200 \
  --eval-every 300 --bf16 --device cuda:0 \
  --output runs/reproduced-lifetime-carry-191
```

Use `--state-mode task_reset` in a separate output directory for the matched
control. For repetition 192, change the seed and use
`runs/discoveryworld/active_experiments/rehearsal_repeat/counterfactual_seed184/update1200.pt`.
The model archives, rather than the inference-only archive, provide the retained
optimizer state. These trainings are in progress; the native comparison will
use their fixed final checkpoints.

## Run

From the repository root, select a fixed endpoint and state condition:

```bash
python -m experiments.discoveryworld.evaluate_active_stream \
  --manifest experiments/discoveryworld/active_lifetime_comparison.json \
  --checkpoint "$CDM_CHECKPOINT" \
  --mixture-checkpoint runs/discoveryworld/calibration/mixture/shared_train_only_seed31/last.pt \
  --state-mode carry \
  --initial-rng-root runs/discoveryworld/active_experiments/lifetime_native/initial_rng \
  --output runs/discoveryworld/active_experiments/lifetime_native/carry \
  --workers 16 --threads 2 --max-actions 400 --max-decisions 160
```

Use `--state-mode reset` for a task-boundary reset. It differs from the earlier
single-instance evaluator's `--reset-state`, which resets before every action.
Separate output directories preserve each condition. Streams run in parallel;
instances within a stream execute in order. `--save-boundary-states` optionally
stores the actual entry and exit tensors for a small implementation check.

## Native interface check

Before running the fixed comparison, the frozen calibration-rehearsal model
was executed on the development sequence
`2200, 2205 → 2201, 2206 → 2210, 2215` with shared initial RNG.

| State across instances | Tasks completed | Total native actions |
|---|---:|---:|
| Carried | 6/6 | 396 |
| Reset | 6/6 | 357 |

All twelve instances began with zero measurement samples and only the two
public support settings. Saved entry states exactly matched either the
previous exit state or the model's initial state, as specified. Every terminal
feedback observation changed the continuing state. The first instance had
identical actions; each later instance differed from its first decision.
This verifies the native state interface. These weights had not been trained
across task boundaries, so this check is not a test of lifetime-training gains.
The records are under
`runs/discoveryworld/active_experiments/lifetime_native_development/`.
