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
