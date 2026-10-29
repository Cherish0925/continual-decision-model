# Continual experimental decisions: results

Each native condition follows the same ordered task instances. Training repetitions are reported separately.
[Method and stage-level results](methods.csv) · [Data, paired intervals and training records](summary.json)

[Task-boundary state controls](state_controls.md) separate the carried neural state from replay and online adapters on the same instances and checkpoints.

| Condition | Completed | Actions / instance | Measurements / instance | CPU-evaluated elapsed s / instance |
| --- | ---: | ---: | ---: | ---: |
| `baseline_parent` | 179/184 | 135.95 | 9.71 | 6.22 |
| `baseline_rehearsal` | 179/184 | 116.32 | 10.30 | 5.84 |
| `baseline_explicit` | 180/184 | 128.42 | 47.62 | 5.83 |
| `full_seed201` | 179/184 | 148.58 | 9.19 | 7.93 |
| `no_adaptation_seed201` | 176/184 | 183.89 | 10.16 | 9.25 |
| `task_reset_seed201` | 181/184 | 197.40 | 12.69 | 9.30 |
| `fixed_schedule_seed201` | 176/184 | 182.11 | 10.45 | 8.94 |
| `full_reset_seed201` | 179/184 | 197.12 | 11.55 | 9.39 |
| `full_discard_updates_seed201` | 179/184 | 148.58 | 9.19 | 7.80 |
| `full_seed202` | 174/184 | 193.76 | 9.70 | 10.07 |
| `no_adaptation_seed202` | 176/184 | 193.57 | 9.64 | 10.27 |
| `task_reset_seed202` | 178/184 | 210.30 | 12.18 | 10.02 |
| `fixed_schedule_seed202` | 172/184 | 218.99 | 10.15 | 11.06 |
| `full_reset_seed202` | 178/184 | 201.35 | 11.81 | 9.86 |
| `full_discard_updates_seed202` | 174/184 | 193.80 | 9.68 | 9.56 |

Time includes environment construction, model execution and tool operations on CPU. It is elapsed time under concurrent evaluation, not process CPU time or isolated model latency.

## Internal activities and received-feedback prediction

Counts below are totals across the same native instances. Predictions were recorded before each selected action; their targets are its subsequently returned test success and log(1 + native actions).

| Condition | REFINE | RETRIEVE | LEARN | CONSOLIDATE | Fast / slow nonzero instances | Test success MSE, shared → adapted | Measurement cost MSE, shared → adapted | Test cost MSE, shared → adapted |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `full_seed201` | 9464 | 25 | 0 | 0 | 0 / 0 | 0.1491 → 0.1491 | 0.6749 → 0.6749 | 2.2942 → 2.2942 |
| `no_adaptation_seed201` | 80 | 10089 | 0 | 0 | 0 / 0 | 0.1973 → 0.1973 | 0.5140 → 0.5140 | 2.3581 → 2.3581 |
| `task_reset_seed201` | 736 | 0 | 0 | 13400 | 0 / 184 | 0.1431 → 0.1530 | 0.5508 → 0.2901 | 2.9847 → 2.0319 |
| `fixed_schedule_seed201` | 3041 | 3021 | 3021 | 3021 | 184 / 184 | 0.1149 → 0.1117 | 0.7202 → 0.2237 | 1.9834 → 0.3486 |
| `full_reset_seed201` | 3772 | 144 | 0 | 0 | 0 / 0 | 0.1523 → 0.1523 | 0.4758 → 0.4758 | 2.4185 → 2.4185 |
| `full_discard_updates_seed201` | 9464 | 25 | 0 | 0 | 0 / 0 | 0.1491 → 0.1491 | 0.6749 → 0.6749 | 2.2942 → 2.2942 |
| `full_seed202` | 7075 | 0 | 0 | 4937 | 0 / 184 | 0.1267 → 0.1253 | 0.6962 → 0.2421 | 2.3952 → 0.7264 |
| `no_adaptation_seed202` | 80 | 11960 | 0 | 0 | 0 / 0 | 0.1437 → 0.1437 | 0.7133 → 0.7133 | 2.6823 → 2.6823 |
| `task_reset_seed202` | 725 | 0 | 0 | 13376 | 0 / 184 | 0.1440 → 0.1500 | 0.5759 → 0.3119 | 3.9590 → 2.8163 |
| `fixed_schedule_seed202` | 3273 | 3253 | 3253 | 3253 | 184 / 184 | 0.0829 → 0.0820 | 0.7172 → 0.2044 | 2.0283 → 0.2923 |
| `full_reset_seed202` | 4950 | 0 | 0 | 8776 | 0 / 179 | 0.1335 → 0.1362 | 0.4838 → 0.2400 | 2.3870 → 1.8108 |
| `full_discard_updates_seed202` | 7061 | 0 | 0 | 4943 | 0 / 0 | 0.1267 → 0.1267 | 0.6965 → 0.6965 | 2.3966 → 2.3966 |

LEARN/CONSOLIDATE counts show requested updates; nonzero norms show retained adapter state and may include earlier tasks. The discard-updates intervention can request updates while retaining zero adapters. Success predictions are raw regression outputs. Measurement actions have a zero test-success target and are scored separately. Shared/adapted errors use the same executed actions; policy benefit requires the paired intervention results.

## Training exposure

| Training | Status | Updates | Stream-ticks | External interactions | Finished proxy tasks | Elapsed s |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `full_seed201` | finished | 1200 | 4915200 | 1248481 | 94003 | 4009.5 |
| `no_adaptation_seed201` | finished | 1200 | 4915200 | 1186658 | 89267 | 3956.7 |
| `task_reset_seed201` | finished | 1200 | 4915200 | 1289865 | 77527 | 3935.9 |
| `fixed_schedule_seed201` | finished | 1200 | 4915200 | 988687 | 75514 | 3997.8 |
| `full_seed202` | finished | 1200 | 4915200 | 1265339 | 96948 | 4030.1 |
| `no_adaptation_seed202` | finished | 1200 | 4915200 | 1184795 | 90774 | 3973.9 |
| `task_reset_seed202` | finished | 1200 | 4915200 | 1898761 | 119543 | 4005.0 |
| `fixed_schedule_seed202` | finished | 1200 | 4915200 | 988466 | 72411 | 3897.2 |

Equal stream-tick budgets do not imply equal external training data, FLOPs or elapsed time. Training tasks use recorded/synthetic calibration outcomes and proxy costs; the table above separates them from native execution. Intervention conditions reuse their corresponding full-model training, not additional runs.

## Paired interpretation

Intervals resample complete task streams, retaining every instance within each sampled stream. Initial, switched and returning segments include separate first/subsequent-instance results. Cost differences are reported both for all instances and for the subset completed by both conditions. Training repetitions share test instances and are not pooled as independent test sets.

Full model, training repetition 201. Differences are full minus reference, with 95% stream-bootstrap intervals. Completion differences are percentage points; cost differences are native actions per instance.

| Reference | Gained / lost completions | Completion difference | Action difference, all | Action difference, joint success (n) |
| --- | ---: | ---: | ---: | ---: |
| `baseline_parent` | 2 / 2 | +0.00 [-2.13, +2.20] | +12.63 [+0.92, +24.59] | +12.62 [+1.61, +22.92] (177) |
| `baseline_rehearsal` | 1 / 1 | +0.00 [+0.00, +0.00] | +32.26 [+14.19, +50.00] | +32.21 [+13.99, +50.04] (178) |
| `baseline_explicit` | 1 / 2 | -0.54 [-2.35, +1.12] | +20.16 [+3.20, +35.47] | +19.57 [+2.32, +35.21] (178) |
| `no_adaptation_seed201` | 4 / 1 | +1.63 [-0.55, +4.02] | -35.31 [-51.34, -19.85] | -35.82 [-51.81, -20.00] (175) |
| `task_reset_seed201` | 0 / 2 | -1.09 [-2.67, +0.00] | -48.82 [-61.03, -37.46] | -51.16 [-63.46, -39.61] (179) |
| `fixed_schedule_seed201` | 3 / 0 | +1.63 [+0.00, +3.47] | -33.53 [-46.56, -20.88] | -33.45 [-47.27, -20.03] (176) |
| `full_reset_seed201` | 1 / 1 | +0.00 [-1.58, +1.65] | -48.54 [-62.87, -34.24] | -49.17 [-64.07, -34.44] (178) |
| `full_discard_updates_seed201` | 0 / 0 | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] (179) |

Full model, training repetition 202. Differences are full minus reference, with 95% stream-bootstrap intervals. Completion differences are percentage points; cost differences are native actions per instance.

| Reference | Gained / lost completions | Completion difference | Action difference, all | Action difference, joint success (n) |
| --- | ---: | ---: | ---: | ---: |
| `baseline_parent` | 2 / 7 | -2.72 [-5.85, +0.00] | +57.80 [+44.98, +70.63] | +57.26 [+44.52, +69.56] (172) |
| `baseline_rehearsal` | 1 / 6 | -2.72 [-5.00, -1.01] | +77.43 [+56.86, +97.61] | +74.02 [+52.92, +95.09] (173) |
| `baseline_explicit` | 1 / 7 | -3.26 [-6.45, -0.51] | +65.34 [+45.58, +83.50] | +62.21 [+41.89, +81.21] (173) |
| `no_adaptation_seed202` | 2 / 4 | -1.09 [-3.23, +1.11] | +0.19 [-14.62, +15.29] | -2.27 [-18.02, +14.06] (172) |
| `task_reset_seed202` | 3 / 7 | -2.17 [-5.32, +0.52] | -16.54 [-28.99, -3.67] | -19.49 [-32.85, -5.89] (171) |
| `fixed_schedule_seed202` | 4 / 2 | +1.09 [-1.12, +4.00] | -25.24 [-33.98, -16.50] | -25.54 [-35.46, -15.49] (170) |
| `full_reset_seed202` | 1 / 5 | -2.17 [-5.24, +0.54] | -7.59 [-18.05, +2.90] | -11.43 [-22.72, -0.53] (173) |
| `full_discard_updates_seed202` | 0 / 0 | +0.00 [+0.00, +0.00] | -0.04 [-0.20, +0.07] | -0.05 [-0.22, +0.07] (174) |

## Exploratory follow-ups

These comparisons were proposed after examining the primary results and reuse the same task instances. They are not independent confirmation or checkpoint selection.

- `full_replay_feedback_seed201`: Check whether the actual-native versus replay 1/5 feedback-cost scale changes greedy internal activity selection.
- `full_sample_seed201`: Sample internal activities from the trained policy while keeping external actions greedy; retain fast/slow adapter updates.
- `full_sample_discard_updates_seed201`: Sample internal activities from the trained policy while keeping external actions greedy; discard fast/slow adapter updates after computing them.
- `full_sample_seed202`: Second training repetition fixed after the 201 findings and before the 202 native outcomes: Sample internal activities from the trained policy while keeping external actions greedy; retain fast/slow adapter updates.
- `full_sample_discard_updates_seed202`: Second training repetition fixed after the 201 findings and before the 202 native outcomes: Sample internal activities from the trained policy while keeping external actions greedy; discard fast/slow adapter updates after computing them.

| Condition | Activity policy (base seed) | Completed | Actions / instance | REFINE / RETRIEVE | LEARN / CONSOLIDATE | Fast / slow nonzero instances |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `full_replay_feedback_seed201` | greedy | 179/184 | 148.58 | 9457 / 25 | 0 / 0 | 0 / 0 |
| `full_sample_seed201` | sample (233) | 180/184 | 145.70 | 3409 / 1108 | 2239 / 1821 | 184 / 184 |
| `full_sample_discard_updates_seed201` | sample (233) | 180/184 | 145.53 | 3413 / 1103 | 2239 / 1816 | 0 / 0 |
| `full_sample_seed202` | sample (233) | 174/184 | 193.86 | 4452 / 1331 | 1908 / 4053 | 184 / 184 |
| `full_sample_discard_updates_seed202` | sample (233) | 174/184 | 194.71 | 4453 / 1335 | 1906 / 4078 | 0 / 0 |

Prediction MSE, shared → adapted, on each condition's own executed actions. Cost errors use actual native action counts:

| Condition | Test success | Measurement log-cost | Test log-cost |
| --- | ---: | ---: | ---: |
| `full_replay_feedback_seed201` | 0.1491 → 0.1491 | 0.6749 → 0.6749 | 2.2942 → 2.2942 |
| `full_sample_seed201` | 0.1495 → 0.1499 | 0.6529 → 0.2288 | 2.2811 → 0.5495 |
| `full_sample_discard_updates_seed201` | 0.1498 → 0.1498 | 0.6548 → 0.6548 | 2.2790 → 2.2790 |
| `full_sample_seed202` | 0.1266 → 0.1264 | 0.6891 → 0.2019 | 2.3835 → 0.4632 |
| `full_sample_discard_updates_seed202` | 0.1263 → 0.1263 | 0.6903 → 0.6903 | 2.3879 → 2.3879 |

Paired differences are candidate minus reference, with the same stream-bootstrap intervals and grouping used above.

| Candidate − reference | Completion difference (pp) | Action difference, all | Action difference, joint success (n) |
| --- | ---: | ---: | ---: |
| `full_replay_feedback_seed201` − `full_seed201` | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] (179) |
| `full_sample_seed201` − `full_seed201` | +0.54 [+0.00, +1.69] | -2.89 [-6.86, +0.11] | -1.88 [-5.56, +0.52] (179) |
| `full_sample_seed201` − `full_sample_discard_updates_seed201` | +0.00 [+0.00, +0.00] | +0.16 [-0.13, +0.66] | +0.17 [-0.13, +0.67] (180) |
| `full_sample_seed202` − `full_seed202` | +0.00 [+0.00, +0.00] | +0.11 [-1.20, +1.31] | +0.11 [-1.28, +1.37] (174) |
| `full_sample_seed202` − `full_sample_discard_updates_seed202` | +0.00 [+0.00, +0.00] | -0.84 [-2.40, +0.02] | -0.89 [-2.56, +0.02] (174) |

## Regenerate

```bash
python -m experiments.continual_experiment.summarize \
  --training-root runs/continual-experiment/joint_activity_20261028 \
  --native-root runs/discoveryworld/continual_experiment/native \
  --output experiments/continual_experiment/results \
  --resamples 10000 --bootstrap-seed 201
```
