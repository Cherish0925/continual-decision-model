# Active measurement and persistent state

The persistent policy learned to choose measurements before the experimental
record was complete. Both training runs completed all 52 new instances from
training relation families, using 100.15 and 90.69 native actions per instance;
full measurement and posterior selection required 242.13. Resetting the trained
policy's state reduced completion to 49/52 in both runs. Transfer to the reserved
relation varied substantially between training runs.

The comparison uses the same 64 Reactor Lab Normal instances and saved initial
environment randomness, with a 400-native-action limit. Costs include public
discovery, navigation, instrument use and scalar adjustment. Official task
completion is the outcome; the eight-hertz replay reward is not used to score
native execution. These results use original recorded-task training, before
counterfactual calibration rehearsal.

| Training seed | Method | New instances, seen relations (52) | Reserved relation (12) | All completions (64) | Mean native actions, all instances |
| ---: | --- | ---: | ---: | ---: | ---: |
| 181 | Persistent core | 52/52 | 1/12 | 53/64 | 155.17 |
| 181 | Same weights, state reset | 49/52 | 2/12 | 51/64 | 186.88 |
| 181 | Separately trained ledger | 50/52 | 1/12 | 51/64 | 149.45 |
| 182 | Persistent core | 52/52 | 8/12 | 60/64 | 128.52 |
| 182 | Same weights, state reset | 49/52 | 5/12 | 54/64 | 172.77 |
| 182 | Separately trained ledger | 51/52 | 0/12 | 51/64 | 144.91 |
| — | Full measurement, posterior selection | 52/52 | 10/12 | 62/64 | 245.64 |
| — | Full measurement, learned setting selector | 51/52 | 10/12 | 61/64 | 258.36 |

The reset intervention preserves the model weights and explicit measurement
ledger. The separately trained ledger control resets recurrent state and
workspace during training and evaluation, with the same architecture,
initialization, visible records and schedule. All methods share the numerical
prior and public execution tools.

## Cost on jointly completed instances

The following costs condition on both methods completing an instance. They
describe execution efficiency on that common subset and must be read alongside
the full-sample completion counts above. The comparison set differs by row.

| Training seed | Reference | Jointly completed instances | Persistent actions | Reference actions | Difference, persistent minus reference |
| ---: | --- | ---: | ---: | ---: | ---: |
| 181 | Same weights, state reset | 50 | 97.00 | 128.88 | −31.88 |
| 182 | Same weights, state reset | 54 | 97.28 | 130.69 | −33.41 |
| 181 | Separately trained ledger | 51 | 96.18 | 85.59 | +10.59 |
| 182 | Separately trained ledger | 51 | 87.90 | 79.88 | +8.02 |

State reset increases cost even within the jointly completed subsets: the
paired 95% bootstrap intervals for the persistent-minus-reset difference are
[−49.68, −15.18] and [−48.26, −19.78] actions. These intervals resample scenario
instances within each training run. They do not estimate variation between
training runs; the two runs use the same evaluation instances.

Persistent state improves execution relative to resetting the same model.
The separately trained ledger remains competitive and is cheaper on common
successes; full measurement completes more reserved-relation instances.
These results motivate testing whether alternative calibration practice can
improve transfer while retaining efficient acquisition.

## Result files

The [overview CSV](active_overview.csv), [paired costs](active_paired_costs.csv),
and individual results for [seed 181](active_native_seed181.csv) and
[seed 182](active_native_seed182.csv) are available directly in this repository.
The [release records](../RELEASE.md) restore
`runs/discoveryworld/active_experiments/comparison/`.
Each seed directory contains `instances.csv`, `summary.json`,
`paired_instances.csv` and `paired_details.json`, preserving every failure and
termination reason. All eight native runs finished without execution errors.
The base summaries use [summarize_active.py](../summarize_active.py); additional
tables apply its same pairing and bootstrap to the stated subsets.
