# Calibration rehearsal improves native experimental coverage

Rehearsing alternative calibration relations increased native task completion
from 54/64 with ordinary continued training to 62/64. It completed eight additional
instances without losing any ordinary-training successes. It also matched the
full-measurement control's exact set of 62 completed instances, using 115.31
rather than 245.94 native actions per instance, a 53.1% reduction in mean actions.

Both training arms resumed the same persistent model and optimizer. Each used
200 updates of hindsight supervision and 1,000 actor–critic updates, with the
same recorded measurements, model inputs, architecture and frozen numerical
tool. Rehearsal changed only the simulated query settings: a randomly proposed
quantity defined an affine relation through two observed working settings.
Degenerate fits and settings outside the public 0–10,000 Hz range were rejected.
The 50% proposal rate produced 38.25% accepted rehearsal worlds. No new native
training measurements were collected for this comparison.

The confirmation used 64 fresh Reactor Lab Normal instances, matched initial
randomness and a 400-native-action limit. The additional relation was absent
from the original native training records but present in the rehearsal
hypothesis family. This tests transfer from generated practice to native
execution, rather than a relation unseen in every form of training.

| Method | Original native relations (52) | Additional relation (12) | All completions (64) | Mean native actions, all instances |
| --- | ---: | ---: | ---: | ---: |
| Frozen persistent parent | 50/52 | 0/12 | 50/64 | 151.75 |
| Ordinary continued training | 50/52 | 4/12 | 54/64 | 133.28 |
| Calibration rehearsal | 50/52 | 12/12 | 62/64 | 115.31 |
| Rehearsal, same weights with state reset | 47/52 | 10/12 | 57/64 | 139.47 |
| Full measurement, posterior selection | 50/52 | 12/12 | 62/64 | 245.94 |

Resetting the rehearsal model's GRU and workspace preserves its weights and
explicit measurement ledger. The complete model recovered five completions
relative to this intervention, with no losses. All methods retained every
failure in the reported costs and completed evaluation without execution errors.

## Cost on jointly completed instances

| Reference | Joint successes | Rehearsal actions | Reference actions | Difference |
| --- | ---: | ---: | ---: | ---: |
| Ordinary continued training | 54 | 98.00 | 83.89 | +14.11 |
| Frozen parent | 50 | 97.48 | 82.24 | +15.24 |
| Same weights with state reset | 57 | 98.79 | 107.47 | −8.68 |
| Full measurement | 62 | 106.13 | 241.00 | −134.87 |

Rehearsal expands coverage while adding exploration on some already solvable
instances; it does not make every instance faster. Against ordinary training,
the eight completion gains and zero losses give an exact paired two-sided
probability of 0.0078. The all-instance action difference is −17.97, with a
paired bootstrap 95% interval of [−43.92, 5.72]. Against full measurement, the
action difference is −130.63 [−147.77, −113.03], with identical success and failure
identities. These intervals resample paired instances within continuation seed
183; they do not estimate variation across training repetitions.

Public data: [per-instance results](calibration_rehearsal_instances.csv),
[relation-wise summaries](calibration_rehearsal_summary.csv), and
[paired costs and uncertainty](calibration_rehearsal_pairs.csv).
The [confirmation protocol](../active_rehearsal_comparison.json) fixes instances
and budgets. Full local outputs, including pairwise instance records and the
matched-configuration record, are under
`runs/discoveryworld/active_experiments/rehearsal_comparison/seed183/`.
