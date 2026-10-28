# Calibration rehearsal improves native experimental coverage

Rehearsing alternative calibration relations reached 62/64 native task completions
in both training repetitions, compared with 54/64 and 60/64 for ordinary continued
training. It added eight and two completions, respectively, without losing any
ordinary-training successes. Both rehearsal models matched the full-measurement
control's exact set of 62 completed instances, using 115.31 and 111.70 native
actions per instance instead of 245.94: reductions of 53.1% and 54.6%.

The two repetitions used separately trained persistent parents: continuation
seed 183 started from parent 181, and seed 184 from parent 182. Within each
repetition, ordinary training and rehearsal resumed the same model and optimizer.
Each used 200 updates of hindsight supervision and 1,000 actor–critic updates,
with batch size 512, the same recorded measurements, model inputs, architecture
and frozen numerical tool. Training updated 26,653,443 parameters, including
23,108,608 in the persistent core. This route uses structured public observations;
no visual encoder was trained.

Rehearsal changed only the simulated query settings: a uniformly proposed
quantity defined an affine relation through two observed working settings.
The original support settings and recorded measurements remained unchanged.
Degenerate fits and settings outside the public 0–10,000 Hz range were rejected.
The 50% proposal rate produced 38.25% and 38.40% accepted rehearsal worlds in
repetitions 183 and 184. Both used the final 1,200-update checkpoint. No new
native training measurements were collected for this comparison.

Both repetitions used the same 64 Reactor Lab Normal confirmation instances,
matched initial randomness, a 400-native-action limit and at most 160 policy
decisions. They are two training repetitions on 64 instances, not 128 independent
test instances. The full-measurement control is one shared evaluation, reused
as a paired reference in both repetitions. The additional relation was absent
from the original native training records but present in the rehearsal
hypothesis family. This tests transfer from generated practice to native
execution, rather than a relation unseen in every form of training.

| Repetition | Method | Original native relations (52) | Additional relation (12) | All completions (64) | Mean native actions, all instances |
| --- | --- | ---: | ---: | ---: | ---: |
| 183 | Frozen persistent parent 181 | 50/52 | 0/12 | 50/64 | 151.75 |
| 183 | Ordinary continued training | 50/52 | 4/12 | 54/64 | 133.28 |
| 183 | Calibration rehearsal | 50/52 | 12/12 | 62/64 | 115.31 |
| 183 | Rehearsal, same weights with state reset | 47/52 | 10/12 | 57/64 | 139.47 |
| 184 | Frozen persistent parent 182 | 50/52 | 10/12 | 60/64 | 117.02 |
| 184 | Ordinary continued training | 50/52 | 10/12 | 60/64 | 111.41 |
| 184 | Calibration rehearsal | 50/52 | 12/12 | 62/64 | 111.70 |
| 184 | Rehearsal, same weights with state reset | 48/52 | 12/12 | 60/64 | 133.70 |
| Shared | Full measurement, posterior selection | 50/52 | 12/12 | 62/64 | 245.94 |

Resetting the rehearsal model's GRU and workspace preserves its weights and
explicit measurement ledger. The complete models recovered five and two
completions relative to this intervention, with no losses. This compares the
same trained weights under different state-retention conditions. All methods
retained every failure in the reported costs and completed evaluation without
execution errors.

## Cost on jointly completed instances

| Repetition | Reference | Joint successes | Rehearsal actions | Reference actions | Difference |
| --- | --- | ---: | ---: | ---: | ---: |
| 183 | Ordinary continued training | 54 | 98.00 | 83.89 | +14.11 |
| 184 | Ordinary continued training | 60 | 99.97 | 92.17 | +7.80 |
| 183 | Frozen parent | 50 | 97.48 | 82.24 | +15.24 |
| 184 | Frozen parent | 60 | 99.97 | 98.15 | +1.82 |
| 183 | Same weights with state reset | 57 | 98.79 | 107.47 | −8.68 |
| 184 | Same weights with state reset | 60 | 98.92 | 115.95 | −17.03 |
| 183 | Full measurement | 62 | 106.13 | 241.00 | −134.87 |
| 184 | Full measurement | 62 | 102.40 | 241.00 | −138.60 |

Rehearsal expands coverage while adding exploration on some already solvable
instances. Relative to ordinary training, its all-instance action differences
are −17.97 and +0.30, with paired bootstrap 95% intervals of [−43.92, 5.72]
and [−17.17, 17.09]. The corresponding exact paired two-sided completion tests
give p = 0.0078 and p = 0.50. The improvement is smaller from the stronger
second parent; these results do not establish a uniform speed advantage over
ordinary training.

Relative to full measurement, the action differences are −130.63
[−147.77, −113.03] and −134.23 [−152.06, −115.25], with identical success and
failure identities in both repetitions. The intervals resample paired instances
separately within each trained model. They do not estimate training-run
variation, and the repeated use of the same instances is not pooled as
independent evidence.

Public data: [per-instance results](calibration_rehearsal_instances.csv),
[relation-wise summaries](calibration_rehearsal_summary.csv), and
[paired costs and uncertainty](calibration_rehearsal_pairs.csv).
The [confirmation protocol](../active_rehearsal_comparison.json) fixes instances
and budgets. Full local outputs are under
`runs/discoveryworld/active_experiments/rehearsal_comparison/seed183/` and
`runs/discoveryworld/active_experiments/rehearsal_comparison/seed184/`.
