# Learning physical predictions across experiments

The model preserves uncertain scientific relationships across tasks, then
recalibrates them from the next task's measurements. These beliefs determine
predicted device settings and enter the experimental action policy. This changes
what online learning affects: acquired knowledge can change physical predictions,
in addition to predictions of operation cost or success.

## Physical belief and task memory

The [belief model](../../cdm/experimental_belief.py) maintains affine hypotheses
over measured quantities. For each hypothesis it represents uncertainty about
the intercept and slope. A new task starts with the shared learned prior and a
bounded set of previously inferred task posteriors. A learned coefficient-drift
distribution allows an earlier relation to recur with a new calibration.
Current measurements and publicly observed working settings determine the
relative evidence for each candidate context and relation.

The stored object is an uncertain predictive relationship. It does not contain
a scenario seed, a hidden relation label, or a future target setting. Within a
task, inference is recomputed from the available ledger, so reading the same
ledger twice does not count the evidence twice. Completing a task stores its
posterior once. Reset and computed-and-discarded controls remove this
cross-task information while retaining within-task calibration.

The implementation accepts an arbitrary feature count. This experiment uses
nine measured quantities and single-feature affine relationships, an explicit
hypothesis class. Its 73 shared trainable parameters describe coefficient
priors, observation noise, relation weights, context drift and memory weight.
Task-specific Bayesian updates change the predictive state. These updates are
distinct from SGD updates of the shared neural action core.

## Predictive training

[Data preparation](data.py) uses 1,663 recorded training worlds. Labels are
settings publicly observed while a reactor was active. Density, temperature,
quantum gap and spectra use their first recorded returns. Radiation uses the
minimum of up to 32 actual recorded measurements. This offline prediction study
reveals all selected readings before predicting a setting; acquiring those
readings is not free in the separate native task evaluation.

Each training stream contains two tasks under relationship A, two under B, then
two under A again. Each task has a fresh physical calibration. The model first
uses the two initial working settings to predict the first target. Only after
scoring that prediction is its public working setting revealed. The second
target is then predicted, revealed and incorporated into task memory.
Generating relation identities arrange the stream but never enter the model.

Half of training streams retain the recorded tasks. The other half practice
alternative affine relationships: a selected measured quantity and the two
original support settings define a new relationship, which generates target
settings. This extends the original four training relations to all nine
quantities. It is generated practice on recorded measurements, not additional
native simulator observations.

Two training repetitions use seeds 211 and 212. Both inherit the same numerical
prior, with independently initialized drift and memory parameters and independent
stream samples. Each runs 500 updates with 64 streams of six tasks, or 192,000
task presentations. The loss is predictive negative log likelihood in Hz, and
gradients pass through remembered task posteriors. A matched episodic model has
the same initialization, stream samples and budget, but resets memory at every
task. Final checkpoints use the fixed final update without validation selection.
All four trainings completed; [pretraining.csv](results/pretraining.csv) records
their budgets and measured runtime.

## Held-out physical predictions

The comparison uses 64 held-out source worlds, arranged into 128 six-task streams
per condition. A source world never repeats within one stream. Worlds recur
across streams, so the 1,536 predictions per condition are not 1,536 independent
physical instances. All compared methods receive the same stream and public
evidence.

Three conditions separate the recorded setting distribution from broader
calibrations:

- **Native:** recorded readings and observed working settings.
- **Alternative relationships:** generated affine relations through the two
  recorded support settings, using the held-out measured quantities.
- **Wider calibration:** the alternative tasks additionally receive a new
  per-task frequency scale in 0.35–2.0 and offset in −1,000–1,000 Hz. Some
  predictions lie outside the simulator's control range; this is a numerical
  prediction stress test, not a native task completion result.

The following table reports the mean absolute error of the mean under the
highest-probability relation. Slash-separated values are the two training
repetitions. Mixture means, predictive likelihoods and the other interventions
are included in [prediction.csv](results/prediction.csv).

| Model and deployment memory | Native MAE, Hz | Alternative MAE, Hz | Wider-calibration MAE, Hz |
| --- | ---: | ---: | ---: |
| Original numerical prior, no cross-task memory | 83.72 | 818.50 | 999.05 |
| Initial physical memory, before new training | 46.03 / 46.04 | 728.90 / 728.18 | 923.54 / 923.54 |
| Episodic training, memory retained at evaluation | 47.06 / 47.06 | 725.78 / 727.70 | 905.61 / 915.40 |
| Through-memory training, memory reset at evaluation | 169.54 / 181.45 | 858.74 / 861.09 | 1,010.82 / 1,009.58 |
| Through-memory training, memory retained | 118.03 / 137.23 | 639.83 / 628.43 | 860.62 / 838.93 |

Preserving learned physical posteriors improves predictions relative to resetting
the same model. Training through memory further improves the generated
relationship and wider-calibration conditions relative to episodic training.
It does not preserve the best native point predictions: initial memory and
episodic training are better on that measure. Lower predictive likelihood loss
alone does not establish a uniformly better experimental policy.

On the first prediction after A returns, the two through-memory models have
native MAE of 1.12 and 1.11 Hz, versus 175.21 and 197.96 after resetting memory,
and 404.26 and 467.82 when retaining only the last context. All 128 returning
predictions are within 2 Hz of the recorded successful setting. Initial memory
already reaches 1.22 and 1.23 Hz, with 126/128 within that interval. These
comparisons attribute much of this recovery to the new memory representation;
they do not attribute all of it to training through memory. A recorded
successful setting is not the simulator's hidden exact resonance, so this
interval count is not a task completion count. Phase-specific results are in
[prediction_phases.csv](results/prediction_phases.csv).

## What remains difficult

Two calibration pairs generally fit an affine function of each candidate
quantity. They need not identify which quantity actually determines the output.
Memory can supply a useful prior, but an unfamiliar calibration can still
require a distinguishing experiment.

The [error decomposition](results/prediction_mechanisms.csv) makes this problem
visible. In the wider-calibration condition, 70.1–72.3% of first-target
predictions select the wrong relationship. Those selections account for more
than 99.99% of absolute error. Conditioning the numerical prediction on the
generating relationship gives only 0.077 Hz MAE; this is an explanatory
calculation using evaluation metadata, not an available policy input. After the
first target's public working setting is revealed, wrong-relation selection
falls to 1.2%, and second-target MAE falls to 9.40 Hz. All selected features are
visible in this prediction protocol, so missing-feature treatment cannot explain
this particular failure.

The remaining ambiguity connects prediction to active experimentation. A failed
setting supplies information about which predictions are inconsistent with
actual feedback. Evaluating and using that information requires the measured
setting and its activation result; hidden target distances are unnecessary.
The frozen pretraining and native comparisons above do not yet condition their
physical posterior on those failed-setting intervals.

## Initial native task comparison

The pretrained belief was connected to the existing calibration-rehearsal action
policy without changing that policy's weights. Thirty new development instances
form five ordered task streams. Measurements are chosen and paid for in the
actual simulator; each method has the same public interfaces and 400-action
budget. This comparison tests the transfer from full-reading prediction to
partially observed action.

| Method | Completed | Mean native actions, including failures |
| --- | ---: | ---: |
| Calibration-rehearsal parent, seed 183 | 30/30 | 107.33 |
| Explicit relation-memory reference | 29/30 | 145.83 |
| Initial physical memory | 30/30 | 107.17 |
| Pretrained physical memory, seed 211 | 30/30 | 102.27 |
| Same weights, computed memory updates discarded | 30/30 | 105.40 |
| Pretrained physical memory, seed 212 | 30/30 | 102.27 |
| Same weights, computed memory updates discarded | 30/30 | 105.40 |

Both pretrained memories produce the same trajectories on this cohort. Relative
to discarding their updates, they save 94 native actions, concentrated in three
instances. First-return instances do not improve. The five-stream uncertainty
interval for the comparison with the seed-183 parent includes zero. This is a
development result that supports further joint training, not evidence of a
general capability advantage. The second action parent, seed 184, separately
completes 30/30 at 124.50 actions; it is the starting point for the second joint
training repetition.

Native records are under `runs/physical_belief/native_development/`, including
the per-instance comparisons in `comparisons/`. The joint action-training and
independent confirmation study use separately recorded conditions.

## Reproduce the physical prediction study

From the repository root, extract the existing measured-data and numerical-prior
artifacts, then run:

```bash
python -m experiments.physical_belief.train \
  --data runs/discoveryworld/active_experiments/data_large \
  --mixture-checkpoint runs/discoveryworld/calibration/mixture/shared_train_only_seed31/last.pt \
  --output runs/reproduced_physical_belief/seed211 \
  --device cuda:0 --seed 211 --updates 500 --batch 64 --segment-length 2

python -m experiments.physical_belief.evaluate_prediction \
  --data runs/discoveryworld/active_experiments/data_large \
  --checkpoint runs/reproduced_physical_belief/seed211/last.pt \
  --initial-checkpoint runs/reproduced_physical_belief/seed211/initial.pt \
  --reference-mixture runs/discoveryworld/calibration/mixture/shared_train_only_seed31/last.pt \
  --output runs/reproduced_physical_belief/seed211/prediction_unique \
  --device cuda:0 --streams 128 --segment-length 2
```

Repeat with seed 212. Add `--mode reset` to training for the matched episodic
control. Prediction records retain the actual inputs' source identities,
predictions made before label disclosure, and subsequent labels. Generate the
compact tables with [summarize_prediction.py](summarize_prediction.py).
