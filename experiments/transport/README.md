# Learned transport of unfinished numerical state

Run this CPU-only mechanism experiment with Python 3.12 and the pinned NumPy dependency:

```bash
python -m pip install -r experiments/transport/requirements.txt
python experiments/transport/run_experiment.py
```

The command writes raw measurements, five frozen trained models, configuration,
environment metadata and a generated report to this directory. It does not
download data or use a GPU. The completed reference run used Python 3.12.14 and
NumPy 2.3.5. Read [RESULTS.md](RESULTS.md) for measured findings and their scope.

## What is trained

For a changing quadratic objective with symmetric positive definite Hessian H,
the candidate learns a nine-parameter inverse polynomial. Its three polynomial
coefficients each depend quadratically on clipped log condition number. Training
uses 1,800 independent paired-context samples per seed, exact linear solves as
supervision, and relative vector squared error with ridge regularization. A
damping factor is selected from a fixed grid on separate validation streams.

A separate 18-parameter model conditions additionally on log interruption depth.
It learns the change in an unfinished gradient-descent state following a target
change. Its labels recompute the same-depth trajectory from a fixed zero seed.
The diagnostic includes interruption depths absent from training. These labels
are different from inverse/equilibrium transport; matching them is not itself
evidence of improved decisions.

## Evaluation and controls

Five training seeds share six paired 128-tick evaluation streams in each of five
regimes. Evaluation includes stationary inputs, slow changes, unseen large
target changes, held-out curvature, and abrupt joint shifts. All policies have
the current H, b and spectral bounds. A deliberately stale baseline ignores the
newest three ticks. Full exact solutions and analytical inverse transfers are
explicit accuracy references, not equal-cost competitors.

The candidate, current-residual and readout-only controls use five dense matrix
vector products per tick. Strong alternatives include five gradient or
Nesterov steps and four CG iterations after an initial residual. A separate GD
budget is calibrated to approximate candidate CPU latency on validation data.
Actual CPU wall time includes coefficient evaluation and vector operations and
is separate from simulated environment ticks. It is not a GPU benchmark.

`results/main_raw.csv` and `results/finite_progress_raw.csv.gz` retain all individual
measurements. Summary files report aggregate scores; paired bootstrap intervals
resample independent streams, not repeated deterministic baseline seeds.
`results/config.json` fixes data seed namespaces, sample counts and hyperparameters.
`results/frozen_models_seed_*.npz` stores all learned coefficients. Random
initialization and synthetic data are deterministic; CPU timing and the
latency-calibrated GD step count may vary across machines and runs.

## Scientific interpretation

This construction uses established polynomial approximation and
predictor–corrector ideas. It is a numerical probe, not an originality claim or a
CDM prototype. The first tested equilibrium-transport mechanism loses to strong
numerical baselines. The finite-progress experiment tests a distinct state
target and provides a concrete next hypothesis; a recurrent model with latent
state distinct from emitted action and outcome-based evidence is still needed.
The main environment stream is exogenous; emitted decisions do not change future
observations. It tests optimization tracking, not closed-loop control.

Sanity assertions check positive curvature, exact target generation, causal
prefix invariance, quadratic objective identities, zero innovation, exact
solution accuracy, and the finite-depth sensitivity formula.

An independent full rerun reproduced fixed-work scores and fitted coefficients within
the tolerances recorded in [reproduction.json](results/reproduction.json). Timing and
the timing-calibrated iteration count are excluded from numerical identity checks.
