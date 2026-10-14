# Numerical transport experiment

This CPU experiment fits small polynomial corrections for changing quadratic objectives and compares them with optimization baselines. A separate reconstruction test studies a finite-step gradient state.

## Run

The recorded environment used Python 3.12.14 and NumPy 2.3.5.

```bash
python -m pip install -r experiments/transport/requirements.txt
python experiments/transport/run_experiment.py --output-dir runs/transport-local
```

The output directory contains `RESULTS.md` and `results/`. Without an argument, the default is `runs/transport` relative to the working directory. Reusing a run directory replaces generated files.

## Models and data

The inverse polynomial has nine fitted parameters. Its coefficients depend quadratically on clipped log condition number. Each of five seeds uses 1,800 paired training samples and ridge-regularized relative squared error. Damping is selected on separate validation streams.

An 18-parameter model additionally conditions on interruption depth. It learns the change in a gradient iterate after a target change, using a fixed zero seed. Training depths are 1, 2, 4, 8, and 16; unseen test depths are 3, 6, and 12.

Six paired 128-tick test streams per regime cover stationary inputs, slow change, large target changes, held-out curvature, and joint shifts. All methods receive current problem parameters and spectral bounds. Test parameters are frozen after fitting.

## Comparisons and interpretation

Compare ordinary and accelerated gradient, conjugate gradient, approximate latency-matched gradient, the same polynomial on the whole residual, and correction applied only to the output. Report exact solutions and analytical transfer separately.

[RESULTS.md](RESULTS.md) records the original run. The tested correction loses to the stronger solvers. Finite-step reconstruction and task optimization also have different targets: an exactly reconstructed unfinished solution can retain substantial objective error.

This experiment uses exogenous problem streams and the stored iterate is also the emitted action. [EXPERIMENTS.md](../../EXPERIMENTS.md) specifies the next neural experiment with a distinct latent workspace and action-dependent observations.

## Reproduction

Original raw rows, fitted coefficients, seed namespaces, configuration, timing, and summaries remain in `results/`. CPU latency and the timing-calibrated iteration budget can vary across machines.

The original full-rerun record is [reproduction.json](results/reproduction.json). The documentation/output-directory revision is checked in [report_update_check.json](results/report_update_check.json). Both compare fixed-work numerical results separately from timing.

The code checks positive curvature, causal prefixes, residual/objective identities, zero context innovation, exact solutions, and finite-step sensitivity.
