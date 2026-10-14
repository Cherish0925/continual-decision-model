# Numerical transport experiment

A five-seed CPU experiment on changing 16-dimensional quadratic objectives. The fitted polynomial correction underperforms the stronger solvers tested. A separate finite-step reconstruction task explains why copying unfinished computation can preserve its optimization error.

## Problem and fitted models

For f_t(z) = 1/2 z^T H_t z − b_t^T z, emit a decision vector at each simulated tick. H_t is symmetric positive definite. The correction uses the context innovation q_t = (b_t − H_t z) − r_previous, with the cached residual evaluated at the same prior output. Apply a learned degree-two inverse polynomial to q_t, then one gradient correction. All methods receive current H, b, and the same spectral bounds.

The inverse model has nine fitted parameters. A separate 18-parameter model also conditions on interruption depth and learns the change in a same-depth gradient iterate. Both use polynomial approximation and prediction-correction methods.

## Training and evaluation

Five independent training seeds; 1800 paired samples per inverse model and 1800 per finite-step model. Damping is selected on three validation streams. Six independent test streams per regime have 128 ticks each. Training, validation, main tests, and reconstruction tests use separate random-seed namespaces. Parameters are frozen before test.

The main score is cumulative excess quadratic objective divided by the cumulative excess objective of always emitting zero; lower is better. Exact solutions supply labels and scores and are separately reported as numerical reference methods.

Selected damping by seed: 1101: 1.0, 2202: 1.0, 3303: 1.0, 4404: 1.0, 5505: 1.0.

Validation-calibrated gradient iterations at approximately the candidate CPU latency: 1101: 9, 2202: 8, 3303: 9, 4404: 8, 5505: 9. This iteration budget depends on machine timing.

## Main results

| Regime | Learned transport | Current-residual control | Readout-only control | Warm GD (5) | Latency GD | Nesterov (5) | CG (4) |
|---|---:|---:|---:|---:|---:|---:|---:|
| stationary_control | 0.00395836 | 6.24199e-05 | 0.00395836 | 0.000254041 | 7.43802e-05 | 5.9758e-05 | 2.51845e-05 |
| slow_change_id | 0.0039617 | 0.000112282 | 0.00578505 | 0.000480644 | 0.000145379 | 0.000110057 | 4.77494e-05 |
| fast_target_ood | 0.010211 | 0.00648263 | 0.108338 | 0.0251891 | 0.00832038 | 0.00614741 | 0.00304872 |
| curvature_ood | 0.011107 | 0.000660152 | 0.0146944 | 0.00203678 | 0.000882149 | 0.000575853 | 0.000325265 |
| abrupt_joint_ood | 0.0304954 | 0.00812249 | 0.0973423 | 0.0223902 | 0.0104726 | 0.00720906 | 0.00355481 |

The correction loses to Nesterov, CG, the same-weights current-residual control, and validation-latency-calibrated GD in all five regimes. It beats ordinary GD5 only under fast target changes. Persisting the correction helps relative to readout-only correction during changes, but does not overcome the stronger methods.

Ratios below are means of per-stream ratios after averaging training seeds. They differ from ratios of the aggregate means above. Bootstrap intervals resample the six independent test streams; repeated deterministic baseline seeds are not independent replications.

| Regime | Learned / CG gap | Stream-bootstrap 95% interval |
|---|---:|---:|
| stationary_control | 524 | [162, 948] |
| slow_change_id | 125 | [68, 204] |
| fast_target_ood | 3.86 | [3.07, 5.04] |
| curvature_ood | 34.1 | [30.7, 37.6] |
| abrupt_joint_ood | 8.61 | [7.84, 9.56] |

## Finite-step state reconstruction

For fixed H and target change Δb, the change in the k-step gradient trajectory is (I − (I − H/L)^k) H^{-1} Δb. This differs from equilibrium displacement H^{-1} Δb. The depth-conditioned model trains on k = 1, 2, 4, 8, 16 and tests on unseen k = 3, 6, 12. Relative state error divides reconstruction distance by the exact state displacement.

| Test distribution | Learned finite-step error | Learned inverse error | Exact inverse error | Exact finite tangent error |
|---|---:|---:|---:|---:|
| seen_depth_id | 0.0590994 | 0.825425 | 1.22616 | 1.82961e-16 |
| unseen_depth_curvature_ood | 0.100362 | 0.793753 | 4.80829 | 1.957e-16 |
| unseen_depth_id | 0.0558032 | 0.371473 | 0.644856 | 1.86998e-16 |

On unseen in-distribution depths, the same-depth reference itself has mean objective gap **0.727103**. Exact reconstruction has zero state error while retaining that gap. The learned finite-step update has mean gap 0.76091, compared with 0.495945 for the learned inverse and 0.379438 for exact equilibrium transport.

The different rankings follow from the target: reproducing an unfinished iterate and improving its objective are different tasks. This experiment corrects the training choice. Neural training will optimize subsequent task outcomes, with state reconstruction as an auxiliary comparison. The exact finite tangent uses k gradient steps and is reported as a variable-work reference.

## Computation and timing

The candidate uses five H-vector products per tick: current residual (1), polynomial basis (2), corrected-state residual (1), and output residual cache (1). GD5 and Nesterov5 use five; CG4 uses an initial residual and at most four search-direction products. Coefficient generation and vector operations add overhead. Equal product counts therefore require a separate latency comparison.

The current-residual control applies the same fitted polynomial to the whole residual. The readout-only control emits the correction but retains its uncorrected one-step state. Both use five products. Exact-solve references have a different cost.

CPU timings include each method loop and trace copies; data generation, offline training, and scoring are separate. The simulator advances once per tick, independently of CPU microseconds. The stale baseline uses observations three ticks old by construction.

| Method | Median CPU microseconds / tick |
|---|---:|
| stale_3_ticks_gd5 | 12.702 |
| restart_gd5 | 12.402 |
| warm_gd1 | 3.276 |
| warm_gd5 | 12.430 |
| warm_gd_latency | 21.033 |
| warm_nesterov5 | 20.101 |
| warm_cg4 | 30.413 |
| warm_jacobi_cg4 | 30.473 |
| learned_transport | 21.928 |
| learned_current_residual | 22.130 |
| readout_only_transport | 22.744 |
| scalar_transport | 8.036 |
| analytic_previous_inverse | 14.004 |
| analytic_current_inverse | 13.926 |
| exact_current_solution | 6.445 |

## Interpretation and next experiment

In stationary conditions, context innovation is zero and the candidate spends its five-product budget on only one corrective step. Under changes, its retained update still loses to the stronger solver controls. This polynomial construction is not selected as the neural CDM core.

The problems are small, fully observed quadratics with supplied spectral bounds. Streams are exogenous and the stored iterate equals the action vector. The next experiment uses a distinct learned latent workspace, nonlinear recurrence, action-dependent observations, and measured inference time.

Current-residual, readout-only, and latency controls were added after inspecting the initial result; candidate parameters, hyperparameters, and test streams remained unchanged. The finite-step test uses a fixed zero seed. Five trained models and six test streams per regime define the replication structure.

## Files

The results directory contains main_raw.csv, main_summary.csv, paired_comparisons.csv, finite_progress_raw.csv.gz, finite_progress_summary.csv, validation.csv, latency_calibration.csv, frozen_models_seed_*.npz, metadata.json, and config.json. The implementation checks causal prefixes, residual/objective identities, zero innovation, finite-step sensitivities, and exact solutions.
