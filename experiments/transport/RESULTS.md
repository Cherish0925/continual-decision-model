# Transport of unfinished numerical state: CPU mechanism study

This is a completed small numerical experiment, not a CDM implementation, agent benchmark, GPU benchmark, or originality claim. All learned parameters are fitted on synthetic training data and frozen before test.

## Question and candidate

For f_t(z) = 1/2 z^T H_t z − b_t^T z, an actor must emit a decision every environment tick. The candidate transports its prior iterate using the evidence innovation q_t = (b_t − H_t z) − r_previous, where the cached residual is evaluated at the same prior output. It applies P_theta(H_t, L_t) q_t, then one current-evidence gradient correction. P is a learned degree-two polynomial in I − H/L, with coefficients conditioned on clipped log condition number. Eigenvalue bounds (minimum 1, maximum L) are supplied equally to all methods.

This is an established-style predictor–corrector construction with fitted coefficients. The experiment tests a mechanism and its limitations; polynomial inverse approximation is not claimed as a new mathematical method.

## Training and evaluation

Five independent training seeds; 1800 paired samples per inverse model and 1800 per finite-progress model. Damping is chosen only on three validation streams. Tests share six independent streams per regime, each 128 ticks, across every method and training seed. Main test streams and diagnostic samples have separate seed namespaces from training and validation. OOD tests hold out target-shift magnitudes and/or curvature range.

The main score is cumulative excess quadratic objective divided by the cumulative excess objective of always emitting zero (lower is better). Residuals and state norms are also saved. Exact solutions are used only for labels, scoring, and explicitly labeled oracle references. They are unavailable to learned/control policies.

Selected damping by seed: 1101: 1.0, 2202: 1.0, 3303: 1.0, 4404: 1.0, 5505: 1.0.

Validation-calibrated GD iterations at approximately the candidate CPU latency: 1101: 9, 2202: 8, 3303: 9, 4404: 8, 5505: 9. This additional control uses more matvecs where the candidate’s overhead permits them; timing noise means test latency is only approximately matched.

## Main results

| Regime | Learned transport | Current-residual control | Readout-only control | Warm GD (5) | Latency GD | Nesterov (5) | CG (4) |
|---|---:|---:|---:|---:|---:|---:|---:|
| stationary_control | 0.00395836 | 6.24199e-05 | 0.00395836 | 0.000254041 | 7.43802e-05 | 5.9758e-05 | 2.51845e-05 |
| slow_change_id | 0.0039617 | 0.000112282 | 0.00578505 | 0.000480644 | 0.000145379 | 0.000110057 | 4.77494e-05 |
| fast_target_ood | 0.010211 | 0.00648263 | 0.108338 | 0.0251891 | 0.00832038 | 0.00614741 | 0.00304872 |
| curvature_ood | 0.011107 | 0.000660152 | 0.0146944 | 0.00203678 | 0.000882149 | 0.000575853 | 0.000325265 |
| abrupt_joint_ood | 0.0304954 | 0.00812249 | 0.0973423 | 0.0223902 | 0.0104726 | 0.00720906 | 0.00355481 |

Ratios are means of per-stream ratios after averaging training seeds, not ratios of the aggregate means in the preceding table. Ratios below 1 favor learned transport. Confidence intervals resample the six independent test streams after averaging training seeds; training-seed ranges are separately saved, so repeated deterministic baseline runs are not treated as independent evidence.

| Regime | Learned / CG gap | Stream-bootstrap 95% interval |
|---|---:|---:|
| stationary_control | 524 | [162, 948] |
| slow_change_id | 125 | [68, 204] |
| fast_target_ood | 3.86 | [3.07, 5.04] |
| curvature_ood | 34.1 | [30.7, 37.6] |
| abrupt_joint_ood | 8.61 | [7.84, 9.56] |

## Finite-progress diagnostic

A separate frozen learned polynomial also conditions on interruption depth k. For fixed H and a change Δb, the exact change in the unfinished k-step gradient trajectory is (I − (I − H/L)^k) H^{-1} Δb. It generally differs from equilibrium transport H^{-1} Δb. The model trains on depths 1, 2, 4, 8, 16 and is tested on unseen depths 3, 6, 12. The score is distance to the recomputed same-depth state, normalized by the exact state displacement. This measures preservation of finite computational progress, not quality of the final action.

| Diagnostic | Learned finite-progress error | Learned inverse error | Exact inverse error | Exact finite tangent error |
|---|---:|---:|---:|---:|
| seen_depth_id | 0.0590994 | 0.825425 | 1.22616 | 1.82961e-16 |
| unseen_depth_curvature_ood | 0.100362 | 0.793753 | 4.80829 | 1.957e-16 |
| unseen_depth_id | 0.0558032 | 0.371473 | 0.644856 | 1.86998e-16 |

Finite-state fidelity is a diagnostic target. The objective comparison reverses the fidelity ranking on unseen in-distribution depths: learned finite-progress transport has mean objective gap 0.76091, versus 0.495945 for learned inverse transport and 0.379438 for exact equilibrium transport. Lower gap is better. This is direct evidence that matching the old computation depth alone is not the desired task objective. The exact finite tangent is computed with k gradient steps, so it is an accuracy reference with variable work.

## Compute accounting and timing

Main learned transport uses five dense H-vector products per tick: current residual (1), degree-two polynomial basis (2), corrected-state residual (1), and output residual cache update (1). Warm GD and restarted Nesterov each use five; CG and Jacobi CG use one initial residual plus at most four search-direction products. Early convergence may use fewer. Scalar transport uses three, and warm GD1 uses one. Restart GD5 begins at zero. The deliberately stale baseline performs five gradient steps on evidence three ticks old, although all policies have the current evidence available.

The analytical previous/current inverse references use a dense linear solve plus three matvecs per tick; exact current solution uses a solve. These are not cost-matched winners. Candidate vector arithmetic, coefficient evaluation, and context clipping are additional work beyond matvec counting; raw wall-time measurements include them.

Two additional five-product controls use exactly the same learned parameters. Current-residual correction applies the polynomial to all current residual rather than only evidence innovation. Readout-only transport emits a transported action but carries forward only its one-step untransported GD state; its last product maintains that state’s residual. These distinguish a useful learned preconditioner from evidence transport and persistent transport from a transient output correction within this toy.

CPU wall time covers each complete algorithm loop and trace copies, excludes offline training, environment generation and scoring, and is reported separately from simulated ticks. These small NumPy timings are descriptive, not hardware-independent latency or GPU measurements. The benchmark advances the environment once per tick; it does not turn actual CPU microseconds into simulated time. The three-tick stale baseline is an explicit delay condition, not measured hardware latency.

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

## Interpretation and limits

The learned equilibrium-style transport fails to beat Nesterov or CG in every tested regime. It beats warm GD5 only in the fast-target regime. The same-weights current-residual control beats it in every regime, and the validation-latency-calibrated GD control also beats it in every regime. Persistent transport improves on the readout-only control under changes, but this does not overcome the strong baselines. This is a negative result for the proposed main mechanism, not evidence for an effective CDM core. Stationary streams expose the cost of spending work on evidence transport: with no innovation, the candidate pays its five-product budget but advances only one correction.

The finite-progress study isolates a real distinction between an unfinished trajectory and its equilibrium. Better finite-state reconstruction alone does not establish better decisions, autonomous agency, persistent learning, or a novel architecture. The next research gate is a learned finite-progress state transfer that improves closed-loop outcomes under equal measured compute, with nonlinear recurrences and changes in both evidence and objectives.

All problems are small, fully observed, strongly convex quadratics with known spectral bounds; generalization to nonlinear, partial-observation, long-horizon agents is untested. The streams are exogenous: decisions do not alter future observations, so the main experiment is optimization tracking, not closed-loop control. The persistent solver state equals its decision vector, so this does not establish communication or revision of a distinct general-purpose latent workspace. The finite diagnostic uses a fixed zero initial iterate and a linear recurrence; it does not address sensitivity identifiability for variable branch seeds. Six test streams per regime give limited distributional coverage. The current-residual, readout-only and latency controls were added after inspecting the initial result; candidate parameters, hyperparameters, seeds, and test streams were not changed. No GPU speedup is claimed.

## Artifacts

`results/main_raw.csv`, `main_summary.csv`, `paired_comparisons.csv`, `finite_progress_raw.csv.gz`, `finite_progress_summary.csv`, `validation.csv`, `latency_calibration.csv`, `frozen_models_seed_*.npz`, `metadata.json`, and `config.json`. The script contains causal prefix, residual/objective, zero-innovation, finite-tangent, and exact-oracle sanity assertions.
