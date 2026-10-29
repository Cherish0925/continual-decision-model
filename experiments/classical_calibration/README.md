# Bayesian active calibration reference

This reference selects measurements and device settings without a CDM neural
policy, recurrent workspace or learned activity controller. It uses the same
nine affine hypotheses, train-only numerical prior, public instrument returns,
initial device settings and native execution tools. Each instance starts from
the task's initial state and has a budget of 400 actual environment actions.

For hypothesis $q$, standardized resonance satisfies
$y=b_q+a_q x_q$. Coefficients have a Gaussian prior. A received activated setting
$s$ constrains resonance to $(s-2,s+2)$ Hz; coefficient inference approximates
this interval by variance $4/3$ Hz squared. Input uncertainty contributes
$E[a_q^2]\,\mathrm{Var}(x_q)$ to observation variance. Three iterations recompute
the posterior from the same original prior while updating this slope-dependent
variance. Each observed label contributes once. The procedure is a Gaussian
moment approximation, not exact inference with uncertain covariates.

Unmeasured quantities retain the training marginal mean and variance. The
isolated radiation meter adds one-sided noise and rounds its return. The
reference uses the intersection of the corresponding feasible intervals from
returned samples; non-radiation quantities retain rounding uncertainty. If
observed radiation spread exceeds this simple noise model, the minimum return
supplies the local interval used by the implementation. This is
an approximate measurement model and does not establish a bound on unobserved
background radiation.

For the current target, returned failed settings remove their activation
intervals from the latent resonance distribution. A relation remains available
after failure. Candidate settings comprise coefficient-posterior predictions
and interval boundaries, projected onto the actual public control grid. The
chosen setting maximizes remaining activation probability with a small travel
cost adjustment. The predicted quantity is latent resonance: the activation
window is applied once, not added again as future label noise.

The measurement policy compares a candidate test with instrument plans that
complete missing readings for known supports and the current target. Eight
posterior samples estimate each plan's increase in the best remaining
activation probability. The gain is divided by expected execution cost;
measurement costs start from a prior development cohort and update from actual
returns. Repeated radiation measurements use a variance-reduction approximation.
This is a one-step acquisition rule over measurement bundles, not an exact
long-horizon planner or a reproduction of an existing neural experiment-design
method. All actual measurements and local setting probes consume the native
action budget.

Two configurations are fixed before confirmation:

- **Reset:** fresh coefficient prior for each task.
- **Forgetting memory:** up to 16 earlier task posteriors remain as mixture
  components alongside the fresh prior. Context mass decays by 0.95 per task;
  normalized coefficient covariance receives 0.0025 diagonal inflation.

Both configurations use the same online measurement-cost estimates. The
memory comparison concerns physical relationships, not a reset of all runtime
statistics. Neither configuration performs neural parameter updates or
internal activity selection. It executes decisions synchronously; elapsed
wall time therefore has no artificial two-tick result-delivery delay. We
report decision-computation and execution time separately.

## Run

```bash
python -m experiments.classical_calibration.evaluate_native \
  --mode forgetting \
  --manifest experiments/continual_physics/native_confirmation.json \
  --initial-rng-root runs/continual_physics/native_confirmation/initial_rng \
  --output runs/continual_physics/classical_confirmation/forgetting \
  --workers 6
```

Change `--mode` and the output directory to run the reset comparison. The
normalization/prior and training marginal defaults are the shared training
artifacts used by the native CDM experiments. No task seed, hidden resonance,
scenario relation identifier or evaluator score enters the decision model.

Development used the five new ordered streams in
`../continual_physics/native_development.json`. The two implementation
corrections made before confirmation were slope-dependent measurement-noise
refitting and separating latent resonance from noisy successful settings.
No confirmation instance is used for selecting those corrections or parameters.
Machine-readable development and confirmation results live under `results/`.
