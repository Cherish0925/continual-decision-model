# Continuing computation under new observations

The current study asks how a model should revise unfinished inference when observations or goals change. The proposed neural model jointly learns a persistent workspace, shared refinement, and a small observation-conditioned update. Its training objective is subsequent task performance under elapsed-time costs.

| Document | Content |
| --- | --- |
| [Mechanism](COMPUTATION_TRANSPORT.md) | Update equations, sensitivity analysis, and training |
| [Related methods](TRANSPORT_PRIOR_ART.md) | Prediction-correction, latent planning, action correction, and recurrent derivatives |
| [Numerical experiment](../experiments/transport/README.md) | Runnable polynomial-correction experiment |
| [Results](../experiments/transport/RESULTS.md) | Solver comparisons and finite-step reconstruction |

The numerical correction loses to the stronger solvers tested. In the separate reconstruction test, an exact same-depth reference retains an average objective gap of 0.727103 on unseen depths. Recovering that state accurately therefore retains its unfinished optimization. This explains why state fidelity and task performance can rank methods differently.

The next experiment trains nonlinear latent computation in an action-dependent environment. It compares ordinary recurrence, a learned initializer, action-only correction, and persistent revision under matched information and measured runtime.

The full CDM programme, including proactive observation and continuing learning, is specified in [RESEARCH.md](../RESEARCH.md). Server continuation starts at [RUNNING.md](../RUNNING.md).
