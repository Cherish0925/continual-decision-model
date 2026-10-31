# CDM research

The research objective is a model that continues to observe, compute, act and
learn while its environment changes. [RESEARCH.md](../RESEARCH.md) defines that
objective and the [current execution priority](../RESEARCH.md#current-execution-priority).
[HANDOFF.md](../HANDOFF.md) records completed results and running experiments.

| Study | Contents |
| --- | --- |
| [Response updates and later experiments](../experiments/continual_experiment/RESPONSE_LEARNING.md) | Update and full joint comparisons complete; useful response fitting survives core training; strong-reference and long-term gains remain unresolved |
| [Active physical and chemical learning](../experiments/continual_experiment/ACTIVE_TRAINING.md) | Completed joint active-feedback training; response updates did not establish a task advantage |
| [Execution-aware physical learning](../experiments/continual_physics/EXECUTION.md) | Completed native-cost training and comparisons with recurrent and Bayesian policies |
| [Shared-core training](../experiments/continual_experiment/SHARED_TRAINING.md) | Physical actions, recorded chemical feedback and cross-domain state continuation |
| [Chemical response learning](../experiments/continual_chemistry/README.md) | History-conditioned predictions, actual experimental choices and planning |
| [Physical prediction and action](../experiments/physical_belief/README.md) | Reusable uncertain relationships and jointly trained experimental decisions |
| [Classical active calibration](../experiments/classical_calibration/RESULTS.md) | Bayesian measurement selection and relationship-memory reference |
| [Recurrent state and workspace](../experiments/continual_experiment/results/neural_state_controls.md) | Separate effects of carrying recurrent state and workspace across tasks |
| [Continual computation](CONTINUAL_COMPUTATION.md) | Earlier formulation of computation reuse, operation values and comparisons |
| [Process lifetimes](../experiments/lifetime/README.md) | Structured control and training through deployed learning updates |
| [Transport mechanism](COMPUTATION_TRANSPORT.md) | Earlier context-correction candidate and sensitivity derivations |
| [Transport related work](TRANSPORT_PRIOR_ART.md) | Prediction-correction, latent planning and recurrent derivatives |
| [Transport results](../experiments/transport/RESULTS.md) | Completed quadratic solver comparisons |
| [Phase review](PHASE_REVIEW.md) | Historical interpretation of the process-lifetime studies |

[DESIGN.md](../DESIGN.md) describes the architecture.
[TRAINING.md](../TRAINING.md) describes the data, objectives and optimization.
[RUNNING.md](../RUNNING.md) links executable studies.
