# Continual computation

The current research asks how intermediate computation can remain useful while observations, goals, and the environment change. CDM combines a fresh recurrent context, persistent workspace, computation records, and learned reading and revision. The complete model and learning sequence are specified in [DESIGN.md](../DESIGN.md) and [TRAINING.md](../TRAINING.md).

| Study | Contents |
| --- | --- |
| [Continual computation](CONTINUAL_COMPUTATION.md) | Central hypothesis, condition-factorized reading, operation values, and comparisons |
| [Neural prototype](../cdm/model.py) | GRU context, shared Transformer workspace, bounded record memory, action/activity/value heads |
| [Supervised pilot](../experiments/neural_pilot/README.md) | Sequential action learning with restart, carry, and memory reading; [five-seed results](../experiments/neural_pilot/results/REPORT.md) |
| [Changing grid](../experiments/continual_grid/README.md) | Explicit planning-state reuse while the environment advances |
| [Transport mechanism](COMPUTATION_TRANSPORT.md) | Earlier context-correction candidate and sensitivity derivations |
| [Transport related work](TRANSPORT_PRIOR_ART.md) | Prediction-correction, latent planning, action correction, and recurrent derivatives |
| [Transport results](../experiments/transport/RESULTS.md) | Completed quadratic solver comparisons and finite-step reconstruction |

The analytical grid comparison and supervised neural pilot address different questions. The next experiment trains the neural model inside the changing environment, followed by learned activity selection and continued parameter adaptation. [RUNNING.md](../RUNNING.md) contains the executable commands and current results.
