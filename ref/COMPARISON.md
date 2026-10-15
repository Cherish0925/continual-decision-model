# Model comparisons

The current CDM experiment asks whether a model can learn to reuse unfinished computation when goals, observations, and dynamics change. The proposed mechanism stores intermediate work with its generating conditions, then reads and revises it using the current context. The comparison is task return at a measured computation and response-time budget.

Source editions and reading depth are recorded in [papers.json](papers.json). The architecture is in [DESIGN.md](../DESIGN.md); tasks, training conditions, and measurements are in [EXPERIMENTS.md](../EXPERIMENTS.md).

## Closest mechanisms

| Prior work | Established mechanism | CDM experiment |
| --- | --- | --- |
| [RMT](https://arxiv.org/abs/2207.06881), [Deep Repeated ConvLSTM](https://arxiv.org/abs/1901.03559) | Persistent recurrent state and repeated internal computation | Compare a shared recurrent policy with an explicit workspace and computational memory. |
| [RIM](https://arxiv.org/abs/1909.10893), [GateL0RD](https://arxiv.org/abs/2110.15949) | Modular or sparse latent updates trained for task performance | Compare learned retention and revision with selective recurrent updates. |
| [Incrementally-Computable Neural Networks](https://arxiv.org/abs/2307.14988) | Reuse of intermediate neural values after input edits | Compare reuse of an existing calculation with adaptation of its task-dependent intermediate state. |
| [LAP](https://arxiv.org/abs/2505.03077) | Observation-conditioned updates of a persistent latent plan | Compare retained computation with iterative latent replanning. |
| [A2C2](https://arxiv.org/abs/2509.23224) | Fresh-observation correction of actions from a slower base policy | Separate immediate action correction from revision that also improves subsequent computation. |
| [Successor features](https://arxiv.org/abs/1606.05312), [deep SF and GPI](https://proceedings.mlr.press/v80/barreto18a.html) | Policy-dependent reuse across changed rewards | Test goal changes separately from changes to transitions; include policy improvement. |
| [LPA*](https://idm-lab.org/bib/abstracts/Koen04a.html), [D* Lite](https://idm-lab.org/bib/abstracts/Koen02e.html) | Incremental repair of structured search | Compare with explicit graph planning where a graph model is available. |
| [Finding the Time to Think](https://arxiv.org/html/2606.26463v1) | Learned planning budgets with actions during computation | Compare fixed and learned schedules while the environment advances. |
| [MIRA](https://arxiv.org/abs/2610.02525) | Learned research allocation over persistent records | Compare activity selection and delayed returns at the appropriate timescale. |
| [Are Large Reasoning Models Interruptible?](https://arxiv.org/abs/2510.11713) | Evaluation of interrupted and changing-context reasoning | Vary the arrival time of new information during computation. |

These methods establish the individual roles of recurrence, reuse, selection, and adaptation. The CDM research contribution must be the specified learning mechanism and its measured behavior under changing conditions.

## Retrieval and conditioned reuse

Memory capacity and retrieval alone can recover useful prior work. The experiment separates four operations:

| Operation | What is recovered | Current information |
| --- | --- | --- |
| Snapshot retrieval | A stored workspace selected by a fixed retrieval rule | Combine it with the current context; do not restore an old position, goal, or observation history. |
| Context retrieval | A workspace selected by a learned query and memory keys | Give the retriever the same current context as the candidate model. |
| Conditioned reuse | A read operation selects and transforms stored work using its generating conditions and the current context | Train selection and revision together with the downstream task. |
| Fresh refinement | Additional computation after retrieval or initialization | Apply the same refinement budget and action interface. |

A learned retriever can already select task-relevant records. The additional question is whether transforming or composing retrieved intermediate results improves decisions over direct retrieval followed by the same refiner. Compare both while holding the memory capacity, training data, observation access, and measurement procedure fixed. Report parameter counts and actual lookup, read, refinement, and action latency.

Use three change regimes. Full restoration tests reuse when previous conditions recur. Partial restoration changes only some conditions, testing whether useful components can be retained. Continuous change tests revision when a matching snapshot is unlikely. All methods receive fresh context through the action path. A condition label used only by the simulator is available for analysis or training supervision, not as an unreported test-time model input.

## Activity and learning

[Continual computation](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/11/cc_aij_horvitz.pdf), [Metacontrol](https://arxiv.org/abs/1705.02670), and [Thinker](https://arxiv.org/abs/2307.14993) supply foundations for preparing future work and choosing internal operations. Activity learning should predict achievable future return from observing, computing, retrieving, or acting, including elapsed time. A change in the action distribution alone does not measure computational benefit.

[DreamerV3](https://www.nature.com/articles/s41586-025-08744-2) and [TD-MPC2](https://arxiv.org/abs/2310.16828) provide world-model comparisons. [TTT-E2E](https://arxiv.org/abs/2512.23675), [Nested Learning](https://arxiv.org/abs/2512.24695), [SEAL](https://arxiv.org/abs/2506.10943), and [Meta-TTL](https://arxiv.org/abs/2604.00830v4) concern parameter adaptation and learning procedures. These are distinct from retaining a workspace during one episode; assess transfer after learning separately from immediate memory reuse.

The earlier low-rank state-update study remains documented in [COMPUTATION_TRANSPORT.md](../research/COMPUTATION_TRANSPORT.md) and [its nearest comparisons](../research/TRANSPORT_PRIOR_ART.md). Low-rank correction is one possible implementation to compare, rather than the definition of CDM.
