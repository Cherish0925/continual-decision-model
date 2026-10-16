# Related methods for continuing computation

Read: 14 October 2026.

The proposed neural mechanism jointly learns a persistent workspace, shared refinement, and a compressed observation-conditioned update. The relevant prior methods address reuse, correction, planning, and the propagation of sensitivities.

## Closest methods

| Work and inspected material | Mechanism | Comparison to implement |
| --- | --- | --- |
| Simonetto et al., 2016, [prediction-correction](https://arxiv.org/html/1509.05196v2), Section II, equations 3–8 | Moves a potentially nonoptimal iterate with a changing objective while approximately preserving its residual, then corrects it | Explicit numerical prediction-correction; transporting unfinished iterates is already part of this method |
| Diehl, Bock, and Schlöder, 2005, [real-time iteration](https://epubs.siam.org/doi/10.1137/S0363012902400713), publisher abstract | Newton-type refinement during physical execution, with contraction and optimality analysis | Warm-start control with a supplied or learned dynamics model |
| Sambharya et al., 2024, [learned warm starts](https://www.jmlr.org/papers/volume25/23-1174/23-1174.pdf), abstract and architecture | Learns an initializer through a finite number of solver iterations | Initializer from current context versus revision using the previous workspace |
| Noh et al., 2025, [Latent Adaptive Planner](https://arxiv.org/html/2505.03077v2), variational-replanning section, equations 10–11 | Keeps a Gaussian latent-plan posterior, uses it as the next prior, and performs incremental replanning | Persistent latent planning versus persistent learned refinement |
| Black, Galliker, and Levine, 2025, [RTC](https://arxiv.org/html/2506.07339v1), Section 4 | Generates a new action chunk during execution and incorporates already committed actions | Asynchronous execution and action compatibility |
| Sendai et al., 2025, [A2C2](https://arxiv.org/html/2509.23224v1), Section 3, equations 3–5 | Corrects actions using current observations, older policy features, and chunk position | Action-only correction versus revision retained for future inference |
| Sommer and Schilling, 2026, [Reinforced Planning](https://arxiv.org/html/2608.18669v1), Section 5, equation 10 and implementation | Learns residual plan refinement through a latent world model; inspected experiments use open-loop planning | Learned refinement with repeated observations and action-dependent outcomes |
| Chen, Callaway et al., 2026, [recurrent metareasoning](https://www.biorxiv.org/content/10.64898/2026.04.14.718499v1), indexed primary passages and [author code](https://github.com/SixingChen028/recurrent-meta-reasoning) | Recurrent actor–critic learning selects computation and external behavior | Learned activity selection and revision of selected work after observations |
| Williams and Zipser, 1989, [RTRL](https://doi.org/10.1162/neco.1989.1.2.270), original paper | Propagates recurrent state–parameter derivatives through time | Exact finite-step sensitivities where computationally feasible |
| Tallec and Ollivier, 2018, [UORO](https://arxiv.org/pdf/1702.05043), Sections 3.1–3.2, equations 5–12 | Stochastic low-rank approximation of recurrent derivatives | Sensitivity compression, update cost, and a task-trained response map |

These works supply the mathematical and algorithmic foundations. The neural study tests their interaction with a learned representation and task-outcome training.

## Mathematical relation

For \(z_{k+1}=F_\theta(z_k,c)\), the derivative satisfies
\(S_{k+1}=J_zF_\theta S_k+J_cF_\theta\).
A low-rank map approximates its action on \(\Delta c\). This is a compressed first-order response calculation. Fixed-point sensitivity and finite-step sensitivity describe different reference states.

The CDM experiment must therefore test whether the representation and learned update save useful computation on later decisions. The [mechanism note](COMPUTATION_TRANSPORT.md) provides the full derivation, including context-dependent seeds and local error bounds.

## Experimental comparisons

1. Give action heads current observations and measure the complete observation-to-action time.
2. Compare revision with carry-and-refine, ordinary recurrence, a learned initializer, and restart.
3. Compare action-only correction with an update retained for subsequent inference.
4. Compare the low-rank operation with an unconstrained learned update and numerical sensitivities where feasible.
5. Vary smooth changes, irrelevant observations, local changes, new goals, abrupt changes, and stationary periods.
6. Train on repeated interruptions and evaluate action-dependent outcomes in unfamiliar environments.

When policies choose their observations, match initial information, sensor access, and resource budgets. Use common fixed streams for state-update comparisons, then separate closed-loop runs for active observation.

The completed quadratic experiment fits a small correction with a fixed numerical representation. The proposed neural model will learn the representation, refinement, revision, and policies jointly.
