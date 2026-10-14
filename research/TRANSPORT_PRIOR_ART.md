# Prior art for transporting unfinished computation

Reviewed: 2026-10-14. Scope: primary sources closest to the proposed CDM mechanism, rather than the complete CDM literature. Statements about the proposed mechanism below are research hypotheses. No claim of priority or empirical superiority follows from this audit.

## Candidate being evaluated

CDM would maintain a persistent internal iterate while evidence and external time continue to change. A learned, inexpensive transport operation would adapt that unfinished computation to new evidence; a shared refiner would continue work from the adapted state. An action readout would use current evidence and the latest usable internal state at every control opportunity.

The specific candidate is to jointly learn the internal representation, refinement operator, and a compressed evidence-sensitivity map, using losses on interrupted task trajectories. The intended benefit is to preserve useful computation under a fixed end-to-end budget, including the cost of transport itself.

## Ten close primary works

### 1. Prediction-correction for time-varying optimization

Andrea Simonetto, Aryan Mokhtari, Alec Koppel, Geert Leus, Alejandro Ribeiro. *A Class of Prediction-Correction Methods for Time-Varying Convex Optimization*. IEEE Transactions on Signal Processing, 2016.

[Paper](https://arxiv.org/abs/1509.05196) · [Full text](https://arxiv.org/html/1509.05196v2)

**Verified:** Section II, equations (3)–(8). The predictor follows the changing optimization problem while approximately preserving its gradient residual; a bounded number of gradient or Newton corrections then reduces that residual.

**Overlap:** Transporting a current iterate before continuing refinement is established. Crucially, the derivation starts from an arbitrary, potentially nonoptimal iterate. “We transport unfinished work rather than the optimum” is therefore insufficient by itself to distinguish CDM.

**Remaining question:** Can a learned neural representation and low-cost transport retain useful task computation when the objective and evidence interface are learned, observations arrive irregularly, and actions remain due throughout inference? The classical paper does not supply that particular learned architecture or its evaluation.

### 2. Real-time iteration in nonlinear optimal control

Moritz Diehl, Hans Georg Bock, Johannes P. Schlöder. *A Real-Time Iteration Scheme for Nonlinear Optimization in Optimal Feedback Control*. SIAM Journal on Control and Optimization 43(5), 1714–1736, 2005.

[Publisher](https://epubs.siam.org/doi/10.1137/S0363012902400713)

**Verified:** Publisher abstract. A Newton-type online scheme continually refines approximate optimal feedback while the controlled process runs; the work analyzes contraction and loss of optimality.

**Overlap:** Acting before optimization converges, reusing computation, and refining during physical execution are longstanding control ideas.

**Remaining question:** CDM would need to learn an internal computation interface shared across task families, rather than merely implementing a learned approximation to an existing control solver. RTI is an essential comparison wherever an explicit dynamics and cost model permit it. The paper's control guarantees cannot be transferred to an unconstrained neural implementation.

### 3. Learned warm starts

Rajiv Sambharya, Georgina Hall, Brandon Amos, Bartolomeo Stellato. *Learning to Warm-Start Fixed-Point Optimization Algorithms*. Journal of Machine Learning Research 25(166), 2024.

[Publication](https://www.jmlr.org/beta/papers/v25/23-1174.html) · [Full paper](https://www.jmlr.org/papers/volume25/23-1174/23-1174.pdf)

**Verified:** Abstract and architecture description. A neural network maps problem parameters to an initial iterate; training differentiates through a fixed number of subsequent algorithm iterations.

**Overlap:** Learning a state that becomes useful after a limited refinement budget, rather than requiring an exact solution, is established.

**Remaining question:** Does conditioning transport on the existing unfinished state preserve valuable instance-specific work beyond a strong learned initializer from the latest evidence? This must be measured directly. A transport network that ignores its previous state could otherwise reduce to a warm-start predictor.

### 4. Incremental latent replanning

Donghun Noh and colleagues. *Latent Adaptive Planner for Dynamic Manipulation*. 2025.

[Proceedings](https://proceedings.mlr.press/v305/noh25a.html) · [Full text](https://arxiv.org/html/2505.03077v2)

**Verified:** Variational-replanning section, equations (10)–(11). LAP maintains a Gaussian posterior over a trajectory-level latent plan, uses the previous posterior as the next prior, and applies a small variational update as observations arrive. Its reported implementation uses one replanning gradient step after a longer initial inference phase.

**Overlap:** Persistent latent plans, incremental evidence updates, and reduced replanning computation are already explicit.

**Remaining question:** CDM's candidate transports the internal progress of a learned refiner and resumes that refiner. That distinction requires actual state semantics, training objectives, and evidence of retained computational progress; merely renaming LAP's posterior a workspace would not establish it.

### 5. Asynchronous action chunk execution

Kevin Black, Manuel Y. Galliker, Sergey Levine. *Real-Time Execution of Action Chunking Flow Policies*. 2025.

[Paper](https://arxiv.org/abs/2506.07339) · [Full text](https://arxiv.org/html/2506.07339v1)

**Verified:** Section 4. RTC generates a new chunk while the previous chunk executes, accounts for actions already committed during inference, and uses inpainting and soft masking to maintain chunk compatibility.

**Overlap:** Concurrent inference and action, observation staleness, and reuse of a previous action sequence are established practical mechanisms.

**Remaining question:** Transport of an internal unfinished iterate should preserve future refinement ability, beyond continuity of the external action chunk. CDM evaluation should count committed actions and observation-to-execution delay explicitly. Comparing only with synchronous inference would be weak.

### 6. Fast correction from current observations

Kohei Sendai, Maxime Alvarez, Tatsuya Matsushima, Yutaka Matsuo, Yusuke Iwasawa. *Leave No Observation Behind: Real-time Correction for VLA Action Chunks*. 2025.

[Paper](https://arxiv.org/abs/2509.23224) · [Full text](https://arxiv.org/html/2509.23224v1)

**Verified:** Section 3, equations (3)–(5). A2C2 adds a lightweight, per-control-step correction head conditioned on current observations, the base action, its position within a chunk, and base-policy features. It trains the head without retraining the base policy.

**Overlap:** A fast path that combines fresh evidence with a slower model's old output or latent features is already a direct solution to part of CDM's motivating problem.

**Remaining question:** Does modifying the continuing internal computation improve later decisions beyond a matched action-only correction head? This is an essential ablation, not an optional comparison. The distinction is state that remains useful for further reasoning versus correction that only changes the current output.

### 7. Learned plan refinement

Armin Sommer, Jannik Schilling. *Reinforced Planning with Latent World Models*. arXiv preprint, August 2026.

[Paper](https://arxiv.org/abs/2608.18669) · [Full text](https://arxiv.org/html/2608.18669v1)

**Verified:** Section 5, equation (10). RP1 learns a residual update to a multistep action plan using the plan, a critic value, and its gradient. Training uses imagined world-model trajectories. The implementation section explicitly states that its experiments use open-loop planning.

**Overlap:** Learned iterative improvement of plans through a world model cannot be claimed as CDM's invention.

**Remaining question:** The proposed CDM setting introduces interruptions, changing evidence during computation, and repeated internal-state migration before later refinement. The inspected RP1 experiments do not establish performance in that setting. This is a specific experimental distinction, not a claim that RP1 cannot be extended.

### 8. Recurrent metareasoning

Sixing Chen, Frederick Callaway and colleagues. *Learning to select computations in recurrent neural circuits*. bioRxiv preprint, April 2026.

[Paper](https://www.biorxiv.org/content/10.64898/2026.04.14.718499v1) · [Author code](https://github.com/SixingChen028/recurrent-meta-reasoning)

**Verified:** Indexed primary abstract/method passages and author repository. The model uses recurrent meta-reinforcement learning to control computational choices and external behavior. The repository describes an actor–critic recurrent architecture and choice/planning case studies. Direct full-paper retrieval was blocked during this audit, so details beyond those passages were not independently inspected.

**Overlap:** Learning what and when to compute, with recurrent internal state, is already the explicit subject of this work.

**Remaining question:** CDM must specify how selected work survives a change in evidence and what incremental transport contributes. Merely adding a computation-selection policy does not answer that question.

### 9. Forward sensitivity in recurrent networks

Ronald J. Williams, David Zipser. *A Learning Algorithm for Continually Running Fully Recurrent Neural Networks*. Neural Computation 1(2), 270–280, 1989.

[DOI](https://doi.org/10.1162/neco.1989.1.2.270) · [Original-paper mirror](https://gwern.net/doc/ai/nn/rnn/1989-williams-2.pdf)

**Verified:** Original paper. RTRL propagates sensitivities through the recurrent state using the chain rule to obtain online gradients.

**Overlap:** Recursively tracking the derivative of an evolving state is established automatic differentiation. Replacing differentiation with respect to weights by differentiation with respect to conditioning evidence does not create a new chain rule.

**Remaining question:** The use of a trained, compressed sensitivity map to migrate inference state and improve downstream decisions is a separate design question. Its value must come from useful approximation, representation learning, and budgeted task performance.

### 10. Compressed recurrent sensitivities

Corentin Tallec, Yann Ollivier. *Unbiased Online Recurrent Optimization*. 2017 preprint; ICLR 2018.

[Paper](https://arxiv.org/abs/1702.05043) · [Full paper](https://arxiv.org/pdf/1702.05043)

**Verified:** Sections 3.1–3.2, equations (5)–(12). UORO approximates the recurrent state–parameter Jacobian with a stochastic rank-one representation, reducing storage while maintaining an unbiased estimator under its formulation.

**Overlap:** Low-rank compression of recurrent sensitivity is established. Low-rank factorization alone cannot carry CDM's novelty claim.

**Remaining question:** A deterministic, task-trained transport approximation need not share UORO's estimator or objective, but must be distinguished from low-rank autodifferentiation. Compare approximation error, memory, update cost, and effect on interrupted-task outcomes. UORO's unbiasedness does not automatically apply to such a learned map.

## The mathematical boundary

For a refiner with fixed conditioning during a branch,

\[
z_{k+1}=F_\theta(z_k,c),\qquad S_k=\frac{\partial z_k}{\partial c},
\]

ordinary forward differentiation gives

\[
S_{k+1}=J_zF_\theta\,S_k+J_cF_\theta.
\]

Initialize \(S_0=0\) only when the branch seed is independent of \(c\); otherwise differentiate the seed too. At a small conditioning change, the candidate transport is

\[
\widetilde z_k=z_k+A_\phi(z_k,c,k)\,\Delta c,
\qquad A_\phi=UV^\top.
\]

If \(A_\phi=S_k\), this is a first-order Taylor approximation to the same-progress branch under changed conditioning. A finite-progress target \(F_{c+\Delta c}^{k}(z_0)\) is different from an equilibrium sensitivity target. Both distinctions and derivative recurrences are mathematical facts, not novel algorithms. Moreover, the prediction-correction literature already includes transportation of nonconverged iterates.

The candidate contribution is therefore the *jointly learned computational object*: the refiner, its state representation, and an inexpensive map trained to preserve task-useful progress across interruptions. Raw hidden-state agreement is not enough; a representation may be easy to match while carrying little useful work. The final objective must evaluate subsequent refinement and actions. Large evidence changes also require learned finite corrections or renewed computation; local linearization gives no general guarantee for abrupt mode changes.

## Experiments that discriminate the candidate

1. **Same-information comparison.** All methods receive the same observations at the same physical times. Account for encoder, transport, refinement, and readout latency; the environment advances during computation.
2. **State necessity.** Compare learned transport with a strong initializer from current evidence, plain carry-and-refine, a matched generic recurrent update, and restart. Remove or shuffle the transported state to measure whether past computation is actually used.
3. **Sensitivity necessity.** Compare exact finite-progress Jacobian-vector products, low-rank numerical approximations, a learned unstructured edit, and the trained low-rank map. An exact derivative is a correctness reference, not automatically the best task policy.
4. **Output versus computation.** Compare with A2C2-style action-only correction at the same budget, then test whether transport improves *later* decisions and reduces the subsequent refinement needed to reach a matched quality.
5. **Changes and budgets.** Evaluate smooth drift, irrelevant events, local contradictions, goal changes, and abrupt regime switches across multiple interruption depths and delays. Include stationary tasks to quantify overhead when transport is unnecessary.

On explicit optimization problems, include prediction-correction and RTI where applicable. A gain against stale or restarted inference alone does not distinguish the proposal from these traditions. Use paired rollouts or shared prefixes when possible to identify retained computation rather than easier observations.

## Claim that could be supported by evidence

> We jointly train a recurrent inference core and a compressed state-transport operator to preserve useful finite-budget computation across changes in conditioning, improving decisions while observations and actions continue asynchronously.

This is a candidate claim. It becomes defensible only with a specified mechanism, reproducible implementation, and the comparisons above. The mathematical sensitivity diagnostic and a classical optimization pilot can verify components and expose failure modes; neither by itself establishes a trained CDM or general autonomous intelligence.
