# Transporting unfinished computation in a Continual Decision Model

**Status: research proposal with a conditional local derivation, 14 October 2026.** This document specifies a candidate mechanism, its training signal, and a falsifiable claim. It does not report a trained CDM, establish priority, or prove stability of the full model. Fixed-point differentiation and sensitivity-based real-time optimization are established foundations [1–3].

## Research hypothesis

A persistent model should retain useful computational progress when evidence, executed actions, or goals change. Ordinary recurrence can learn this implicitly. The hypothesis here is more specific: **jointly learning a representation, a refinement operator, and a cheap transport of partially completed computation can improve decisions under interruption at a fixed wall-clock budget.**

The proposed transport predicts how the *current stage of computation* should change when its context changes. It is trained across early and late interruption points, rather than only around converged solutions, and the corrected state persists into subsequent work. The potential contribution is co-learning this representation, refinement, and compressed update under interrupted task outcomes. Neither continuous execution, nonoptimal-iterate transport, nor sensitivity correction alone is a new contribution: Simonetto et al. already transport nonoptimal iterates along an approximately constant-residual manifold [4].

## Model and execution

Let the model retain an event state $h$, a fixed-capacity workspace $z\in\mathbb R^n$, and progress features $p$. A multimodal event encoder and gated matrix recurrence update $h$ from observations, executed actions, elapsed time, and goals. A projection produces the current context $c\in\mathbb R^d$. Context coordinates remain consistent within a parameter version.

The workspace holds decision-relevant intermediate computation. Its meaning is learned through outcome prediction, action supervision, and interrupted rollouts; assigning semantic names to latent slots does not establish their contents. A shared Transformer block supplies a refinement operator $F_\theta(z,c)$. Progress features include elapsed time, recent internal-step counts, residual summaries, and predicted benefit of further refinement. These are available features, not an oracle giving a unique computational age after a history of context changes.

When context changes from $c$ to $c'=c+\Delta c$, transport the workspace before continuing:

$$
\widetilde z=T_\phi(z,c,p,\Delta c)
=z+U_\phi(z,c,p)\,[V_\phi(z,c,p)^\top\Delta c],
\qquad z^+=F_\theta^{m}(\widetilde z,c').
$$

Here $U\in\mathbb R^{n\times r}$ and $V\in\mathbb R^{d\times r}$. The implementation applies the factors without constructing a dense $n\times d$ matrix. Rank $r$ is an experimental variable; low rank is a compression hypothesis, not a property guaranteed by the problem. The additive structure gives $T(z,c,p,0)=z$ exactly. A context change can still require substantial new refinement, especially across discrete hypothesis or goal changes.

A fast policy reads fresh context and the most recently completed workspace, $a\sim\pi_{\rm act}(c',z,p)$, while an activity policy selects observation, retrieval, refinement budget, or learning activity. After transport completes, the fast policy can read $\widetilde z$ without waiting for $m$ refinements. Deep computation and physical action therefore overlap semantically; a cheap action readout need not await a complete plan. Both raw fresh context and workspace remain available, so stale internal work cannot obscure an available observation by construction of the interface alone. Whether the policy uses that observation correctly is learned and measured.

At each short compute boundary, the model incorporates arrived events and reconsiders activity. Without new external input it may continue refinement, generate a measurement request, or work on retained goals. The environment and clock keep advancing during computation. No text generation or new user prompt is required for the next internal step. Bounded device kernels give explicit interruption granularity; a single permanently executing kernel is unnecessary.

## Why sensitivity is a useful starting point

For a fixed context, suppose an idealized refinement has a smooth fixed point:

$$
z^*(c)=F_\theta(z^*(c),c).
$$

When $I-J_zF$ is invertible, implicit differentiation gives

$$
D_cz^*=(I-J_zF)^{-1}J_cF.
$$

For a smooth nondegenerate local minimizer of an objective $E(z,c)$ with invertible $\nabla_{zz}^2E$, the analogous expression is $D_cz^*=-(\nabla_{zz}^2E)^{-1}\nabla_{zc}^2E$. These established identities explain why simply copying the previous state can waste work: the appropriate state itself moves with the context. An exact sensitivity corrects its first-order displacement. The proposed model learns a compressed approximation; it does not compute a dense inverse during deployment.

The more relevant target for CDM is *finite progress*. Starting from a common context-independent seed $z_0$, define $z_{k+1}(c)=F_\theta(z_k(c),c)$. Its exact tangent satisfies

$$
S_0=0,\qquad
S_{k+1}=J_zF_\theta(z_k,c)S_k+J_cF_\theta(z_k,c).
$$

For a context-dependent seed, use $S_0=D_cz_0$ instead. An accurate transport should approximate $S_k\Delta c$ at the interruption depth $k$. Far from convergence, $S_k$ need not resemble fixed-point sensitivity. This standard forward-differentiation recurrence supplies a diagnostic and training target, not a new derivative or a novelty distinction by itself. For a smooth trajectory, Taylor expansion gives $z_k(c+\Delta c)=z_k(c)+S_k\Delta c+O(\|\Delta c\|^2)$; its constants may grow with depth. Whether the compressed learned state retains enough information to infer this tangent is part of the hypothesis.

The first branch diagnostic uses a globally fixed seed. With variable seeds, $(z_k,c,k)$ need not determine $S_k$: for $F(z,c)=cz$, $k=1$, and $c=0$, every seed gives $z_1=0$ but $S_1=z_0$. Full rollouts therefore retain a bounded history/anchor summary among the progress features and measure approximation error; compression can leave irreducible ambiguity. No exact recovery from arbitrary compressed histories is assumed.

A concrete alternative is a persistent low-rank response state $K$, updated alongside $z$ to approximate the tangent recurrence, with $K\Delta c$ providing transport. This carries sensitivity history explicitly instead of regenerating it from $z$ alone. Compressed recurrent derivatives have RTRL/UORO antecedents [7]; here the candidate is co-learning that bounded response state for useful context adaptation. Its additional memory and update cost require matched controls.

For a fixed-seed branch, suppose $\|A(z,c,p_k)-S_k(c)\|\le\eta_k+L_{A,k}e_k$, where $e_k=\|z-z_k(c)\|$, and $\|D_c^2z_k\|\le H_k$. Taylor expansion directly yields

$$
\|T(z,c,p_k,\Delta c)-z_k(c')\|
\le (1+L_{A,k}\delta)e_k+\eta_k\delta+H_k\delta^2/2.
$$

After $j$ further refinements, a local Lipschitz factor $L$ for $F_{c'}$ multiplies this bound by $L^j$; bounded implementation error adds $\xi\sum_{i=0}^{j-1}L^i$. This finite-progress comparison does not require convergence, but it permits error amplification when $L>1$. It is distinct from the equilibrium tracking claim below.

## A conditional local tracking bound

The following elementary bound identifies what an accurate transport could improve. It is not a novelty claim or a guarantee for arbitrary Transformer dynamics.

Assume, in a shared local neighborhood and fixed parameter version:

1. $F_c$ is a contraction in $z$ with uniform factor $0<q<1$ and fixed point $z^*(c)$; all states and context line segments considered remain in this neighborhood.
2. $z^*$ is twice differentiable, with $\|D^2z^*\|\le H$.
3. The approximate sensitivity $A(z,c,p)=UV^\top$ satisfies $\|A(z,c,p)-D_cz^*(c)\|\le\eta+L_A\|z-z^*(c)\|$ for the progress states covered by the claim.
4. Each implemented refinement differs from $F_c$ by a vector of norm at most $\xi$.

Write $e=\|z-z^*(c)\|$ and $\delta=\|\Delta c\|$. After transport and $m\ge0$ refinements at $c'$, the error obeys

$$
e^+\le q^m\left[(1+L_A\delta)e+\eta\delta+\frac H2\delta^2\right]
+\xi\frac{1-q^m}{1-q}.
$$

**Proof.** Taylor expansion gives $z^*(c')=z^*(c)+D_cz^*(c)\Delta c+R$, with $\|R\|\le H\delta^2/2$. Subtract this from $z+A\Delta c$, apply assumption 3 and the triangle inequality, then apply the contraction bound $m$ times. The per-step implementation errors form the geometric sum shown above.

For ordinary warm-starting with $\widetilde z=z$, the corresponding upper bound is $q^m(e+L_*\delta)+\xi(1-q^m)/(1-q)$ when $\|D_cz^*\|\le L_*$. Thus exact transport at equilibrium removes the first-order *event-induced* error. With learned error $\eta>0$, the term $\eta\delta$ remains first order; with incomplete prior computation, the $e$ terms can dominate. A smaller upper bound is not proof of pointwise superiority.

For variable budgets $m_t$, a uniform bound follows conditionally if $q^{m_t}(1+L_A\delta_t)\le\rho<1$ and the remaining additive terms are bounded by $b$: $e_t\le\rho^te_0+b(1-\rho^t)/(1-\rho)$. This condition can fail during extended periods without refinement. The statement concerns local solver-state tracking, not physical-system stability, general reasoning, or guaranteed long-horizon task success. A Lipschitz action readout converts state error into a local action-deviation bound, but still does not establish control stability.

## Training the representation and transport

Begin with paired branches from a globally fixed seed. Sample a context, interruption depth $k$, and a context change representing an observation, executed action, or goal update. Compare the transported partial state with a reference that recomputes the same $k$ refinement steps from that seed under the changed context. This reference spends training-only computation and uses the newly available context, not future outcomes as student inputs.

Train across several continuation depths $j$ with

$$
\mathcal L_{\rm branch}
=\mathbb E_{k,j}\,d\!\left(
F_{c'}^j(T_\phi(z_k(c),c,p_k,\Delta c)),
\operatorname{sg}[F_{c'}^{k+j}(z_0)]\right).
$$

The distance combines normalized latent discrepancy and disagreement of grounded prediction/action readouts. Stop-gradient reference branches use the same parameter snapshot; parameter updates occur after the branches are evaluated. Local Jacobian-vector targets provide an optional direct sensitivity diagnostic. Include unchanged, irrelevant, small continuous, and substantial context changes, with both early and late interruptions. Do not require exact global commutation of transport and refinement: changed contexts can change the refinement dynamics themselves.

The fixed-context paired branch is only an initialization and diagnostic. Full training unrolls repeated interruptions, retained transport states, changing goals, and actions that alter later observations. At each branch point, clone the same causal history; an expensive teacher may refine more using only the information available there. Evaluate the transported branch through subsequent physical outcomes, rather than pretending every evolving history has an equivalent fixed-context depth $k$. This is where representation, refinement, transport, and activity selection must be co-trained.

Consistency alone admits useless representations. Joint training therefore includes measured-outcome prediction, task-return optimization on physically timed rollouts, and supervision of intermediate actions. Context/state normalization makes sensitivity diagnostics interpretable within fixed coordinates; it does not make them representation invariant. The practical objective is $\mathcal L_{\rm task}+\lambda_p\mathcal L_{\rm prediction}+\lambda_b\mathcal L_{\rm branch}$, with separate reporting of every term. Branch imitation must not outweigh actual task improvement.

Train the activity policy on later return after accounting for elapsed time, observations, and computation. Persistent prediction uncertainty should sometimes be resolved through new evidence and sometimes through deeper calculation; the environment must contain both cases. Consolidation then distills effective deeper decisions into fast policies and updates reusable predictive knowledge. Replay and distillation are established methods; their benefit here must appear in subsequent tasks, not merely reproduction of earlier latent states.

## Experimental decisions and GPU constraints

The decisive controls are a generic recurrent model with identical inputs and matched resources, workspace reset, ordinary warm-start, a parameter-matched unconstrained residual update, analytic sensitivity where available, and the learned transport. Give every baseline the same fresh-evidence action head; compare action-only correction inspired by A2C2 [6] against persistent workspace correction. Include direct-context-only readout and workspace shuffling to detect policies that bypass persistent computation. Match wall-clock budgets as well as parameters: transport has a cost. Measure downstream return, event-to-applied-action latency, continuation quality versus interruption depth, and retained benefit on unfamiliar changes. Remove progress conditioning and paired-branch training separately. A transport that only helps near a known optimum does not validate the finite-progress claim.

Keep workspace capacity, event blocks, and transport rank bounded. Matrix recurrence supports efficient event processing; short Transformer refinement blocks and batched low-rank products suit GPUs. Generating $U,V$ also costs compute and memory and belongs in profiling. Multiple environments provide training parallelism; recurrent refinement remains causally sequential. Real-time blocks are chosen for latency, not copied from long training chunks. Separate state-producing parameter updates from actor execution and reconstruct or migrate state when those parameters change.

This mechanism addresses one central difficulty within the broader CDM: maintaining useful thought while the world moves. Persistent goals, multimodal observation, active measurement, tool use, fast action, background learning, and experience consolidation remain parts of the complete model. Their integration becomes scientifically substantive if transported computation measurably improves those activities across tasks. A local sensitivity result is a starting explanation, not a substitute for that evidence.

The accompanying numerical study concerns inverse-polynomial transport controls and finite-progress diagnostics. It does not implement or train the proposed joint neural $F/U/V$/action core. Its results must be described separately from this architecture proposal.

## Foundations and priority boundary

1. Bai, Kolter, and Koltun. [Deep Equilibrium Models](https://papers.nips.cc/paper/2019/hash/01386bd6d8e091c2ab4c7c7de644d37b-Abstract.html), NeurIPS 2019. Fixed-point representation and implicit differentiation.
2. Diehl, Bock, and Schlöder. [A Real-Time Iteration Scheme for Nonlinear Optimization in Optimal Feedback Control](https://doi.org/10.1137/S0363012902400713), SIAM Journal on Control and Optimization, 2005. Iterative refinement while the controlled process runs; local analysis.
3. Frey, Nurkanović, and Diehl. [Advanced-Step Real-time Iterations with Four Levels](https://arxiv.org/abs/2403.07101), 2024. Preparation/feedback phases and contemporary real-time-iteration error analysis.
4. Simonetto et al. [A Class of Prediction-Correction Methods for Time-Varying Convex Optimization](https://arxiv.org/abs/1509.05196), IEEE Transactions on Signal Processing, 2016. Section II, equations 4–7 explicitly treat nonoptimal iterates and iso-residual transport; higher-order tracking improvements are prior results.
5. Noh et al. [Latent Adaptive Planner for Dynamic Manipulation](https://arxiv.org/abs/2505.03077), CoRL 2025. Persistent latent-plan posterior and variational replanning as observations arrive.
6. Sendai et al. [Leave No Observation Behind: Real-time Correction for VLA Action Chunks](https://arxiv.org/abs/2509.23224), 2025. A2C2 corrects actions using fresh observations, timing, and base-policy features.
7. Tallec and Ollivier. [Unbiased Online Recurrent Optimization](https://arxiv.org/abs/1702.05043), ICLR 2018. An established compressed online recurrent-derivative method; its parameter-gradient purpose differs from the proposed context transport.

These references establish close antecedents, not an exhaustive novelty search. Related recurrent computation, learned optimization, and predictive coding also require comparison before asserting priority. The candidate distinction is the jointly learned, reusable latent computation and its task-trained compressed transport, not the individual mathematical identities or the mere availability of fresh-evidence action correction.
