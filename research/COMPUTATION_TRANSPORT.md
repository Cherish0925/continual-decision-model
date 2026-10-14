# Observation-conditioned revision of ongoing computation

**Proposed neural mechanism, 14 October 2026.**

A model carries a partially developed internal representation while observations and goals change. The proposed operation updates that representation before further inference, so that later decisions can use both current observations and useful prior computation. The complete model is specified in [DESIGN.md](../DESIGN.md).

## Learned revision and refinement

Let \(z\in\mathbb R^n\) be the persistent workspace, \(c\in\mathbb R^d\) the context, and \(p\) a summary of timing and computation history. A shared Transformer supplies refinement \(F_\theta(z,c)\). For context change \(\Delta c\), define

\[
\widetilde z=T_\phi(z,c,p,\Delta c)
=z+U_\phi(z,c,p)\,[V_\phi(z,c,p)^\top\Delta c],
\qquad
z^+=F_\theta^m(\widetilde z,c+\Delta c).
\]

The factors have shapes \(n\times r\) and \(d\times r\). Their product is applied without materializing a dense \(n\times d\) matrix. Rank is an experimental parameter. Factor generation and application both contribute to the total cost.

The updated workspace persists into subsequent inference. A fast action policy reads current context and the latest completed workspace; an activity policy selects further computation, observations, retrieval, or action. New events enter at short computation boundaries.

The research hypothesis is that joint learning of \(z\)'s representation, \(F\), and \(T\) improves task performance per unit elapsed time. The additive update is one candidate. Compare it with a generic recurrent update, an unstructured residual update, and an initializer from current context.

## Finite-step sensitivity

For a fixed context and a context-independent seed \(z_0\), write

\[
z_{k+1}(c)=F_\theta(z_k(c),c),\qquad
S_k=\frac{\partial z_k}{\partial c}.
\]

Forward differentiation gives

\[
S_0=0,\qquad
S_{k+1}=J_zF_\theta(z_k,c)S_k+J_cF_\theta(z_k,c).
\]

For a context-dependent seed, \(S_0=D_cz_0\). A local change in context gives

\[
z_k(c+\Delta c)=z_k(c)+S_k\Delta c+O(\|\Delta c\|^2).
\]

This identifies a concrete supervised target for approximating how an unfinished computation responds to new context. It also explains why a sensitivity at convergence may be inappropriate early in inference.

The current workspace alone need not identify its sensitivity. For \(F(z,c)=cz\), after one step at \(c=0\), every seed gives \(z_1=0\), while \(S_1=z_0\). The first reconstruction experiment therefore fixes the seed. For general histories, retain a bounded history summary or compare a persistent low-rank response state updated alongside \(z\). The latter approximates the derivative recurrence directly.

These derivatives use the chain rule underlying RTRL and compressed recurrent-gradient methods. The proposed application learns a representation and compressed response for interrupted decision tasks.

## Local approximation error

Let \(A=UV^\top\), \(e_k=\|z-z_k(c)\|\), and \(\delta=\|\Delta c\|\). Suppose, within a neighborhood containing the compared states,

\[
\|A(z,c,p_k)-S_k(c)\|\le\eta_k+L_{A,k}e_k,
\qquad \|D_c^2z_k\|\le H_k.
\]

Taylor expansion and the triangle inequality give

\[
\|T(z,c,p_k,\Delta c)-z_k(c+\Delta c)\|
\le(1+L_{A,k}\delta)e_k+\eta_k\delta+\frac{H_k}{2}\delta^2.
\]

If subsequent refinement is locally \(L\)-Lipschitz, \(j\) further steps multiply this bound by \(L^j\). Per-step implementation error bounded by \(\xi\) adds \(\xi\sum_{i=0}^{j-1}L^i\). Thus refinement can reduce or amplify state error depending on its local dynamics.

At a smooth fixed point \(z^*(c)=F_\theta(z^*(c),c)\), invertibility of \(I-J_zF\) gives

\[
D_cz^*=(I-J_zF)^{-1}J_cF.
\]

For a smooth nondegenerate local minimizer of \(E(z,c)\) with invertible Hessian,

\[
D_cz^*=-(\nabla_{zz}^2E)^{-1}\nabla_{zc}^2E.
\]

These established sensitivity formulas motivate the approximation; neural inference need not converge to a fixed point.

For the special case where \(F_c\) is a contraction with factor \(q<1\), assume \(\|D_c^2z^*\|\le H\) and
\(\|A-D_cz^*\|\le\eta+L_A\|z-z^*(c)\|\).
With \(e=\|z-z^*(c)\|\), the transported state after \(m\) refinements satisfies

\[
e^+\le q^m\left[(1+L_A\delta)e+\eta\delta+\frac H2\delta^2\right]
+\xi\frac{1-q^m}{1-q}.
\]

To obtain the bound, expand \(z^*(c+\Delta c)\), subtract the transported state, and apply contraction to each refinement step. Implementation errors sum geometrically. Exact sensitivity removes the first-order context-displacement term at equilibrium; approximation error and unfinished computation remain.

If the coefficient \(q^{m_t}(1+L_A\delta_t)\) stays below \(\rho<1\) and the additive terms are bounded by \(b\), iteration gives \(e_t\le\rho^te_0+b(1-\rho^t)/(1-\rho)\). These are local state-error calculations. Closed-loop performance is measured in the environment.

## Training for task outcomes

Train the representation, refinement, revision, and policies through repeated interruptions and subsequent actions. At a sampled branch point, clone the same causal history and available observations. Compare the revised state with a recomputed or more deeply refined branch.

An auxiliary fixed-seed branch loss is

\[
\mathcal L_{\rm branch}
=\mathbb E_{k,j}\,
d\!\left(
F_{c'}^j(T_\phi(z_k(c),c,p_k,\Delta c)),
\operatorname{sg}[F_{c'}^{k+j}(z_0)]
\right),\qquad c'=c+\Delta c.
\]

The distance can combine latent discrepancy and grounded prediction/action disagreement. Both branches use the same parameter snapshot, with the reference stopped in backpropagation. Future outcomes score the resulting decisions; they are not added to earlier student inputs.

A same-depth reference may remain far from a good solution. The numerical study's exact reference has mean objective gap 0.727103 on unseen depths. Perfect reconstruction preserves that gap. Use branch matching as auxiliary supervision and compare it with task-outcome training, rather than making latent agreement the primary objective.

Joint training uses task return, outcome prediction, and optional branch loss. Include local and large context changes, irrelevant observations, early and late interruptions, and long sequences of repeated updates. Allow the model to reorganize or advance computation when doing so improves decisions.

## Experiments and computational cost

The principal comparisons are ordinary recurrence, workspace reset, carry-and-refine, a learned initializer, unstructured residual revision, action-only correction, and low-rank revision. Give every applicable action head current observations. Measure later decisions as well as immediate response.

Shuffle or remove the workspace to test its use. Vary interruption depth, rank, history summary, and revision training. Compare exact Jacobian-vector products where feasible, counting their computation separately.

Maintain fixed workspace capacity and short refinement blocks. Measure factor generation, revision, encoding, retrieval, action readout, memory traffic, and scheduling. Batched environments supply parallelism during training; refinement within one trajectory remains sequential.

## Related methods and current implementation

Prediction-correction already transports nonoptimal iterates; real-time iteration refines control solutions during execution; latent replanning preserves plan distributions; action-correction methods use fresh observations; recurrent differentiation propagates sensitivities. [TRANSPORT_PRIOR_ART.md](TRANSPORT_PRIOR_ART.md) records the relevant methods and comparisons.

Fixed-point differentiation is also developed in [Deep Equilibrium Models](https://papers.nips.cc/paper/2019/hash/01386bd6d8e091c2ab4c7c7de644d37b-Abstract.html). [Advanced-Step Real-time Iterations](https://arxiv.org/abs/2403.07101) provides a further comparison for preparation and feedback computations in optimal control.

The existing [numerical code](../experiments/transport/run_experiment.py) fits polynomial corrections on explicit quadratic problems. Its results concern that implementation. The jointly trained neural workspace and policies described here are the next implementation.
