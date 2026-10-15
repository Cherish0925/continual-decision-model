# Reusing computation while the world changes

## Research question

A calculation consumes time. During that time, an observation may arrive, a goal may change, or an action may alter the environment. Restarting discards useful work; carrying the whole previous state can retain conclusions tied to an old situation. The CDM hypothesis is that learned representations of generating conditions can make partial computation reusable across these changes.

The proposed mechanism stores intermediate computation together with the context, goal, environment description, parameter version, and depth that produced it. A learned reader combines current input with those records, revises the selected computation, and continues shared refinement. The action head always reads current context.

The first neural implementation is in [cdm/model.py](../cdm/model.py). Its reader factors the *score* into goal, environment, context, and depth terms. It uses one distribution over whole-workspace records and then a gate per slot. Independent record selection for different slots is a later comparison. This distinction determines what the current model can learn and what the experiments should claim.

## What is retained

Let \(\mathcal H_t\) denote the received interaction history, including applied actions and elapsed time. The model maintains context \(c_t\), workspace \(Z_t\), and bounded computation memory \(M_t\). Each computation step is

\[
Z^{k+1}=F_\theta(Z^k,c_t).
\]

A retained state \(U_j\) records an intermediate stage of this process. It is useful if its content reduces the work needed for a good future action. Its usefulness may depend on a goal, environment condition, observation history, model version, or the amount of refinement already performed.

Three kinds of persistence should be measured separately:

| Persistence | Stored object | Main comparison |
| --- | --- | --- |
| Ongoing state | Current context and workspace | Restart versus carry |
| Computation reuse | Earlier intermediate states and their conditions | Whole snapshots versus conditional reading and revision |
| Learned ability | Parameters changed by experience | Frozen parameters versus specified adaptation |

Retaining more state can help even without a new reuse mechanism. Memory capacity, storage bandwidth, and lookup cost therefore accompany return and latency comparisons.

## Conditional reading

For record \(j\), the current prototype computes

\[
s_j=s_g(g_t,g_j)+s_d(d_t,d_j)+s_c(c_t,c_j)+s_k(k_j).
\]

Softmax weights over compatible records form a retrieved workspace. A gate per slot selects a mixture of direct reuse, learned revision, and fresh initialization. Further refinement lets slots interact under the current context.

The scientific question concerns the learned relationship between generating conditions and reusable computation. A goal change can preserve dynamics-related information. An environment change can preserve some goal-related calculation. Interventions on these factors, their joint changes, and their recurrence provide the training and test distribution.

The initial reader can learn to combine records, but one shared distribution can also blur different candidate continuations. Compare it with per-slot reading, a small set of separately retained branches, and a larger unstructured workspace at equal memory and computation cost. These experiments identify whether factor-specific selection contributes beyond capacity and attention.

The observation prefix associated with a record says which history was available when it was written. It does not identify which individual inputs caused a particular latent component. Source-specific observation scopes are an extension for asynchronous sensor and tool streams.

## The value of further computation

For candidate operation \(b\), define

\[
Q_{\mathrm{op}}(\mathcal H_t,Z_t,M_t,b)
=\mathbb E\left[
\int_0^{\Delta t_b}e^{-\beta s}u_{t+s}\,ds-C_b
+e^{-\beta\Delta t_b}V(\mathcal H_{t+\Delta t_b},Z',M')
\right].
\]

The operation can be sensing, retrieval, refinement, action, or waiting. Its duration changes what happens before its output is available. Its consequence includes the environment transition, newly received observations, updated computation, and later decisions.

A computation that changes the action is not necessarily useful; it may change it in the wrong direction or too late. Useful computation raises expected task return after its time and resource costs. Conversely, an operation may be useful without changing the immediate action, for example by preparing a later intervention.

To estimate relative operation value, branch from the same interaction state:

\[
\Delta Q(b,b_0)=\mathbb E[G_b-G_{b_0}].
\]

Both branches start with the same received history, current environment, workspace, records, and pending operations. They advance over a common physical-time horizon, and use a specified continuation policy after the alternative operations. Shared exogenous randomness can reduce variance. Action-dependent transitions still follow each branch's own actions.

This comparison supplies learning targets for the activity policy. Dividing predicted improvement by duration gives a simple scheduling baseline. Duration-aware actor–critic learning can account for longer consequences, interacting operations, and future opportunities to compute.

## Information and computation

At the same decision state, one can write

\[
V^*_{\mathrm{full}}-V_{\mathrm{model}}
=
\left(V^*_{\mathrm{full}}-V^*_{\mathrm{received}}\right)
+
\left(V^*_{\mathrm{received}}-V_{\mathrm{model}}\right).
\]

All terms require the same task, time horizon, action options, and cost convention. The first reference has full state information; the second uses only the received information with ideal planning. Finite planners approximate these references with their own stated information and compute budgets.

This decomposition describes two sources of performance loss. It does not say which available observation or computation can remove them. That choice depends on achievable improvement and duration, as represented by \(Q_{\mathrm{op}}\).

Internal refinement can improve an estimate from fixed observations without supplying another independent measurement. Repeated records enter a computation reader, not the sensor stream. In the event-driven extension, duplicate handling and prediction calibration are tested on repeated reading and repeated refinement of unchanged inputs. A metadata field alone does not specify the model's probabilistic update.

## Training the mechanism

The initial supervised pilot trains action readouts after each shared refinement step. Its task is sequential contextual recall: observe action cues associated with contexts, then answer cue-free queries under changed goals. It trains recurrent state, workspace, record scoring, revision, and gating through action prediction. Activity and value heads are present but are not trained by that action loss.

The dynamic-grid study separately compares explicit planning states under changing doors, goals, and obstacles. Moving the neural prototype into that interaction loop supplies the next task-based experiment. Train actions at several refinement depths, then add paired operation-value targets and recurrent policy optimization.

The [quadratic study](../experiments/transport/RESULTS.md) found that its learned correction lost to stronger solvers. Its exact same-depth state reference also retained unfinished optimization error. Reproducing an internal state is therefore an auxiliary training target whose usefulness must be assessed through subsequent decisions. The primary objectives are action quality, return, adaptation speed, and measured runtime.

## Experimental comparisons

| Comparison | What it tests |
| --- | --- |
| Restart after each new context | Value of retaining computation |
| Carry current workspace | Value of conditional reuse beyond ordinary persistence |
| Exact snapshot retrieval with fresh context | Value beyond caching a returned situation |
| Nearest-context retrieval | Value beyond a similarity lookup |
| Current-context action correction only | Value of changing future internal computation |
| Successor features with policy improvement | Reuse of goal-related computations through an established factorization |
| Unstructured memory at matched capacity | Value of explicit condition factors |
| Conditional reading with and without revision | Whether retrieved computation needs learned adaptation |

These are the full comparison plan; the experiment READMEs list the subset already executable. All policies receive current observations at their action readout. A snapshot restores earlier computational content while keeping current position, goal, and observations. A complete context return can favor ordinary caching; partial return and continuous change provide the main tests of adaptive reuse.

For successor features, \(Q^\pi(s,a)=\psi^\pi(s,a)^\top w\) depends on the policy \(\pi\). Changing reward weights revalues that policy; generalized policy improvement is the relevant comparison for obtaining a better current policy. The [literature comparison](../ref/COMPARISON.md) places these methods alongside incremental planning, latent planning, modular recurrence, and learned metareasoning.

Run paired environment seeds, equal tuning budgets, and matching information access. Report task return, goal-reaching time, recovery after changes, refinement work, memory-read cost, and end-to-end latency. Compute ticks in a simulator and hardware milliseconds answer different questions; measure both and connect them through an explicit timing model when evaluating real-time operation.

A successful mechanism should improve this quality–time tradeoff on held-out factor combinations, not only reconstruct familiar contexts. Negative or mixed outcomes determine which representation, reader, or task distribution needs to change.
