# Experiments

The programme tests whether persistent state, continuing inference, proactive activity selection, and learning improve decisions under physical time constraints. The first complete model uses a nonlinear instrument; later studies add a second domain and richer modalities.

## 1. Nonlinear instrument

Implement a partially observed process with controllable input, unknown parameters, sensor bias, delayed measurements, and disturbances. One starting family is a two-state thermal system:

\[
C_1\dot T_1=\eta u-k_{12}(T_1-T_2)-k_1(T_1-T_{\rm amb})+d_1(t),
\]
\[
C_2\dot T_2=k_{12}(T_1-T_2)-k_2(T_2-T_{\rm amb})+d_2(t).
\]

Add a nonlinear actuator response or temperature-dependent heat transfer, with model parameters sampled across runs. Use a reproducible numerical integrator and record its step size. The learning task observes delayed sensor readings rather than the full simulator state.

Actions include changing power, requesting a cheap noisy measurement, requesting a more accurate delayed measurement, calibrating a sensor, or waiting. The model may refine its internal state while a measurement is pending. The simulator advances during every model operation.

Use three related tasks:

| Task | Objective |
| --- | --- |
| Identification | Estimate a response property within measurement and elapsed-time budgets |
| Adaptive control | Track target conditions while hidden dynamics or disturbances change |
| Experimental search | Choose interventions to locate a desired response efficiently |

Vary dynamics, sensor noise, delay, cost, disturbances, and goals independently. Include local changes that leave much of a plan useful and changes that require extensive recomputation. Hold out whole parameter regimes and combinations.

Calibrate simulated compute durations from measured model operations. Then evaluate selected policies with actual wall-clock inference and action application. Report these two timing modes separately.

## 2. Architecture comparisons

Use shared observation encoders and output spaces where compatible, and report parameter counts and total training resources.

| Model | Comparison |
| --- | --- |
| Generic recurrent policy with extra internal steps | Whether the training distribution already teaches useful continuing inference |
| Attention–recurrence hybrid | Whether the context backbone improves state capacity and cost |
| Restarted workspace | Cost and quality of rebuilding computation after an observation |
| Persistent workspace with ordinary updates | Benefit of continuing from prior state without a dedicated revision operation |
| Learned initializer from current context | Whether prior workspace adds useful information beyond a good starting state |
| Action-only correction | Whether persistent revision improves later decisions beyond a fast corrected output |
| Unconstrained residual revision | Whether the low-rank form helps beyond a matched learned update |
| Low-rank revision candidate | Joint learning of representation, refinement, and observation-conditioned revision |
| Explicit world-model planner | Whether imagined trajectories and learned planning control outperform implicit refinement |

For action-only correction, use A2C2's principle of current-observation correction with older policy features. Attribute the original method and describe any reimplementation. Include warm-start MPC or real-time iteration when an explicit model is available; specify whether that model is learned or supplied.

Give every applicable action policy the current observations. Match total actor latency in addition to parameters, and plot performance across budgets. Include a large one-pass model to compare extra capacity with repeated shared computation.

## 3. What to vary

**Persistent computation.** Reset, shuffle, or bypass the workspace while retaining the same observation context. Compare actions immediately after an observation and after subsequent refinement. This measures whether the workspace carries reusable computation.

**Observation and inference.** Compare fixed, tuned heuristic, and learned schedules. Some tasks require a new measurement; others can be resolved from existing information. Measure task return, measurement use, and time, including activity-selection overhead.

**Revision and training.** Vary rank, interruption depth, event rate, goal changes, and shared-refinement depth. Compare task-outcome training, auxiliary branch matching, and their combination. Include stationary periods to measure overhead.

**Continual learning.** Compare frozen parameters, memory retrieval, parameter updates, and background replay. Measure adaptation to successive changes, performance on earlier tasks, and transfer after transient state resets.

For fixed-stream comparisons, use identical observations and timing. For active observation, use the same initial information, available sensors, and budgets; policies may then collect different observations as a consequence of their choices.

## 4. Measurements

| Aspect | Measurements |
| --- | --- |
| Task quality | Return, success, tracking error, identification error, measurement efficiency |
| Response | Observation-arrival-to-applied-action latency, capture-to-action age, p50/p95/p99, deadline misses |
| Computation | Actor device time, total training GPU-hours, FLOPs when available, transfers, parameter count |
| State | Retained bytes, memory growth, effect of resets and workspace substitutions |
| Learning | Acquisition speed after each change, forgetting, forward transfer, later-task performance |
| Activity selection | Return from requested observations and extra computation; prediction calibration |
| Operation | Long-run throughput, latency drift, state growth, task quality over time |
| Energy | Measured device energy where instrumented, reported separately from compute proxies |

Use independent training seeds and paired environment conditions. Start with five training seeds for the primary model comparison. Report variation across both trained models and evaluation environments; successive events from one trajectory are correlated.

Report immediate response and later action quality together. A method can react quickly while discarding useful planning, or reason accurately while reacting too late.

## 5. Transfer and scale

After the instrument study, choose a second domain whose observations and actions differ. Candidates are partially observed navigation, manipulation, or an executable digital investigation environment. [The benchmark review](ref/WORLD_MODELS.md) maps existing tasks to these requirements.

For digital research, reconstruct executable tasks from versioned tools and environments before using archived trajectories to evaluate new actions. Track interruption, resumption, tool cost, and whether retained computation helps across changing results.

Increase model size, training interactions, sequence length, modality complexity, and run duration separately. Use logarithmically spaced operation horizons extending from episodes to multi-hour or longer runs. Report what carries across each task boundary.

At cluster scale, retain comparisons at equal interactions and compute as well as absolute best performance. The main research result is the task-quality versus response-time relationship, together with transfer and learning behavior.

## 6. Completed numerical study

[experiments/transport/RESULTS.md](experiments/transport/RESULTS.md) records the existing CPU experiment on changing quadratic objectives. The tested polynomial correction underperformed strong solvers. A separate finite-step reconstruction task demonstrated the consequences of matching an incomplete reference solution.

That study supplies a reproducible calculation and a training-target correction. The experiments above implement the proposed neural model, action-dependent observations, and real-time execution.
