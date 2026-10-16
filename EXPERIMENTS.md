# Experiments

The main comparison now concerns a continuously operating model that learns observation, inference, action, and learning choices together. Persistent-computation studies provide components and comparisons for this objective.

## Integrated model campaign

This is the next main experiment. The studies below remain completed results and comparisons; further maze variants are not prerequisites for this campaign. The proposed campaign has not yet been run.

Train one recurrent/Transformer core with action-conditioned prediction and the complete activity interface in [DESIGN.md](DESIGN.md). Keep the same shared core and activity parameterization across task families, using modality/task-specific input and output modules. Train both a mixed-family model and matched separate-family controls so multi-domain training is not confused with transfer.

| Interaction family | What the agent experiences | What it must acquire or choose |
| --- | --- | --- |
| Changing navigation | Partial maps, changing goals, delayed surveys, persistent local transition rules | Useful sensing, continued inference, adaptation to unfamiliar rules, reuse on returning conditions |
| Nonlinear instrument | Delayed and noisy signals, unknown response parameters, drift, actuator dynamics | Action-conditioned prediction, measurement selection, identification, tracking, experimental search |
| Delayed digital tools | Queryable records, executable operations, changing task requirements, pending results | Useful queries, tool arguments, subgoals, work during waiting, retention of learned task relations |

A digital environment must return results of executed operations and score observable task outcomes. A scripted text transcript alone does not support evaluating new actions. Instruments are one task family, not a replacement for the model research objective. An established control or embodied benchmark should also be ported into the same timing interface, with its modified interaction semantics reported.

For each family, vary unknown but learnable dynamics, familiar dynamics, and irreducible stochastic disturbances. Include intervals in which no update is useful. Withhold complete parameter combinations, operation compositions, goal types, and event timings. Include both known-family transfer and a held-out task family under a declared adaptation budget. Randomize scene changes independently of training-window boundaries; neither a new request nor a returning condition implicitly resets the model.

### Primary comparisons

| Comparison | Purpose |
| --- | --- |
| Strong recurrent policy with persistent state | Adaptation obtainable through ordinary recurrent activations |
| Learned world model with a fixed observation/learning schedule | Benefit of prediction and online learning without learned activity selection |
| Frozen model with learned activity selection | Benefit of adaptive computation and observation without deployment parameter updates |
| Same CDM with fixed learning schedule | Contribution of choosing when and what to learn, with matched available update resources |
| Full CDM | Jointly trained continuing state, activity, action, prediction, and local adaptation |

Use implementations appropriate to each domain; an established method's native benchmark results are not directly comparable to a modified timed task. Match observation access, training interactions, model capacity or measured compute, and tuning budgets. Include ordinary retrieval-based improvement as a memory comparison. A larger single-pass model tests whether extra capacity is sufficient.

The principal removal experiments restart persistent computation, disable adapter updates, or replace learned learning choices with a fixed schedule. Distinguish removal at evaluation on the same checkpoint from retraining the ablated model with the matched recipe; the former also changes the policy's operating distribution. Report these separately from independently trained external baselines. Avoid a full Cartesian sweep of every architectural option; each comparison should answer a specific question about the primary hypothesis.

### Continuing time and continuing experience

Train with an event-time simulator that advances through measured or explicitly specified operation durations. Confirm behavior in an execution mode whose environment clock advances independently while the model computes or learns. Fast control continues during expensive operations. Measure capture-to-action and incorporation-to-action latency separately, including p50/p95/p99, the age of the observation used, and control error during learning.

A lifetime retains context, workspace, records, local adapters, and optimizer state across requests and scenario changes. Evaluation agents adapt independently from their own experience. A separate collective-adaptation setting may pool experiences, but is not reported as single-instance improvement.

Evaluate all primary methods on one-hour uninterrupted runs, then the full model and strongest alternatives on eight-hour runs. A 24-hour run compares the same model and its strongest alternative under repeated and novel conditions. These are proposed durations for actual elapsed execution; accelerated simulated lifetimes and parallel agent-ticks are also reported, with their own units.

### Acquisition, initiative, and consolidation

Measure task performance before a new regularity is encountered, after interaction, when an earlier condition returns, and on held-out related tasks. Report completed tasks or control error alongside net reward, compute, observations, and learning cost. Curves must separate savings from better task execution.

Give the agent a continuing objective and introduce intervals without a new user request. It can select useful measurements, subgoals, or experiments during those intervals. Evaluate the resulting later task performance and compare with reactive-only operation. Count useful outcomes, not self-generated text or the number of operations.

For consolidation, compare equal-total-budget continuations that spend an idle interval on replay, deeper inference, additional observation, or waiting. Later tasks must test the acquired skill without replaying the exact training label. Parameter-only, state-only, and retrieval-only reset comparisons identify what was retained. Measure energy during quiet operation rather than inferring sleep-like behavior from zero refinement count.

### Scale and deliverables

[TRAINING.md](TRAINING.md#sequence-optimization-and-gpu-execution) specifies proposed model sizes, interaction budgets, parallelism, and lifetime lengths. The campaign produces one integrated implementation, reusable configurations, five-seed primary comparisons, adaptation/retention curves, measured quality–time–resource tradeoffs, and released checkpoints with their data-generation procedures.

The central question for this campaign is whether the same model improves future task performance through its chosen activities while maintaining timely action. Its contribution must survive comparison with fixed-schedule learning and strong recurrent/model-based alternatives. Report model size, interactions, and elapsed duration alongside the measured capabilities.

## 1. Dynamic grid: reuse under changing conditions

[experiments/continual_grid/](experiments/continual_grid/README.md) implements the first interactive planning study. A fully observed grid contains changing doors, moving obstacles, and changing goals. A bounded block of internal computation advances environment time; each method then receives the current context and emits an action. Computation uses the observation available when it starts. Action selection uses the observation available when it finishes.

| Regime | Change | What it tests |
| --- | --- | --- |
| Full return | A previous map and goal return | Reusing a whole computational state |
| Partial return | Some earlier conditions return while others differ | Reusing parts of earlier computation |
| Continuous change | Conditions keep changing | Updating computation without exact recurrence |

The agent's position and observations are always current. Retrieving an old plan never restores an old physical state.

Compare restart on context change, continued computation, exact-context snapshots, nearest-context retrieval, and factorized computational memory. All use the same Bellman operator, initialization family, action space, fresh-context action interface, and computation budgets. The factorized method assembles a warm start from records with matching goal and local map conditions. It is an explicit analytical construction; the neural model must learn its representation and reading rule in a subsequent comparison.

Measure return, goals reached, blocked moves, action regret against a current-map reference planner, cell backups, retained memory, and end-to-end CPU time. Count memory reading and writing. A complete reference solve is used only for evaluation, outside the candidate's computation. Simulated computation durations and measured wall-clock durations are reported separately.

Use paired exogenous traces for five seeds and more than one computation budget. Actions create different positions; pairing matches the external change sequence, not the resulting observations. Archive individual run rows, configuration, and summaries.

## 2. Neural core: sequential learning

[experiments/neural_pilot/](experiments/neural_pilot/README.md) trains the recurrent context, shared Transformer workspace, memory readout, and action head on a small sequential task. Random context-specific cues appear first and are queried later under changing goals. Held-out episodes resample cue assignments and query order.

Compare the same model with a restarted workspace, a persistent workspace, and persistent workspace plus computational memory. Every variant retains its recurrent observation context and sees the same input sequences. Supervise action readouts at each refinement depth. Report held-out queries separately from visible cues, together with training steps, parameters, and runtime per seed.

This study measures trainability and sequential task performance. Activity selection, action-dependent observations, and environment change during neural inference enter the next study. Action cross-entropy does not train the activity or value heads.

## 3. Neural planning studies and component comparisons

Connect the neural core to the grid, keeping the environment clock independent of observation, refinement, memory access, and action. First imitate a planner restricted to the model's observations. Then train action and activity policies for return under elapsed time.

The first neural instance is [experiments/dynamic_maze](experiments/dynamic_maze/README.md). Convolutional latent planners act every tick in fully observed room mazes whose doors, obstacles, and goals change while they refine, with five seeds per variant. It implements restart, carry, restart on goal moves, exact snapshot and nearest-context retrieval, learned per-cell and whole-map reading of stored workspaces, a planner whose workspace is factorized by premise with two ablations, and carry trained only on static mazes. It also compares rules for sharing refinements and surveys among many agents, planners trained on partial views with and without an age input, and online learning on larger mazes. The [continual agent](experiments/continual_agent/README.md) adds learned activity selection: a PPO-trained head chooses refinements, surveys and moves each tick under a partial view, and unused compute pays for online learning. The recurrent action policy without a workspace, action-only correction, successor features with policy improvement, and the larger one-pass model remain to be implemented. Measurements: [maze](experiments/dynamic_maze/RESULTS.md), [agent](experiments/continual_agent/RESULTS.md).

The existing maze activity policy chooses refinement count, a survey request, and move/stay. The integrated campaign additionally trains learning, retrieval, and subgoal choices. The action head remains available after every completed computation block. Introduce partial observation and measurement cost before drawing conclusions about proactive sensing.

| Comparison | Purpose |
| --- | --- |
| Recurrent action policy | What ordinary recurrent state can learn |
| Restarted workspace | Effect of discarding previous computation |
| Persistent workspace | Value of continuing computation |
| Snapshot retrieval with fresh context | Value of restoring a complete previous workspace |
| Nearest-context retrieval | Value of ordinary similarity-based reuse |
| Action-only correction | Correcting output without revising retained computation |
| Successor features with generalized policy improvement | Reward-conditioned reuse of policy-dependent predictions |
| Condition-factorized computational memory | Selective, context-conditioned reuse and revision |
| Larger one-pass model | Additional capacity versus repeated shared computation |

The analytical grid implements the planning-reuse subset. Action-only neural correction and successor features are specified comparisons, not existing implementations. Attribute each reimplementation and its differences from the original method.

Match training interactions and tuning budgets. Report parameter counts, actual inference time, and performance across computation budgets. Better performance at equal refinement count may still cost more after memory access.

### Learning the choice of computation

Fork two rollouts from identical environment state, model state, and future exogenous randomness. One performs an additional computation; the other follows a specified alternative. Both advance the environment by their operation durations and receive only observations available at their respective times. Compare subsequent return over the same elapsed-time horizon.

Use these differences to train activity selection. An action change alone is not an improvement. A larger distance from a reference policy does not imply that additional computation can recover that distance cheaply. The deployed policy selects from predicted achievable improvements and their costs.

### Changes and ablations

- Change goals, dynamics, observation availability, and interruption depth separately and jointly.
- Reset or shuffle the workspace while retaining current recurrent context.
- Compare whole-record retrieval, factorized retrieval, revision, and no memory.
- Compare fixed schedules, tuned heuristics, and learned activity selection.
- Compare task-return training with intermediate-state reconstruction as an auxiliary loss.
- Repeat computation on the same observation stream; measure accuracy and probability calibration without treating repeated reads as new observations.
- After parameter updates, compare unchanged, migrated, and rebuilt computational memory.

Interpret results at matched budgets. If snapshots and factorized memory perform similarly, retain the simpler method until a task demonstrates a benefit from finer reuse. If ordinary recurrence is sufficient, concentrate on activity selection and longer tasks. If all reuse methods lose to restart, examine change frequency, computation depth, and representation learning before scaling. A gain in one regime identifies where a method helps; generalization requires additional regimes and tasks.

## 4. Autonomous instrument

As part of the integrated campaign, use different observations and actions: a partially observed nonlinear process with delayed sensors and controllable inputs. One starting family is a two-state thermal system:

\[
C_1\dot T_1=\eta u-k_{12}(T_1-T_2)-k_1(T_1-T_{\rm amb})+d_1(t),
\]
\[
C_2\dot T_2=k_{12}(T_1-T_2)-k_2(T_2-T_{\rm amb})+d_2(t).
\]

Add nonlinear actuator response or temperature-dependent heat transfer. Sample dynamics, sensor bias, delays, and disturbances. Record the integrator and step size. Inputs are delayed sensor readings, not hidden simulator state.

Actions change power, request measurements of differing cost and delay, calibrate a sensor, or wait. Refine predictions while measurements are pending. Tasks include identification, tracking, and experimental search. Hold out complete parameter combinations. Include warm-start model-predictive control and real-time optimization, distinguishing supplied from learned dynamics.

## 5. Long-term learning and scale

Compare frozen parameters, retrieval, online updates, and background replay across task sequences. Measure adaptation, performance on earlier tasks, transfer after transient-state reset, and learning compute. Separate state updates from parameter updates.

Develop longer operation and input/output adapters against the same timestamped event interface. Record retained memory, throughput, latency drift, parameter versions, and task quality. Include structured sensors, visual input, instructions, and tool results in the main interface; video/audio and richer communication extend that interface without changing its persistent-state or activity semantics.

| Aspect | Measurements |
| --- | --- |
| Task quality | Return, success, time to goal, tracking or prediction error |
| Response | Observation-to-applied-action latency, observation age, p50/p95/p99 |
| Computation | Device time, training GPU-hours, transfers, memory access, model size |
| Learning | Adaptation, forgetting, transfer, performance after state resets |
| Activity | Benefit of requested observations and additional computation |
| Resources | Persistent-state bytes, retained records, measured energy where available |

Use five independent training seeds for the primary comparison and separate held-out environment seeds. Consecutive events are not independent experimental repetitions. Report variation across models and environments. Scale model size, interactions, sequence length, and modality complexity separately.

## 6. Earlier numerical study

[experiments/transport/RESULTS.md](experiments/transport/RESULTS.md) retains the changing-quadratic experiment. Its polynomial correction underperformed strong optimization baselines. Matching a finite-depth reference improved reconstruction while worsening task quality relative to some alternatives. These results motivate task-based learning of ongoing computation and remain unchanged alongside the new studies.
