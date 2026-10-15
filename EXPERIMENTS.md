# Experiments

The central question is whether preserving and adapting intermediate computation improves decisions while observations, goals, and available computation change. Three executable studies lead into the full neural control experiment: numerical state revision, analytical planning reuse, and supervised training of the neural core.

## 1. Dynamic grid: reuse under changing conditions

[experiments/continual_grid/](experiments/continual_grid/README.md) implements the first closed-loop planning study. A fully observed grid contains changing doors, moving obstacles, and changing goals. A bounded block of internal computation advances environment time; each method then receives the current context and emits an action. Computation uses the observation available when it starts. Action selection uses the observation available when it finishes.

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

## 3. Neural control: the primary model comparison

Connect the neural core to the grid, keeping the environment clock independent of observation, refinement, memory access, and action. First imitate a planner restricted to the model's observations. Then train action and activity policies for return under elapsed time.

The activity policy chooses observation, computation, or external action. The action head remains available after every completed computation block. Introduce partial observation and measurement cost before drawing conclusions about proactive sensing.

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

Interpret results at matched budgets. If snapshots and factorized memory perform similarly, retain the simpler mechanism until a task demonstrates a benefit from finer reuse. If ordinary recurrence is sufficient, concentrate on activity selection and longer tasks. If all reuse methods lose to restart, examine change frequency, computation depth, and representation learning before scaling. A gain in one regime identifies where a mechanism helps; generalization requires additional regimes and tasks.

## 4. Autonomous instrument

After neural navigation, add different observations and actions: a partially observed nonlinear process with delayed sensors and controllable inputs. One starting family is a two-state thermal system:

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

Extend operation from episodes to hours before adding modalities. Record retained memory, throughput, latency drift, parameter versions, and task quality. Introduce image, audio, tool, and sensor encoders through the timestamped event interface.

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
