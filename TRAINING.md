# Data and training

The [supervised pilot](experiments/neural_pilot/README.md) trains the implemented [CDM core](cdm/model.py) on sequential contextual recall. It compares workspace reset, workspace carry, and memory reading using action cross-entropy after each refinement. Activity and value heads are not trained by this objective. The sequence below extends that starting point to environment interaction, learned activity, and continued adaptation. Current commands are in [RUNNING.md](RUNNING.md).

## Interaction data

The training unit is a time-ordered interaction containing received observations, computation operations, applied actions, and outcomes. Include useful work between observations and observations arriving before a calculation finishes.

| Field group | Stored information |
| --- | --- |
| Identity | Run, trajectory, task family, simulator version, seed |
| Timing | Observation capture and arrival; operation start and finish; action application |
| Input | Modality, source, units, payload reference, goal update |
| Activity | Observation request, refinement, memory read, tool call, or wait |
| Action | Proposed and applied action, policy probability, pending request identifier |
| Outcome | Measurement, reward, elapsed time, resource use, related action |
| Computation record | Generating context, observation scope, goal, parameter version, depth, retained payload |
| Training state | Policy version, memory capacity, reset locations, adaptation steps |

Sequence batches follow arrival order while retaining capture time. Numeric events can use columnar storage; larger media use indexed assets. Received observations and imagined trajectories have distinct data types.

Split collection runs, layouts, dynamics, goal combinations, and random seeds before constructing training windows. Evaluate new factor combinations separately from longer sequences and changed timing. Repeated contexts must occur within trajectories so reuse can be learned; held-out trajectories and task combinations remain separate.

## Environment distribution

The initial grid distribution contains changing doors, goals, and obstacle positions. Each operation consumes a specified interval. Use three regimes: full context return, return of only some conditions, and continued change without exact repetition. Every method receives the current observations at its action head.

Reference planners initialize behavior from the information assigned to them. Record whether a reference uses the agent's observation history, a reconstructed map, or full simulator state. A planner with extra information can be a teacher or performance reference without becoming a student input.

Later instrument tasks add nonlinear dynamics, noisy and delayed measurements, pending requests, and actions that affect future observations. Tool environments add executable operations and delayed results. Existing digital-research logs can support prediction and imitation after import; evaluating a new action policy requires an environment in which its consequences can be measured.

## 1. Useful actions at every computation depth

Train the context encoder, workspace initializer, shared refinement, and action head before introducing computation memory. Use reference actions or action values at depths \(k\in\{0,1,2,4,8\}\), with the depth-zero head reading current context.

An initial supervised objective is

\[
\mathcal L_{\mathrm{init}}=
\sum_{k\in\mathcal K}w_k\,
\mathcal L_{\mathrm{action}}\bigl(\pi_\theta(c_t,Z_t^k),y_t^k\bigr)
+\lambda_p\mathcal L_{\mathrm{prediction}}.
\]

Prediction targets are future received observations, rewards, and outcomes conditioned on action and elapsed time. Labels must reflect the information and time available to the corresponding computation branch. Depth weights prevent the model from training only its final readout.

Measure action quality against refinement depth and elapsed time. This establishes whether additional internal computation helps before learning to allocate it.

## 2. Learned reuse under changing conditions

Add bounded computation memory while holding observation schedules and compute budgets fixed. Train the record encoder, context-factor projections, compatibility features, payload revision, and attention reader through the action and prediction losses.

Generate training pairs that change goals, dynamics, or observation content separately and jointly. For each branch, retain the actual generating conditions of its stored computation. Include exact returns, partial returns, and changes for which restarting is useful. These interventions teach the reader which parts of earlier work transfer.

A fixed retention rule is the initial baseline. Compare learned conditional reading with restart, ordinary state carry, exact snapshot lookup, and unconditioned memory attention at matched capacity and read budget. The action-only correction baseline receives the same new context while leaving the stored computation unchanged.

Train for action quality and later return. Hidden-state matching is an optional auxiliary loss with a specified teacher; matching a shallow state can reproduce that state's unfinished error. A more deeply computed teacher is compared with outcome-only learning rather than assumed to be the correct latent representation.

For the environment-training extension, collapse exact duplicate records before retrieval and compare with the current ring-buffer reader. Evaluate repeated refinement on unchanged observations and repeated retrieval of the same computation. Prediction quality should improve only when computation improves the estimate; adding a copied record must not act as another measurement.

## 3. Value of the next operation

For an interaction history \(\mathcal H_t\), create continuations beginning from the same simulator state, observation history, workspace, memory, and pending operations. One continuation performs candidate operation \(b\); another performs comparison operation \(b_0\). Both environments advance during their operations and then use a specified continuation policy.

\[
\widehat{\Delta Q}(b,b_0\mid\mathcal H_t)
=\widehat G_b-\widehat G_{b_0}.
\]

Returns use a common physical-time horizon or discount convention and include operation costs. Pair exogenous randomness when possible while allowing action-dependent transitions to differ. Reference computation is recorded as training cost.

Use these targets to initialize predicted operation values. Improvement divided by operation duration is an optional scheduling heuristic; optimize duration-aware return for the learned policy. Differences between performance gaps at unrelated times do not supply this target.

For tasks in which approximate planning references are available, report information-limited and computation-limited performance separately. Their magnitudes describe the current shortfall, whereas operation values estimate what a particular next activity can achieve.

## 4. Joint activity and action learning

Train recurrent actor–critic policies, initially PPO, from fresh rollouts. Let an operation last \(\Delta t_i\), use \(\gamma_i=\exp(-\beta\Delta t_i)\), and accumulate reward during the interval. Bootstrap from the state after that duration.

\[
\mathcal L=
\mathcal L_{\mathrm{policy}}
+\lambda_v\mathcal L_{\mathrm{value}}
+\lambda_p\mathcal L_{\mathrm{prediction}}
+\lambda_q\mathcal L_{\mathrm{operation}}
-\lambda_h\mathcal H(\pi).
\]

Choose coefficients on validation trajectories and record their gradient scales. The policy selects observation, retrieval, computation, action, and wait operations. The fast action readout remains available between longer activities.

The distribution needs situations where sensing is useful, where further computation is useful, and where immediate action is best. Include irrelevant or noisy inputs so surprise alone does not determine the activity. Compare fixed budgets, simple benefit-per-time scheduling, and the learned policy with equal training and tuning budgets.

## 5. Continued learning

First evaluate frozen parameters with evolving context, workspace, and memory. Then update prediction and policy heads online; compare adapters and shared-core updates after the smaller updates work.

Replay supports prediction, rehearsal, and distillation. PPO uses fresh policy trajectories; policy learning from retained data requires a specified off-policy method. Background learning consumes recent and earlier experience, optionally augmented with model-generated rollouts.

Evaluate retained task performance, new-task acquisition, and transfer after resetting transient state. Report environment interactions, update count, and total training cost. Actors use identified parameter snapshots. When state-producing parameters change, reconstruct state from a recent event prefix or evaluate a learned conversion, including its cost.

## Sequence optimization and GPU execution

Use truncated backpropagation through time with numerical state carried between windows. Sweep unroll length and test longer dependencies. Reconstruct training state with a preceding burn-in segment when replay or parameter changes require it.

Keep workspace size, record capacity, and retrieval count fixed within a batch. Group sequences by length and operation count. BF16, activation checkpointing, compilation, and CUDA graphs are evaluated against an FP32, uncompiled reference for the relevant model and hardware.

Parallelize independent environments, sequence batches, seeds, and variants. For a model that fits on one GPU, use one process per device and [DistributedDataParallel](https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html), explicitly sharding trajectories. Evaluate [FSDP](https://docs.pytorch.org/docs/stable/distributed.fsdp.fully_shard.html) when model and optimizer storage require sharding.

Begin with five independent seeds on individual devices. Scale each run only when model size or rollout throughput benefits. A larger cluster can run matched baseline sweeps, teacher collection, and held-out evaluation concurrently. Choose the allocation from measured simulator, learner, and communication throughput.

Measure batch-one observation-to-action latency separately from batched throughput. Include encoding, recurrence, memory lookup, record adaptation, refinement, action readout, and transfers. Report median and tail latency, parameter count, persistent-state bytes, peak memory, interactions per second, and GPU-hours. Simulated operation ticks and measured hardware time are separate quantities until a measured timing model connects them.

## Run output

For environment-training runs, record configuration, source commit, dependency and simulator versions, seeds, training curves, checkpoints, per-seed task scores, timing distributions, and evaluation trajectories. The current pilot writes configuration, versions, seeds, curves, and task scores; checkpoint saving is optional. Store large checkpoints and datasets externally with versioned download locations and hashes in GitHub. Use isolated output directories so new runs retain the earlier experiments.
