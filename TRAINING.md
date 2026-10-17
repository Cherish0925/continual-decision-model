# Data and training

The [supervised pilot](experiments/neural_pilot/README.md) trains the implemented [CDM core](cdm/model.py) on sequential contextual recall. It compares workspace reset, workspace carry, and memory reading using action cross-entropy after each refinement. Activity and value heads are not trained by this objective. The next main training run jointly trains environment interaction, activity selection, and adaptation over continuing lifetimes. The numbered sections below specify objectives and initialization dependencies; they are not separate research projects that must each reach their best score before integration. Current commands are in [RUNNING.md](RUNNING.md).

## Interaction data

The training unit is a lifetime containing changes of goals and operating conditions, delayed results, idle periods, and returns to earlier conditions. It contains time-ordered received observations, computation operations, applied actions, and outcomes. A tensor window used for backpropagation does not end that lifetime or clear its state, experience, or adaptive parameters. Include useful work between observations and observations arriving before a calculation finishes.

| Field group | Stored information |
| --- | --- |
| Identity | Run, trajectory, task family, simulator version, seed |
| Timing | Observation capture and arrival; operation start and finish; action application |
| Input | Modality, source, units, payload reference, goal update |
| Activity | Observation request, refinement, memory read, tool call, learning update, subgoal, or wait; selected target and budget |
| Action | Proposed and applied action, policy probability, pending request identifier |
| Outcome | Measurement, reward, elapsed time, resource use, related action |
| Computation record | Generating context, observation scope, goal, parameter version, depth, retained payload |
| Training state | Policy version, memory capacity, reset locations, adaptation steps, instance-owned adapters and optimizer state |

Sequence batches follow arrival order while retaining capture time. Numeric events can use columnar storage; larger media use indexed assets. Received observations and imagined trajectories have distinct data types.

Split collection runs, layouts, dynamics, goal combinations, and random seeds before constructing training windows. Evaluate new factor combinations separately from longer sequences and changed timing. Repeated contexts must occur within trajectories so reuse can be learned; held-out trajectories and task combinations remain separate.

## Environment distribution

The initial grid distribution contains changing doors, goals, and obstacle positions. Each operation consumes a specified interval. Use three regimes: full context return, return of only some conditions, and continued change without exact repetition. Every method receives the current observations at its action head.

Reference planners initialize behavior from the information assigned to them. Record whether a reference uses the agent's observation history, a reconstructed map, or full simulator state. A planner with extra information can be a teacher or performance reference without becoming a student input.

The main distribution includes instrument tasks with nonlinear dynamics, noisy and delayed measurements, pending requests, and actions that affect future observations, alongside navigation and executable digital tool tasks. These families use the same event/action interface and shared core. Learned activity and adaptation are trained across these families, rather than postponed until navigation is complete. Existing digital-research logs can support prediction and imitation after import; evaluating a new action policy requires an environment in which its consequences can be measured.

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

The lifetime trainer implements these continuations as copies of a process that share its future disturbances, readings and condition changes, with the continuation policy sampling under shared noise ([`runtime.branch_values`](experiments/lifetime/runtime.py)); their returns train an activity-value head. Whether such targets carry signal is measured, not assumed: [`branches.py`](experiments/lifetime/branches.py) values the same states twice with independent randomness. One sampled continuation per option repeats with a correlation below 0.1 at horizons of 4 to 32 ticks, so a single choice's consequence is buried in the variability of what follows; continuations on the policy's most probable choices over 4 ticks, averaged over four independent sets, repeat with a correlation of 0.75. The policy follows the head in proportion to the share of fresh continuations' variance it predicts.

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

Choose coefficients on validation trajectories and record their gradient scales. The policy selects observation, retrieval, computation, action, learning, subgoal, and wait operations. Learning choices include an experience source, an adapter group, and a bounded update budget. Separate action and activity heads allow continued control while slower work runs. The fast action readout remains available between longer activities.

The distribution needs situations where sensing is useful, where further computation is useful, and where immediate action is best. Include irrelevant or noisy inputs so surprise alone does not determine the activity. Compare fixed budgets, simple benefit-per-time scheduling, and the learned policy with equal training and tuning budgets.

## 5. Learning to acquire and retain abilities

Use shared parameters \(\theta\) across training lifetimes and local adaptation state \(\eta^{(j)}\) for lifetime \(j\). A learning action selects a batch \(B_t\) from that lifetime's received experience and an update budget. A first implementation uses ordinary gradient updates on selected prediction/action adapters:

\[
\eta_{t+1}=\eta_t-\alpha_t\nabla_\eta
\mathcal L_{\mathrm{received}}(\theta,\eta_t;B_t).
\]

This is a starting update rule, not a novelty claim. Its targets are received observations, executed-action outcomes, and available task feedback. Model-generated rollouts and fixed-teacher distillation are separate auxiliary sources. Retain the source of each target in the training data.

The outer actor–critic training optimizes the shared context, refinement, prediction, action, and activity parameters across lifetimes that actually contain these inner updates. PPO estimates the return of choosing learning without requiring gradients through the optimizer. A differentiable inner update is an explicit later comparison, not an unstated property of PPO. The next transition includes the changed adapters, resulting predictions, pending operations, and elapsed time. When recomputing policy probabilities, restore or reconstruct the adapter state that existed at that decision; the final adapters of a lifetime cannot be substituted for earlier ones. Store the adapter/optimizer state at training-window boundaries and the intervening update data. The discrete activity selector uses policy-gradient return targets, while action and prediction losses also train the shared representation; this does not claim an exact gradient through all lifetime adaptation.

Compute value targets across task changes and learning effects. A backpropagation-window boundary uses the next decision state's value; only the specified end of a lifetime is terminal. Use duration-aware discounting and continuation values at horizons long enough to include later tasks. If an update only pays off after the training horizon, that training setup cannot teach its value.

For a subset of simulated histories, branch from the same external state, model state, experience, and optimizer state: learn now versus spend the interval on another operation. Apply the same continuation policy and compare subsequent utility over a common elapsed-time horizon. This can initialize or train operation-value estimates. The deployed selector uses its own received history, not simulator parameters or the branch comparison.

Experience selection initially chooses among recent, returning-task, and task-relevant retained records. Sample examples across all selected streams and records; [the corrected replay sampler](experiments/continual_agent/replay.py) addresses the prefix restriction in the existing operation experiment. A learned retrieval query or mixture may then optimize which experiences improve subsequent return.

The lifetime trainer implements the differentiable comparison (`--objective deployed`): within each training window it replays the learning steps the agent took, at the ticks it took them, each on the buffer it read then, and trains the shared parameters and the step size through every step by the loss of each tick's prediction under the adapter then in effect. The same replay restores the adapter at every decision, so recomputed policy probabilities read the adapter that was in effect. Precise readings, when they return, replace the noisy reading of the transition they measured, and likelihoods weigh each by its sensor's noise.

Compare acquisition with state-only adaptation, retrieved experience, fixed-schedule parameter updates, and learned update selection. Carry learned adapters across changing conditions and revisit earlier conditions without resetting them. Test transfer on held-out combinations. Separate reset experiments identify whether improvement resides in transient activations, retrieved records, or adapted parameters.

### Per-instance and collective learning

Parallel training can aggregate experience to update \(\theta\). In single-instance deployment evaluation, each agent's \(\eta\), optimizer, replay, and computation memory use only that agent's received interactions. Vectorizing these independent copies is valid; pooling their experience into one adaptive model answers a different question.

The current operation script uses fleet experience and shared parameter updates. Retain those results as collective adaptation. The integrated trainer must represent instance-specific adapters explicitly for the single-agent claim. A collaborative mode can later share experience and report the information and communication it receives.

During quiet intervals, choices include waiting, gathering useful observations, replay, and deeper-to-shallower distillation. Evaluate their contribution on subsequent tasks and their measured device time and energy. Repeated updates, a running process, or a falling replay loss are not substitutes for acquired capability.

## Sequence optimization and GPU execution

Use truncated backpropagation through time with numerical state carried between windows. Sweep unroll length and test longer dependencies. Reconstruct training state with a preceding burn-in segment when replay or parameter changes require it.

Keep workspace size, record capacity, and retrieval count fixed within a batch. Group sequences by length and operation count. BF16, activation checkpointing, compilation, and CUDA graphs are evaluated against an FP32, uncompiled reference for the relevant model and hardware.

Parallelize independent environments, sequence batches, seeds, and variants. For a model that fits on one GPU, use one process per device and [DistributedDataParallel](https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html), explicitly sharding trajectories. Evaluate [FSDP](https://docs.pytorch.org/docs/stable/distributed.fsdp.fully_shard.html) when model and optimizer storage require sharding.

The main campaign is a shared-core model in the 30–50 million parameter range, followed by a 150–250 million parameter scale comparison. These are proposed engineering budgets, not measured optimum sizes. Begin the campaign with approximately 10^8 aggregate external environment transitions per principal configuration, divided across five training seeds and reported per seed; an extended 10^9-transition run measures whether further data and capacity improve the same objectives. Report internal refinements and learning updates separately from external transitions.

Target 1,024–4,096 parallel training streams where the environment and memory footprint permit. Backpropagation minibatches are smaller than the collection fleet: use 128–512-event unrolls with carried state and lifetimes of 10^4–10^5 external events. Choose the physical-time discount and value horizon to cover the subsequent tasks on which learning is expected to pay off; the unroll length is not itself that horizon. Measure the effect of unroll length on delayed learning benefits. State, experience, and local adapters remain live between unrolls. A short numerical and throughput check initializes the campaign; it does not replace the main run or count as its research result.

On two H100s, prioritize the integrated model and its strongest matched baseline; schedule the independently timed actor/environment measurement explicitly. On a larger cluster, run five training seeds and matched baselines concurrently, and allocate separate capacity to held-out continuous operation. Replicate modest models for independent experiments before sharding them. A 4–8-device training group is appropriate only if measured rollout or optimizer throughput benefits; compute the completion estimate from measured transitions per second and GPU-hours, not GPU count alone. Use fixed shapes, batched adaptation, bounded retrieval, and grouped refinement counts to limit per-agent Python dispatch.

Measure batch-one observation-to-action latency separately from batched throughput. Include encoding, recurrence, memory lookup, record adaptation, refinement, action readout, and transfers. Report median and tail latency, parameter count, persistent-state bytes, peak memory, interactions per second, and GPU-hours. Simulated operation ticks and measured hardware time are separate quantities until a measured timing model connects them.

## Run output

For environment-training runs, record configuration, source commit, dependency and simulator versions, seeds, training curves, checkpoints, per-seed task scores, timing distributions, and evaluation trajectories. The current pilot writes configuration, versions, seeds, curves, and task scores; checkpoint saving is optional. Store large checkpoints and datasets externally with versioned download locations and hashes in GitHub. Use isolated output directories so new runs retain the earlier experiments.
