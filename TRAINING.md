# Data and training

**Status:** training specification for the neural architecture in [DESIGN.md](DESIGN.md). The existing fitted numerical models are described in [experiments/transport/](experiments/transport/README.md).

## Training unit

The basic unit is a time-ordered interaction containing observations, ongoing computation, executed actions, and outcomes. Training must include useful activity between observations and new observations arriving before reasoning completes.

| Field group | Stored information |
| --- | --- |
| Identity | Run, trajectory, task family, simulator version, random seed |
| Timing | Observation capture and arrival; computation start and finish; action application |
| Input | Modality, channel, units, missingness, payload reference, goal updates |
| Activity | Requested observation, internal operation, retrieval, subgoal, tool call, or wait |
| Output | Proposed and applied actions; policy probabilities; pending request identifiers |
| Outcome | Received measurement, task reward, resource use, related action identifier |
| Learning state | Parameter version, retained-memory configuration, reset points |

Store compact numeric events in columnar files and larger media in indexed assets. Sequence batches follow arrival order. Capture times remain available so the model can interpret delayed observations. Keep real observations and world-model predictions distinguishable.

Split by environment dynamics, task families, goals, and collection runs before constructing sequence windows. Use separate training, validation, and test trajectories. Test timing shifts independently from task-content shifts.

## Data sources

**Simulated instruments and control.** Generate randomized nonlinear dynamics, delayed and costly measurements, multiple pending operations, disturbances, changing targets, and interventions that alter later observations. Include periods with no new input and periods with rapid changes.

**Recorded expert interaction.** Use controllers, planners, or stronger models to initialize action and activity choices. Record what each teacher observed and when. A teacher using simulator state may supply training labels; its additional information is recorded separately from student inputs.

**Digital research trajectories.** Existing research-system logs are a candidate source of tool choices, long dependencies, interruptions, and resumed investigations. Import source versions, timestamps, actions, results, and task identity before selecting examples. The project has not yet imported such a corpus. Historical logs support prediction and imitation; new action policies require an executable environment to evaluate the consequences of alternative choices.

**Background experience.** Replay real trajectories and optionally generate imagined trajectories with a learned world model. Track real and imagined data separately and evaluate transfer back to actual interaction.

## Learning sequence

### 1. Prediction and initial behavior

Train the event encoder and persistent state to predict future observations, measured outcomes, and rewards conditioned on executed actions and elapsed time. Use grounded readouts so predictions retain task-relevant information. Initialize action and measurement choices with behavior cloning where demonstrations exist.

Train shared refinement at several depths and supervise intermediate action readouts. A partially computed workspace should already support a useful decision. Fixed-depth runs establish the initial quality–time curves.

### 2. Continuing inference under interruption

Train the context encoder, workspace, revision, and action heads on repeated observation and goal changes. The model carries its state through each change and receives return from later actions.

A paired branch can provide an auxiliary target: clone the same causal history, apply the new context, and compare transported computation with a recomputed or more deeply refined branch. Both branches use information available at the branch point. Teacher computation used only during training is counted as training cost.

The first numerical study shows that copying a same-depth state can preserve its optimization error. Branch matching is therefore optional auxiliary supervision. Compare it with outcome-only training and with a more refined teacher, including the effect on later decisions.

### 3. Learned observation and computation

Train the activity and action policies through recurrent actor–critic learning, initially PPO. Use elapsed-time discounting \(\gamma_i=\exp(-\beta\Delta t_i)\); rewards include utility accumulated during the interval and explicit observation/computation costs.

An initial joint loss is

\[
\mathcal L=
\mathcal L_{\rm PPO}
+\lambda_v\mathcal L_{\rm value}
+\lambda_p\mathcal L_{\rm prediction}
+\lambda_b\mathcal L_{\rm branch}
-\lambda_e\mathcal H(\pi).
\]

During supervised initialization, behavior cloning replaces the policy-gradient term. Choose coefficients on validation tasks; record each loss and its gradient scale. Report task utility, time, sensing cost, and compute cost separately from their scalar combination.

The activity distribution must contain cases where more observation helps, cases where more computation helps, and cases where immediate action is best. Include noisy or irrelevant streams so prediction error alone does not determine activity.

### 4. Learning across tasks

Extend rollouts across changing and recurring task families. Compare frozen parameters, retained records, fast adaptation, and updates to shared parameters. Begin with prediction and policy heads; then compare adapters and full-core updates.

Use fresh rollout batches for PPO. Replay supports prediction, rehearsal, and distillation. Using replay for policy improvement requires a specified off-policy algorithm.

Measure learning after transient-state resets, on revisited tasks, and on unfamiliar tasks. Later train the choice of learning operation itself: which experience to rehearse, which skill to practice, or when an update is worth its cost.

## Sequence optimization

Use truncated backpropagation through time while retaining the numerical state across windows. Sweep unroll length and evaluate dependencies longer than the training window. Reconstruct replay state with a preceding burn-in segment under the current parameters.

Actors use identified parameter snapshots. Log the policy version that generated each trajectory and the state treatment when parameters change. Keep state local to its environment and preserve observation/action chronology when batching.

Use BF16 where supported and verify recurrent-state and loss numerics against an FP32 reference. Accumulate sensitive reductions in appropriate precision. Measure activation memory before choosing sequence length and checkpointing.

## Multi-GPU training

Parallelize independent environments, sequence batches, model seeds, and experiment variants. These are complementary uses of a cluster.

For models that fit on one device, start with one process per GPU and [DistributedDataParallel](https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html). Explicitly shard trajectories across workers; DDP synchronizes gradients rather than partitioning the input automatically.

For models whose parameters, gradients, and optimizer states dominate memory, evaluate [FSDP](https://docs.pytorch.org/docs/stable/distributed.fsdp.fully_shard.html). Choose the strategy from measured memory and communication cost. Keep a compact deployed actor independent of training sharding when measuring real-time latency.

An illustrative 32-GPU allocation after the single-device implementation works:

| Work | GPUs | Purpose |
| --- | ---: | --- |
| Five independent training runs | 20 | Four GPUs per run; same model and data specification |
| Interaction collection | 6 | Batched environment rollouts and fresh policy trajectories |
| Evaluation | 4 | Held-out tasks, long runs, and batch-one timing |
| Teacher trajectories and data preparation | 2 | Training targets and additional interaction data |

This is an allocation example, not a measured optimum. CPU-bound simulation, small models, or expensive vision encoders can require a different balance. Rotate the cluster across comparison methods and reserve comparable tuning and training budgets.

Record effective batch size, unroll length, optimizer steps, environment interactions, parameter count, GPU-hours, hardware topology, communication time, and policy-version delay. Scaling the GPU count changes data collection and optimization; retain a fixed-compute comparison as well as the larger run.

## Inference optimization

Profile observation encoding, recurrence, revision-factor generation, refinement, retrieval, action readout, and transfers separately, then measure the full path. Short shared blocks permit frequent event incorporation. Group training sequences by shape and operation count where useful.

Evaluate compilation and CUDA graph execution for repeated fixed-shape blocks after the uncompiled implementation is correct. Dynamic activity selection can change shapes and execution paths; measure padding and regrouping costs. Consult the [PyTorch compiler documentation](https://docs.pytorch.org/docs/stable/torch.compiler_cudagraph_trees.html) for the installed version.

Measure batch-one latency on the intended actor hardware separately from batched training or rollout throughput. More GPUs can accelerate training and evaluation without reducing the sequential work required for one decision.

## Recorded output

Each run produces the configuration, source commit, environment and dependency versions, training curves, checkpoints, seed-specific scores, timing distributions, and evaluation trajectories. Save large checkpoints and datasets in versioned external storage with download locations and file hashes recorded in GitHub.

The implementation sequence and current runnable command are in [RUNNING.md](RUNNING.md).
