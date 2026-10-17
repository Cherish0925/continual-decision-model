# CDM: objective, implementation, and results

Status at **17 October 2026**. Planner numbers come from release [v0.2.0](https://github.com/Cherish0925/continual-decision-model/releases/tag/v0.2.0); activity heads retrained with the corrected PPO bootstrap and the integrated learning-while-operating runs come from release [v0.2.1](https://github.com/Cherish0925/continual-decision-model/releases/tag/v0.2.1); the continual core and its lifetime campaign from release [v0.3.0](https://github.com/Cherish0925/continual-decision-model/releases/tag/v0.3.0).

**CDM aims to be a continuously operating model that learns both what to do in the world and how to spend its own computation and experience.** Observations, requests, tool results, and conversations enter the same ongoing process. Goals, unfinished inference, and learned abilities should outlast an individual output.

The present results are a trained navigation prototype with persistent computation and learned choices of observation and refinement, and a continual core: one model that keeps its state through process lifetimes and chooses at every tick its control, its measurements and the use of its compute for inference or for an update of its own parameters, while the process keeps moving. Training its state to predict what each action will do makes this model better than a recurrent policy, whether or not the policy reads the predictions; the instance's own parameter updates have not yet improved its control, except a little on unfamiliar dynamics when the shared network is trained through those updates. The setting called "full", in which every activity is learned, is the most complete configuration of this prototype; it is not yet the continuously proactive model the project aims for. General multimodal operation, autonomous research, and long-term skill acquisition remain research objectives.

## The intended model

The model should maintain an understanding of its environment, goals, pending work, and its own capabilities; incorporate new input while inference is unfinished; initiate observations and actions without a fresh prompt; and keep useful decisions available while more expensive computation proceeds. It should also work in a request-response setting.

Fast interaction, deeper inference, memory retrieval, and parameter learning operate on different timescales. During quiet periods, useful work can include observation, prediction, replay, experiments, and consolidation into faster skills. Low activity should reduce actual compute and power use. Merely keeping a process alive, performing parameter updates, or producing a small output does not establish these benefits.

The runtime supplies time, observations, devices, tools, and execution. The research objective is to train state updates, activity choices, and adaptation in the model. The proposed roles of a fast context update, persistent inference workspace, action-conditioned prediction, activity policy, memory and learning, and task-specific outputs remain relevant. Their final architectures are not fixed by the current maze implementation.

## Implementations

| Implementation | Current structure | Trained behavior |
| --- | --- | --- |
| [Sequence core](cdm/model.py) | GRU context, latent slots refined by a shared Transformer, bounded whole-workspace records, conditioned reading and slotwise reuse/revision/initialization, action/activity/value heads | Action prediction on a twelve-event contextual-recall task. The activity and value heads are not trained in that study. |
| [Spatial planner](cdm/planner.py) | Shared gated convolutional refinement; spatial workspace; optional goal-cell read, learned rebuild gate, stored workspaces, map/goal split, and observation-age input | Supervised optimal-action and distance prediction in changing mazes; inference continues across changes. |
| [Continual agent](experiments/continual_agent/README.md) | Spatial planner, last observed map and its age, request/survey counters, policy over refinement count, survey request, and move/stay | PPO trains the activity head with the planner frozen. A separate operation experiment updates planner parameters under a hand-written activity schedule. |
| [Continual core](cdm/core.py) | GRU fast state; eight persistent slots refined by a shared Transformer block over the fast state and recent experience records, normalised after every refinement, optionally through a gate that keeps, revises or restarts each slot; action-conditioned prediction of the next reading for every action through an instance-owned low-rank adapter, optionally a bank of blocks weighed by a selector of the state, with learned sensor noise; per-action control scores; heads for precise measurement and for compute (refinements or an adapter learning step); critic; optional activity-value head | Recurrent PPO over instrument and thermal process lifetimes of 16,384 ticks whose state, experience and adapters persist; trained choice of measurement, refinement and learning ([lifetime experiment](experiments/lifetime/README.md)). A second round, in training, trains it through its deployed learning steps, its workspace across events and copies of processes, on instruments that also ask for their largest output, thermal processes and vehicles behind walls. |

The sequence core and spatial planner are separate models. The continual agent wraps the spatial planner. The continual core integrates fast state, persistent workspace, world prediction, activity choice and instance learning in one trained model for structured process events, including vehicles that navigate around walls; language, digital tools and retrieval of computation records are not yet part of it.

## Capability coverage

### Perception, inference, and action

| Intended capability | Present implementation and result | Work remaining |
| --- | --- | --- |
| Persistent current state | Workspace and map belief persist across requests and idle intervals within a site. The continual core keeps fast state, workspace, experience records and adapters through 16,384-tick lifetimes spanning condition changes, requests and idle periods; slot normalisation keeps the workspace at one scale. | Carry or adapt state across sites and task types; operation over hours. Current site changes call `begin()` again. |
| Continuing latent inference | Repeated shared neural refinement, with action readout after the selected refinements. No language decoding is required. The continual core's workspace persists across events and can be trained across 16 of them, with a gate that keeps, revises or restarts each slot. | Show that continued inference pays on tasks that need it; persistent candidate plans; intermediate interruption under independently arriving events. |
| Incorporation of new input | New maps/goals enter the next simulated tick; surveys return after two ticks. | General timestamped multimodal event handling, arbitrary arrival times, concurrent pending tools, and interruption inside a longer computation. |
| Multimodal input | Structured maps, goals, position, observation ages, and counters. The sequence core accepts numeric features. | Text, image, video, audio, physical sensors, modality encoders, and their joint training. |
| Action-conditioned world prediction | Distance and action heads in a maze planner. The continual core predicts the next reading for each of 81 actions and scores actions by their predicted outcomes. | Multi-step and reward predictions, imagined experiments and rollouts. |
| Timely decisions and control | One move/stay decision per simulated tick; measured GPU planner-step times. The continual core acts every tick while refinements, learning steps and precise measurements take time; `realtime.py` runs it against processes on their own wall clock, with learning on a separate CUDA stream. | Deadline-aware training and capture-to-applied-action timing on physical devices. |
| Proactive observation and computation | PPO selects 0/1/2/4 refinements, survey/no survey, and move/stay. It observes during idle periods and varies computation with the request. In the continual core it selects precise measurements, refinement counts and learning steps; copies of a process under shared randomness value each option, and an activity-value head trained on them can guide the choice. Option values from one sampled set of copies do not repeat (correlation below 0.1); greedy, averaged short copies do (0.75). | Learned choices of retrieval, tool use and longer internal activities; showing that values from copies improve on PPO. |
| Autonomous goal pursuit | Pursues externally supplied navigation requests without a prompt for every action. | Generate and revise subgoals, initiate useful investigations, and maintain long research objectives. |
| Reactive and conversational use | External goals and requests can arrive while the agent runs. | Language interaction, dialogue, instruction understanding, and use as a general LLM-agent component. |
| Self-assessment and regulation | Activity head reads policy features, value-related features, map age, and request/survey progress. | Learned estimates of capabilities, uncertainty, expected progress, operation duration, and learning benefit that guide behavior across tasks. |
| Output interfaces | Discrete navigation actions; activity choices; internal latent state. | Continuous controls, tool arguments, experiments, structured plans, language, speech, and image outputs. |
| Environment and tool integration | Simulated grid observations and surveys; simulated instruments and thermal processes with delayed precise measurements. | Robot/device interfaces, files, operating-system actions, digital tools, and asynchronous experimental instruments. |

### Memory, learning, and scale

| Intended capability | Present implementation and result | Work remaining |
| --- | --- | --- |
| Reuse of computation | Ring-buffer records, snapshot and learned readers; controlled comparisons. Useful on returning contexts, with limited additional benefit in the strongest maze planner. | Selective reuse beyond fixed map/goal structure, learned writing/eviction, longer retention, reuse after parameter updates, and transfer to other tasks. |
| Experience and knowledge | Recent belief/goal sequences and survey outcomes are replayed; old-size experience can be retained across sites. | Durable experience abstraction, a transferable knowledge/skill store, and external knowledge retrieval. |
| Offline training | Supervised sequence/planner training and PPO activity training run successfully. | Unified training distribution and objectives across model operations and domains; larger sequence and model scales. |
| Learning during operation | Replay, frozen-teacher distillation, solver supervision, and hindsight from surveys are implemented. Separate full-view adaptation improves performance; in the integrated stream with a full view, learning from unused compute raises 9×9 reward by 0.026 per 100 ticks without loss on 5×5. In the continual core, instance adapter updates as trained raise error slightly (+0.0015 ± 0.0003 against discarding them); trained through its own step, the network makes them neutral, and useful on slower dynamics than it was trained on (−0.0031 ± 0.0016). Trained through the steps it actually takes, with precise readings as targets, a one-seed pilot's updates lower error by a third against discarding them at matched compute (0.312 against 0.466). | Confirm with seeds and held-out dynamics; retention of returning conditions. |
| Choosing when and what to learn | Unused refinement credits trigger a gradient step at a fixed threshold. The continual core's activity head chooses adapter learning steps, trained jointly with control, measurement and refinement; precise readings replace the cheap reading of the transition they measured and weigh by their noise; step size grows with the gradient up to a learned bound. | Learned choice of experiences; learning that serves returning conditions (adapter blocks, in training). |
| Quiet-period consolidation | Low-activity observation is learned; replay and deeper-to-shallower distillation can run from spare credits. The continual core's evaluation removes computation between requests to measure what it contributes to later ones, and real-time runs record power and energy per process step. | Show that idle work improves subsequent tasks; longer-term skill consolidation, broader knowledge acquisition, and measured power reduction. |
| Long-term self-improvement | A frozen copy's deeper computation improves its fast student on larger mazes. | Repeated acquisition of new skills and knowledge from interaction, retained improvements across tasks, and multi-stage self-improvement. |
| Generalization | New maze instances, larger mazes, changed door dynamics, and recombined map/goal conditions. Held-out instrument dynamics; one continual core trained on instrument and thermal processes together, and on these with vehicles (in training). | Transfer across modalities and to digital tools. |
| GPU-friendly computation | BF16, batched PyTorch, fixed-size workspaces within a run, concurrent seeds/variants on two H100s. Measured training throughput, stage times and operation timing of the 35.9 M continual core. | End-to-end timing of adaptive activity and learning; distributed optimization, larger models, power measurements, and edge deployment. |
| Open research artifacts | Source, result tables, per-seed measurements, retained planner weights, activity weights, configurations, and training curves are public. | Preserve failed training attempts in future aggregate reports; broader external comparisons and a paper supported by the completed experiments. |

## Results that are already available

### Dynamic planning and adaptation

The main study has 14 variants with five training seeds each, plus three partial-view variants with three retained seeds each: **79 retained planners**. Main evaluations use 1,024 held-out episodes per regime and maze size, each with 96 ticks. A 5×5-room maze has 21×21 grid cells; 9×9 rooms has 37×37 cells.

- **Low-budget continuous change:** at one refinement per tick, restarting makes 16.04% non-optimal moves and earns 0.68 return; carry makes 2.03% and earns 1.61; the map/goal-split planner makes 0.40% and earns 1.73, against 1.74 for the current-map optimal policy.
- **Across all three change regimes at the training size:** the split planner makes 0.27–0.42% non-optimal moves and earns 1.96/1.87/1.73, versus 1.97/1.87/1.74 for the reference.
- **After a goal move:** goal-cell reading raises affected-cell accuracy after one refinement from 43.0% to 92.6%. Adding the learned rebuild gate raises immediate accuracy before further refinement from 12.9% to 83.0%, and one-refinement accuracy to 93.6%. The hand split reaches 82.5% and 96.3%, respectively. Input encoding, rebuilding, and readout still execute at zero additional refinements.
- **Larger mazes:** the single workspace with goal reading transfers best at one refinement, with 1.53–2.35% non-optimal moves on 9×9 rooms, versus 5.45–7.51% for carry. The learned gate is not uniformly better than the simpler goal-read model on this test.
- **Stored workspaces:** exact snapshots reduce full-return errors from 2.60% to 1.42%, but help little on partial returns. Learned readers reach roughly 1.2–1.3% on full/partial returns; goal reading alone reaches 0.5–0.7% across regimes.
- **Shared refinement allocation:** a hand-written priority based on the last change of the action distribution reaches 0.13–0.18% non-optimal moves with 1.5 refinements per agent on average, versus 0.20–0.26% with uniform four-refinement allocation at the training size. This is a scheduling heuristic, not a learned activity policy.
- **Partial observation:** additional observations improve return more than additional refinement in the tested schedules. Two partial-view training runs stalled near goal-independent accuracy and were replaced; the published 79-run collection contains the retained runs, not all attempts.

See [maze results](experiments/dynamic_maze/RESULTS.md), [training records](experiments/dynamic_maze/results/training.json), and [per-seed evaluations](experiments/dynamic_maze/results/eval_seeds.json). Action accuracy, affected-cell accuracy, request completion, and return describe different outcomes.

### Learned activities

Each of three planner variants has **three PPO training seeds**, 120 iterations, and 2,048 parallel streams; the value bootstrap at each segment end uses the decision state after the segment's last action. The held-out evaluation contains four sites of 128 ticks each per stream.

| Activity policy, on the gated planner | Net reward per 100 ticks | Refinements per tick | Surveys per 100 ticks | Requests reached |
| --- | ---: | ---: | ---: | ---: |
| One refinement every tick | 1.453 | 1.000 | 0.00 | 80.01% |
| Hand-written adaptive schedule | 1.575 | 0.678 | 3.47 | 82.00% |
| Learned activity head (three seeds) | 1.623 ± 0.005 | 0.378 | 6.77 | 82.45% |

Across the three planners, learned heads earn 1.623, 1.571 and 1.595, compared with 1.575–1.581 for the hand-written schedule and 1.449–1.478 for fixed one-refinement operation: better on two planners, equal on the third. They use 42–51% fewer refinements than the hand-written schedule and request about twice as many surveys; total hardware savings have not been measured. Request completion is 81.03–82.45%. The single-seed heads of release v0.2.0, trained with the earlier boundary bootstrap, earned 1.556–1.587.

For the gated planner, computation averages 2.18 refinements on a request-arrival tick and 0.06 when idle, and surveys increase as the stored map ages; all three seeds learn this pattern.

The approximately one million evaluation agent-ticks are parallel samples, not one uninterrupted million-tick run. Each site starts with a full map, a new workspace, and 16 warm-up refinements; subsequent observations are partial. Within a site, state persists through requests and idle periods.

Source: [activity results](experiments/continual_agent/RESULTS.md) and [execution code](experiments/continual_agent/agent.py).

### Learning during operation

A separate two-seed full-view experiment adapts a 5×5-trained split planner to 9×9 mazes. After 200 updates, non-optimal moves fall from **1.90% to 0.91%** with a frozen copy making deeper predictions, or to **0.73%** with the exact solver. Replaying old-size episodes with solver supervision retains the old-size error near 0.39%, versus 0.38% initially.

In the integrated stream with a full view on 9×9 mazes, frozen-copy distillation paid from unused compute raises reward by 0.026 per 100 ticks on average over six paired runs (two planners, three seeds; standard error 0.0075), with no change on 5×5 afterwards. In the integrated partial-view stream, replay, frozen-copy distillation, and survey hindsight produce no return improvement beyond the reported site variation of about ±0.03. This experiment uses a hand-written schedule (`policy: null`, `schedule: scaled`), not the trained activity head. It carries weights and replay across sites while reinitializing the current workspace.

Distillation toward the learner's own changing deeper output severely degrades both tested planners; fixed teachers work better in these runs. This comparison concerns the tested targets and optimization settings, not a general impossibility of learning from self-generated computations.

Sources: [online adaptation](experiments/dynamic_maze/RESULTS.md#6-learning-during-operation), [integrated operation](experiments/continual_agent/RESULTS.md#learning-while-operating), and [operation configuration](experiments/continual_agent/results/operate/spare_frozen_distill+hindsight.json).

### GPU performance

Reported measurements below use an idle H100, BF16, **5×5 rooms**, and one refinement per tick.

| Planner | Parameters | One agent: mean / p95 | 256 agents: mean per batch |
| --- | ---: | ---: | ---: |
| Carry | 0.60 M | 0.912 / 0.935 ms | 1.86 ms |
| Goal read | 0.61 M | 1.008 / 1.035 ms | 2.34 ms |
| Goal read and learned rebuild | 0.66 M | 1.244 / 1.278 ms | 3.14 ms |
| Map/goal split | 0.48 M | 1.439 / 1.470 ms | 2.65 ms |

The split planner at one refinement and carry at two take 2.65 and 2.67 ms per 256-agent batch, with 0.27–0.42% and 0.82–0.99% non-optimal moves respectively on the training-size mazes.

The timed loop includes map assimilation, refinement, action readout, and a simulated move. Inputs are already on the GPU; initialization and its 16 refinements occur before timing, and the first 20 timed ticks are discarded. The archived latency file contains **only 5×5-room measurements**. No 9×9 latency, sensor-to-actuator latency, full learned-agent timing under online updates, or power saving follows from this table.

Source: [latency measurements](experiments/dynamic_maze/results/latency.json) and [timing function](experiments/dynamic_maze/evaluate.py).

### Continual core on process lifetimes

Eight settings of one 35.9 M-parameter model, five seeds of 2·10⁷ interactions each, evaluated over one uninterrupted 16,384-tick lifetime on 512 newly sampled instruments whose operating conditions change and return without announcement. Error is the tracking error while a request is active; a controller that knows the true steady response scores 0.115.

| Setting | Error | Estimated GPU µs per tick |
| --- | ---: | ---: |
| Recurrent policy | 0.146 ± 0.008 | 4.15 |
| Recurrent policy trained to predict, reading no prediction | 0.109 ± 0.002 | 4.15 |
| Predictions only | 0.106 ± 0.002 | 4.15 |
| + learned workspace computation | 0.106 ± 0.001 | 9.54 |
| + learning on a fixed schedule | 0.107 ± 0.003 | 8.66 |
| + choosing when to learn (full setting) | 0.108 ± 0.001 | 8.94 |
| World model with fixed schedules | 0.112 ± 0.003 | 7.45 |
| CDM trained through its learning step | **0.097 ± 0.004** | 8.82 |

- Prediction supervision lowers error by 25 % even when the policy reads no prediction; supplying the predictions to the policy adds 0.003, within the spread of seeds. On these instruments, workspace refinement, online learning and the choice of when to learn add nothing measurable on top, and learned allocation of observation and compute beats fixed schedules by 0.005. The GPU column is estimated from per-operation timings.
- Training the shared network through the instance's learning step gives the best model, here and on held-out instruments with slower dynamics, noisier sensors and more frequent changes; its deployed updates are neutral on the training family, so the gain is not explained by the updates themselves.
- With each learning step kept but its update discarded (same simulated time and nominal cost, gradient computation skipped), the full setting tracks better by 0.0015 ± 0.0003, and after a condition change applying the updates costs 0.004–0.005 for 10–100 ticks. A smaller test-time step removes the harm. A step size learned through the single-step meta loss converges to 0.30 in three seeds, and those models' deployed updates hurt. Candidate causes are properties of the code (fixed-distance steps, noisy targets, mixed conditions in the buffer, one-step meta-training); their effects still need training comparisons.
- About 0.06–0.07 of the noise-free next output remains unexplained, not all of it necessarily inferable from the agent's observations; the adapter of the model trained through its step removes a little of it, the plain adapter adds to it.
- One set of weights runs instrument and thermal processes without being told which is which, at 0.023 and 0.013 above single-family models that saw twice as much of each family; single-family models do not transfer.
- Against processes advancing on an independent 10 ms clock, absorbing every published observation and learning from the actions actually applied, the median response stays near 3.6 ms up to 64 instances; learning on a separate stream raises the p99 to 11–13 ms and leaves 5–8 % of steps behind the newest observation, which is not a deadline guarantee. At 512 instances learning takes the GPU from control and 24–92 % of steps run behind.
- A persistent state must keep one scale: an unnormalised residual workspace grew without bound over a lifetime and made fresh lifetimes 0.175 against 0.110 in training. Every refinement now normalises the slots.

Sources: [lifetime results](experiments/lifetime/RESULTS.md), [per-seed results](experiments/lifetime/results/per_seed.json), and the [corrections](https://github.com/Cherish0925/continual-decision-model/issues/1) applied before training.

### Earlier studies and literature

| Study | Completed result | Role in the project |
| --- | --- | --- |
| [Neural pilot](experiments/neural_pilot/results/REPORT.md) | Five seeds: computational memory 56.20% query accuracy, restarted workspace 38.44%, carried workspace 36.03%. | Trainability and memory use in the GRU/Transformer core; not autonomous control or a matched-latency result. |
| [Analytical dynamic grid](experiments/continual_grid/RESULTS.md) | Five seeds, three regimes, two budgets; reuse helps under low computation, but the tested local factorization does not beat nearest-context retrieval. | Explicit comparison of computation reuse; not a neural-model result. |
| [Numerical transport](experiments/transport/RESULTS.md) | The proposed correction loses to stronger numerical solvers; reconstructing an unfinished state can preserve its errors. | Completed negative result; it does not support a claim of superior state transport. |
| [Related research](ref/README.md) | 122 primary-source records: 41 section-level readings, 76 abstracts, four official summaries, one indexed excerpt. | Literature collection and comparison plan; not 122 full-paper replications or a completed originality comparison. |

Recurrence, gating, memory, iterative planning, active observation, and computation scheduling have prior work. The current results establish behavior within this task family. A method contribution must show what the proposed state representation, update rule, or learning objective adds beyond the closest alternatives. Channel-update patterns alone do not identify the contents of those channels or establish learned causal decomposition.

## Remaining work that directly serves the objective

**Make the instance's own learning pay.** Trained through the steps it actually takes, the continual core's updates lowered error by a third in a one-seed pilot. The second-round campaign tests this with seeds, held-out dynamics, single-step meta-training, cheap targets only and no learning, and tests adapter blocks for returning conditions. Measure acquisition, retention on returning conditions, and total compute together.

**Keep control on time at scale.** Control runs beside learning on its own stream up to tens of instances; at hundreds, learning must be bounded so that control keeps its deadlines. Measure capture-to-action timing on physical devices.

**Extend the shared core across task families.** The core runs instruments, thermal processes and vehicles behind walls on one interface, and instruments now also ask for their largest output. Add delayed digital tools, and show where continued inference pays.

The PPO rollout-boundary bootstrap now uses the next decision state after each segment's last transition; the activity results above come from the corrected training. Report all partial-view training attempts, including the two replaced runs, when comparing training reliability.

The remaining comparison set includes an ordinary recurrent action policy, output-only correction, a larger one-pass network, established gated planning/state-update models, and successor-feature reuse where applicable. Match observation access, training/tuning budgets, and actual runtime. Hours-long operation, distributed training, multimodal learning, tools, robotics, edge hardware, and autonomous physical/digital research remain subsequent implementations with distinct measurements.

## Publication

Release v0.2.0 provides a 170 MB archive of 79 retained planners with weights/configurations/training curves and three activity-head checkpoints; a 5 MB archive of per-seed maze evaluations including recovery curves and the reference policy; and SHA-256 checksums. Small result files and analysis are in the repository.

Raw terminal transcripts, the two replaced partial-view runs, and temporary exploratory outputs are not included in that release. The release completes the main retained maze/agent artifact set. It does not mean the full CDM research objective has been completed.
