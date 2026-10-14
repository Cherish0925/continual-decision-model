# First Technical Design

**Status: proposed architecture and experimental specification.**

This document records an initial implementation baseline. The subsequent [literature investigation](ref/README.md), [closest-work comparison](ref/COMPARISON.md), and [architecture experiments](ref/EXPERIMENTS.md) refine its research questions. The Mamba-2 core and component combination below are starting choices, not a final architecture or an originality claim.

The first implementation should test one concrete proposition: a persistent recurrent model can learn to allocate observation, internal computation, and external action more effectively than the same model following a fixed schedule.

The proposed starting point is a **recurrent core with an attention workspace and a learned activity policy**. It combines inexpensive state updates with selective computation. This is a CDM design hypothesis; the component literature is collected in the [research programme](RESEARCH.md#selected-foundations).

## 1. Model structure

| Component | First implementation | Purpose |
| --- | --- | --- |
| Event encoders | Small encoders for numeric sensors, images, and goal descriptors | Preserve modality, timing, and missing-observation information. |
| Persistent core | Mamba-2 event recurrence | Carry a learned representation across the full interaction. |
| Attention workspace | A fixed set of latent slots attending to recent event features and retrieved records | Compare evidence, goals, candidate interventions, and pending outcomes. |
| Internal refinement | A shared attention/MLP block applied for a variable number of steps | Develop the current representation without requiring a new external observation. |
| Activity policy | Categorical activity selection plus activity-specific arguments | Choose observation, computation, retrieval, action, or waiting. |
| Prediction and value heads | Action-conditioned latent transition, outcome prediction, and value estimation | Evaluate consequences and train decision-relevant representations. |
| Action heads | Continuous control and discrete instrument commands | Express decisions in the environment's native action space. |
| Experience memory | A bounded collection of episode records with learned retrieval scores | Recover useful evidence beyond the recurrent state's effective capacity. |

Mamba-2 is the initial candidate for event recurrence because it provides a concrete structured state space architecture [1]. Local attention offers direct access to recent evidence; Griffin provides a precedent for mixing recurrence and attention [2]. RWKV-7 is a valuable alternative backbone [3]. These choices should be compared after the first complete experiment runs, with the rest of the model held fixed.

Start with eight recurrent blocks of width 512 and sixteen workspace slots. Use a shared refinement block with one, two, four, and eight steps in the initial fixed-budget comparisons. These are launch settings for a compact pilot. Measure the resulting parameter count, recurrent-state footprint, and device performance before expanding the model.

The recurrent core updates on arriving events and executed actions. Refinement updates the workspace using a snapshot of that state. At a computation boundary, a learned projection summarizes the refined workspace as an internal event, which the recurrent core processes through its ordinary state update. A new observation can therefore revise the same ongoing state that supports later thought and action.

All prediction, value, and decision heads read this shared representation. The design keeps activity selection within the learned model; the surrounding runtime supplies event delivery, clocks, and execution.

## 2. State and interfaces

Distinguish five objects in the implementation:

- **Recurrent state:** per-layer state carried between updates.
- **Workspace:** latent slots holding the current computational context.
- **Recent evidence:** a bounded buffer of encoded observations with timestamps.
- **Experience memory:** retained records available for selective retrieval.
- **Parameters:** learned functions shared across trajectories.

Goals, pending measurements, uncertainty, and resource estimates enter the shared representation through typed features. Auxiliary predictions can test whether the state retains these quantities. Their full representation remains learned.

The initial model interface is:

```python
new_state, decision, predictions = model.step(
    state=state,
    events=arrived_events,
    elapsed_time=elapsed_time,
    goal=goal,
    resources=resource_observation,
)
```

`decision` contains an activity type and its arguments. An observation decision names a channel and measurement configuration; an action specifies a command; a computation decision requests another refinement step; a wait decision selects the next reconsideration interval. Ongoing physical actions can continue while a different internal activity is selected.

State persists across communication turns and temporary gaps in input. A new independent experimental run creates new transient state. Memory and parameter retention across runs are controlled separately by the evaluation protocol.

The core accepts modality features rather than requiring every input to become text. Later extensions can use pretrained visual, audio, and language encoders, with their versions and costs included in each experiment. Language and other expressive outputs can be attached as decoders conditioned on model state. The first experiment uses numeric sensors, a simple instrument image, and structured goals so that the core mechanisms can be investigated directly.

## 3. Time and execution

The environment continues to advance during inference. Observation arrival, model completion, and action application are separate events.

The runtime follows four rules:

1. Record when an observation was captured and when it became available.
2. Charge encoding, retrieval, refinement, and action generation for their elapsed time.
3. Incorporate newly available evidence at the next computation boundary, then reconsider the activity.
4. Record when an action actually affects the environment and which state produced it.

The first implementation is serial at model computation boundaries: an ongoing actuator command persists while the core computes, and new observations wait until the current step completes. This gives interruption a measurable maximum granularity. Concurrent perception and action generation can be investigated after this timing model is established.

Use an event-driven simulator with explicit operation durations for reproducible training. Calibrate those durations from the implementation, then evaluate with actual wall-clock inference. Sweep slower hardware profiles, variable inference delays, sensor jitter, and bursty arrivals. Simulator time must advance during internal refinement as well as during external actions.

A useful policy must learn when deeper thought can still affect the outcome. Duration-aware returns discount future value by elapsed environmental time rather than by the number of internal steps. Report the raw task score, consumed resources, and scalar training return separately so the trade-off remains visible.

## 4. First environment: an experimental instrument

Build a simulated physical process with hidden dynamics and a controllable input. A practical starting case is a thermal instrument with an unknown response curve, sensor bias, observation delay, and changing operating conditions.

The model receives a target operating condition and an experimental objective. It can change the input, select a measurement channel or resolution, conduct a calibration, or wait for an outstanding result. Measurements consume time and a finite experiment budget. Some conditions can be inferred from cheap observations; others require a deliberate intervention.

The task alternates between identifying useful response properties and reaching a desired operating state. Changes in hidden parameters make remembered experience valuable while requiring renewed inference. The model sees only available measurements; hidden simulator state supports scoring and diagnostic analysis.

Create three task families within this environment:

- **Identification:** estimate a response property with a fixed measurement and time budget.
- **Adaptive control:** track changing targets while identifying altered dynamics.
- **Experimental search:** choose interventions that locate a desired response efficiently.

Vary dynamics, noise, delay, measurement cost, goal composition, and change frequency independently. Hold out entire dynamics families and combinations of factors. Later environments should add manipulation and digital investigation, testing whether the same state and activity mechanisms remain useful beyond this instrument.

## 5. Data engineering

The primary data unit is a timestamped trajectory. Store observations, decisions, execution, and feedback as linked events.

| Field group | Required information |
| --- | --- |
| Identity | Run, trajectory, task family, environment version, and random seed |
| Time | Event time, arrival time, decision start and completion, action application time |
| Observation | Channel, modality, payload reference, units, missingness, and measurement configuration |
| Context | Goal, available resources, pending-operation identifiers, and previous executed actions |
| Decision | Selected activity, arguments, policy distribution, and requested computation |
| Outcome | Task feedback, measured consequence, resource cost, and causal action identifier |
| Model | Parameter version, encoder version, state-reset boundaries, and memory configuration |

Keep compact event metadata in columnar files and large media payloads in separate indexed assets. Build sequence batches from arrival order, preserving the time of the underlying events. A delayed result belongs to the moment it becomes available to the model, with a link to the action that produced it.

Generate initial trajectories with random exploration, fixed schedules, and a model-based controller using the same available observations. Teacher planning can be expensive during dataset generation; student evaluation counts the student's own computation. Retain unsuccessful and interrupted trajectories because they reveal the cost of poor scheduling and misleading hypotheses.

Split by run and underlying mechanism before producing training windows. Keep related trajectories and derived media in the same split. For continual-learning experiments, fix the chronological sequence in advance and expose only the permitted experience prefix at each evaluation point.

Inspect distributions of action coverage, delays, missing channels, task outcomes, and sequence lengths. Audit a sample of full trajectories visually against the simulator. Next-event targets may come from later in a training sequence, while the corresponding model input contains only information already available at that decision.

## 6. Training sequence

### Stage A: learn representations and useful initial behavior

Train encoders, recurrent state, and prediction heads on ordered trajectories. Predict future sensor features, measured outcomes, and elapsed-time-dependent transitions conditioned on executed actions. Add behavior cloning from the model-based teacher to initialize action and observation choices.

Use observation reconstruction and outcome prediction alongside latent prediction so that an uninformative latent representation cannot satisfy the entire objective. Start with fixed internal computation, then train across the proposed refinement depths. Supervise intermediate readouts to make partial computation useful.

The first world model predicts observations and outcomes. A subsequent experiment uses imagined action sequences to train or improve decisions, drawing on the world-model approach in Dreamer [4]. The effect of imagination should be measured separately from the effect of recurrent state.

### Stage B: learn activity selection through interaction

Use recurrent actor–critic training, starting with [proximal policy optimization](https://arxiv.org/abs/1707.06347), to optimize the activity and action policies jointly. Include their conditional log probabilities in the policy objective, with duration-aware value targets and returns.

The scalar objective combines task achievement with explicit measurement and computation costs. Auxiliary prediction losses continue to train the shared state. Record the component losses and gradient scales; tune their relative weights on held-out training tasks before the final comparisons.

The scheduler receives the current state, pending-result information, elapsed time, and resource observations. It learns whether another internal step, a new measurement, an action, or waiting has greater expected value. Begin with a small activity vocabulary and expand it only when the existing choices are distinguishable in learned behavior.

### Stage C: learn across changing tasks

Introduce long streams with recurring and novel dynamics. Compare frozen parameters, retained memory, and ongoing parameter updates. Mix recent experience with earlier trajectories during replay, logging the sampling distribution and total update budget.

PPO policy updates use fresh rollouts. Replayed experience trains prediction and rehearsal objectives; any later use of replay for policy improvement requires a separately specified off-policy algorithm. This keeps the initial training procedure well defined.

Begin parameter adaptation with the prediction and policy heads, then compare adapters and full-core updates. Evaluate retention and the speed of acquiring new competence throughout the stream. Replay, regularization, and selective feature renewal are candidate interventions if learning capacity declines; the plasticity results in [8] motivate these comparisons.

Learned selection of learning opportunities is a later extension: the model could choose which experience to rehearse or when an update is worth its compute cost. First establish a measured benefit from ordinary online updates with a fixed schedule.

### Sequence optimization

Use truncated backpropagation through time while carrying forward the numerical state. Cutting the gradient history does not require resetting the modeled situation. Sweep training sequence length and evaluate dependencies substantially longer than a training window.

For replay, reconstruct state from a preceding burn-in segment under current parameters. Measure sensitivity to burn-in length and parameter drift; avoid treating cached hidden states from old weights as exact current states. Record the parameter version and state treatment when online updates occur.

Start with batched environment simulation and mixed precision where numerical checks support it. Profile encoders, recurrence, refinement, retrieval, and output separately. Optimize the dominant measured cost, then repeat the end-to-end timing measurement.

## 7. Memory and deeper computation

Add experience retrieval after establishing the no-retrieval recurrent baseline. Store compact records containing context, intervention, outcome, and timestamps. Use a fixed capacity and retention rule initially; learn retrieval scores from their contribution to downstream prediction and decisions. Then compare learned retention against recency and uniform sampling at equal memory budgets.

The internal refinement block should first operate over actual evidence and candidate actions. Later, add explicit imagined transitions and compare their contribution. Shared latent recurrence [5] and CTM's internal dynamics [6] offer distinct alternatives for implementing internal computation; the initial workspace is a simpler experimental choice.

Keep three compute comparisons separate: a larger single-pass model, additional shared refinement steps, and a better observation or memory read. Their relative value may change with the task and available time. That variation is precisely what the activity policy should learn.

## 8. Evaluation and ablations

Evaluate baselines with shared encoders, training data, environmental information, and resource accounting:

| Comparison | Question |
| --- | --- |
| Fixed-window Transformer vs recurrent core | Does persistent state improve the quality–resource trade-off? |
| State retained vs periodically reset | Is long-running state responsible for the improvement? |
| Fixed vs learned observation schedule | Does the model acquire more useful evidence? |
| Fixed vs learned refinement depth | Does it spend computation where it helps? |
| No memory vs matched-capacity retrieval | Does stored experience add value beyond recurrent state? |
| Prediction only vs imagined trajectories | Does model-based internal planning improve decisions? |
| Frozen vs adapting parameters | Does lasting learning improve subsequent tasks? |

Present both equal-budget comparisons and full performance curves. A parameter-matched model may use a different amount of computation; report both quantities rather than treating either as a universal proxy.

Measure:

- **Task performance:** cumulative utility, success rate, control error, and measurement efficiency.
- **Responsiveness:** observation-to-applied-action latency at p50, p95, and p99; deadline misses; age of the latest relevant evidence used by each action.
- **Computation:** FLOPs or measured device time, peak memory, recurrent-state size, retrieval overhead, and energy where instrumentation is available.
- **Adaptation:** experience required to regain a target performance level after a change.
- **Continual learning:** retention, forward transfer, and new-task learning speed across the full stream.
- **Self-assessment:** calibration of predicted success and predicted benefit from extra computation or observation.
- **Generalization:** held-out dynamics, unseen goal combinations, new timing conditions, and transfer to another environment family.

Use at least five training seeds for the primary pilot comparison and paired evaluation scenarios. Report uncertainty over independent training runs; repeated decisions within a trajectory are correlated observations. Increase replication when uncertainty prevents the central comparison from being resolved.

## 9. Implementation order and decisions

| Step | Deliverable | Decision enabled |
| --- | --- | --- |
| 1 | Event-driven instrument, trajectory recorder, and fixed-schedule controller | Does the task expose meaningful observation and timing trade-offs? |
| 2 | Encoders, recurrent core, prediction heads, and fixed-depth workspace | Does persistent state carry information useful to later decisions? |
| 3 | Learned activity policy and duration-aware training | Does learned allocation outperform a tuned fixed schedule? |
| 4 | Retrieval and continuing parameter updates, evaluated separately | Which mechanism produces useful adaptation and lasting transfer? |
| 5 | Second environment family and measured device execution | Which benefits survive a change of task and runtime conditions? |

The next engineering decision is to implement the instrument and event schema, then establish the recurrent baseline. The first decisive result will be a task-quality versus elapsed-time comparison between fixed and learned activity schedules on the same model. That result will guide the next investment in backbone scale, memory, or internal reasoning depth.
