# Phase results: the continual core after two rounds

**Architecture update, 19 October:** [DESIGN.md](../DESIGN.md) now selects the first multimodal architecture, [the configuration](../configs/cdm_multimodal_v1.json) fixes its dimensions, and [TRAINING.md](../TRAINING.md) defines its integration and optimization. The hypotheses below record the choices considered after the process experiments. The selected memory path retrieves experience; restoring complete old adapters remains a comparison rather than the default.

Written 19 October 2026, at revision of the second-round results. It records where the project stands against its objective, what the experiments show, what failed and why, and what the next phase must do. Numbers come from [the lifetime results](../experiments/lifetime/RESULTS.md); this document links them rather than restating every table.

## A. Goal and value

### a1. Goal and question

The objective is a model that keeps perceiving, reasoning, acting and learning while its world changes. It keeps its state across interactions, chooses its next observation, computation or learning step itself, and stays able to act while slower work runs ([RESEARCH.md](../RESEARCH.md)).

The question for the next publication is narrower: when the environment keeps changing and computation is limited, how can a model keep and revise inference that is still in progress, turn what it experiences while acting into ability it can reuse, and keep its decisions timely? Choosing to observe, act, keep reasoning or learn serves this question; it is not a separate goal.

### a2. Value

Robots, electromechanical systems, autonomous experiments and digital research all run longer than one request and change while a model thinks. A model that restarts its reasoning at every input, or learns only offline, wastes the computation and experience such settings produce. The value of the project lies in showing that a single model can carry both across time and benefit from them.

### a3. Priorities and conditions of success

A capability counts only when a measured behaviour shows it, on tasks where other methods can also do well:

1. learning during operation improves later performance;
2. a situation met before is recovered faster than a new one;
3. continuing and revising unfinished inference beats restarting when new observations arrive;
4. decisions stay timely against an independent clock, counting every cost;
5. what was learned transfers to new combinations of tasks.

Publication requires these on public tasks, against strong existing methods at matched data and measured compute, with scaling behaviour. So far only (1) and (4) are shown, and only on simulated process families of our own.

### a4. Constraints and resources

- **Hardware.** Two H100 GPUs are in use. Eight A100 GPUs on separate nodes are not yet confirmed for this project.
- **Cost per run.** One second-round run of 2·10⁷ interactions takes 1.5–3.4 hours on a shared GPU.
- **This phase's compute.** The second round used 35 runs and 66 run-hours, plus evaluations.
- **Setting.** All work is digital simulation. Code, results and weights are public in this repository.

## B. Understanding and judgment

### b1. What the experiments show

**First round** (eight settings, five seeds):

- The gain over a recurrent policy comes from prediction supervision shaping the state, not from reading the predictions: 0.146 → 0.109 → 0.106.
- Workspace refinement, online learning and the choice of when to learn added nothing.
- The instance's own updates slightly hurt.

**Second round** (training through deployment, three seeds for the main variants, on a harder task mix):

- The instance's own updates now carry the result: 0.153 with them, 0.307 with them discarded at matched compute. The gain holds on every held-out family and against a real clock.
- After a condition change, error falls from 0.40 to 0.09 with learning and stays at 0.29 without.
- A condition met before is re-acquired faster, but only when the updates are kept.
- The workspace's computation contributes (0.046 when frozen in the best model). Unrolling it across events added nothing, and the trained keep, revise or restart gates do not use their choice.
- Values of activity options from copies of a process repeat only over a few ticks and when averaged, and a head trained on them did not predict fresh states.
- A bank of adapter blocks collapsed onto one block.
- One model runs three process families, behind single-family models.

### b2. Thinking and scope

The work sits among meta-learning that trains a learner through its updates (MAML, learned optimisers), recurrent meta-reinforcement learning (RL²), test-time training, fast weights, model-based control, and real-time execution of slow policies (real-time chunking).

CDM's emphasis is their coupling in time: the same model acts, observes, computes and learns in one continuing process whose timing matters. Our only distinctive positive result so far — learning that pays because training replays the learning steps actually taken — is a meta-learning result. It has not yet been compared with these methods on their own tasks.

### b3. Insight

**Aligning training with what runs is the lever that has worked, three times:**

- the reset correction that made PPO reconstruct the states collection saw;
- replaying the deployed sequence of learning steps;
- giving precise readings to the learner as targets.

**The converse is just as consistent: a mechanism whose benefit the training signal does not reach is not learned.**

- The workspace gate kept one option. Insufficient benefit from conditional revision is a possible explanation to test.
- The adapter bank's selector collapsed. Short-window learning may limit retention, but this has not been established as the cause.
- Activity values could not guide the policy through a term that had no gradient.

### b4. Judgment and research taste

The first phase spread across too many mechanisms at once: an unrolled workspace, a gate, values from copies, an adapter bank, three families and an imagined-rollout planner. Several were trained at full scale before checking, at small scale, that their training signal moved the behaviour they were meant to train. That is how a zero-gradient term reached a campaign.

One evaluation (removing refinement options) confounded the removal with a change of the rest of the activity. It was replaced by computing refinements but leaving the workspace unchanged.

These are errors of method, not of compute. The rules adopted follow from them:

- check each training term at small scale before scaling;
- measure first whether a task gives a capability any room to pay;
- leave a branch as soon as results show no gain.

### b5. Global view

| Intended capability | Status |
| --- | --- |
| Continuous operation, input, output and decisions | Done on low-dimensional simulated processes: a 10 ms loop, 64 instances, decisions on control, measurement, compute and learning at every step |
| Improvement from experience during operation | Done within a lifetime: errors halved |
| Experience kept until a situation returns; consolidation; skills | Not achieved |
| Continuing and revising unfinished inference | Computation persists and pays; revision of unfinished work is not shown |
| Proactive goals, experiments and research | Not implemented beyond choosing measurement, compute and learning |
| Multimodal input, language, tools | Not part of the two trained rounds; image/text input integration was added afterward, as specified in [DESIGN.md](../DESIGN.md). General language/tool behavior remains untrained. |

## C. Problem and theory

### c1. The problem

A decision model in a moving world faces three timescales at once:

- **events** arrive and age;
- **unfinished computation** becomes stale or remains useful;
- **experience** becomes knowledge only if it is kept and reused.

Computation takes time, during which the world moves. Learning changes parameters that also produce the state the model carries.

### c2. First principles

A capability is acquired only if two conditions hold:

1. the task makes it pay;
2. training carries that payoff back to the parameters that produce it.

The comparisons above show that learning paid once the objective replayed the deployed updates, and failed while it did not. The activity-value policy term had no gradient. The gate and adapter bank also failed to improve performance, but the proposed explanations for their collapse require separate experiments. Long horizons weaken the second condition without removing it: the critic bootstraps across windows, but a discount of 0.995 leaves 0.24 % weight on returns 1,200 ticks away, and direct gradients through computation and updates span at most one window.

### c3. Structure

The current core keeps one recurrent state, eight workspace slots, the 32 latest experience records, a buffer of 64 transitions and one low-rank adapter.

What it lacks is structure linking these:

- which events an inference used;
- which past situation an experience belongs to;
- how a parameter change alters the meaning of a state already held.

### c4. Variables, relations and causality

Effects are identified by paired evaluations of the same model on the same random stream:

- updates discarded with every learning step still computed;
- refinements computed with the workspace left unchanged;
- no computation between requests;
- learning off.

Seeds of the baselines vary widely (predictions only: 0.233–0.350), so comparisons between variants rest on paired differences and several seeds.

### c5. Model and representation

Today all inputs pass into one recurrent vector. The workspace slots carry no record of their premises. The only learned memory is one adapter that keeps following recent experience. These are engineering choices that have not been studied as principles.

### c6. Hypotheses for the next phase

Each comes with a test that can fail.

| Hypothesis | Test |
| --- | --- |
| **H1.** Units of ongoing work that record the events they depend on can be revised locally when one of those events changes, if training compares continuing with restarting at equal compute. | Recovery of correct predictions after a new observation, against restart and against keeping |
| **H2.** Restoring the adapter used in a situation when that situation is recognised gives retention where a learned selector collapses. | Recovery curve and number of observations needed on a returning situation, against relearning |
| **H3.** Consolidating repeated experience and fast parameters into shared parameters during idle periods improves later work at the same budget. | Later performance with idle compute counted |
| **H4.** Replaying deployed operations (learning steps, retrievals, consolidation) in training is the common method that makes each of them pay. | Each operation with and without its replay |

## D. Creation and solving

### d1. What is new and verified

- **A training objective that replays the learning steps an agent actually took.** Each step reads the buffer it read, takes effect when it did, and lets the online loss train the shared network and the step size through it. A test shows the replay reproduces the deployed adapters tick by tick.
- **An evaluation protocol with matched-compute controls.** Discarded updates, frozen workspace and no idle compute, each on common random numbers.
- **A measurement of whether counterfactual values of activities are a usable signal.** Copies of a process share its randomness; repeatability is measured between independent sets.
- **A real-time runtime.** It absorbs every observation in order, learns from the action actually applied, and records power.

### d2. Strategy

1. Write a complete, trainable design before further campaigns: how each kind of information is represented and updated, what trains each part, and the smallest experiment that would show each design choice wrong.
2. Measure, with simple reference methods, how much room each capability has to pay in a task, before training the model for it.
3. Move the main experiment to public tasks with strong existing methods; keep the instrument family for analysing mechanisms.

### d3. Methods and status

| Method | Status |
| --- | --- |
| Deployed-step replay | Works |
| Precise targets with sensor noise | Helps |
| Gradient-scaled steps | In every learning variant; not separately ablated |
| Unrolled workspace with gate | No gain |
| Recomputed predictions | Part of the unrolled package, no gain |
| Values from copies | Repeatable only short and averaged; following them untested |
| Adapter bank with selector | Collapsed |
| Imagined-rollout control | Worse than the control head |
| Three families in one model | Works, behind single-family models |

### d4. Algorithms

The replay groups the learning steps of a window by order across streams, so that the k-th steps of all streams run as one batch. The adapter in effect at each tick is then gathered from these versions. Precise readings are placed at the transition they measured, from the tick they return. Learning steps scale the gradient by η/(‖g‖+κ), with η and κ learned. Details are in [objectives.py](../experiments/lifetime/objectives.py) and the [lifetime README](../experiments/lifetime/README.md#training-through-deployment).

### d5. Checks of construction

Tests establish:

- replayed adapters equal deployed ones;
- the unrolled workspace equals collection;
- copies of a process stay identical under identical actions;
- the environments' random streams are unchanged by the refactoring;
- the activity-value gradient moves probability toward better options.

The last test was added after an external review found the zero gradient. A test that this gradient is non-zero should have existed before the campaign.

### d6. Experiments and computation

- **Second round:** 35 training runs (66 run-hours), with evaluations on the training family, three held-out families, three process families and four evaluation settings.
- **Analyses:** a branch-repeatability analysis, gate probes and a real-time measurement with power.
- **Process costs:** two runs ran out of memory and were rerun. Two runs were exact duplicates, because the term that was to distinguish them had zero weight.

### d7. Connections across fields

- **Adaptive control and system identification** describe what the instance adapter does, and offer baselines for it.
- **Complementary learning systems** motivate separating fast, retrieved and slow knowledge: a fast store for recent episodes, slow integration into shared structure, replay between them. These are hypotheses H2 and H3, not established properties of our model.

## E. Execution and evolution

### e1. Initiative

Both rounds were planned, implemented, run and analysed end to end, including corrections found during the work: the real-time recording, the duplicated evaluation of refinement, and the out-of-memory reruns. The pause in this phase came from the direction itself, which a further campaign could not settle.

### e2. Implementation

| Part | File |
| --- | --- |
| Core model | [cdm/core.py](../cdm/core.py) |
| Process families | [cdm/instrument.py](../cdm/instrument.py), [cdm/thermal.py](../cdm/thermal.py), [cdm/navigation.py](../cdm/navigation.py) |
| Runtime | [runtime.py](../experiments/lifetime/runtime.py) |
| Objectives | [objectives.py](../experiments/lifetime/objectives.py) |
| Trainer | [train.py](../experiments/lifetime/train.py) |
| Evaluation, probes and analysis | [evaluate.py](../experiments/lifetime/evaluate.py), [probe.py](../experiments/lifetime/probe.py), [branches.py](../experiments/lifetime/branches.py), [planner.py](../experiments/lifetime/planner.py) |
| Real time and timing | [realtime.py](../experiments/lifetime/realtime.py), [latency.py](../experiments/lifetime/latency.py) |
| Tests | [tests/](../tests) |

### e3. Continuity

The work ran through:

1. the maze planners and the continual agent;
2. the first lifetime round;
3. the corrections of [issue #1](https://github.com/Cherish0925/continual-decision-model/issues/1);
4. the second round;
5. this review.

Each step's records stay in the repository, with failed and superseded attempts reported as such.

### e4. Feedback and adjustment

| Review | Correction |
| --- | --- |
| PPO state reconstruction at resets; evaluation accounting | Corrected before the first round's reported results |
| Overstated causes and the name of the full setting | Rewritten as hypotheses; "full setting" is a setting, not the model |
| Prediction supervision versus readout | The `predict_hidden` baseline separated them |
| Real-time recording of the applied action | Corrected and re-measured |
| The zero-gradient term | Corrected and tested |
| The confounded refinement removal | Replaced by a frozen-workspace evaluation |
| Overreaching explanations: the training window as the cause of the bank's collapse; the environment as lacking any room for initiative | Withdrawn |

### e5. Route reconstruction

The route changes from adding mechanisms to self-made environments toward three things:

- a stated computational design;
- tasks where each capability is first shown to have room to pay;
- public tasks with strong methods as the main experiment.

The instrument family remains for analysing mechanisms.

### e6. Accumulated knowledge

The results, the rules in b4, the hypotheses in c6, and the trained models with their evaluations are kept here and in the releases, so that the next phase starts from verified objects rather than from memory.

## F. Results and transfer

### f1. Findings

1. Prediction supervision, not prediction readout, carries a recurrent controller's gain from action-conditioned prediction.
2. Training through the deployed sequence of learning steps turns an instance's own updates from slightly harmful into the source of half of its performance, on familiar and unfamiliar dynamics and against a real clock.
3. Persistent workspace computation pays in a trained model. Unrolling it across events and gating it did not add to that.
4. Counterfactual values of single activity choices are buried in trajectory variability beyond a few ticks.
5. A learned selector over adapter blocks collapses when trained only through short-window prediction.

### f2. Explanations

- **Why learning failed in the first round and works now.** The first-round objective trained the predictor and the step size for a sequence of updates the agent never took. This is supported by the comparison of objectives.
- **Why the bank collapsed.** No training signal within a window rewards keeping separate blocks. This is a hypothesis, not yet tested.

### f3. Methods

The methods in d1, available in this repository with tests.

### f4. A working result

On the instrument family, a controller learns during operation and halves its error relative to the same model without its updates. It answers in 3.6 ms for up to 64 instances at a 10 ms period while learning beside control on the same GPU.

### f5. Generalisation and transfer

- The gain from learning holds on held-out dynamics.
- One model runs three process families. It is behind specialists, and on thermal processes level with predictions only.
- No single-family model transfers to another family.

### f6. Value and impact

The phase established, with controls, that learning during operation can be made to pay by training through it. That is the core claim of CDM in its simplest form. It did not yet establish the other capabilities the project aims for, nor compare the method with existing ones on their tasks. Its lasting value depends on the next phase carrying it into that comparison.
