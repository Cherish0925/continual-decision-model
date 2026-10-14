# Research Programme

The [reference collection](ref/README.md) develops the background for this programme, including [research positioning](ref/POSITIONING.md), [closest-work comparisons](ref/COMPARISON.md), and [experiments for choosing the architecture](ref/EXPERIMENTS.md).

## 1. The scientific objective

Continual Decision Model research asks how a single trainable model can sustain useful activity as observations, goals, and environmental dynamics change. The target is a model that maintains an evolving understanding, initiates purposeful activity, learns from outcomes, and transfers that learning to subsequent situations.

The central unit of study is a **trajectory of decisions through time**. An individual answer or action captures only part of the capability. The important questions concern what the model remembers, what it chooses to investigate, how it responds to interruption, and whether experience improves later behavior.

This objective joins three problems: representation of an ongoing situation, selection of external and internal activities, and adaptation across experience. Their interaction is the main research subject. A better memory may reduce the need for computation; a better experiment may simplify a decision; a better learned skill may make a previously expensive plan immediate.

## 2. Persistent state as an evolving belief

The model rarely observes the full state of its environment. Its internal state must therefore represent what is known, what remains uncertain, and what may have changed since the last observation.

Useful state includes several kinds of information:

- Beliefs about entities, relationships, hidden dynamics, and likely outcomes.
- Current goals, intermediate intentions, and criteria for progress.
- Pending actions, outstanding measurements, and unresolved dependencies.
- Recent evidence and references to relevant earlier experience.
- Estimates of uncertainty and the resources needed for the next activity.

These are functional requirements for learned representations. They need not become separate prose fields or manually defined modules.

**Hypothesis:** learned recurrent state can preserve decision-relevant information over long interactions more efficiently than repeatedly reconstructing it from a fixed context window.

**Evidence:** performance as the delay between relevant evidence and its use increases, together with state size, retrieval cost, and observation-to-action latency. Controlled state resets and withheld memory access reveal which mechanism carries the information. An advantage must survive comparison with a Transformer given an equivalent resource budget.

## 3. Proactivity as activity selection

A proactive model chooses activities because they advance a goal or resolve uncertainty that matters to future decisions. It may inspect a sensor, ask a question, retrieve a previous result, think further, act immediately, or wait for an already running experiment.

This requires more than an internal clock that repeatedly invokes inference. The selection policy must learn the consequences of initiating, continuing, switching, and stopping activities. Choosing a productive subgoal is part of the same problem: the model should connect an intermediate investigation to the larger objective it serves.

**Hypothesis:** learning activity selection improves task progress per unit of time and information cost compared with a fixed observation and computation schedule.

**Evidence:** compare learned and fixed schedules using the same backbone, sensors, tools, and total budgets. Include situations in which further thought helps, situations in which fresh evidence helps, and situations in which immediate action is best. Evaluate waiting as a legitimate choice when the next useful event is already on its way.

An initial training signal will value achieved outcomes and charge for consumed resources. Predicted information value can guide exploration; its contribution should be checked against actual later improvement. Otherwise, an interesting-looking signal can distract the model from the investigation it is meant to advance.

## 4. Thought, prediction, and the world

Internal computation should support actionable distinctions: which explanation fits the evidence, which outcome follows an intervention, which observation would separate two possibilities, and which plan remains useful after a change.

Two forms of recurrence deserve separate study. **Recurrence across events** carries experience forward. **Recurrence within computation** develops a representation before the next external decision. Their interaction could let a model pause an investigation, incorporate new information, and resume with revised assumptions.

**Hypothesis:** a model that can vary internal computation will achieve a better trade-off between task quality and response time than one using the same depth for every decision.

**Evidence:** measure complete quality–latency curves, including the cost of selecting computation depth. Compare additional latent steps, additional retrieved evidence, and additional environmental observations. Intervene on intermediate representations and predictions to determine what information makes later decisions better.

Action-conditioned prediction provides a concrete purpose for this computation. The model should learn how interventions change future observations and task outcomes. Dreamer demonstrates learning behavior through a world model [4]; latent recurrence offers separate examples of computation beyond a single forward pass [5, 6]. CDM will investigate how these ideas work together under an advancing environmental clock.

## 5. Intelligence under time constraints

Inputs arrive at different rates. A camera frame, a spoken instruction, and a delayed experimental result have different temporal meanings. A model also spends time encoding observations, recalling information, computing alternatives, and transmitting actions.

Real-time capability must therefore be defined relative to a task: a useful decision arrives before its opportunity expires. Throughput alone does not establish this. A model can process many observations while acting on an outdated situation.

**Hypothesis:** persistent state and selective computation can support responsive action while preserving the benefits of deeper reasoning when time permits.

**Evidence:** vary input rates, observation delays, inference costs, and action deadlines. Report task outcomes alongside latency distributions, deadline misses, and the age of the evidence underlying executed actions. Asynchronous action execution is an established research direction [7]; CDM also asks how the model should choose what to compute while execution continues.

## 6. Experience that changes future behavior

Continuity has three distinct mechanisms:

| Mechanism | What changes | How to measure its contribution |
| --- | --- | --- |
| State adaptation | Activations representing the current situation | Freeze parameters; compare retained and reset state. |
| Memory accumulation | Stored episodes, observations, or reusable records | Compare access to matched relevant and irrelevant experience. |
| Parameter learning | The functions that predict and select behavior | Reset transient state and compare performance before and after training. |

A successful programme should connect all three. Immediate adaptation keeps the current investigation coherent; memory preserves useful experience; learning makes future tasks easier.

Continual learning also requires attention to both retained competence and capacity to learn new things. Loss of plasticity has been demonstrated in continual supervised and reinforcement learning settings [8]. This motivates experiments that measure the learning curve throughout a long task sequence, including revisits and genuinely new task families.

**Hypothesis:** combining selective experience replay with ongoing adaptation improves learning speed on later tasks while maintaining earlier skills.

**Evidence:** compare frozen parameters, memory-only adaptation, and parameter updates under equal interaction and training budgets. Report forward transfer, retention, adaptation speed, and performance after repeated distribution changes. State changes alone establish online adaptation; a lasting improvement that survives state reset establishes a different result.

## 7. Self-assessment and generalization

The model needs estimates of its own uncertainty and expected effectiveness. These estimates are useful when they change a decision appropriately: requesting a measurement, spending another computation step, revising a plan, or recognizing that a previous method already applies.

We will operationalize self-assessment through predictions of task success, remaining effort, and the value of an additional activity. Calibration and intervention experiments can test these predictions directly. Verbal confidence is one possible output; the primary evidence is how accurately the model allocates effort.

Generalization must cover more than unseen random seeds. Important shifts include new dynamics, new combinations of goals and tools, altered observation timing, unfamiliar sensor combinations, and transfer between task families. Experiments should distinguish performance without adaptation, performance with a fixed adaptation budget, and performance after further training.

## 8. A connected experimental programme

Begin with a continuously running experimental instrument whose dynamics must be inferred from costly and delayed measurements. This makes state, observation choice, intervention, internal computation, and learning measurable in one setting.

Extend the same mechanisms to embodied control and digital investigations. Each extension adds a specific demand: richer perception, continuous actions, external tools, language goals, or longer dependencies. Shared mechanisms and explicit adaptation budgets make the comparison scientifically useful.

The programme should produce four kinds of evidence:

1. **Mechanism:** a controlled comparison shows which model component causes an improvement.
2. **Efficiency:** the improvement persists when interaction, compute, and elapsed time are counted.
3. **Continuity:** useful behavior and learning persist across long runs and repeated changes.
4. **Transfer:** the mechanism remains valuable outside the task family on which it was developed.

The [first technical design](DESIGN.md) turns this programme into an initial architecture and experiment. Its proposed contribution is learned coordination of persistent state, internal computation, and action under a shared time budget. Whether that coordination yields a useful model family is the empirical question.

## Selected foundations

These sources motivate specific design choices; this is a focused starting bibliography.

1. Dao and Gu. **Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality.** 2024. [Paper](https://arxiv.org/abs/2405.21060). Structured state space computation and Mamba-2.
2. De et al. **Griffin: Mixing Gated Linear Recurrences with Local Attention for Efficient Language Models.** 2024. [Paper](https://arxiv.org/abs/2402.19427). Combining recurrence with local attention.
3. Peng et al. **RWKV-7 "Goose" with Expressive Dynamic State Evolution.** 2025. [Paper](https://arxiv.org/abs/2503.14456). An alternative recurrent architecture with expressive state updates.
4. Hafner et al. **Mastering diverse control tasks through world models.** 2025. [Paper](https://www.nature.com/articles/s41586-025-08744-2). Learning prediction and behavior through imagined trajectories.
5. Geiping et al. **Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach.** 2025. [Paper](https://arxiv.org/abs/2502.05171). Shared recurrent computation in latent space.
6. Darlow et al. **Continuous Thought Machines.** 2025. [Research page](https://pub.sakana.ai/ctm/). Internal computation through neural dynamics and synchronization.
7. Black, Galliker, and Levine. **Real-Time Execution of Action Chunking Flow Policies.** 2025. [Paper](https://arxiv.org/abs/2506.07339). Asynchronous execution under inference delay.
8. Dohare et al. **Loss of plasticity in deep continual learning.** 2024. [Paper](https://www.nature.com/articles/s41586-024-07711-7) · [Authors' implementation](https://github.com/shibhansh/loss-of-plasticity). Learning capacity over extended task sequences.
