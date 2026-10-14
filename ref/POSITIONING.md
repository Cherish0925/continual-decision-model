# What CDM Is Trying to Achieve

## 1. The full objective

CDM aims to develop a general model whose perception, internal computation, decisions, and learning form an ongoing process. The model maintains an evolving state, receives asynchronous information, initiates useful activities, produces timely actions or communication, and improves through experience.

The objective concerns the model's learned computation. A runtime still delivers observations, advances clocks, exposes tools, and executes outputs. The research question is how state updates and activity choices become trainable capabilities that transfer across environments.

The target includes eight connected properties:

| Property | Operational meaning |
| --- | --- |
| Persistence | Relevant beliefs, goals, and computational progress survive individual outputs and pauses in observation. |
| Multimodality | Inputs and outputs can use the representations appropriate to text, images, sound, video, sensors, and control. |
| Proactivity | The model initiates observations, investigations, subgoals, or actions from its current objectives and understanding. |
| Internal computation | The model can develop and compare representations without requiring each intermediate step to be verbalized. |
| Timeliness | Useful decisions reach the environment while their opportunity still exists. |
| Learning | Feedback improves predictions and behavior, including competence on later tasks. |
| Self-assessment | Estimates of uncertainty, progress, and expected computational benefit guide subsequent activity. |
| Generalization | Shared mechanisms remain useful under new goals, dynamics, sensory conditions, and task families. |

These properties reinforce one another. Better learning can make later decisions faster. Better observation can replace expensive inference. Better memory can preserve an investigation through interruptions. The architectural challenge is to make these interactions productive within finite resources.

## 2. The distinctions that keep the research coherent

**Persistence has several meanings.** A recurrent activation, an attention cache, an external record, and a modified parameter can all carry information. They differ in capacity, access, update cost, and the conditions under which they are retained. The experiment must specify each one's lifetime.

**State propagation, internal depth, and online adaptation are different operations.** Event recurrence carries information across observations. Depth recurrence spends more computation on a representation. Online parameter or fast-weight adaptation changes a function using experience. One architecture may support all three; success on one does not measure the others.

**Proactivity concerns selection.** A periodic inference call establishes a schedule. A proactive policy chooses which activity to initiate, why it is useful, and when to change direction. Self-generated subgoals remain accountable to the larger task or learning objective.

**Real time is a relationship to the environment.** It depends on sensing, inference, execution, and deadlines together. A high output rate can coexist with poor response to new evidence. Both the action rate and the event-to-effective-response delay matter.

**Adaptation and lasting learning need different experiments.** Improvement with a retained working state can be valuable. Improvement that survives a working-state reset and transfers to another task establishes an additional capability. Neither result should be relabeled as the other.

## 3. The research programme, an architectural hypothesis, and a paper

| Level | Current formulation | What establishes progress |
| --- | --- | --- |
| Programme | General, persistent, proactive, adaptive decision models | Accumulating evidence across mechanisms, timescales, and domains |
| Architectural hypotheses | Compare shared recurrent state, revisable computational state, explicit learned planning, and adaptive parameter memory | Specified mechanisms that improve behavior under controlled comparisons |
| Candidate first paper | Test one mechanism, such as learned revision of ongoing latent computation, in changing, partially observed environments | Reproducible mechanisms, strong baselines, matched resources, and transfer experiments |

The first paper is an entry into the programme. Its environment and mechanism do not define the entire model's intended scope. Robotics, physical experimentation, digital investigation, and local intelligence remain target domains even if initial experiments isolate a smaller set of mechanisms.

The architectural hypothesis is revisable. Alternatives include a single unpartitioned recurrent state, memory-augmented attention, learned planning with an explicit world model, and parameter-memory architectures. Their performance and explanatory value should determine which structure becomes central.

## 4. What we inherit

There are strong precedents for every broad ingredient. Sequence research supplies attention, recurrence, and editable memory. Metareasoning studies the value and choice of computations. Incremental planning reuses valid earlier work. World models support predictions of action consequences. Meta-learning and continual learning address adaptation and retained competence. Agent systems show how memories, tools, and curricula can sustain activity.

Two recent overlaps are particularly instructive. [Continuous Memory Machines](https://arxiv.org/abs/2610.07907) jointly update matrix-valued short- and long-term stores using a Transformer. [BDH-CQ](https://arxiv.org/abs/2608.09888) combines contextual recurrent memory with iterative latent reasoning. These make a generic claim about separating memory from computation insufficient to distinguish CDM.

Likewise, [Finding the Time to Think](https://arxiv.org/html/2606.26463v1) already studies learned planning budgets while the environment advances. [Horvitz's continual computation](https://www.microsoft.com/en-us/research/publication/principles-applications-continual-computation/) supplies an older decision-theoretic lineage for allocating computation over time. The opportunity is to specify a further mechanism and test its incremental contribution.

Attribution should follow the actual dependency: architecture, objective, algorithm, training recipe, code, and benchmark may each come from different sources. A combination can support a research contribution when its interaction solves a defined problem and the evidence explains why. Citation acknowledges inheritance; experiments establish the new result.

## 5. The current mechanism hypothesis

Maintain a learned representation of the observed situation and a persistent representation of an unfinished investigation. New evidence should change both according to its implications. The model learns which intermediate predictions or candidate actions remain useful, which need revision, and where subsequent computation should go.

The proposed distinction between situation state and computational state is functional. It may be implemented through separate slots, separate update paths, or a shared state with learned organization. A two-store design earns its complexity only if it improves outcomes over a matched single-state model.

The central object of study is an **evidence-conditioned update of ongoing computation**. Its inputs may include the current belief, the new observation, executed actions, elapsed time, goals, and the previous computational state. Its outputs determine revised computation and subsequent activity. Observable and imagined information retain their different roles in action-conditioned prediction.

An illustrative case is an instrument comparing possible explanations for a response. While a measurement is pending, the model evaluates candidate interventions. New evidence changes one relevant parameter. The model should use that evidence to update the affected predictions and continue the investigation with appropriate actions. The important result would be better decisions for the same elapsed time and computational resources.

This hypothesis becomes original research only through the particular representation, update rule, training signal, and empirical findings. Naming a revision operation does not establish a new algorithm. The [comparison](COMPARISON.md) includes classical as well as neural alternatives that address parts of this problem.

## 6. Other substantive research opportunities

**Decision-relevant state formation.** Train state retention using its contribution to later decisions and transfer across possible future goals. Test whether this differs usefully from observation reconstruction or generic associative recall at the same capacity.

**Choice of information versus computation.** Train the model to distinguish uncertainty that additional inference can resolve from uncertainty that requires a new measurement. This connects activity selection to the information structure of the task.

**Learning which experience deserves further learning.** Once ongoing parameter adaptation has measurable value, study selective replay, learning schedules, or update rules using future task performance. Existing fast-weight and meta-learning mechanisms are essential comparisons.

These are connected hypotheses, not simultaneous prerequisites for one paper. Each needs its own causal comparison before a combined model can support a broader claim.

## 7. Value and application

The practical value is sustained useful decisions under change. A robot can benefit from accumulated scene understanding and plans that adapt during execution. An experimental model can use waiting intervals productively and transfer earlier findings to subsequent investigations. An edge model can allocate limited computation to the events that matter. A digital research model can maintain unresolved questions and continue work when evidence arrives.

The scientific value would be a better account of how learned state, computation, and adaptation cooperate over time, supported by mechanisms that remain effective across domains. The project should be judged by those improvements, with implementation scale and architectural labels serving the research rather than defining it.
