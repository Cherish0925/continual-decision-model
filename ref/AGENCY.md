# Proactivity, Exploration, and Agent Systems

## 1. What it means to initiate activity

CDM should select useful next activities from its goals, available evidence, and anticipated consequences. This includes obtaining information, refining an internal representation, acting, and acquiring skills that help later. Several literatures already study parts of this behavior, with different choices about where the intelligence resides and what is learned.

The important comparison is operational. Which goals are supplied? Which subgoals are generated? Which choices are learned? What experience persists? How does a choice affect later competence? These questions distinguish a trainable model contribution from a runtime convention without discounting the value of either.

## 2. Intrinsic motivation and self-generated curricula

[Intrinsically Motivated Goal Exploration Processes](https://arxiv.org/abs/1708.02190) organize self-generated goals, intrinsic selection, exploration, and reuse of experience across goals. Their experiments include autonomous skill discovery with a humanoid robot. This is a direct predecessor for claims that a model can choose its own learning activities.

[Curiosity-driven Exploration by Self-supervised Prediction](https://arxiv.org/abs/1705.05363) uses prediction error in a learned feature space as an exploration signal, evaluated in visual game environments. [Random Network Distillation](https://arxiv.org/abs/1810.12894) instead uses error in predicting a fixed random network's features and studies difficult Atari exploration. These are concrete, different ways to make novelty affect behavior.

[DIAYN](https://arxiv.org/abs/1802.06070) discovers diverse skills through an information-theoretic objective and evaluates their downstream use in simulated control. [POET](https://arxiv.org/abs/1901.01753) co-evolves challenges and their solutions, allowing useful solutions to transfer between environments. They show why learning activities and their ordering deserve explicit study.

For CDM, the useful question is how such objectives interact with an ongoing task. A model may seek evidence to improve its present action, practice a skill for later tasks, or explore a promising subgoal. Those choices have different horizons and should be logged and evaluated separately.

We propose an experiment that separates predictable but irrelevant variation, irreducible noise, and information that changes an effective action. Compare novelty, prediction error, learning progress, and downstream decision benefit as activity-selection signals. This is a proposed CDM comparison; no single cited exploration result establishes the best signal for every environment.

## 3. Language-model agents and persistent activity

[ReAct](https://arxiv.org/abs/2210.03629) interleaves language reasoning with actions that obtain external information, evaluated on knowledge and interactive tasks. It is a clear precedent for connecting deliberation, observation, and action within one trajectory.

[Reflexion](https://arxiv.org/abs/2303.11366) retains verbal reflections on feedback in episodic memory to improve later attempts, without updating the base model's weights. Its evidence includes decision-making, coding, and reasoning. It makes the lifetime and location of adaptation especially clear.

[Generative Agents](https://arxiv.org/abs/2304.03442) combine experience records, retrieval, reflection, and planning in an interactive social simulation. Their evaluation concerns believable individual and social behavior. This is relevant to persistent agendas and spontaneous activity, with a different primary objective from optimal physical control.

[MemGPT](https://arxiv.org/abs/2310.08560) manages tiers of memory and control flow around a language model, evaluated on document analysis and multi-session conversation. [Voyager](https://arxiv.org/abs/2305.16291) combines an automatic curriculum, a code-based skill library, and feedback-driven program improvement in Minecraft. These demonstrate forms of continuity and skill accumulation at the agent-system level.

Such systems are useful baselines and possible sources of trajectories. Their external records can be effective memory, and their explicit intermediate actions can reveal how observations influence later decisions. CDM asks an additional model question: which of these functions can be learned as efficient state transitions and activity policies, with measurable generalization?

## 4. A fair boundary between model and runtime

An external loop is necessary to run any interactive model. Its existence does not determine whether a contribution is architectural. The decisive issue is what the loop specifies and what the model learns.

For each implementation, describe who chooses the next observation, the computation budget, the memory operation, the subgoal, and the action. Record which decisions use learned parameters and which follow fixed rules. If a language model chooses a tool or a goal, that is still a learned decision; the comparison should identify its cost and representation rather than dismissing it as a wrapper.

Similarly, persistent latent state is a design choice, not an automatic claim to greater intelligence. Compare a CDM with a strong language-agent baseline that receives the same observations, tools, history access, and time budget. Include an untimed quality comparison to distinguish loss of reasoning quality from delay-induced loss of task performance.

## 5. What proactivity should demonstrate

A convincing result would show purposeful initiation without an instruction for every step. Test whether the model acquires useful evidence before a consequential decision, uses waiting intervals productively, chooses intermediate goals that improve later progress, and abandons low-value activities when circumstances change.

Measure effects over complete trajectories. The number of self-initiated actions is not a sufficient score: a model can be active while making little progress. Relevant measurements include task utility, information and experiment costs, competence acquired, and transfer of the activity-selection strategy.

The broader CDM objective includes both externally supplied goals and learned selection of subgoals or learning opportunities. The first experiments can fix the overarching goal while studying meaningful initiative within it. Later experiments can examine how a repertoire of learned goals and skills develops across tasks. This preserves the ambition while making each capability identifiable.

## Reading depth

The ten source descriptions in this review are based on their primary abstracts and publication records. They establish the reported mechanisms and broad evaluation settings. Reproducing a specific agent or exploration algorithm requires reading and pinning its method, prompts or objectives, implementation, and evaluation configuration before using it as an experimental baseline.
