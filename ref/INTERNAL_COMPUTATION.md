# Internal computation and real-time decisions

This review covers primary sources on repeated computation, latent reasoning, computation selection, and asynchronous execution. Source editions and inspected sections are in [papers.json](papers.json).

## Four distinctions that determine the architecture

The word recurrence covers several different mechanisms.

| Mechanism | What changes | Question it addresses |
| --- | --- | --- |
| Recurrence across observations | A persistent state incorporates a new event and elapsed time. | What remains relevant from the past? |
| Recurrence across internal depth | Shared parameters repeatedly transform a representation. | What additional computation improves the current decision? |
| Adaptive stopping | A rule or learned policy ends an ongoing computation. | Is another iteration worthwhile? |
| Selection of cognitive activity | A policy chooses an observation, simulation, retrieval, computation, or action. | What should the model do next? |

These mechanisms can coexist. A persistent state may retain useful intermediate computation, and a looped attention block may update that state repeatedly. Architectural labels alone do not specify the relationship. In particular, a model can have adaptive depth while reconstructing its working state for each query, or persistent memory while performing only a fixed amount of reasoning.

Three clocks also need separate treatment: arrival of environmental events, execution of internal operations, and changes to learned parameters. A computation can advance the second clock while the first continues running. CDM needs an explicit account of all three, including how their interaction is learned.

## Established foundations: persistence and adaptive depth

[Adaptive Computation Time](https://arxiv.org/abs/1603.08983) introduced a differentiable way for an RNN to learn how many internal steps to perform between receiving input and producing output. Its algorithmic and language-modeling experiments establish adaptive computation as an older foundation, rather than a new consequence of recent reasoning models.

[Universal Transformers](https://arxiv.org/abs/1807.03819) combine shared self-attentive transformations with recurrence across depth and dynamic per-position halting. Consequently, attention plus recurrence plus adaptive stopping is already an established architectural combination.

[PonderNet](https://arxiv.org/abs/2107.05407) learns computational effort through an objective balancing prediction, cost, and generalization. It is a useful stopping-policy reference, especially when testing whether extra iterations provide a systematic benefit.

[Deep Repeated ConvLSTM](https://arxiv.org/abs/1901.03559) is particularly relevant: its state survives between observations, and the same recurrent stack executes multiple times within each environment step. The paper studies learned planning behavior in combinatorial environments and selected Atari games. Both temporal persistence and internal iteration therefore belong in the baseline set.

## Recent latent reasoning: what each mechanism contributes

[Coconut](https://arxiv.org/abs/2412.06769) feeds hidden representations back as input embeddings instead of decoding every intermediate step into language. This provides a direct route around mandatory verbalization; its reported experiments concern reasoning tasks.

[Huginn and recurrent-depth language modeling](https://arxiv.org/abs/2502.05171) study repeated latent computation at substantially larger language-model scale. The reported 3.5-billion-parameter model shows that this direction extends beyond tiny task-specific networks.

[Continuous Thought Machines](https://arxiv.org/abs/2505.05522) introduce neuron-specific temporal processing and synchronization-based representations. The [official technical exposition](https://pub.sakana.ai/ctm/) explains internal ticks that need not follow the input sequence. CTM is a useful architectural alternative, particularly for studying what the dynamics themselves represent.

[HRM](https://arxiv.org/abs/2506.21734) uses interacting recurrent modules with different update frequencies. [TRM](https://arxiv.org/abs/2510.04871) simplifies recursive reasoning to a small network while reporting strong puzzle performance. Together they motivate testing hierarchy, rather than assuming that more cognitive modules necessarily produce better reasoning. Their task results do not establish general competence across scientific research, language, and robotics.

[T²MLR](https://arxiv.org/abs/2607.15178) adds a pathway from the previous token's middle-layer representation into an earlier layer at the next position. This explicitly carries intermediate computation across decoding steps and includes a reported pretrained-model retrofit. It is directly relevant to designs that preserve more than emitted tokens.

[BDH-CQ](https://arxiv.org/abs/2608.09888) distinguishes recurrent contextual memory from a latent reasoning workspace: demonstrations update the former, then the latter iterates on a query. This is a close conceptual neighbor for memory-and-reasoning architectures. Its evaluated setting is public ARC and controlled visual transformations; detailed update rules and the complete training recipe remain proprietary.

The consequential design choice is what persists and what that persistence is trained to achieve. Reusing parameters reduces model size; retaining intermediate work may reduce repeated computation; retaining observations supports inference across time.

## Choosing computation: a substantial existing research line

[Metacontrol for Adaptive Imagination-Based Optimization](https://arxiv.org/abs/1705.02670) already learns both how many internal simulations to perform and which predictive expert to consult. The controller accounts for reliability and computation cost. Choosing the content of thought has therefore been a concrete learning problem for years.

[Thinker](https://arxiv.org/abs/2307.14993) gives agents actions for interacting with a learned world model and learns planning behavior through reinforcement learning. It supplies a natural reference for deciding which imagined trajectories to explore.

[Learning to Select Computations in Recurrent Neural Circuits](https://www.biorxiv.org/content/10.64898/2026.04.14.718499v1) offers a complementary cognitive-modeling direction. The inspected [author repository](https://github.com/SixingChen028/recurrent-meta-reasoning) provides recurrent actor-critic implementations for choice and planning case studies. This review uses that official description; the paper's full text was inaccessible.

[Looped Actor](https://arxiv.org/abs/2609.37432) brings adaptive Transformer depth directly into reinforcement-learning policies. It stops when consecutive action distributions satisfy a predefined KL-stability threshold and evaluates across twenty-two tasks. Distributional stability is a useful operational criterion, but it is not the same as estimating the expected benefit of another computation.

[Finding the Time to Think](https://arxiv.org/html/2606.26463v1) studies planning while the environment continues advancing. A learned gate assigns budgets to a frozen MCTS planner while a fast policy supplies actions. Its explicit extensions include joint planner training and learned dynamics.

For CDM, the remaining question is how these choices participate in a single adaptable model over long experience. The model must discover which internal work matters, connect it to incoming observations, and transfer the resulting strategies.

## Real-time execution changes the learning problem

[Real-Time Chunking](https://arxiv.org/abs/2506.07339) generates a future action chunk while executing the current one, preserving already committed actions. It demonstrates a practical asynchronous interface for diffusion and flow policies.

[Legato](https://arxiv.org/abs/2602.12978) trains continuation into flow policies using partially known action sequences and varying schedules. Its relevance is the alignment between training and the circumstances of deployment.

[FASTER](https://arxiv.org/abs/2603.19199) prioritizes near-term actions during flow sampling and studies effective reaction latency.

[Real-Time Execution with Autoregressive Policies](https://arxiv.org/abs/2606.13355) explores tokenization horizons and constrained decoding for asynchronous control. Real-time requirements therefore do not uniquely select a flow decoder or exclude an autoregressive one.

CDM should evaluate the complete observation-to-action path. A rapid policy head can still use stale visual features; an efficient reasoning block can still monopolize the accelerator; a long action chunk can continue smoothly after its assumptions become obsolete. Measure when new observation changes the action actually applied, alongside task outcome, resource use, and useful internal progress.

## Reusing computation under changing inputs

[Incrementally-Computable Neural Networks](https://arxiv.org/abs/2307.14988) discretizes intermediate activations to limit recomputation after input edits. Its document-classification setting isolates computational reuse from control. CDM should measure whether retained calculations improve later actions, alongside their update cost.

[LAP](https://arxiv.org/abs/2505.03077) retains a latent-plan posterior and updates it with each new observation segment. Its reported replanning uses one gradient step after a longer initialization. [A2C2](https://arxiv.org/abs/2509.23224) instead corrects each action from fresh observations while keeping the base policy fixed. Compare persistent internal revision with this direct correction path.

[Are Large Reasoning Models Interruptible?](https://arxiv.org/abs/2510.11713) evaluates mathematics and programming under interrupted generation and in-flight context changes. This motivates varying when an observation arrives, rather than evaluating only a completed answer under a fixed input.

## CDM computation choices

Compare persistent recurrence, shared latent refinement, explicit imagined trajectories, and observation-conditioned revision. Train their use on tasks where uncertainty can come from missing observations or from calculations that have not yet been performed.

Additional computation is valued through later decisions and its elapsed time. The model should continue useful work, revise assumptions after observations, and request information when further inference cannot resolve the missing quantity. Test both fixed-depth and learned activity policies.

The current computational-memory comparison is in [COMPARISON.md](COMPARISON.md), and the integrated architecture is in [DESIGN.md](../DESIGN.md). The earlier low-rank revision study remains in [COMPUTATION_TRANSPORT.md](../research/COMPUTATION_TRANSPORT.md).
