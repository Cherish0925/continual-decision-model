# Planning, metareasoning, active perception, and the reuse of computation

Research cutoff: 14 October 2026.

## The question CDM should inherit and advance

A Continual Decision Model must decide what to do in the environment and which internal operations are worth performing. These decisions interact: gathering evidence can improve the next plan; thinking consumes time during which the environment changes; an existing plan can provide useful structure for the next decision. The scientific object is therefore a coupled process of perception, computation, action, and adaptation.

This framing has substantial intellectual history. Its value for CDM is to identify mechanisms, assumptions, and experimental comparisons. A persistent neural state can make that history trainable and scalable, but persistence alone does not establish a new decision principle. The following sources trace the most relevant foundations and nearby approaches. Evidence levels and inspected locations are recorded in the accompanying structured bibliography.

## Computational effort as a decision

**Russell and Wefald, 1991 — [Principles of metareasoning](https://www.sciencedirect.com/science/article/pii/000437029190015C).** The foundational move is to treat computational actions as choices requiring justification. This supplies the conceptual basis for deciding which reasoning operation to perform. CDM should explain its learned approximation to that choice rather than present internal action selection itself as new.

**Horvitz, 2001 — [Principles and Applications of Continual Computation](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/11/cc_aij_horvitz.pdf).** Computation can prepare for uncertain future tasks, including during otherwise idle periods. Partial results can be cached and refined later. This is especially close to CDM's proactive ambition: useful thinking can precede the next externally presented problem.

**Hay et al., 2012 — [Selecting Computations: Theory and Applications](https://arxiv.org/abs/1207.5879).** Simulation selection is formulated as a Bayesian selection problem, with sampling results and practical approximations. Its lesson is that the usefulness of a computation depends on the decision it may change, rather than on uncertainty reduction alone.

**Lieder and Griffiths, 2017 — [Strategy selection as rational metareasoning](https://cocosci.princeton.edu/papers/Strategy%20selection%20as%20RM%20Preprint.pdf).** A model learns to predict strategies' performance and select favorable cost-benefit tradeoffs. Behavioral model comparisons and experiments address human strategy choice. This motivates a competence-sensitive computation policy, without prescribing a particular neural architecture.

**Callaway et al., 2018 — [Learning to select computations](https://arxiv.org/html/1711.06892v3).** Bayesian metalevel policy search approximates computation selection using value-of-information structure. The evaluation covers stopping, allocating effort between options, and planning. Counting the overhead of metareasoning is essential when CDM claims an efficiency gain.

**Lin et al., 2015 — [Metareasoning for Planning Under Uncertainty](https://www.ijcai.org/Proceedings/15/Papers/229.pdf).** Planning is explicitly costly while the physical situation can change. The paper formalizes this problem for MDPs and studies approximations based on BRTDP. Charging computation against environmental time has a clear predecessor.

**Godara, Aléman, and Yu, 2024/2025 — [Metareasoning in uncertain environments: a meta-BAMDP framework](https://arxiv.org/html/2408.01253v2).** The formulation incorporates unknown rewards or transitions into metareasoning. The inspected version develops approximations and Bernoulli-bandit applications. It connects computational uncertainty with uncertainty about the environment; those uncertainties should remain distinguishable in CDM.

## Learning how to use a world model

**Weber et al., 2017 — [Imagination-Augmented Agents for Deep Reinforcement Learning](https://arxiv.org/html/1707.06203v2).** A learned encoder extracts useful information from imagined trajectories and combines it with a direct policy path. Sokoban and MiniPacman experiments study imperfect predictions and transfer between tasks. Learned interpretation of simulations is already an established architectural idea.

**Pascanu et al., 2017 — [Learning model-based planning from scratch](https://arxiv.org/abs/1707.06170).** The Imagination-based Planner learns variable-length imagination, accumulates a plan context, and selects among previously imagined states while accounting for computation costs. This is a particularly close predecessor for learned management of deliberation. Evidence-driven revision across incoming observations requires a separate comparison.

**Chung, Anokhin, and Krueger, 2023 — [Thinker: Learning to Plan and Act](https://arxiv.org/abs/2307.14993).** Additional actions allow an agent to interact with a learned world model and learn planning behavior. Sokoban and Atari experiments demonstrate this approach. A CDM contribution must go beyond adding an internal action alphabet or replacing a fixed planner with a learned one.

**Labiosa and Hanna, 2026 — [When to Plan](https://arxiv.org/html/2607.16421v2).** A learned policy chooses reactive control or planning using uncertainty and time costs. The inspected experiments use an accurate supplied model, A*, depth-based cost proxies, and plans executed until option termination. These assumptions define useful comparisons for asynchronous, partially observed CDM experiments.

## Information gathering and preserving prior work

**Bajcsy, 1988 — [Active Perception](https://repository.upenn.edu/bitstreams/14686d28-7a8a-4554-beee-991eec61cbdd/download).** Perception includes control of data acquisition based on interpreted evidence, sensor models, and task goals. This is a foundation for deciding what to observe next. Streaming every available modality into a network does not by itself implement active perception.

**Bajcsy, Aloimonos, and Tsotsos, 2017 — [Revisiting active perception](https://link.springer.com/article/10.1007/s10514-017-9615-3).** This author retrospective organizes the purpose, content, method, time, and location of sensing. It helps specify CDM's perceptual choices and distinguishes sensor control from passive recognition. The journal article appeared online in 2017 and in a 2018 volume.

**Klenske and Hennig, 2016 — [Dual Control for Approximate Bayesian Reinforcement Learning](https://www.jmlr.org/papers/v17/15-162.html).** Actions can improve control and reveal uncertain dynamics. Approximate Bayesian methods produce structured exploration in simulated systems. This is a direct foundation for experiments whose actions both investigate and change a physical process.

**Koenig, Likhachev, and Furcy, 2004 — [Lifelong Planning A*](https://idm-lab.org/bib/abstracts/Koen04a.html).** LPA* reuses unaffected portions of previous search and repairs results when graph costs change. The inspected manuscript also discusses changes during search. Selectively revising unfinished reasoning is consequently a research direction with established algorithmic ancestry.

**Koenig and Likhachev, 2002 — [D* Lite](https://idm-lab.org/bib/abstracts/Koen02e.html).** Incremental heuristic search supports navigation and mapping in unknown terrain by retaining useful search information. It is an important reference baseline wherever CDM's proposed revision mechanism can be expressed as explicit graph replanning.

**Diehl, Bock, and Schlöder, 2005 — [A Real-Time Iteration Scheme for Nonlinear Optimization in Optimal Feedback Control](https://epubs.siam.org/doi/10.1137/S0363012902400713).** Newton-type updates refine control solutions while the process runs. The inspected scheme advances after one iteration per problem, with analysis and a kite-control example. Reusing optimization work while producing timely actions also has a mature control-theoretic lineage.

## What this means for the model design

Three distinctions should organize the architecture. A belief about the world is different from a partially completed computation about possible futures. Both differ from durable knowledge that survives the present episode. They may share learned representations, but their update rules and empirical roles should be identifiable. New observations update the world estimate; they can also invalidate a hypothetical branch or make an unfinished calculation valuable again.

A promising CDM hypothesis is to learn this interaction directly. A persistent workspace would preserve useful predictions, alternatives, and unresolved dependencies. An evidence-conditioned update would decide what to retain, revise, expand, or abandon. A computation policy would allocate effort according to downstream decision value. An action readout would use the currently available result while further updates proceed.

This is a proposed neural implementation and training problem. Classical incremental search already preserves valid work, model predictive control already updates ongoing optimization, and metareasoning already values computation. The potential contribution is a learned mechanism that accomplishes these operations with uncertain representations and models, across task families, at a measured cost. Its advantage must be demonstrated against those neighboring approaches.

Architecture alone cannot supply the training signal. The data must contain cases where additional observation helps, cases where additional thinking helps, and cases where acting immediately is preferable. It must also expose the consequences of retaining a mistaken prediction. A corpus containing only successful final actions leaves these alternatives largely unidentified. Simulator forks, recorded decision alternatives, and controlled changes to incoming evidence can make the distinctions learnable.

## Preserve the project scope; sharpen the first paper

The complete CDM goal remains a general, continuously operating model with persistent state, multimodal interaction, proactive computation and action, ongoing adaptation, and transferable learning. Applications include robotics, instruments, scientific experiments, edge systems, and digital environments. A first paper should establish one mechanism that contributes to this programme while explicitly retaining the broader goal.

A suitable first claim is: **learning to revise persistent deliberation under asynchronous evidence improves the achievable decision-quality versus latency tradeoff across several changing environments.** This is a hypothesis, not an established result or a claim of priority. A successful paper would specify the revision operator, the information it receives, the training objective, and what computation it actually saves. “Transformer plus RNN” would describe some implementation materials.

Use task families with different causes of change: uncertain physical parameters, newly revealed observations, delayed experimental outcomes, and changing goals. Separate variations that preserve most prior computation from variations requiring extensive reconstruction. The resulting performance curve is more informative than a single average score. A genuinely useful mechanism should adapt its degree of reuse to the kind of change.

The core comparison should include restarting deliberation, continuing it unchanged, ordinary recurrent updating, learned selective revision, and strong incremental or warm-start planners where applicable. Match information access, training data, backbone capacity, and compute accounting. Also compare complete practical systems on their own performance-cost frontiers; an architectural ablation and a competitive system comparison answer different questions.

## Evidence that would establish progress

The environment must advance during computation. Record observation time, arrival time, decision time, and action application time. Otherwise a faster software loop can be mistaken for a better model of real-time decision making. Include inference, encoding, memory access, and scheduling overhead when reporting elapsed time.

Test whether retained deliberation causally helps. Remove it, replace it with a state from another trajectory, or selectively alter evidence that should affect a particular prediction. These interventions can distinguish useful computational reuse from an ordinary memory-capacity effect. Evaluate new dynamics, observation schedules, goals, and task compositions rather than only new random seeds.

Report task outcome, action latency distributions, recovery after informative surprises, unnecessary computation, and total resource use. Distinguish adaptation through active state from learning that remains after that state is cleared. A revision mechanism is valuable if it preserves useful work and changes obsolete conclusions quickly; neither perpetual reuse nor perpetual recomputation is the desired behavior.

The strongest outcome would be a reproducible mechanism whose benefit survives these comparisons, together with an explanation of when it succeeds. The full CDM programme can then expand its modalities, environments, and learning timescales around an experimentally supported core.

