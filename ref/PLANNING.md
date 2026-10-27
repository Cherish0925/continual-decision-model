# Planning, computation selection, and active perception

Planning and observation consume time while the environment changes. The methods below study the value of internal computation, active acquisition of information, and reuse of prior planning. The catalogue records reading depth and source editions.

## Computational effort as a decision

**Russell and Wefald, 1991 — [Principles of metareasoning](https://www.sciencedirect.com/science/article/pii/000437029190015C).** The foundational move is to treat computational actions as choices requiring justification. This supplies the conceptual basis for deciding which reasoning operation to perform. CDM should explain its learned approximation to that choice rather than present internal action selection itself as new.

**Horvitz, 2001 — [Principles and Applications of Continual Computation](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/11/cc_aij_horvitz.pdf).** Computation can prepare for uncertain future tasks, including during otherwise idle periods. Partial results can be cached and refined later. This is especially close to CDM's proactive ambition: useful thinking can precede the next externally presented problem.

**Hay et al., 2012 — [Selecting Computations: Theory and Applications](https://arxiv.org/abs/1207.5879).** Simulation selection is formulated as a Bayesian selection problem, with sampling results and practical approximations. Its lesson is that the usefulness of a computation depends on the decision it may change, rather than on uncertainty reduction alone.

**Lieder and Griffiths, 2017 — [Strategy selection as rational metareasoning](https://cocosci.princeton.edu/papers/Strategy%20selection%20as%20RM%20Preprint.pdf).** A model learns to predict strategies' performance and select favorable cost-benefit tradeoffs. Behavioral model comparisons and experiments address human strategy choice. This motivates a competence-sensitive computation policy, without prescribing a particular neural architecture.

**Callaway et al., 2018 — [Learning to select computations](https://arxiv.org/html/1711.06892v3).** Bayesian metalevel policy search approximates computation selection using value-of-information structure. The evaluation covers stopping, allocating effort between options, and planning. Include the computation-selection overhead in runtime measurements.

**Lin et al., 2015 — [Metareasoning for Planning Under Uncertainty](https://www.ijcai.org/Proceedings/15/Papers/229.pdf).** Planning is explicitly costly while the physical situation can change. The paper formalizes this problem for MDPs and studies approximations based on BRTDP. Charging computation against environmental time has a clear predecessor.

**Godara, Aléman, and Yu, 2024/2025 — [Metareasoning in uncertain environments: a meta-BAMDP framework](https://arxiv.org/html/2408.01253v2).** The formulation incorporates unknown rewards or transitions into metareasoning. The inspected version develops approximations and Bernoulli-bandit applications. It connects computational uncertainty with uncertainty about the environment; those uncertainties should remain distinguishable in CDM.

## Learning how to use a world model

**Magnino et al., 2026 — [Learning Robot Metareasoning: Adaptive Computation Budgets for Real-Time Control](https://arxiv.org/html/2610.23662v1).** This preprint couples action commitment and solver time in a semi-Markov policy, using state, the executing plan and solver statistics. The inspected formulation assumes an optimizer whose solutions improve with additional work; reported applications include navigation, quadrotor control and manipulation. For CDM, this identifies two separate learning questions: making internal computation improve a decision, and choosing its duration while acting. The current DiscoveryWorld experiment addresses the former; independently timed process tasks are needed for the latter. Sections I–V inspected on 27 October 2026.

**Weber et al., 2017 — [Imagination-Augmented Agents for Deep Reinforcement Learning](https://arxiv.org/html/1707.06203v2).** A learned encoder extracts useful information from imagined trajectories and combines it with a direct policy path. Sokoban and MiniPacman experiments study imperfect predictions and transfer between tasks. Learned interpretation of simulations is already an established architectural idea.

**Pascanu et al., 2017 — [Learning model-based planning from scratch](https://arxiv.org/abs/1707.06170).** The Imagination-based Planner learns variable-length imagination, accumulates a plan context, and selects among previously imagined states while accounting for computation costs. This is a particularly close predecessor for learned management of deliberation. Observation-conditioned revision across incoming observations requires a separate comparison.

**Chung, Anokhin, and Krueger, 2023 — [Thinker: Learning to Plan and Act](https://arxiv.org/abs/2307.14993).** Additional actions allow an agent to interact with a learned world model and learn planning behavior. Sokoban and Atari experiments demonstrate this approach. A CDM contribution must go beyond adding an internal action alphabet or replacing a fixed planner with a learned one.

**Labiosa and Hanna, 2026 — [When to Plan](https://arxiv.org/html/2607.16421v2).** A learned policy chooses reactive control or planning using uncertainty and time costs. The inspected experiments use an accurate supplied model, A*, depth-based cost proxies, and plans executed until option termination. These assumptions define useful comparisons for asynchronous, partially observed CDM experiments.

## Information gathering and preserving prior work

**Bajcsy, 1988 — [Active Perception](https://repository.upenn.edu/bitstreams/14686d28-7a8a-4554-beee-991eec61cbdd/download).** Perception includes control of data acquisition based on interpreted observations, sensor models, and task goals. This is a foundation for deciding what to observe next.

**Bajcsy, Aloimonos, and Tsotsos, 2017 — [Revisiting active perception](https://link.springer.com/article/10.1007/s10514-017-9615-3).** This author retrospective organizes the purpose, content, method, time, and location of sensing. It helps specify CDM's perceptual choices and distinguishes sensor control from passive recognition. The journal article appeared online in 2017 and in a 2018 volume.

**Klenske and Hennig, 2016 — [Dual Control for Approximate Bayesian Reinforcement Learning](https://www.jmlr.org/papers/v17/15-162.html).** Actions can improve control and reveal uncertain dynamics. Approximate Bayesian methods produce structured exploration in simulated systems. This is a direct foundation for experiments whose actions both investigate and change a physical process.

**Koenig, Likhachev, and Furcy, 2004 — [Lifelong Planning A*](https://idm-lab.org/bib/abstracts/Koen04a.html).** LPA* reuses unaffected portions of previous search and repairs results when graph costs change. The inspected manuscript also discusses changes during search. Selectively revising unfinished reasoning is consequently a research direction with established algorithmic ancestry.

**Koenig and Likhachev, 2002 — [D* Lite](https://idm-lab.org/bib/abstracts/Koen02e.html).** Incremental heuristic search supports navigation and mapping in unknown terrain by retaining useful search information. It is an important reference baseline wherever CDM's proposed revision mechanism can be expressed as explicit graph replanning.

**Diehl, Bock, and Schlöder, 2005 — [A Real-Time Iteration Scheme for Nonlinear Optimization in Optimal Feedback Control](https://epubs.siam.org/doi/10.1137/S0363012902400713).** Newton-type updates refine control solutions while the process runs. The inspected scheme advances after one iteration per problem, with analysis and a kite-control example. Reusing optimization work while producing timely actions also has a mature control-theoretic lineage.

## Structured reuse and computational uncertainty

[Successor features](https://arxiv.org/abs/1606.05312) express a policy's value as $Q^\pi(s,a)=\psi^\pi(s,a)^\top w$. With unchanged dynamics, changing reward weights revalues that policy. Generalized policy improvement combines the available policies; reweighting alone does not produce an optimal policy for an arbitrary new goal. The [deep SF and GPI extension](https://proceedings.mlr.press/v80/barreto18a.html) learns reusable features from task rewards and expands the skill library. Include SF and GPI as a structured goal-transfer baseline.

[Universal value function approximators](https://proceedings.mlr.press/v37/schaul15.html) learn $V(s,g)$ and can factor it into state and goal embeddings; the [successor representation](https://doi.org/10.1162/neco.1993.5.4.613) makes values a linear read of goal-independent occupancy. [Value iteration networks](https://arxiv.org/abs/1602.02867) and [gated path planning networks](https://arxiv.org/abs/1806.06408) embed value-iteration-like recurrent convolution in a policy and train it end to end on mazes with fixed maps and goals. The [changing-maze study](../experiments/dynamic_maze/README.md) builds on both lines: a gated convolutional planner whose map workspace persists through goal moves and is re-targeted by a read at the goal cell while doors and obstacles change during computation.

[Posterior and Computational Uncertainty in Gaussian Processes](https://arxiv.org/abs/2205.15449) separates finite-data uncertainty from uncertainty introduced by approximate computation within a defined GP family. Its covariance results depend on that formulation. For CDM, repeated reasoning processes existing observations; it does not supply additional independent observations. A record's source label alone does not implement this distinction: the memory-reading and state-update operations must specify how repeated use is handled.

## CDM planning choices

Compare an implicit latent workspace with an explicit world model and learned planning control. Both should retain relevant observations and ongoing computation across real events. Use tasks where an action changes the physical process and can reveal its hidden dynamics.

The training distribution needs choices among measuring, thinking, acting, and waiting. Recorded final actions alone do not identify which unchosen activities would have helped; simulator branches and executable tasks can supply those comparisons.

Measure immediate response and later decisions after new observations. Compare restart, ordinary recurrence, incremental planning, learned initialization, and persistent revision with the same task access and complete runtime accounting. The detailed protocol is in [EXPERIMENTS.md](../EXPERIMENTS.md).
