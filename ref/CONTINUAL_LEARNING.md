# Learning that persists: memory, adaptation, and acquired ability

Research cut-off: 14 October 2026. This chapter examines the learning mechanisms most relevant to a Continual Decision Model. The accompanying catalogue records the primary source and the level of evidence inspected for each work. Here, full-text inspection means the relevant method and experiment sections, not a reproduction of the results.

## The distinction CDM needs

An online model can change in several ways. These changes may look similar in a demonstration but support different scientific claims.

| What changes | What it can provide | What establishes its contribution |
|---|---|---|
| Activations, recurrent states, or explicit memory contents | A current belief, remembered evidence, an unfinished computation, or a learned adaptation procedure executed within a trajectory | Performance with identical parameters and controlled state resets or substitutions |
| Fast parameters used as memory | Context compressed into a learned function, updated during deployment | Benefit attributable to the fast update, its retained contents, and its computational cost |
| Parameters retained across tasks | Skills, representations, or update rules that improve later problems | Transfer to fresh problems after removing task-specific context and accounting for all retained stores |

These are operational distinctions, not an immutable division between tensors. A weight matrix can be the recurrent state of a larger learning system. A persistent activation store can carry useful knowledge across tasks. Conversely, updating weights on a document does not by itself demonstrate an enduring improvement in general capability. The relevant questions are what information survives, for how long, through which update rule, and with what benefit on future experience.

For CDM, this suggests three separate experimental claims: **maintaining useful state**, **adapting within an ongoing process**, and **acquiring transferable ability from that process**. A paper may make a strong contribution to one without having established all three. The project should identify its primary claim before choosing the learning machinery.

## Foundations: learning can occur inside recurrent dynamics

| Work | Established mechanism and evidence | Consequence for CDM |
|---|---|---|
| [Using Fast Weights to Attend to the Recent Past](https://arxiv.org/abs/1610.06258), Ba et al., 2016 | Decaying associative fast weights complement slower learned weights; evaluated in retrieval, visual tasks, and memory-dependent RL. | Intermediate memory timescales already have a concrete computational history. |
| [RL²](https://arxiv.org/abs/1611.02779), Duan et al., 2016 | A recurrent policy carries adaptation state across episodes of an MDP; experiments include bandits, finite MDPs, and visual navigation. | Persistent state and learned exploration are established meta-RL ingredients. |
| [Learning to reinforcement learn](https://arxiv.org/abs/1611.05763), Wang et al., 2016 | Outer-loop RL trains recurrent dynamics that implement an inner learning procedure; seven proof-of-concept experiments examine this mechanism. | Frozen deployment weights can still support meaningful behavioral learning. |
| [MAML](https://arxiv.org/abs/1703.03400), Finn et al., 2017 | Optimizes initial parameters for effective adaptation through a few gradient steps, with classification, regression, and RL experiments. | Preparation for future learning is a training objective, not just a backbone property. |

The important inheritance is a way to formulate the training problem. A CDM can be trained on sequences in which early actions reveal information that becomes useful later. The outer objective can reward the entire adaptation trajectory. This is more informative than training isolated input-output pairs and subsequently asking the model to become proactive. However, reproducing this formulation alone would reproduce an established idea; a new contribution must identify the additional mechanism or capability it explains.

## Context as trainable memory

| Work | Established mechanism and evidence | Consequence for CDM |
|---|---|---|
| [TTT layers](https://arxiv.org/abs/2407.04620), Sun et al., 2024 | A linear model or MLP becomes the hidden state, updated through self-supervised learning; language models span 125M–1.3B parameters. | Updating a small model as recurrent memory is already explicit prior art. |
| [Titans](https://arxiv.org/abs/2501.00663), Behrouz et al., 2024 preprint | Combines trainable neural memory with attention; reports language, reasoning, genomics, time-series, and retrieval experiments. | Attention plus persistent trainable memory cannot constitute the novelty claim by itself. |
| [MIRAS](https://arxiv.org/abs/2504.13173), Behrouz et al., 2025 | Organizes associative memory design around representation, objective, retention, and update algorithm; evaluates several resulting sequence models. | Specify CDM's update objective and retention behavior as precisely as its architecture. |
| [ATLAS](https://arxiv.org/abs/2505.23735), Behrouz et al., 2025 | Updates neural memory using current and past context; evaluates language, recall, and long-context understanding. | Memory optimization over accumulated evidence is an existing design axis. |
| [TTT-E2E](https://arxiv.org/abs/2512.23675), Tandon et al., 2025 | Meta-trains sliding-window Transformers whose selected MLP weights update from next-token loss; studies language-model scaling and retrieval. | A standard backbone with an appropriate learning process is a serious baseline. |
| [Nested Learning / Hope](https://arxiv.org/abs/2512.24695), Behrouz et al., 2025 | Combines self-modifying memory with modules updated at different frequencies; includes class-incremental text classification and long-context evaluations. | Multilevel learning and adaptive memory updates already have direct precedents. |

This family changes the architecture question. Choosing between a Transformer and an RNN is insufficient: one must also choose what is optimized during experience, which observations supply the learning signal, and how the trained initialization anticipates those updates. For CDM, prediction loss is a useful component, but the relevant end result is better decisions under changing information and limited time. A memory objective that reconstructs every detail may spend resources differently from one optimized for future experimental choices.

One candidate research question within this broader programme is: **can the model learn which experience to incorporate, when to incorporate it, and which part of its current computation to revise, using the later value of its decisions as the training signal?** Existing work makes each broad ingredient plausible. Originality would depend on the resulting operator, objective, and evidence, including comparison with ordinary recurrent updates and learned adaptation policies.

## Direct neighbors of self-directed adaptation

| Work | Established mechanism and evidence | Consequence for CDM |
|---|---|---|
| [SEAL](https://arxiv.org/abs/2506.10943), Zweiger et al., 2025 | Generates adaptation data and update directives; downstream post-update performance trains these choices. Experiments address knowledge incorporation and few-shot generalization. | Choosing how to change one's own weights is already an explicit research contribution elsewhere. |
| [Meta-TTL](https://arxiv.org/abs/2604.00830v4), Lou et al., 2026 | Learns self-improvement policies across tasks; reports repeated-interaction language-agent evaluations including Jericho and WebArena-Lite. | CDM must distinguish model-internal adaptation from learned agent-level improvement procedures. |
| [Continuous Memory Machines](https://arxiv.org/abs/2610.07907), Regan et al., 2026 | A shared Transformer jointly updates short- and long-term matrix memories; evaluates algorithmic tasks, mazes, regression, and ImageNet-100. | Two interacting persistent/computational stores are a particularly close architectural neighbor. |

The CMM comparison is important for the present proposal. Calling two stores “world state” and “thinking state” does not establish a different mechanism. CDM would need to specify and test their functional relationship: for example, whether new evidence selectively revises dependent computations while preserving useful work, and whether that behavior improves decisions in an environment that continues evolving during inference. Semantic names are an interpretation to test, not a result delivered by assigning separate arrays.

Likewise, self-directed learning is too broad a novelty claim. A concrete contribution could concern coordinated choices among acquiring evidence, continuing internal computation, updating memory, and taking action. To support that contribution, the learned choice must outperform a matched fixed schedule and an established learned adaptation baseline. Its advantage should survive changes in task identity and observation format.

## Retention and the capacity to learn again

| Work | Established mechanism and evidence | Consequence for CDM |
|---|---|---|
| [EWC](https://arxiv.org/abs/1612.00796), Kirkpatrick et al., 2016 preprint | Slows changes to parameters important for previous tasks; demonstrated on sequential digit classification and Atari tasks. | Preserving learned competence is a distinct objective from remembering an observation. |
| [GEM](https://arxiv.org/abs/1706.08840), Lopez-Paz and Ranzato, 2017 | Uses episodic examples to constrain gradients; evaluates forgetting and transfer on MNIST/CIFAR task sequences. | Retention and forward/backward transfer require separate measurements. |
| [CLEAR](https://arxiv.org/abs/1811.11682), Rolnick et al., 2018 preprint | Combines replay, on/off-policy learning, and behavioral cloning without supplied task boundaries; evaluates Atari and DMLab sequences. | Replay is a substantive continual-control baseline, not an optional implementation detail. |
| [Loss of plasticity](https://github.com/shibhansh/loss-of-plasticity), Dohare et al., 2024 | Demonstrates declining learning ability in continual supervised/RL settings; continual backpropagation renews low-utility units. | Measure learning on new problems after prolonged experience, not only old-task recall. |
| [C-CHAIN](https://arxiv.org/abs/2506.00592), Tang et al., 2025 | Reduces changes to out-of-batch outputs during updates; evaluates continual learning across multiple control/game suites. | Test plasticity interventions alongside the proposed online learning rule. |

## A learning protocol that answers the CDM question

The following is a proposed CDM protocol, not a result of the cited papers. It separates the contribution of the learning process from the contribution of state capacity.

**Begin with repeated, related environments.** Preserve a continuous stream containing observations, executed actions, delayed outcomes, and elapsed time. Vary hidden dynamics and goals independently. Hold out combinations and mechanisms, so that success cannot be explained solely by remembering a task identifier. Include cases where further computation helps and cases where only a new measurement can resolve uncertainty.

**Give adaptation a causal objective.** Train an update on information available at the time it occurs, then score its effect on later decisions. A measurement result cannot supervise an earlier decision before it arrives. Predicted consequences can train representations; task outcomes determine whether the chosen learning activity was worthwhile. Costs must include memory updates, replay, and any teacher or retrieval calls used during deployment.

**Separate persistence experimentally.** Compare frozen parameters with persistent activations, fast-parameter memory with the same observation stream, and across-task parameter consolidation. Reset only the component under study, record every retained store, and evaluate on new problems. If improvements disappear after context removal, report useful adaptation rather than claiming permanent capability growth. If improvements survive, test whether they transfer beyond the experienced instances.

**Measure acquisition, retention, and plasticity together.** Report cumulative task performance, learning speed after successive shifts, performance on revisited tasks, and benefit on new task families. Add response latency and update cost. A method that learns the first change rapidly but becomes difficult to adapt after many changes has not met CDM's long-running objective.

**Test selective revision as one candidate experiment.** Compare ordinary recurrent assimilation, a larger memory, complete recomputation, and selective revision at equal budgets. Introduce evidence that changes one dependency, many dependencies, or nothing relevant. This experiment asks whether the model spends computation where the new evidence matters and carries valid work forward; the broader programme also evaluates activity selection, predictive representation, and acquisition of transferable skills.

The scientific target is an integrated, learned process that converts ongoing experience into better future decisions. The current literature supplies strong tools and close competitors. CDM's contribution should be stated at the level of the new learning mechanism and the capability it establishes, with the existing memory and meta-learning machinery credited as its foundation.
