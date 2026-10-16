# Continual learning, memory, and adaptation

Learning methods differ in what changes during experience and what carries into future tasks. This survey covers recurrent adaptation, parameter memory, replay, learned improvement procedures, and retention. Source sections are recorded in [papers.json](papers.json).

## The distinction CDM needs

An online model can change in several ways. The experiments distinguish their retention and transfer behavior.

| What changes | What it can provide | What establishes its contribution |
|---|---|---|
| Activations, recurrent states, or explicit memory contents | A current belief, remembered observations, an unfinished computation, or a learned adaptation procedure executed within a trajectory | Performance with identical parameters and controlled state resets or substitutions |
| Fast parameters used as memory | Context compressed into a learned function, updated during deployment | Benefit attributable to the fast update, its retained contents, and its computational cost |
| Parameters retained across tasks | Skills, representations, or update rules that improve later problems | Transfer to fresh problems after removing task-specific context and accounting for all retained stores |

A weight matrix can be the recurrent state of a larger learning system. A persistent activation store can carry useful knowledge across tasks. The relevant questions are what information survives, for how long, through which update rule, and with what benefit on future experience.

## Foundations: learning can occur inside recurrent dynamics

| Work | Mechanism and reported experiments | Consequence for CDM |
|---|---|---|
| [Using Fast Weights to Attend to the Recent Past](https://arxiv.org/abs/1610.06258), Ba et al., 2016 | Decaying associative fast weights complement slower learned weights; evaluated in retrieval, visual tasks, and memory-dependent RL. | Intermediate memory timescales already have a concrete computational history. |
| [RL²](https://arxiv.org/abs/1611.02779), Duan et al., 2016 | A recurrent policy carries adaptation state across episodes of an MDP; experiments include bandits, finite MDPs, and visual navigation. | Persistent state and learned exploration are established meta-RL ingredients. |
| [Learning to reinforcement learn](https://arxiv.org/abs/1611.05763), Wang et al., 2016 | Outer-loop RL trains recurrent dynamics that implement an inner learning procedure; seven proof-of-concept experiments examine this mechanism. | Frozen deployment weights can still support meaningful behavioral learning. |
| [MAML](https://arxiv.org/abs/1703.03400), Finn et al., 2017 | Optimizes initial parameters for effective adaptation through a few gradient steps, with classification, regression, and RL experiments. | Preparation for future learning is a training objective, not just a backbone property. |

The important inheritance is a way to formulate the training problem. A CDM can be trained on sequences in which early actions reveal information that becomes useful later. The outer objective can reward the entire adaptation trajectory. This is more informative than training isolated input-output pairs and subsequently asking the model to become proactive.

## Context as trainable memory

| Work | Mechanism and reported experiments | Consequence for CDM |
|---|---|---|
| [TTT layers](https://arxiv.org/abs/2407.04620), Sun et al., 2024 | A linear model or MLP becomes the hidden state, updated through self-supervised learning; language models span 125M–1.3B parameters. | Updating a small model as recurrent memory is already explicit prior art. |
| [Titans](https://arxiv.org/abs/2501.00663), Behrouz et al., 2024 preprint | Combines trainable neural memory with attention; reports language, reasoning, genomics, time-series, and retrieval experiments. | Compare neural memory updates with recurrent state and retrieved records. |
| [MIRAS](https://arxiv.org/abs/2504.13173), Behrouz et al., 2025 | Organizes associative memory design around representation, objective, retention, and update algorithm; evaluates several resulting sequence models. | Specify CDM's update objective and retention behavior as precisely as its architecture. |
| [ATLAS](https://arxiv.org/abs/2505.23735), Behrouz et al., 2025 | Updates neural memory using current and past context; evaluates language, recall, and long-context understanding. | Compare the memory objective and its use of accumulated context. |
| [TTT-E2E](https://arxiv.org/abs/2512.23675), Tandon et al., 2025 | Meta-trains sliding-window Transformers whose selected MLP weights update from next-token loss; studies language-model scaling and retrieval. | A standard backbone with an appropriate learning process is a serious baseline. |
| [Nested Learning / Hope](https://arxiv.org/abs/2512.24695), Behrouz et al., 2025 | Combines self-modifying memory with modules updated at different frequencies; includes class-incremental text classification and long-context evaluations. | Multilevel learning and adaptive memory updates already have direct precedents. |

This family changes the architecture question. Choosing between a Transformer and an RNN is insufficient: one must also choose what is optimized during experience, which observations supply the learning signal, and how the trained initialization anticipates those updates. For CDM, prediction loss is a useful component, but the relevant end result is better decisions under changing information and limited time. A memory objective that reconstructs every detail may spend resources differently from one optimized for future experimental choices.

One candidate research question within this broader programme is: **can the model learn which experience to incorporate, when to incorporate it, and which part of its current computation to revise, using the later value of its decisions as the training signal?** Existing work makes each broad ingredient plausible.

## Direct neighbors of self-directed adaptation

| Work | Mechanism and reported experiments | Consequence for CDM |
|---|---|---|
| [SEAL](https://arxiv.org/abs/2506.10943), Zweiger et al., 2025 | Generates adaptation data and update directives; downstream post-update performance trains these choices. Experiments address knowledge incorporation and few-shot generalization. | Choosing how to change one's own weights is already an explicit research contribution elsewhere. |
| [Meta-TTL](https://arxiv.org/abs/2604.00830v4), Lou et al., 2026 | Learns self-improvement policies across tasks; reports repeated-interaction language-agent evaluations including Jericho and WebArena-Lite. | CDM must distinguish model-internal adaptation from learned agent-level improvement procedures. |
| [Continuous Memory Machines](https://arxiv.org/abs/2610.07907), Regan et al., 2026 | A shared Transformer jointly updates short- and long-term matrix memories; evaluates algorithmic tasks, mazes, regression, and ImageNet-100. | Two interacting persistent/computational stores are a particularly close architectural neighbor. |

Compare CMM’s interacting memory stores with the proposed context and workspace states. Test how each update changes later actions, how much useful computation persists, and what the update costs while the environment advances.

## Retention and the capacity to learn again

| Work | Mechanism and reported experiments | Consequence for CDM |
|---|---|---|
| [EWC](https://arxiv.org/abs/1612.00796), Kirkpatrick et al., 2016 preprint | Slows changes to parameters important for previous tasks; demonstrated on sequential digit classification and Atari tasks. | Preserving learned competence is a distinct objective from remembering an observation. |
| [GEM](https://arxiv.org/abs/1706.08840), Lopez-Paz and Ranzato, 2017 | Uses episodic examples to constrain gradients; evaluates forgetting and transfer on MNIST/CIFAR task sequences. | Retention and forward/backward transfer require separate measurements. |
| [CLEAR](https://arxiv.org/abs/1811.11682), Rolnick et al., 2018 preprint | Combines replay, on/off-policy learning, and behavioral cloning without supplied task boundaries; evaluates Atari and DMLab sequences. | Include replay as a continual-control baseline. |
| [Loss of plasticity](https://github.com/shibhansh/loss-of-plasticity), Dohare et al., 2024 | Demonstrates declining learning ability in continual supervised/RL settings; continual backpropagation renews low-utility units. | Measure learning on new problems after prolonged experience, not only old-task recall. |
| [C-CHAIN](https://arxiv.org/abs/2506.00592), Tang et al., 2025 | Reduces changes to out-of-batch outputs during updates; evaluates continual learning across multiple control/game suites. | Test plasticity interventions alongside the proposed online learning rule. |

## CDM learning comparisons

Use long sequences of related and unfamiliar environments. Compare retained activations, retrieved experience, fast-parameter memory, and across-task parameter updates. Reset each retained component separately and test new tasks after transient-state resets.

Measure acquisition after successive changes, forgetting on revisited tasks, transfer, update cost, and the continued ability to learn. Background replay and distillation should improve subsequent interaction within a recorded compute budget.

[TRAINING.md](../TRAINING.md) specifies data, recurrent optimization, replay, actor parameter versions, and multi-GPU execution. [EXPERIMENTS.md](../EXPERIMENTS.md) defines the task sequences and measurements.
