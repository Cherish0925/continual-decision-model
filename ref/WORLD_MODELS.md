# World models, embodied state, and interaction

This survey connects action-conditioned prediction, persistent embodied memory, streaming interaction, and task evaluation. Source editions and inspected sections are in [papers.json](papers.json).

## Predictive state and learned control

The historical foundation already contains much of the broad CDM vocabulary. [World Models](https://arxiv.org/abs/1803.10122) combines learned visual compression, an RNN predictor, and a compact controller; its experiments include transferring a policy trained in a learned environment. Prediction, recurrent state, and latent imagination therefore belong to the inherited foundation.

[MERLIN](https://arxiv.org/abs/1803.10760) makes memory formation serve predictive modelling in a goal-directed agent and studies severe partial observability in virtual environments. Its relevance is the training of *what to remember*, rather than memory capacity alone. CDM should compare representational objectives, not merely append a larger memory.

[MuZero](https://arxiv.org/abs/1911.08265) learns the reward, policy, and value quantities needed by its search procedure. This supplies an important alternative to reconstructing the entire observable world: representations may be shaped by the decisions they must support. Its experiments concern Atari and board games.

[DreamerV3](https://danijar.com/project/dreamerv3/) learns an environment model and improves behaviour through imagined experience, using one algorithm configuration across more than 150 tasks. Its cross-task evaluation concerns a shared learning recipe; CDM additionally studies learning across a continuous task sequence.

[TD-MPC2](https://arxiv.org/abs/2310.16828) uses trajectory optimisation in an implicit latent world model. It evaluates continuous control broadly and also trains one model across 80 tasks. This makes it a substantial control and multitask baseline, especially when comparing the cost of online planning with a policy distilled from planning.

[Dawid and LeCun](https://arxiv.org/abs/2306.02572) explain energy-based models, latent variables, and hierarchical joint-embedding prediction as ingredients of autonomous intelligence.

[V-JEPA 2](https://arxiv.org/abs/2506.09985) connects video representation learning to action-conditioned prediction and image-goal robot planning. The paper explicitly discusses long-horizon rollout error and planning cost. Its lesson for CDM is to evaluate whether predictive representations preserve action-relevant information, alongside the time required to turn predictions into useful actions.

These approaches suggest several legitimate training objectives. Observation reconstruction preserves sensory detail. Latent prediction can remove nuisance detail. Reward and value prediction prioritise a task. Action-conditioned prediction supports intervention. For CDM, the representation should be judged by how well it supports future decisions after an informative change, including cases where the relevant distinction was not useful to the previous task.

## Persistent embodied state is an immediate comparison

[OpenVLA](https://arxiv.org/abs/2406.09246) demonstrates a reusable vision-language-action policy pretrained on diverse robot demonstrations and adapted to new settings. It is a practical reference for perception-to-action learning and data transfer. Matching its training resources and sensory access matters more than treating a generic language-only model as the sole embodied baseline.

[Gato](https://arxiv.org/abs/2205.06175) uses one network with shared weights across modalities, tasks, and embodiments.

[AtlasVLA](https://arxiv.org/abs/2608.06729) is especially close. Its persistent spatial memory and ego-working memory track observations and task progression, conditioning a diffusion action model. The paper evaluates wrist-camera manipulation and ablates its memory components. Compare this memory with the continuing computation stored in the CDM workspace.

Compare slots, recurrent matrices, spatial state, and retrieval through their effects on decisions after new observations. Measure retained useful computation as well as scene recall.

## Continuous interaction and choosing when to respond

[Moshi](https://arxiv.org/abs/2410.00037) models user and model speech through parallel streams to support full-duplex conversation, including overlapping interaction. Its demonstrated contribution concerns speech generation and conversational timing. This is a relevant architecture precedent for asynchronous input/output, while independent inquiry and lifelong skill acquisition remain separate evaluation questions.

[StreamChat](https://arxiv.org/abs/2501.13468) uses hierarchical memory and parallel scheduling for streaming video interaction, and introduces a video benchmark called StreamBench. It is a useful systems comparison for what can be obtained through memory organisation and execution scheduling without training a new continual decision core.

[StreamReady](https://arxiv.org/abs/2603.08620) goes further toward the timing problem: a learned readiness token and head decide when observations support answering; ProReady-QA supplies observation windows. Its setting still provides a question whose future observations must be awaited; selecting a new inquiry is a distinct target.

For CDM, time should enter the learning problem in operational terms. Encoding an observation, retrieving a memory, refining a hypothesis, producing an action, and waiting for a sensor all consume time. The environment can change during these activities. A comparison that pauses the world during inference measures a different problem from a deployed controller or scientific instrument.

Useful timing metrics include event-to-applied-action delay, observation age at action, missed opportunities, and task value per unit elapsed time. Raw generation speed is incomplete: a fast answer can arrive before its observations, and a highly accurate plan can become obsolete before execution. Costs must include the perception and memory modules used by each method.

## Evaluation families and what they establish

No single existing benchmark covers the entire CDM objective. The first study should combine interpretable mechanism tests with an established application benchmark and a transfer setting. The following families provide complementary tasks.

| Family | Existing anchor | Measured behavior | What our protocol must add |
|---|---|---|---|
| Partial observability and prediction | MERLIN; DreamerV3 | Using hidden state to improve decisions | Interruptions during computation; continuously changing dynamics |
| Continuous control | TD-MPC2 | Action quality and planning efficiency | Measured inference delays and observation scheduling |
| Lifelong manipulation | [LIBERO](https://arxiv.org/abs/2306.03310) | Knowledge transfer across task sequences | Explicit separation of state persistence and parameter learning |
| Long-horizon manipulation | [CALVIN](https://arxiv.org/abs/2112.03227) | Language-conditioned skill composition and generalisation | Mid-plan observation changes and recovery without restarting all work |
| Streaming understanding | StreamChat; StreamReady | Causal use of an unfolding stream and answer timing | Self-selected observations and consequences of choosing them |
| Long-term interaction memory | [LongMemEval](https://arxiv.org/abs/2410.10813) | Recall, temporal reasoning, and changed facts | Behavioural adaptation beyond answering history questions |
| Environment experience | [LongMemEval-V2](https://arxiv.org/abs/2605.12493) | Observation retrieval from web-agent trajectories | Executing improved workflows in subsequent interactions |
| Open-ended curriculum | [POET](https://arxiv.org/abs/1901.01753) | Generating challenges and transferring solutions | A model-level account of how activity selection and retained experience interact |

LIBERO is directly relevant because its original objective is lifelong robot learning, not only aggregate success after joint training. CALVIN supplies long-horizon language-conditioned manipulation and held-out generalisation. These should be used with the original evaluation distinctions preserved. A convenient subset can serve development, and held-out task families measure transfer.

LongMemEval evaluates questions over interaction histories. LongMemEval-V2 evaluates environment-specific experience through a context-gathering and downstream question-answering formulation. Use executable downstream tasks to study how this retained experience changes actions. POET is relevant to choosing learning opportunities, with challenge generation and solution transfer as explicit parts of its algorithm.

There are two unrelated benchmarks named StreamBench. The video benchmark above is attached to StreamChat, arXiv:2501.13468. The language-agent continuous-improvement benchmark is arXiv:2406.08747. Bibliographic keys and experimental scripts must identify the paper, rather than use the name alone.

## CDM environment choices

Start with a nonlinear instrument whose actions change future observations. Add one established application domain and a held-out task family. Match observation access and resources for fixed-stream comparisons; allow policies to select different measurements in active-observation experiments.

Compare predictive objectives, persistent state, implicit refinement, and explicit world-model planning. Track task outcomes, observation-to-action delay, acquisition, retention, and transfer. The concrete environment and model comparisons are in [EXPERIMENTS.md](../EXPERIMENTS.md).
