# World models, embodied state, streaming interaction, and evaluation

Scope: primary research available by 14 October 2026. This chapter connects CDM to learned control, persistent embodied memory, multimodal interaction, and experience-based evaluation. The accompanying bibliography records whether an entry was checked through its abstract, selected full-text sections, or an author-maintained summary. Comparisons below distinguish demonstrated results from the research questions we propose.

## What CDM should contribute

CDM aims to maintain a trainable decision process across observations, internal computation, actions, and feedback. Its ambition includes retaining what it knows, carrying forward unfinished work, deciding what deserves attention, and improving through experience. A useful first scientific contribution must specify which part of that ambition becomes a new capability and explain the mechanism responsible.

For this research programme, three objects deserve separate treatment: an estimate of the environment, the state of an ongoing computation, and experience that changes later behaviour. They interact, but their evidence differs. A map can remember an occluded object without preserving an unfinished inference. A resumed inference can save compute without learning a transferable skill. A trained policy can transfer between tasks without choosing its own observations. These distinctions determine the experiments and prevent an attractive architecture diagram from doing the work of a scientific claim.

One candidate mechanism for an initial study is **continuing useful internal work while revising it in response to a changing world**. This addresses one part of the broader CDM objective; it is a proposed direction, not an established novelty claim. The contribution would have to identify the stored computational object, the update operation, its training signal, and a measurable advantage over recomputation, ordinary recurrent updates, and existing persistent-memory policies.

## Predictive state and learned control

The historical foundation already contains much of the broad CDM vocabulary. [World Models](https://arxiv.org/abs/1803.10122) combines learned visual compression, an RNN predictor, and a compact controller; its experiments include transferring a policy trained in a learned environment. Prediction, recurrent state, and latent imagination therefore belong to the inherited foundation.

[MERLIN](https://arxiv.org/abs/1803.10760) makes memory formation serve predictive modelling in a goal-directed agent and studies severe partial observability in virtual environments. Its relevance is the training of *what to remember*, rather than memory capacity alone. CDM should compare representational objectives, not merely append a larger memory.

[MuZero](https://arxiv.org/abs/1911.08265) learns the reward, policy, and value quantities needed by its search procedure. This supplies an important alternative to reconstructing the entire observable world: representations may be shaped by the decisions they must support. Its reported domain evidence concerns Atari and board games.

[DreamerV3](https://danijar.com/project/dreamerv3/) learns an environment model and improves behaviour through imagined experience, using one algorithm configuration across more than 150 tasks. That breadth supports a general learning recipe; it should not be described as a single continually deployed policy learning all those domains throughout one lifetime.

[TD-MPC2](https://arxiv.org/abs/2310.16828) uses trajectory optimisation in an implicit latent world model. It evaluates continuous control broadly and also trains one model across 80 tasks. This makes it a substantial control and multitask baseline, especially when comparing the cost of online planning with a policy distilled from planning.

[Dawid and LeCun](https://arxiv.org/abs/2306.02572) explain energy-based models, latent variables, and hierarchical joint-embedding prediction as ingredients of autonomous intelligence. This is conceptual lineage. A proposal for an integrated architecture and empirical evidence that the complete architecture works are different kinds of contribution.

[V-JEPA 2](https://arxiv.org/abs/2506.09985) connects video representation learning to action-conditioned prediction and image-goal robot planning. The paper explicitly discusses long-horizon rollout error and planning cost. Its lesson for CDM is to evaluate whether predictive representations preserve action-relevant information, alongside the time required to turn predictions into useful actions.

These approaches suggest several legitimate training objectives. Observation reconstruction preserves sensory detail. Latent prediction can remove nuisance detail. Reward and value prediction prioritise a task. Action-conditioned prediction supports intervention. None should be selected solely because its name resembles the project ambition. For CDM, the representation should be judged by how well it supports future decisions after an informative change, including cases where the relevant distinction was not useful to the previous task.

## Persistent embodied state is an immediate comparison

[OpenVLA](https://arxiv.org/abs/2406.09246) demonstrates a reusable vision-language-action policy pretrained on diverse robot demonstrations and adapted to new settings. It is a practical reference for perception-to-action learning and data transfer. Matching its training resources and sensory access matters more than treating a generic language-only model as the sole embodied baseline.

[Gato](https://arxiv.org/abs/2205.06175) uses one network with shared weights across modalities, tasks, and embodiments. Thus, mixed text and control outputs are established territory. CDM needs evidence for the temporal organisation of those abilities in addition to their presence within one parameter set.

[AtlasVLA](https://arxiv.org/abs/2608.06729) is especially close. Its persistent spatial memory and ego-working memory track observations and task progression, conditioning a diffusion action model. The paper evaluates wrist-camera manipulation and ablates its memory components. A world-state/working-state split by itself is therefore insufficient differentiation; CDM must specify whether and how it preserves *unfinished computational progress*.

This distinction should not force a particular implementation prematurely. Slots, recurrent matrices, sparse object states, and learned retrieval can each represent useful state. The decisive issue is the meaning and consequence of an update. If a model changes its prediction after a new observation, that alone does not establish a new revision mechanism. We need cases where selective revision retains valid work and changes the affected consequences more efficiently than strong generic updates.

## Continuous interaction and choosing when to respond

[Moshi](https://arxiv.org/abs/2410.00037) models user and model speech through parallel streams to support full-duplex conversation, including overlapping interaction. Its demonstrated contribution concerns speech generation and conversational timing. This is a relevant architecture precedent for asynchronous input/output, while independent inquiry and lifelong skill acquisition remain separate evaluation questions.

[StreamChat](https://arxiv.org/abs/2501.13468) uses hierarchical memory and parallel scheduling for streaming video interaction, and introduces a video benchmark called StreamBench. It is a useful systems comparison for what can be obtained through memory organisation and execution scheduling without training a new continual decision core.

[StreamReady](https://arxiv.org/abs/2603.08620) goes further toward the timing problem: a learned readiness token and head decide when evidence supports answering; ProReady-QA supplies evidence windows. This directly overlaps any broad claim that CDM first learns when to respond. Its setting still provides a question whose future evidence must be awaited; selecting a new inquiry is a distinct target.

For CDM, time should enter the learning problem in operational terms. Encoding an observation, retrieving a memory, refining a hypothesis, producing an action, and waiting for a sensor all consume time. The environment can change during these activities. A comparison that pauses the world during inference measures a different problem from a deployed controller or scientific instrument.

Useful timing metrics include event-to-applied-action delay, evidence age at action, missed opportunities, and task value per unit elapsed time. Raw generation speed is incomplete: a fast answer can arrive before its evidence, and a highly accurate plan can become obsolete before execution. Costs must include the perception and memory modules used by each method.

## Evaluation families and what they establish

No single existing benchmark covers the entire CDM objective. The first study should combine interpretable mechanism tests with an established application benchmark and a transfer setting. The following families provide complementary evidence.

| Family | Existing anchor | Suitable evidence | What our protocol must add |
|---|---|---|---|
| Partial observability and prediction | MERLIN; DreamerV3 | Using hidden state to improve decisions | Interruptions during computation; continuously changing dynamics |
| Continuous control | TD-MPC2 | Action quality and planning efficiency | Measured inference delays and observation scheduling |
| Lifelong manipulation | [LIBERO](https://arxiv.org/abs/2306.03310) | Knowledge transfer across task sequences | Explicit separation of state persistence and parameter learning |
| Long-horizon manipulation | [CALVIN](https://arxiv.org/abs/2112.03227) | Language-conditioned skill composition and generalisation | Mid-plan evidence changes and recovery without restarting all work |
| Streaming understanding | StreamChat; StreamReady | Causal use of an unfolding stream and answer timing | Self-selected observations and consequences of choosing them |
| Long-term interaction memory | [LongMemEval](https://arxiv.org/abs/2410.10813) | Recall, temporal reasoning, and changed facts | Behavioural adaptation beyond answering history questions |
| Environment experience | [LongMemEval-V2](https://arxiv.org/abs/2605.12493) | Evidence retrieval from web-agent trajectories | Executing improved workflows in subsequent interactions |
| Open-ended curriculum | [POET](https://arxiv.org/abs/1901.01753) | Generating challenges and transferring solutions | A model-level account of how activity selection and retained experience interact |

LIBERO is directly relevant because its original objective is lifelong robot learning, not only aggregate success after joint training. CALVIN supplies long-horizon language-conditioned manipulation and held-out generalisation. These should be used with the original evaluation distinctions preserved. A convenient subset can serve development, but a final claim about generality needs task-family separation.

LongMemEval evaluates questions over interaction histories. LongMemEval-V2 evaluates environment-specific experience through a context-gathering and downstream question-answering formulation. The latter is closer to practical agent expertise, yet neither a correct retrieved answer nor a compressed history automatically establishes learned control. POET is relevant to choosing learning opportunities, with challenge generation and solution transfer as explicit parts of its algorithm.

There are two unrelated benchmarks named StreamBench. The video benchmark above is attached to StreamChat, arXiv:2501.13468. The language-agent continuous-improvement benchmark is arXiv:2406.08747. Bibliographic keys and experimental scripts must identify the paper, rather than use the name alone.

## A discriminating first study

We propose a continuing experimental environment with hidden dynamics, delayed measurements, changing goals, and costly observations. Its role is to expose the mechanism: some changes alter one local dependency; others invalidate an entire plan. Further tasks should retain the same underlying decision problem while changing observations and action semantics.

Use four matched alternatives: full recomputation from available evidence; ordinary recurrent state updating; retained work with no specialised revision operation; and the proposed selective continuation mechanism. A fixed-stream ablation supplies identical observations to isolate the update mechanism. A separate closed-loop active-sensing evaluation matches initial information, sensory encoders, observation permissions, action opportunities, and resource budgets, while allowing each policy to choose its own observation sequence. Include a strong memory-augmented policy, because gains over a memoryless baseline would establish too little.

The decisive tests are causal. Remove access to retained computation while preserving factual memory. Replace relevant new evidence with equally sized irrelevant input. Change a hidden relationship while keeping appearances similar. Compare behaviour when thinking can solve uncertainty with behaviour when only another measurement can resolve it. These interventions ask whether the model benefits from the intended mechanism.

Report task outcomes together with the resources and adaptation that produced them. Track saved computation after local changes, response to major changes, unnecessary observations, transfer to new dynamics, and the quality of later decisions. Evaluate persistent-state adaptation with frozen weights separately from experience-driven parameter updates. For the latter, measure both acquisition of new skills and retention of earlier capabilities.

A successful first result would demonstrate a reusable principle for continuous decision-making under changing evidence. Broad applications motivate the work, while the comparison identifies exactly what the new mechanism contributes. The next architecture decision should therefore follow the state semantics and evaluation protocol, rather than precede them.
