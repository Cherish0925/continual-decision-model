# Sequence Architectures and Associative Memory

Research cutoff: 14 October 2026.

A Continual Decision Model needs a state that remains useful while observations, actions, delayed results, and internal computation continue. Sequence architectures provide ways to represent and update that state. Their relevance depends on what they preserve, what they can revise, and how much time each update consumes. Language modeling accuracy alone does not settle those questions.

This review follows three connected developments: preserving accessible representations across segments; compressing history into recurrent dynamics; and treating the recurrent state as an editable associative memory. The accompanying records distinguish inspected full-text sections from abstract-level evidence. Architectural implications below are our synthesis, rather than results attributed to the cited papers.

## 1. The distinctions that determine an architecture

Several properties often grouped under “memory” should be separated:

| Property | Architectural question | Consequence for CDM |
| --- | --- | --- |
| Storage | Are past items individually addressable or superposed in a shared state? | Determines retrieval fidelity and interference. |
| Update | Does the model accumulate, decay, overwrite, or reorganize information? | Determines whether changing circumstances can revise an earlier belief. |
| Readout | Can a current query select the relevant information? | Retained information must remain usable for decisions. |
| Computation | Is recurrence across observations, internal reasoning steps, or both? | Persistence and additional thinking require distinct evaluation. |
| Time | Are transitions indexed by tokens, segments, or elapsed time? | Asynchronous control needs event timing and action duration. |
| Learning | Which state or parameters change, under which objective? | Remembering an episode and acquiring transferable competence are different outcomes. |

Attention and recurrence are therefore not cleanly separable into “reasoning” and “memory.” Attention supports content-dependent interaction among available representations. Recurrence determines how representations persist and transform. Either can participate in reasoning, and several architectures admit both attention-like and recurrent formulations.

## 2. Keeping the past accessible

[LSTM](https://www.bioinf.jku.at/publications/older/2604.pdf) established gated memory cells that preserve learning signals across long delays. Its original synthetic experiments are a foundation for persistent state, rather than a complete account of autonomous decision-making.

[Neural Turing Machines](https://arxiv.org/abs/1410.5401) couple a neural controller to external memory through differentiable attention. Their copying, sorting, and associative-recall experiments establish learned read-write operations as an early architectural mechanism. For CDM, external storage can remain part of a trainable model rather than only a separate retrieval service.

[Differentiable Neural Computers](https://www.nature.com/articles/nature20101) add content lookup, temporal links, and usage-based allocation to neural read-write memory. The original paper demonstrates synthetic question answering, graph reasoning, and reinforcement-learning block puzzles. It is a direct ancestor for organizing persistent information and retaining plans that guide subsequent actions.

[Relational Recurrent Neural Networks](https://arxiv.org/abs/1806.01822) use attention among memory slots before gated state updates. Their evidence includes partially observed reinforcement learning, program evaluation, and language modeling. This precedes the idea of a persistent workspace whose contents interact relationally, rather than merely accumulating independent records of the past.

The original [Transformer](https://arxiv.org/abs/1706.03762) established attention-based sequence transduction with strong parallel training, demonstrated principally on translation and parsing. It provides the starting point for comparing content-addressable interaction against compressed recurrent storage. The original experiments should not be read as evidence about lifelong adaptation.

[Transformer-XL](https://arxiv.org/abs/1901.02860) carries cached hidden representations across segments and uses relative positions. Its language-modeling results establish a practical form of recurrence within a Transformer architecture. Reusing earlier activations is different from repeatedly rewriting a compact belief state, although both provide continuity across input boundaries.

[Recurrent Memory Transformer](https://arxiv.org/abs/2207.06881) places learned memory vectors at segment boundaries and passes updated vectors forward. Section 3 describes read/write placement and cross-segment backpropagation; experiments include algorithmic tasks and language modeling. This is a particularly close precedent for a persistent attention workspace. Adding recurrent memory tokens to CDM would reuse an established mechanism.

[Block-Recurrent Transformers](https://arxiv.org/abs/2203.07852) implement recurrence over blocks using attention, persistent state vectors, and gates. Their long-sequence experiments cover books, scientific papers, and source code. This offers a direct comparison for Transformer-based state transitions, including the computational trade-off between processing whole blocks and responding to individual events.

[Temporal Latent Bottleneck](https://arxiv.org/abs/2205.14794) combines a fast Transformer stream with slower recurrent latent slots, exchanging information through cross-attention. Its experiments include perception and sequential decision-making. Fast local processing with slowly updated compressed state is therefore an established design; CDM must specify what its timing and learning mechanisms add.

The important design choice is how much history remains individually recoverable. Keeping every item enables flexible later queries but increases storage and access cost. A compact state makes ongoing updates economical but must decide which distinctions to preserve before future queries are fully known. For CDM, goals can change after an observation, so useful memory cannot be defined exclusively by the current output.

## 3. Recurrent dynamics and their computational forms

[S4](https://arxiv.org/abs/2111.00396) develops an efficient structured parameterization of state-space sequence models, evaluated on long-range and multimodal sequence benchmarks. It establishes the value of deliberately designed dynamics for long dependencies. Its continuous-system formulation does not by itself specify the semantics of a physical controller's clock.

[Mamba](https://arxiv.org/abs/2312.00752) makes state-space parameters input-dependent, allowing content-sensitive retention and propagation with a hardware-aware recurrent algorithm. Its evidence covers language, audio, and genomics. This is relevant to choosing which incoming events should alter persistent state.

[Mamba-2](https://arxiv.org/abs/2405.21060) connects structured state-space models and attention through structured state-space duality, yielding a more efficient core implementation and competitive language modeling. The conceptual lesson is that “Transformer versus RNN” can conceal related computations expressed through different factorizations.

[Mamba-3](https://arxiv.org/abs/2603.15569) improves discretization, introduces complex-valued dynamics, and uses multiple inputs and outputs to improve state tracking and decoding efficiency. Sections 3 and the efficiency analysis are directly relevant to CDM: arithmetic intensity and state movement matter alongside operation counts. Its reported improvements concern its evaluated sequence models, rather than autonomous physical operation.

[RetNet](https://arxiv.org/abs/2307.08621) presents parallel, recurrent, and chunkwise forms of retention, with language-modeling and efficiency comparisons. It illustrates why the training computation need not resemble the deployment computation. CDM should exploit this distinction where its update rule permits it.

[Gated Linear Attention](https://arxiv.org/abs/2312.06635) combines matrix-valued recurrent state, data-dependent gates, and hardware-aware training. Its moderate-scale language-modeling results reinforce a practical point: nominally cheaper attention can still lose to optimized softmax implementations unless memory traffic and parallel execution are addressed.

Together these papers motivate measuring both what a state transition can express and how efficiently it runs. A richer transition can improve tracking while increasing sequential dependence. Conversely, a scan-friendly transition may impose restrictions that become important when updates must depend on the model's current hypotheses.

## 4. From accumulation to editable associations

[Transformers are RNNs](https://arxiv.org/abs/2006.16236) expresses kernelized linear attention as recurrent computation. [Linear Transformers Are Secretly Fast Weight Programmers](https://arxiv.org/abs/2102.11174) identifies its fast-weight interpretation and introduces delta-style corrections to improve finite-memory association. These are essential precedents: calling a matrix “fast weights” does not create a new memory mechanism.

The distinction is intuitive. Additive memory repeatedly deposits associations into shared storage. A delta update first checks the association already represented by a key, then writes a correction toward the new value. This gives the update information about the memory it is changing.

[Parallelizing Linear Transformers with the Delta Rule](https://arxiv.org/abs/2406.06484) turns that richer update into a practical training algorithm using products of generalized Householder transformations and a compact WY representation. Its experiments include a 1.3B model trained on 100B tokens and attention hybrids. Expressive updates and scalable training have to be developed together.

[Gated DeltaNet](https://arxiv.org/abs/2412.06464) combines adaptive decay with targeted delta corrections and evaluates recurrent and hybrid variants. [Kimi Linear](https://arxiv.org/abs/2510.26692) refines decay with channel-wise Kimi Delta Attention and combines KDA with global Multi-Head Latent Attention. Its reported 48B-total, 3B-active architecture supplies a larger-scale hybrid precedent, with matched training comparisons.

[Gated DeltaNet-2](https://arxiv.org/abs/2605.22791) further separates channel-wise erasure from channel-wise writing while retaining efficient chunkwise training. Its 1.3B experiments compare language modeling and retrieval against several recurrent alternatives. Independent erase/write decisions are therefore established prior art; a CDM contribution would need a more specific mechanism or capability.

[RWKV-7](https://arxiv.org/abs/2503.14456) develops generalized delta updates with vector-valued controls and studies state tracking alongside language modeling. Its formal expressivity results have explicit computational assumptions and should remain separate from empirical claims about scientific reasoning.

[Switching Linear Attention](https://arxiv.org/abs/2609.39034), released in September 2026, derives a mixture-based recurrent update from online expectation-maximization. It evaluates associative recall, in-context language learning, and language modeling. Multiple dynamically selected memory components are another relevant neighbor for proposals that partition CDM state into specialized memories.

[Self-Referential Weight Matrices](https://proceedings.mlr.press/v162/irie22b.html) learn to modify their own weights using outer products and delta updates, with evidence from few-shot learning and multitask reinforcement learning. This is a direct precedent for learning the state-update process itself, extending the relevant comparison beyond language backbones.

These developments connect naturally to TTT, Titans, and MIRAS, reviewed in the learning and adaptation part of this survey. A state can itself parameterize a learned function. The research question then includes the online objective and update algorithm, rather than only the shape of the stored tensor.

## 5. What hybrid evidence establishes

[Griffin](https://arxiv.org/abs/2402.19427) mixes real-gated linear recurrence with sliding-window attention and evaluates language models across scales. Section 2.4 explicitly makes its gates independent of the previous recurrent state to retain efficient execution. This is both a strong hybrid baseline and a precise example of the tension between unrestricted state-dependent updates and parallel training.

Hybrid success does not reveal which component supplies a particular capability. [How Linear Attention Remembers](https://arxiv.org/abs/2609.33093) uses causal interventions in pretrained recurrent and hybrid models. Within its controlled recall settings, competition and overlapping writes matter, and recalled identities in tested hybrids are supported predominantly by full-attention KV state. The study explicitly limits the causal interpretation of separately trained checkpoint comparisons.

For CDM, this suggests a clear experiment: intervene separately on recurrent state, attention storage, and adaptive parameters while preserving the remainder of a trajectory. Measure what information and behavior each intervention changes. Distinguish accurate recall, revising an obsolete belief, maintaining an unresolved objective, and learning a reusable strategy. Aggregate task scores cannot identify these mechanisms on their own.

## 6. Implications for CDM

The closest sequence-level starting points are a gated-delta hybrid, a modern selective state-space hybrid, and a recurrent-memory Transformer. Each offers a different hypothesis about the representation that should persist. They should share observation encoders, action access, training trajectories, and comparable compute budgets in an initial comparison.

The deeper opportunity concerns **decision-relevant state formation**. A CDM must preserve information needed to predict the consequences of possible actions, maintain unfinished work, and update its plans when delayed outcomes arrive. Next-token loss, association reconstruction, and retrieval accuracy provide useful signals, but the choice of what to retain should also be tested against subsequent decision quality.

Real time introduces another requirement. A model should distinguish receiving a new measurement, predicting unobserved evolution during a delay, and refining an internal hypothesis. Repeating the same transition without clarifying these roles risks confusing additional computation with additional evidence. An implementation can share parameters while explicitly representing event type, elapsed time, previous actions, and unresolved outcomes.

Three comparisons would make an architectural proposal substantive. First, determine whether selective state updates improve adaptation under changing dynamics at fixed memory. Second, determine whether internal computation improves action choice after accounting for its elapsed time. Third, determine whether experience produces transferable improvement after temporary context is cleared.

The practical objective is a measured frontier: decision quality against observation-to-action latency, memory footprint, and training cost. Full-attention, bounded-attention, and recurrent models should all remain eligible. The literature supports strong components and informative trade-offs; CDM's distinctive contribution must emerge from how those components learn to sustain a useful decision process across time.

