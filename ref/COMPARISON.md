# Closest Work and the Claims CDM Must Earn

The literature already contains most of CDM's broad ingredients. The useful question is which specific interaction between state, computation, activity selection, and learning remains inadequately explained or implemented. This document compares mechanisms and evidence rather than assigning unsupported capability scores.

The linked thematic reviews and [catalogue](papers.json) record reading depth. A capability outside a paper's inspected experiments is an open comparison, not an assertion that its architecture cannot express it.

## 1. Persistent state and ongoing computation

| Nearest work | Established overlap | Consequence for CDM |
| --- | --- | --- |
| [RMT](https://arxiv.org/abs/2207.06881), [Block-Recurrent Transformers](https://arxiv.org/abs/2203.07852), [Griffin](https://arxiv.org/abs/2402.19427) | Attention and recurrent state already coexist in trainable sequence architectures. | A Transformer–RNN combination is a backbone choice. Compare state capacity, retrieval, update cost, and downstream decisions. |
| [Deep Repeated ConvLSTM](https://arxiv.org/abs/1901.03559) | Recurrence across interaction and additional internal computation support learned planning behavior. | Include a generic recurrent policy with extra internal steps before attributing gains to a specialized computational state. |
| [Continuous Memory Machines](https://arxiv.org/abs/2610.07907) | A shared Transformer updates short- and long-term matrix stores. | Two stores and their joint update are not sufficient distinctions. Test whether CDM's proposed organization adds decision-relevant behavior. |
| [BDH-CQ](https://arxiv.org/abs/2608.09888) | Recurrent contextual memory is coupled with iterative latent reasoning. | Compare state lifetimes and how new evidence enters computation. Some implementation details are proprietary in the paper; do not label an approximation a reproduction. |
| [T²MLR](https://arxiv.org/abs/2607.15178) | Temporal middle-layer recurrence carries internal representations forward. | Distinguish temporal reuse from repeated processing of the current input; evaluate both under identical information access. |

These works weaken an architectural novelty claim based solely on persistent memory plus a thinking loop. They also suggest a productive question: which state organization makes new evidence useful soon enough to change an unfolding decision?

## 2. Choosing, preserving, and revising computations

| Nearest work | Established overlap | Consequence for CDM |
| --- | --- | --- |
| [Continual computation](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/11/cc_aij_horvitz.pdf) | Computation can anticipate future problems and preserve partial results. | Proactive background thinking and cached progress have a direct conceptual predecessor. |
| [LPA*](https://idm-lab.org/bib/abstracts/Koen04a.html), [D* Lite](https://idm-lab.org/bib/abstracts/Koen02e.html), [real-time iteration](https://epubs.siam.org/doi/10.1137/S0363012902400713) | Planning or optimization reuses earlier work as the problem evolves. | Compare incremental repair and warm starts wherever the task permits them. Neural state revision must justify its additional machinery. |
| [Learning model-based planning from scratch](https://arxiv.org/abs/1707.06170) | A learned policy varies imagination, aggregates plan context, revisits imagined states, and trades reward against computation cost. | Choosing how to think is already a learned capability. Specify what persists across real events and what new update mechanism is being tested. |
| [Metacontrol](https://arxiv.org/abs/1705.02670), [Thinker](https://arxiv.org/abs/2307.14993) | Learned control chooses or organizes model-based internal work. | Compare useful internal actions and their training signals, including the benefit of an explicit predictive model. |
| [When to Plan](https://arxiv.org/html/2607.16421v2), [Finding the Time to Think](https://arxiv.org/html/2606.26463v1) | Planning choice and budget allocation are learned under computational constraints. | Account for scheduling overhead and elapsed environmental time. Adaptive depth alone cannot establish CDM's contribution. |
| [Looped Actor](https://arxiv.org/abs/2609.37432) | A depth-recurrent actor uses a predefined KL stability threshold for early stopping. | Compare fixed depth, stability-based stopping, and learned computation value separately. |

The plausible research gap is more specific than “remember thoughts and continue.” One candidate is a trainable update that revises unfinished latent computation according to new evidence, with useful transfer beyond the training environment. The mechanism must explain how information changes a representation and improves subsequent choices. A named revision gate without this evidence remains an architectural proposal.

## 3. Interaction and learning

| Nearest work | Established overlap | Consequence for CDM |
| --- | --- | --- |
| [AtlasVLA](https://arxiv.org/abs/2608.06729) | Persistent world and ego state supports embodied action. | Distinguish remembering the scene from retaining the progress of an unresolved decision; include an embodied memory baseline. |
| [StreamReady](https://arxiv.org/abs/2603.08620) | Streaming response readiness is learned for supplied questions. | Choosing when to answer is relevant, while choosing the next investigation requires an additional experiment. |
| [DreamerV3](https://www.nature.com/articles/s41586-025-08744-2), [TD-MPC2](https://arxiv.org/abs/2310.16828) | Learned predictive representations support effective control and planning. | Ask whether an explicit world model is more effective than implicit latent refinement for the intended tasks. Cross-task algorithm performance is distinct from one model learning continuously across tasks. |
| [TTT-E2E](https://arxiv.org/abs/2512.23675), [Nested Learning / Hope](https://arxiv.org/abs/2512.24695) | Test-time updates and multiple adaptation timescales are already research subjects. | Specify which state or parameters change, which objective updates them, and what survives a task boundary. |
| [SEAL](https://arxiv.org/abs/2506.10943), [Meta-TTL](https://arxiv.org/abs/2604.00830v4) | Self-improvement procedures themselves can be learned. | “Learning how to learn” needs a narrower mechanism claim and comparisons with learned adaptation policies. |
| [IMGEP](https://arxiv.org/abs/1708.02190), [POET](https://arxiv.org/abs/1901.01753), [Voyager](https://arxiv.org/abs/2305.16291) | Exploration goals, curricula, and accumulated skills can sustain activity. | Evaluate the usefulness of self-initiated activity. Model-level implementation is a research choice, not proof that system-level agency is ineffective. |

## 4. Claim ledger

| Possible statement | Current status | Evidence needed before claiming a contribution |
| --- | --- | --- |
| CDM combines attention and recurrence. | Established architectural family. | A justified implementation choice; no originality claim from the combination alone. |
| CDM has multiple memory timescales and variable internal depth. | Established ingredients with close combined precedents. | An explicit advantage of the proposed organization over simpler and closely related alternatives. |
| CDM proactively selects observations, thought, and action. | Programme objective with substantial prior art. | Specify the action space, learned objective, resource accounting, and improvements over strong metacontrol policies. |
| CDM learns to revise ongoing computation under incoming evidence. | Candidate mechanism. | Exact update rule, information-matched controls, state interventions, classical repair comparisons, and transfer. |
| CDM learns what to retain for future decisions. | Candidate representation objective. | Gains across unseen goals beyond reconstruction or recall objectives at matched capacity. |
| CDM decides whether to seek information or spend more computation. | Candidate integration of established ideas. | Controlled uncertainty conditions and better closed-loop decisions after counting both costs. |
| CDM learns from a lifetime of experience. | Long-term target. | Retention, acquisition, forward transfer, and plasticity across task sequences; explicit state reset and persistence protocols. |
| CDM is a general intelligence architecture. | Ambition, not an experimental result. | Evidence across domains, modalities, dynamics, and learning horizons substantially beyond the first paper. |

## 5. What would make the work distinctive

A convincing paper could contribute a new state update, a learning objective, an activity-selection mechanism, or a revealing empirical result about their interaction. A useful combination can also be a contribution when its mechanism and advantage are established. The paper must explain which part is new and which existing ideas make it possible.

The research should remain open to an unexpectedly simple outcome. If an ordinary recurrent model learns the desired behavior under a better training distribution, that is evidence against unnecessary architectural specialization. If explicit planning wins, CDM should incorporate that finding. If specialized revision helps only in a narrow task, its claim should have that scope.

Relatedness and originality are compatible. Attribution makes the lineage clear; controlled evidence establishes the additional result. The [experiment programme](EXPERIMENTS.md) turns these distinctions into decisions about what to build next.
