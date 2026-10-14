# Model comparisons

CDM combines persistent state, continuing inference, activity selection, and learning. The comparisons below separate those functions and connect each architectural choice to an experiment. Source editions and reading depth are recorded in [papers.json](papers.json).

## Persistent state and reasoning

| Methods | Relevant mechanism | CDM comparison |
| --- | --- | --- |
| [RMT](https://arxiv.org/abs/2207.06881), [Block-Recurrent Transformers](https://arxiv.org/abs/2203.07852), [Griffin](https://arxiv.org/abs/2402.19427) | Attention with persistent recurrent state | State capacity, downstream action quality, update cost |
| [Deep Repeated ConvLSTM](https://arxiv.org/abs/1901.03559) | State across observations and repeated internal computation | Generic recurrence with additional inference steps |
| [Continuous Memory Machines](https://arxiv.org/abs/2610.07907) | Joint Transformer updates of short- and long-term matrix stores | State organization and interaction between stores |
| [BDH-CQ](https://arxiv.org/abs/2608.09888) | Recurrent contextual memory and iterative latent reasoning | How incoming observations change continuing computation; label approximations where implementation details are unavailable |
| [T²MLR](https://arxiv.org/abs/2607.15178) | Reuse of middle-layer representations across tokens | Temporal reuse versus repeated processing of current context |
| [AtlasVLA](https://arxiv.org/abs/2608.06729) | Persistent world and ego state for embodied action | Scene memory versus stored inference progress |

Attention plus recurrence is the initial model family. A specialized workspace should improve on a comparable recurrent policy through its learned update and training.

## Activity, planning, and learning

| Methods | Relevant mechanism | CDM comparison |
| --- | --- | --- |
| [Continual computation](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/11/cc_aij_horvitz.pdf) | Anticipatory computation and retained partial results | Productive activity before the next observation or request |
| [LPA*](https://idm-lab.org/bib/abstracts/Koen04a.html), [D* Lite](https://idm-lab.org/bib/abstracts/Koen02e.html), real-time iteration | Repair and reuse of plans or optimization state | Incremental planning when explicit problem structure is available |
| [Imagination-based planning](https://arxiv.org/abs/1707.06170), [Metacontrol](https://arxiv.org/abs/1705.02670), [Thinker](https://arxiv.org/abs/2307.14993) | Learned choice and organization of internal computation | Implicit workspace versus explicit world-model planning |
| [When to Plan](https://arxiv.org/html/2607.16421v2), [Finding the Time to Think](https://arxiv.org/html/2606.26463v1), [Looped Actor](https://arxiv.org/abs/2609.37432) | Planning schedules, budgets, or adaptive stopping | Fixed depth, threshold stopping, and learned activity selection including overhead |
| [DreamerV3](https://www.nature.com/articles/s41586-025-08744-2), [TD-MPC2](https://arxiv.org/abs/2310.16828) | Learned prediction and control | Action-conditioned representations and planning cost |
| [TTT-E2E](https://arxiv.org/abs/2512.23675), [Nested Learning](https://arxiv.org/abs/2512.24695) | Parameter memory and multiple update timescales | State retention, memory updates, and cross-task learning |
| [SEAL](https://arxiv.org/abs/2506.10943), [Meta-TTL](https://arxiv.org/abs/2604.00830v4) | Learned improvement procedures | Choice of learning activity and resulting future competence |
| [IMGEP](https://arxiv.org/abs/1708.02190), [POET](https://arxiv.org/abs/1901.01753), [Voyager](https://arxiv.org/abs/2305.16291) | Exploration goals, curricula, and retained skills | Useful self-initiated activity and transfer |
| [StreamReady](https://arxiv.org/abs/2603.08620) | Learned response timing for supplied questions | When to respond and when to initiate an observation or investigation |

## Architecture alternatives

Four routes remain useful: a shared recurrent policy; a context state with a revisable workspace; explicit world-model planning; and adaptive parameter memory. Compare them with shared task access and recorded capacity, training resources, and elapsed time.

The focused revision experiment compares joint learning of the representation, refiner, and update against ordinary recurrence, learned initialization, and action-only correction. Its [closest methods](../research/TRANSPORT_PRIOR_ART.md) include prediction-correction, LAP, A2C2, and learned warm starts.

[EXPERIMENTS.md](../EXPERIMENTS.md) defines the common tasks and measurements. Final architecture choices follow task performance, response time, learning behavior, and transfer across environments.
