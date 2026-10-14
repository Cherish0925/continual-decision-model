# Experiments That Decide the Architecture

CDM's full objective is broader than any first experiment. The immediate goal is to compare concrete explanations of persistent, proactive decision-making and discover which mechanisms merit further development. No backbone is selected as universally best by this survey.

## 1. Four architectural routes

| Route | Working hypothesis | First discriminating experiment |
| --- | --- | --- |
| A. Shared recurrent state with attention | Suitable training lets one state support perception, internal work, and action. | Compare one state against structured states at matched capacity and actual latency. |
| B. Persistent state with revisable computational workspace | Preserving and updating intermediate work improves decisions after interruptions or new evidence. | Vary how much earlier work remains valid; compare restart, continuation, generic recurrence, and learned revision. |
| C. Explicit world model with learned planning control | Predictive rollouts and their learned allocation provide a better representation of unfinished decisions. | Compare explicit imagined trajectories against implicit latent refinement using the same real observations and interaction budget. |
| D. Adaptive parameter memory with activity selection | A learned update process compresses experience into representations that improve future decisions. | Compare retained activations, retrieved records, fast-weight updates, and persistent parameter learning under explicit reset protocols. |

These routes can eventually combine. First compare them in configurations that expose the value of their individual mechanisms. State dimension, trainable parameter count, retained memory, and training cost must remain visible rather than being hidden inside an architecture label.

For sequence backbones, start with a well-understood recurrent baseline and an attention–recurrence hybrid. The [sequence review](SEQUENCE_MODELS.md) motivates later comparisons with selective state spaces and delta-rule memory. Latest publication date and language-model perplexity do not determine the best online controller.

## 2. Model interface and time semantics

Represent an event as modality, content, timestamp, and source. Keep goals, pending external actions, and resource measurements available to the state update. A minimal formulation is

\[
S_{k+1}=F_\theta(S_k,e_k,u_k,\Delta t_k),\qquad
u_k\sim\pi_\theta(\cdot\mid S_k,g_k,b_k).
\]

Here, \(u_k\) selects a concrete activity: an external action, an observation request, an internal operation, retrieval, an adaptation step, or waiting. An internal operation need not require a new sensory event. The state may be shared or structured; that is an experimental choice. This notation defines an interface, not a novel algorithm.

Use a semi-Markov account of activity duration when decisions take different amounts of time. The environment advances during sensing, inference, and execution. Record when evidence becomes available, when the model incorporates it, and when a changed action takes effect. A simulator that pauses during reasoning tests a different problem.

## 3. Candidate mechanism studies

**Ongoing computation.** Generate matched event sequences with irrelevant, partially relevant, and globally invalidating changes. Measure whether preserved computational state helps after each kind of change. Compare restart-on-event, continue-without-special-revision, a generic recurrent update, and a proposed evidence-conditioned update. Add incremental planners or warm-start optimization wherever the task has a suitable explicit formulation. A privileged-state planner is useful as a separately labeled diagnostic reference.

**Decision-relevant memory.** Hold the observation history fixed while varying later goals. Compare state trained for reconstruction, future prediction, and downstream decision value. A delayed recall score alone cannot establish the best decision representation. Intervene on stored components to test whether purported computational progress actually contributes beyond remembered observations.

**Information versus computation.** Construct cases where uncertainty comes from missing observations, expensive inference, or both. The activity policy should discover when another measurement is useful and when internal work is sufficient. Compare fixed schedules, tuned thresholds, and learned metacontrol; include their overhead.

**Lasting adaptation.** Introduce repeated task families with changing dynamics and delayed feedback. Evaluate acquisition, forgetting, forward transfer, and recovery of learning capacity. Reset working state while retaining only the declared learned components. Separately reset all adaptive components. This identifies where improvement lives.

Each study has a distinct hypothesis. Success in one does not substitute for the others.

## 4. Data and training

Begin with event trajectories from simulated instruments and partially observed control. Include missing readings, variable delays, concurrent pending actions, changing goals, and evidence that arrives during computation. Vary these factors independently so a model cannot solve every case through a single timing heuristic.

Store timestamped observations, requested and executed activities, outcomes, rewards, costs, and state-reset boundaries. Keep simulator truth separate from model inputs. Teacher plans may supervise training, but evaluation policies must receive only their declared observations. Hold out dynamics, causal structures, goal combinations, and timing regimes rather than merely random trajectories from identical systems.

Use prediction and demonstrated activity choices to initialize representations and policies. Then optimize closed-loop outcomes with explicit duration and resource costs. If recurrent policy optimization is used, account for sequence context and policy age; replayed data for predictive learning and data for policy gradients need separate validity rules. Learnable memory updates require their own objective and gradient horizon to be documented.

A useful conceptual objective is expected task return minus the measured costs of computation, sensing, and interaction. Report a quality–cost frontier across budgets; a single tuned cost coefficient can conceal poor tradeoffs. Counterfactual improvement from extra computation can be estimated on forked simulator states for training or diagnosis, but its collection cost must be reported.

Long-term parameter adaptation enters after a fixed-parameter stateful model provides a reliable baseline. This staging isolates its effect; it does not remove continual learning from CDM's objective.

## 5. Fair comparisons

For fixed-stream ablations, give methods the same observations and event timing. For active sensing, use the same initial information, sensor permissions, action space, and resource budget; resulting observations should differ because the policy chooses them.

Core comparisons should include a generic recurrent policy, an attention–recurrence model, a recurrent-depth policy, an explicit world-model planner, and the proposed mechanism. Add the closest memory architecture when making a memory-organization claim. Include a competent language-model agent when the application supports it, with its external memory, tools, retries, and inference cost counted. An unavailable implementation should be described as a reimplementation or architectural approximation as appropriate.

Match both training opportunity and deployment constraints. Report parameter counts, retained bytes, training interactions, optimizer steps, accelerator time, and tuning budget. Do not force every method into an unnatural identical implementation; allow documented reasonable tuning under a shared budget.

## 6. Evaluation ladder

| Stage | Evidence sought | Scope of the conclusion |
| --- | --- | --- |
| Mechanism experiments | Causal benefit of state, revision, information choice, or adaptation | The tested mechanism under controlled conditions |
| Established application tasks | Useful behavior in manipulation, control, streaming interaction, or memory | Performance on the selected benchmark and protocol |
| Transfer | Robustness to new dynamics, goals, delays, and a distinct task family | Generalization across the measured changes |
| Extended operation | Sustained performance, bounded resources, and retained learning | Continuity across the measured horizon |

Use the [world-model and benchmark review](WORLD_MODELS.md) to select application tasks. LIBERO and CALVIN address aspects of robot transfer or long-horizon behavior; streaming and conversational memory benchmarks address other parts of the objective. None alone establishes the full CDM target. Add event timing and resource instrumentation explicitly where the original benchmark lacks them.

Measure task return or success, event-to-effective-response latency, deadline misses, action continuity, retained-state size, memory growth, and energy when directly measurable. Report median and tail latency on specified hardware at batch size one. Separate measured energy from FLOP or runtime proxies. A throughput improvement does not imply a better response-time distribution.

Use paired environment seeds, independent training seeds, uncertainty intervals, and per-condition results. Start with a modest multi-seed pilot, then determine the run count from observed variability and the effect being tested. Repeated events within one trained run are not independent model replications.

## 7. Research decisions after the pilot

Keep specialized state revision only if its advantage survives generic recurrence, comparable memory, and appropriate planning baselines. Keep learned scheduling only if decision gains cover its overhead. Keep a new adaptation mechanism only if retained competence or acquisition improves under the declared reset protocol.

When an effect disappears, identify whether the mechanism, training signal, data distribution, or intended use case was responsible. Publishable understanding can come from that distinction as well as from a winning architecture. The next design should follow measured evidence while the broader CDM programme continues to connect perception, thought, action, and learning.
