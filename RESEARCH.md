# Research objective

Continual Decision Model (CDM) is a research programme for models that maintain an ongoing interaction process: perceiving, computing, choosing activities, acting, and learning. A conversation is one source of events in that process. Goals and unfinished work continue across outputs and intervals without input.

**The central question is how a continuously operating model can learn to choose its observations, internal computation, actions, and learning so that it acts effectively now and acquires abilities useful later, while the environment continues to change.**

The next main experiment trains these choices in the same persistent model. Navigation, instruments, and delayed digital tools supply different interactions for that model. They are applications and comparisons, not successive replacements for the research objective.

## Required behavior

| Capability | Model behavior |
| --- | --- |
| Persistent state | Retain goals, observations, pending operations, and unfinished inference across outputs. |
| Multimodal interaction | Incorporate text, images, audio, video, sensors, and tool results with source and timing information. |
| Proactive activity | Choose the next observation, computation, retrieval, tool call, action, learning update, or wait from the current goal and state. |
| Goal-directed initiative | Form and revise subgoals and useful investigations within a continuing objective, including intervals without a new request. |
| Continuing inference | Develop an internal calculation without another prompt; incorporate new input before the calculation finishes. |
| Timely action | Produce control signals, decisions, plans, and communication at the task's required cadence. |
| Action-conditioned prediction | Predict future observations, outcomes, and rewards under candidate actions; use these predictions to choose actions and experiments. |
| Continued learning | Use experience to improve later predictions, decisions, and adaptation to new tasks; consolidate repeated inference into faster skills. |
| Self-assessment | Maintain learned estimates of capabilities, uncertainty, resource use, and progress, and use them to choose activities and research direction. |
| Compute allocation | Predict the benefit and duration of available operations and choose among them. |
| Transfer | Reuse learned behavior under unfamiliar goals, dynamics, observation delays, and task combinations. |

Reactive use and proactive operation belong to the same model: an instruction or conversation can enter an existing process, and the process can continue useful activity between such events. The current implementation of each capability is listed in [STATUS.md](STATUS.md).

An instruction can set or change a goal. The model then selects activities toward that goal without requiring a separate instruction for each step. A runtime delivers events, maintains time, and executes requests; the learned model determines what to request, how to use it, and when to act.

Continuous operation does not require maximum computation at all times. A model may use a small action path during routine operation, deepen its inference when needed, or rehearse experience while waiting. Compact decisions reduce decoding work, but perception, reasoning, and memory access still contribute to response time.

## Scientific hypothesis

An internal operation changes the model's ability to act: observation changes available information, refinement develops a current inference, retrieval recovers earlier work, and learning changes future predictions or policies. Their benefits have different durations. Training must account for the external actions and opportunities that occur while an operation runs, as well as the tasks on which its result will later be used.

**Primary hypothesis:** jointly training persistent state updates and choices of observation, inference, action, and experience-based parameter updates can improve task performance across a changing lifetime at the same elapsed-time and resource budget as strong recurrent policies, fixed learning schedules, or activity policies over frozen models.

The claim requires both timely decisions and retained improvement. A higher reward produced solely by a cheaper calculation is useful compute allocation; acquisition requires improved later behavior from experience. State, retrieved experience, and adapted parameters can all support acquisition, and their contributions must be measured separately. In stationary conditions where the model already performs well, useful behavior may involve little additional learning.

The existing computation-reuse hypothesis remains a component of this programme: intermediate results conditioned on their generating observations and goals may be reused or revised more efficiently than restarting or retrieving whole snapshots. Its formulation and completed studies remain in [research/CONTINUAL_COMPUTATION.md](research/CONTINUAL_COMPUTATION.md). Maze-specific map/goal splitting does not define the general model.

### Relation to existing work

| Existing work | Established direction | CDM comparison |
| --- | --- | --- |
| [RL²](https://arxiv.org/abs/1611.02779) | A recurrent policy can learn an adaptation procedure encoded in its activation state across episodes. | Strong recurrent adaptation with persistent state and fixed deployment weights. |
| [MAML](https://proceedings.mlr.press/v70/finn17a.html) | Train shared parameters so a small number of subsequent gradient updates improves performance on a new task. | Adaptation through gradient updates is established; compare the learned update rule and the decisions it changes. |
| [Test-Time Training](https://proceedings.mlr.press/v119/sun20b.html) | Update predictive parameters from a self-supervised objective on incoming test data. | Compare which received evidence supplies the online loss and whether that loss improves subsequent task performance. |
| [VariBAD](https://arxiv.org/abs/1910.08348) | Learn task inference and condition exploration on uncertainty about the task. | A learned belief-conditioned decision policy is a direct alternative to the numerical belief and continuing neural state. |
| [Dual Control for Approximate Bayesian Reinforcement Learning](https://www.jmlr.org/papers/v17/15-162.html) | Select actions considering both their immediate consequences and the information useful for later control. | A numerical controller with the same instruments, hypotheses and interaction budget tests the value added by learned computation. |
| [Deep Adaptive Design](https://proceedings.mlr.press/v139/foster21a.html) | Learn a sequential experimental-design policy that uses earlier measurements to choose later experiments. | Active experiment selection must be compared with established information-seeking policies. |
| [Dreamer](https://www.nature.com/articles/s41586-025-08744-2) | Learned action-conditioned world models support policy learning from imagined trajectories across varied tasks. | A model-based learner under the same observation, interaction, and computation budgets. |
| [Finding the Time to Think](https://arxiv.org/abs/2606.26463) | A learned gate selects planning budgets over a frozen planner while environment time advances. | Learned compute allocation alone, including its independently timed execution. |
| [Latent Adaptive Planner](https://arxiv.org/abs/2505.03077) | A latent plan is revised as new observations arrive during manipulation. | Continuing and revising inference without learning the full set of activities. |
| [Titans](https://arxiv.org/abs/2501.00663) | Neural memory is updated at test time alongside attention. | Adaptive memory versus acquired prediction or action skills. |
| [Dejavu](https://arxiv.org/abs/2510.10181) | Retrieved execution experience improves a frozen embodied policy. | Experience-based improvement with fixed weights. |

These papers motivate concrete alternatives. Combining their components does not by itself establish a new result. The proposed contribution must specify the state update or training rule responsible for improvement and show its effect at equal resources across task families. The comparison above identifies the relevant overlap for the main experiment.

## Objective over physical time

For activities beginning at times \(\tau_i\), optimize task utility accumulated over physical time together with specified sensing and computation costs:

\[
J(\theta)=\mathbb E_\theta\left[
\int_0^T e^{-\beta t}u(x_t,a_t,g_t)\,dt
-\sum_i e^{-\beta\tau_i}
\left(\lambda_o C_i^{\mathrm{obs}}+\lambda_c C_i^{\mathrm{compute}}\right)
\right].
\]

The policy uses received observations and retained state. The simulator state \(x_t\) is available for scoring and reference planners, according to each experiment's specified information access. An operation takes time, during which the environment and previously applied actions continue.

This objective measures both useful decisions and their timing. Utility, elapsed time, observation cost, and computation cost are also reported separately. Cost coefficients specify the task tradeoff rather than hiding it in a single score.

The choice between sensing and computation depends on their achievable improvement and duration. Decomposing performance into an information limitation and a computation limitation can help describe a task, but the size of either term is not itself the value of the next operation.

## Current execution priority

Updated 30 October 2026 against commit `504b750`. The current publication-scale
question is whether a continuing model can learn **which acquired knowledge to
reuse, which prediction to revise, and which operation to perform next**, so that
experience improves later decisions while observations and tasks change.

### Starting point

The [execution-aware study](experiments/continual_physics/EXECUTION.md#completed-native-comparison)
restores competitive task performance. On its 120-instance cohort, the two full
models complete 115 and 117 tasks at 103.61 and 101.58 actions; the corresponding
recurrent references complete 114 and 119 at 103.69 and 100.72 actions.
Retaining all task state improves on clearing it, but neither learned physical
updates nor the full activity mechanism has a stable independent advantage.
These observations supersede the earlier scalar-cost failures as the current
performance starting point; the cohorts differ, so their difference is not an
isolated estimate of the execution-cost correction.

Finish the four already specified [shared-core trainings](experiments/continual_experiment/SHARED_TRAINING.md)
and their fixed comparison. They train physical decisions and recorded chemical
responses through a common core. Chemistry does not yet learn activity choice,
and its replay pauses the physical scheduler. This study measures cross-domain
state interference and reuse. It does not train the complete proactive process.

### Architecture decision

Keep the current recurrent core, shared Transformer workspace, pretrained
perception interfaces and native tools as the reference implementation. The
next architectural decision concerns the **organization and use of state**:

| Representation | Meaning and use |
| --- | --- |
| Current context | Current objects, goal, received results and pending requests; updated when new observations arrive. |
| Active workspace | Unfinished predictions and candidate actions for the current investigation; refinement must affect subsequent predictions or actions. |
| Retrieved knowledge | Bounded summaries of previously observed relationships, their applicability and uncertainty; read and revised from available observations. |
| Adaptive parameters | Updates trained to improve future received-outcome predictions and decisions, including after intervening tasks. |

These are semantic responsibilities, not four mandatory new networks. Reuse
existing storage and interfaces before adding capacity. A shared model may
keep separately addressable investigation states and use one learned update
rule over them. Mixing unrelated task content into one undifferentiated h/z
is an experimental choice, not a definition of general intelligence. If
separated state wins the current study, use that result to improve state
addressing rather than forcing cross-domain interference.

The existing [workspace-to-recurrence connection](cdm/recurrent_feedback.py)
is an available candidate for making completed internal work affect the next
received-event update. Its zero-initialized projection preserves that local
computation before training. Whole-policy equivalence also depends on input
meaning, action masks, task resets and pending-result handling; the
[inherited-interface implementation](experiments/continual_physics/INITIALIZATION.md)
makes those choices explicit. Neither connection alone establishes a new
learning principle.

### Next integrated training

Train both task interfaces on **their own selected experiments and actual
returned outcomes**, using the same observation, prediction, activity and
learning semantics. Recorded chemistry trajectories remain useful pretraining
and replay data. They cannot by themselves teach the return of actions absent
from those trajectories. New chemical choices must receive outcomes from the
environment before becoming supervised response targets.

Use complete continuing investigations as the data unit. Include familiar
relations with new objects, changed coefficients, returns after intervening
tasks, and new relation or composition families where the environment supports
them. Split these forms of generalization explicitly. Alternate variable-length
segments and vary return delays; the fixed A-B-A schedule remains one interpretable
comparison, not the entire training distribution. Neither task identities
hidden by the environment nor future outcomes become policy inputs.

Train three connected quantities in this same stream:

1. **Predictive knowledge:** outcome distributions conditioned on the received
   history and proposed action. Include failures and partial outcomes. Track
   prediction quality on histories selected by the current policy, since a
   planner can repeatedly select an overestimated experiment.
2. **Useful memory and updates:** prediction and task return on later
   investigations, including the first return after interference. Bootstrap
   across nonterminal training windows and sample later-use examples; a long
   running process with only immediate losses does not train long-term utility.
3. **Operation value:** the expected change in subsequent task return from
   observing, refining, retrieving or learning, including its elapsed time and
   cost. On a sampled subset of simulator prefixes, compare continuations from
   identical received history, model state and pending operations. Reuse the
   branch-value machinery in [TRAINING.md](TRAINING.md#3-value-of-the-next-operation).
   Allocate additional samples where estimated differences are noisy rather
   than treating one stochastic continuation as an exact target.

Prediction, action and activity losses update the same representation. Online
updates must change the predictive quantities used by the action policy.
Train replay and slow consolidation on subsequent received outcomes and
behavior, so quiet-period computation can become useful future capability.
Model-generated outcomes remain predictions, not independent observations.

Difficulty alone should not determine compute. Missing information may call
for measurement; a short deadline may call for immediate action; a useful
retrieved result may remove the need to recompute. The learned choice is which
operation improves the remaining investigation under its actual time budget.

### Continuous-time application and comparison

Keep official DiscoveryWorld results under their original protocol. Add a
separately named timing extension in which the digital process, pending
operations and task deadlines advance during inference. The fast action path
remains available while slower reasoning or learning runs. Carry observation
capture/arrival time and action application time through the same interfaces.
Artificial receipt ticks in an otherwise paused environment do not establish
responsiveness to a changing world.

Use one integrated candidate and a small, informative comparison set:

- The strongest inherited policy.
- A separately trained recurrent or addressed-memory model with fixed
  computation and the same observations, tools and training exposure.
- Fixed-schedule learning versus learned operation choice.
- The existing Bayesian references, with their known function-family
  assumptions reported separately.

Use same-weight memory/update interventions to locate a gain after the
complete method is competitive; do not replace independently trained reference
methods with reset interventions. Report completion and actual operation cost,
first-return behavior, learning on new relations, and capture-to-action delay
separately. Report uncertainty across task streams and training repetitions.
Select development configurations without using the final task cohort.

The whole model need not beat a correctly specified Bayesian method in every
known-family task. A useful result may instead be comparable quality with lower
decision cost, adaptation beyond that function family, or better decisions
under changing timing and computation budgets. Match information access and
show the relevant tradeoff rather than weakening a reference.

### Execution and scaling

The next implementation endpoint is one checkpoint that selects experiments
and internal activities in both domains, learns from its own returns, preserves
useful knowledge through interference, and runs in the independently timed
setting. Each mechanism should have a stated contribution to that endpoint.

Use the two H100s for the integrated learner and its strongest matched
reference. Batch environment streams and use CPU workers for native execution;
cache frozen perception only when the input and encoder version are unchanged.
Measure environment, feature encoding, model forward/backward and transfer time
before choosing additional concurrency. Low allocated memory alone does not
show that a GPU has spare compute.

Keep one short implementation/throughput check, then run the full declared
training schedule and automatic evaluations. Evaluate learning curves on
development streams while keeping the final comparison separate. Extend data,
training duration and model width as controlled scale comparisons, retaining
task cost and inference latency. If the additional A100s are available, use
them for independent repetitions, trajectory production and matched reference
runs; introduce heterogeneous synchronous training only after measuring its
throughput.

Completed task returns, reusable capability and time to reach a given
performance level are the progress measures. File count, releases, GPU
occupancy and numbers of repeated executions are supporting engineering
information.

The mechanism remains a research proposal. Learned internal planning already
appears in [Imagination-based Planner](https://arxiv.org/abs/1707.06170), and
amortized experiment choice in [Deep Adaptive Design](https://proceedings.mlr.press/v139/foster21a.html).
The proposed contribution must identify the state/update/training difference
that improves continuing investigations, not claim either established idea
as new.

## Learning across timescales

| Timescale | What changes | Intended effect |
| --- | --- | --- |
| Current interaction | Context and workspace | Incorporate observations and continue inference |
| Revisited contexts | Computation records and retrieved experience | Reduce repeated work and improve adaptation |
| Within one deployment lifetime | Instance-specific prediction/action adapters and selected experience | Acquire local dynamics or skills from that instance's interactions |
| Across training lifetimes | Shared prediction, policy, and state-update parameters | Learn transferable skills and useful adaptation procedures |

During quiet intervals, replay can improve prediction, imagined rollouts can compare candidate actions, and distillation can turn expensive calculations into fast behavior. Received outcomes and model predictions remain distinguishable in the data. Background learning is evaluated by subsequent interaction performance and its total computation cost.

Experiments separate memory retention from parameter learning. They measure behavior on revisited and unfamiliar tasks, and use separate reset comparisons to determine where an acquired ability is stored. A continuously running evaluation retains state across requests and scenario changes. Parallel training may pool experience; a single-instance adaptation evaluation uses only that instance's experience and parameters. Collective learning is reported as a separate setting.

## Applications

Robots need current control while a long task continues. Instruments need active measurements and reasoning during delayed experiments. Edge devices need bounded memory and useful output within a power budget. Digital researchers need tool use, persistent investigations, and the ability to resume earlier work.

These applications share an observation–computation–action process but differ in encoders, outputs, costs, and timing. CDM uses one model interface with task-specific input and output modules. Generality is evaluated through transfer across those differences.

The completed [quadratic transport experiment](experiments/transport/RESULTS.md) found that its fitted correction lost to stronger numerical solvers. It also showed why faithfully reconstructing an unfinished optimization state can preserve its error. The present design therefore trains reusable computation for task outcomes and measures the cost of the complete decision path.
