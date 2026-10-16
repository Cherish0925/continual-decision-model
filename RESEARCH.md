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

Build and train one integrated CDM with a recurrent context, shared Transformer refinement, action-conditioned prediction, bounded memory, a fast action readout, and a policy that can choose learning as well as observation and inference. The architecture is specified in [DESIGN.md](DESIGN.md), its optimization in [TRAINING.md](TRAINING.md), and the main training campaign in [EXPERIMENTS.md](EXPERIMENTS.md#integrated-model-campaign).

The shared model and timestamped event interface are implemented together. Learning choices, continuing action during slower work, and state retention across task changes belong in that implementation. Instrument and tool tasks connect to the same interface and weights; their construction can proceed while model training and GPU execution are developed. A further increase in maze score is not a prerequisite.

Use one reference recurrence initially and one matched matrix-recurrence comparison. Keep learned memory refinement and detailed channel interpretation subordinate to the main task-return experiment. Reuse existing planner checkpoints as comparisons and initialization sources where their observation and action interfaces apply.

The next deliverable is a trainable model with the full activity set, a long-lifetime trainer, independent-time execution, and measured acquisition and retention. Proposed features and training allocations below are specifications, not completed capabilities. [STATUS.md](STATUS.md) records completed results.

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
