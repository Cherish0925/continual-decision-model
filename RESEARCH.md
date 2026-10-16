# Research objective

Continual Decision Model (CDM) is a research programme for models that maintain an ongoing interaction process: perceiving, computing, choosing activities, acting, and learning. A conversation is one source of events in that process. Goals and unfinished work continue across outputs and intervals without input.

**The central question is how to make past and ongoing computation useful for the next decision while observations, goals, and the environment change.**

## Required behavior

| Capability | Model behavior |
| --- | --- |
| Persistent state | Retain goals, observations, pending operations, and unfinished inference across outputs. |
| Multimodal interaction | Incorporate text, images, audio, video, sensors, and tool results with source and timing information. |
| Proactive activity | Choose the next observation, computation, tool call, action, or wait from the current goal and state. |
| Continuing inference | Develop an internal calculation without another prompt; incorporate new input before the calculation finishes. |
| Timely action | Produce control signals, decisions, plans, and communication at the task's required cadence. |
| Continued learning | Use experience to improve later predictions, decisions, and adaptation to new tasks. |
| Compute allocation | Predict the benefit and duration of available operations and choose among them. |
| Transfer | Reuse learned behavior under unfamiliar goals, dynamics, observation delays, and task combinations. |

An instruction can set or change a goal. The model then selects activities toward that goal without requiring a separate instruction for each step. A runtime delivers events, maintains time, and executes requests; the learned model determines what to request, how to use it, and when to act.

Continuous operation does not require maximum computation at all times. A model may use a small action path during routine operation, deepen its inference when needed, or rehearse experience while waiting. Compact decisions reduce decoding work, but perception, reasoning, and memory access still contribute to response time.

## Scientific hypothesis

A useful intermediate result often depends on only part of the current situation. A map relation can remain useful after a goal changes; a goal-specific plan can remain useful after an unrelated observation; a previous calculation can become useful again when operating conditions return.

**Hypothesis:** a model trained to store intermediate computation together with its generating conditions, and to adapt it to the current context, can obtain higher return within the same time budget than restarting, carrying an undifferentiated state, or retrieving whole snapshots.

The proposed mechanism combines a recurrent context, persistent latent workspace, shared refinement, and bounded computation memory. A learned reader conditions reuse on observations, dynamics, goals, and computation progress. An action policy uses the latest context throughout. An activity policy chooses where additional observation or computation is worthwhile.

This is a specific research claim about representation, reuse, and learning. Recurrence, attention, memory, planning, and metareasoning supply established components. The experiments compare the proposed combination with strong forms of those alternatives. The detailed formulation is in [research/CONTINUAL_COMPUTATION.md](research/CONTINUAL_COMPUTATION.md).

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

## First research sequence

1. **Explicit computation reuse.** A changing grid environment separates exact context return, partial return, and continuous change. Explicit planning states make restart, carry, snapshot retrieval, and conditional reuse directly comparable.
2. **Learned continuing computation.** Train a recurrent context and shared Transformer workspace, then add computation records and condition-factorized reading. Supervise actions at several computation depths and optimize later return.
3. **Learned activity.** Give the model control over observation, retrieval, refinement, and action timing. Train operation values from continuations beginning at the same interaction state.
4. **Continued learning and transfer.** Add online parameter adaptation and background consolidation. Extend evaluation to nonlinear instruments, delayed tools, and richer observation modalities.

The explicit grid experiment and the neural prototype are separate implementations. The [five-seed supervised pilot](experiments/neural_pilot/results/REPORT.md) measures learned memory use on sequential action prediction. [Neural planners in a changing maze](experiments/dynamic_maze/RESULTS.md) measure which computation to keep after a change, and a [continual agent](experiments/continual_agent/RESULTS.md) learns how much to compute and when to look while it acts. Instruments and embodied tasks then test more complex dynamics, partial observations, and measured computation delays. The [experiment plan](EXPERIMENTS.md) and [running guide](RUNNING.md) give the executable scope and commands.

## Learning across timescales

| Timescale | What changes | Intended effect |
| --- | --- | --- |
| Current interaction | Context and workspace | Incorporate observations and continue inference |
| Revisited contexts | Computation records and retrieved experience | Reduce repeated work and improve adaptation |
| Across tasks | Prediction, policy, and state-update parameters | Acquire reusable skills and faster learning |

During quiet intervals, replay can improve prediction, imagined rollouts can compare candidate actions, and distillation can turn expensive calculations into fast behavior. Received outcomes and model predictions remain distinguishable in the data. Background learning is evaluated by subsequent interaction performance and its total computation cost.

Experiments separate memory retention from parameter learning. They measure behavior after state resets, on revisited tasks, and on unfamiliar tasks, with specified adaptation budgets.

## Applications

Robots need current control while a long task continues. Instruments need active measurements and reasoning during delayed experiments. Edge devices need bounded memory and useful output within a power budget. Digital researchers need tool use, persistent investigations, and the ability to resume earlier work.

These applications share an observation–computation–action process but differ in encoders, outputs, costs, and timing. CDM uses one model interface with task-specific input and output modules. Generality is evaluated through transfer across those differences.

The completed [quadratic transport experiment](experiments/transport/RESULTS.md) found that its fitted correction lost to stronger numerical solvers. It also showed why faithfully reconstructing an unfinished optimization state can preserve its error. The present design therefore trains reusable computation for task outcomes and measures the cost of the complete decision path.
