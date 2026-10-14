# Research objective

Continual Decision Model (CDM) is a research programme for a trainable model that maintains an ongoing understanding of its environment, chooses its own next activity, produces timely decisions, and improves through experience. Its scope spans perception, reasoning, action, and learning over extended operation.

**The scientific objective is to learn how information and computation should change future decisions while the environment continues to evolve.** The model must learn what to retain, what to investigate, what to compute, when to act, and how to turn experience into reusable ability.

## Capabilities

| Capability | Model behavior |
| --- | --- |
| Persistent state | Goals, observations, pending actions, and unfinished inference remain available across outputs and input gaps. |
| Multimodal interaction | Text, images, video, audio, sensor streams, and action feedback enter with their timestamps and source information. |
| Proactive behavior | The model initiates measurements, tool calls, investigations, subgoals, and actions from its objectives and current understanding. |
| Continuing inference | Internal computation proceeds without a new prompt and incorporates observations that arrive during that computation. |
| Timely output | Control signals, decisions, plans, and communication are produced at the cadence required by the task. |
| Continuing learning | Current experience improves predictions and behavior; accumulated training improves performance on future tasks. |
| Self-assessment | Predictions of uncertainty, progress, cost, and expected success guide the next activity. |
| Transfer | Shared mechanisms work under new goals, dynamics, modalities, timing conditions, and task combinations. |

The model should support both requested tasks and self-initiated activity. A user instruction can set or revise a goal without resetting the process. Given an ongoing goal, the model can work through several observations and actions without further prompting. Subgoals and learning activities should be selected for their contribution to that goal or to later competence.

## A continuous decision process

An instrument offers a concrete example. The model maintains uncertain estimates of a process, chooses a measurement, and uses the delay before its arrival to compare candidate explanations. The result updates its understanding and ongoing inference. It chooses the next intervention, observes its consequences, and retains the experience. On a related process, the acquired knowledge reduces the measurements or computation needed.

The runtime delivers observations, maintains clocks, and executes requests. The learned model determines which observations to request, how to update its internal state, how much to compute, and which action to take. A language-model agent can also make these choices. The CDM question concerns how to train and execute them through persistent state and continuing latent computation, including their quality and cost.

Continuous operation describes continuity of this process. Computation intensity may vary: fast response during changes, deeper inference when useful, and replay or consolidation during quiet periods. The model may wait for a pending result while maintaining state. A compact action message can reduce output-generation cost, but perception and reasoning may still dominate response time.

## The coupled learning problem

An observation can make more thinking useful, or make previous thinking obsolete. Computation can reveal which measurement to request. Experience can make a previously expensive decision immediate. These interactions motivate joint learning rather than optimizing each component only in isolation.

For activities occurring at times \(\tau_i\), use task utility integrated over physical time, with observation and computation costs:

\[
J(\theta)=\mathbb E_\theta\left[
\int_0^T e^{-\beta t}u(x_t,a_t,g_t)\,dt
-\sum_i e^{-\beta\tau_i}
\bigl(\lambda_o C_i^{\rm obs}+\lambda_c C_i^{\rm compute}\bigr)
\right].
\]

The environment state \(x_t\) is available to the simulator for scoring; the policy acts on its observations and retained state. Activities have duration, so the environment changes before their benefits become available. The utility and costs must have specified units and normalization. Report them separately as well as through the training objective.

This duration-aware decision objective draws on control, reinforcement learning, active perception, and metareasoning. The proposed model contribution concerns the learned state transitions and their effect on behavior.

## Architecture research

The current design combines a recurrent context state and a persistent Transformer workspace. The focused question is whether an observation-conditioned update can retain useful ongoing computation while changing the parts affected by a new observation or goal.

The candidate jointly learns:

- the representation in which partial computation is stored;
- the shared operation that develops that computation;
- the operation that revises it after a context change;
- the actions and activity choices that use it.

Its competing explanations are ordinary recurrence, additional memory capacity, a better initializer, improved action correction, and explicit world-model planning. [EXPERIMENTS.md](EXPERIMENTS.md) specifies these comparisons. Alternative architectures remain part of the programme; the low-rank update is one candidate mechanism.

## Learning across timescales

| Timescale | What changes | Intended benefit |
| --- | --- | --- |
| Current interaction | Recurrent state and workspace | Integrate observations, update beliefs, continue inference |
| Accumulated experience | Retrieved records and optional fast parameters | Reuse observations, outcomes, and task-specific adaptations |
| Training across tasks | Shared predictive and policy parameters | Acquire skills and faster adaptation on future problems |

Background computation can replay experience, compare hypotheses with a learned world model, improve predictive representations, and distill useful deeper decisions into a fast policy. This gives an explicit learning interpretation to the proposed sleep-like or dreaming behavior. Imagined outcomes are model predictions; measured outcomes supply additional information about the environment. Evaluate background learning by its effect on subsequent real interactions.

Persistent state, experience retrieval, and parameter updates need separate experiments because each can produce improvement. Test learning after state resets, on revisited tasks, and on unfamiliar tasks; track both retained competence and the ability to learn again.

## Generalization and applications

Robots need responsive perception and actions while a longer task continues. Instruments need active measurement and reasoning during delayed experiments. Edge devices need bounded state and useful decisions within power and memory limits. Digital researchers need tools, long investigations, and the reuse of prior work.

The same model interface should support these settings through different encoders and output heads. Generalization experiments change dynamics, goals, observation delays, sensor combinations, and task families. Distinguish transfer with frozen parameters, adaptation with a fixed budget, and additional training.

The initial nonlinear instrument task establishes a complete learning and interaction loop. A second environment tests whether the mechanism transfers beyond that task. Larger multimodal models and longer training extend this programme after those implementations are available.

## Current result and next model

The completed numerical experiment fits small polynomial corrections for changing quadratic objectives. The tested correction loses to strong solvers. Its state-reconstruction test further illustrates that a training target can preserve an unfinished solution's error. These results motivate task-outcome training and stronger comparisons; the nonlinear neural architecture remains to be implemented.

The complete specification is in [DESIGN.md](DESIGN.md), [TRAINING.md](TRAINING.md), and [EXPERIMENTS.md](EXPERIMENTS.md). The [literature collection](ref/README.md) attributes the existing architectures and algorithms used by this programme.
