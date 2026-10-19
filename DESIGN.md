# Continual Decision Model: first multimodal architecture

**Selected on 19 October 2026.** CDM is a persistent, goal-conditioned model that learns actions and the use of observation, computation and experience. This document fixes the first integrated architecture. [STATUS.md](STATUS.md) records implementation and measured behavior; [TRAINING.md](TRAINING.md) defines optimization.

The main path is **pretrained perception → recurrent state ↔ persistent Transformer workspace → action and activity policies**, with action-conditioned prediction, episodic memory and parameter learning connected to the same state.

## 1. Concrete model

| Part | Selected structure | Role |
| --- | --- | --- |
| Image and short-text perception | Frozen **SigLIP 2 Base, patch16, 224px**, with its paired text tower; trainable projections to width 512 | Reuse pretrained representations of images and instructions |
| Structured observations | Two-layer MLP, width 512; explicit value-presence fields | Encode numeric readings, task fields, executed actions and tool results |
| Time | MLP over log(1 + elapsed time), observation delay and action age | Distinguish fresh and delayed observations |
| Fast state | One **GRU**, width 512, preceded by attention from its previous state to event tokens | Incorporate each new event without clearing unfinished work |
| Ongoing inference | **8 persistent slots × 512**, refined by **one shared Transformer block**, 8 heads, MLP width 2048 | Continue inference using the current state, observation tokens and recent records |
| Prediction | Action-conditioned MLP with rank-8 instance adapters; extend its outputs to future features, measurements, reward and duration | Compare consequences and learn from returned results |
| External action | Goal-conditioned, task-specific action head | Produce control or a typed tool request from the freshest state |
| Internal activity | Factorized policy over observation requests and internal work; target and budget heads as the corresponding operations are added | Choose sensing, refinement, retrieval, learning or waiting |
| Long-term experience | Timestamped episode store, learned retrieval key, top-8 retrieved records, 32 recent working records | Revisit useful experiences beyond the current window |
| Continued learning | Instance prediction/action adapters; replay and distillation into shared prediction/action parameters | Improve future behavior and make useful repeated inference cheaper |

The dimensions are initial engineering choices, not an assertion of an optimal scale. They remain fixed for the first integrated comparison. The machine-readable choices are in [configs/cdm_multimodal_v1.json](configs/cdm_multimodal_v1.json).

The encoder checkpoint is pinned to revision **75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2**. Its FixRes architecture uses the Transformers SiglipModel class. Images produce a 14×14 patch grid; spatial pooling retains a 4×4 grid before projection. Text uses the checkpoint's 64-token context. Longer instructions are divided into ordered events rather than silently truncated. This text tower provides representations, not generative conversation. A language decoder is a separate output module.

SigLIP 2 uses a vision Transformer. A CNN is not required in this model. The existing convolutional maze planner remains an experimental comparison. The selected recurrence is a GRU; a matrix recurrence is a later matched-backbone comparison, not another unresolved prerequisite.

~~~mermaid
flowchart TD
    E["Images, instructions, readings, tool results"] --> P["Pretrained towers and input projections"]
    P --> H["Fast recurrent state"]
    P --> Z["Persistent Transformer workspace"]
    H --> Z
    Z --> Z
    H --> A["Action and activity policies"]
    Z --> A
    Z --> W["Action-conditioned prediction"]
    W --> A
    A --> X["Environment and tools"]
    X --> E
    X --> M["Episode store"]
    M --> Z
    M --> L["Adapter learning and consolidation"]
    L --> W
    L --> A
~~~

## 2. State and event contract

The running model retains

\[
S_t=(h_t,Z_t,M_t,D_t,\eta_t,q_t,g_t),
\]

where \(h_t\) is fast context, \(Z_t\) is unfinished inference, \(M_t\) is working memory, \(D_t\) is accumulated experience, \(\eta_t\) is instance-owned adaptive parameters and optimizer state, \(q_t\) is pending work, and \(g_t\) is the continuing objective and current subgoal.

An event carries its source and type, capture time, arrival time, payload, and related request/action identifier. The model consumes events in arrival order and retains capture time. A returned measurement is attached to the action and conditions that produced it. An imagined observation has its own type and does not replace a received measurement.

The event log stores exact identifiers and media references. The neural input contains encoded content, timing and task-relevant relations. Arbitrary request identifiers are not treated as meaningful real numbers.

In the implemented frontend, numeric and timing tokens are always present; image and text tokens may be absent or masked per stream. Numeric schemas include explicit indicators for missing readings. Token masks prevent absent inputs from affecting attention. Frozen encoder features can be cached by asset and encoder revision; projected features must be recomputed while their projection is being trained.

## 3. Observation, inference and action

For encoded event tokens \(E_t\),

\[
u_t=\operatorname{Attention}(h_{t-1},E_t,E_t),\qquad
h_t=\operatorname{GRU}(P(u_t),h_{t-1}).
\]

Observation updates the fast state. It does not automatically erase the workspace or adapters.

Each refinement uses slot self-attention, cross-attention to fresh context and records, and a feed-forward network:

\[
Z^{k+1}=\operatorname{LayerNorm}
F_\theta\!\left(Z^k,[h_t,E_t,M_t]\right).
\]

The block parameters are shared across refinements. A model can use zero, one, two or four refinements before reconsidering its next internal activity. New input can enter between blocks. During a block already executing on a GPU, the runtime queues arrivals; preemption inside a matrix multiplication is not assumed.

The action head can read \((h_t,Z_{\mathrm{ready}})\) immediately after observation, before any new refinement. Completed refinement replaces \(Z_{\mathrm{ready}}\). The action stream therefore does not depend on completing a long plan or generating a language explanation.

The first model uses residual workspace revision and normalization. The existing keep/revise/restart gate is an ablation, not a required improvement: the available experiments have not established its advantage. Slots have no manually assigned meanings such as “one hypothesis per slot”; such interpretations require a corresponding training objective.

## 4. Prediction and planning

The implemented process predictor estimates the next scalar reading's mean and variance for each candidate action. Its first layer has an instance-owned rank-8 adapter. Action scores can read these predictions. The numeric distance-to-setpoint term is specific to process tasks and is disabled for unrelated tool tasks.

The integrated prediction extension uses a width-512 latent transition initialized from \([h_t,\operatorname{pool}(Z_t)]\). A residual two-layer MLP conditions each transition on an action embedding and elapsed time. Separate readouts predict frozen observation features, numeric measurements, task reward and operation duration. Future feature targets come from the frozen perception towers; prediction targets are tied to actual returned observations.

Short imagined action sequences use repeated transitions of this same model. Their outcomes inform the action head or supply training examples for a faster policy. The first implementation does not require pixel generation or a large language model in every control step.

## 5. Proactive activity and goals

The model receives a continuing objective. It can then initiate an observation, experiment, tool call or internal calculation without another instruction.

External action and internal work use separate heads. An activity has an operation, target and budget. Available operations are:

- request a sensor reading, image, document or tool result;
- refine the workspace or compare candidate actions;
- retrieve earlier experience;
- update an instance adapter from received outcomes;
- rehearse experience or distill useful computation;
- wait until a chosen reconsideration time or an arriving event.

The existing code learns measurement and refinement/learning choices. Retrieval, general tool targets, subgoals and reconsideration time extend this policy; they are not yet trained capabilities.

A subgoal is a task-grounded target with an estimated completion value and duration. In digital experimentation, candidates include measuring a variable, testing a candidate condition, acquiring a required object or repeating a measurement. The policy selects and revises them using downstream task return. Free-form goal descriptions alone are not an objective.

Value, prediction uncertainty, expected duration, progress and pending operations provide computational self-assessment. Routine states can take a short path; difficult or informative states can receive more computation. Quiet periods can use replay or wait. Useful continuity does not require full GPU utilization.

## 6. Memory, learning and consolidation

Three forms of persistence have different jobs:

| Form | Stored content | How it changes |
| --- | --- | --- |
| Activations | Current context and unfinished inference | Observation and refinement |
| Episodic memory | Received events, executed actions, outcomes, useful computation summaries and their generating context | Append during interaction; retrieve by learned compatibility |
| Parameters | Local prediction/action adapters and shared abilities | Gradient updates and distillation |

The selected long-term reader retrieves eight event/experience records and projects them into the workspace context. Store the observation interval, goal and parameter version of a computation summary. Its relevance is learned from subsequent prediction and task performance. Recording a dependency does not by itself determine whether old reasoning remains useful.

Within an adaptation interval, shared state-producing parameters remain fixed. Prediction adapters can change without changing the coordinates of the current workspace. Action adapters are the next parameter group; the existing lifetime learner updates prediction adapters only.

Training across lifetimes optimizes the shared representation and the learning procedure together. Quiet-period consolidation initially updates prediction/action parameters using old and recent episodes plus task-grounded distillation. Updating the shared state encoder during a live run also requires recomputing state under the new version. A bounded replay prefix is an approximation to full history and is evaluated as such.

Retrieving an old experience is preferred to blindly overwriting current adapters with an old snapshot. Newer learning must remain available when a previous condition returns.

## 7. Training and GPU execution

[TRAINING.md](TRAINING.md) specifies one integrated progression: initialize multimodal projections and action/prediction heads from trajectories; jointly train action and activity over persistent lifetimes; differentiate through selected local learning steps; consolidate on mixed old and recent experience.

Use BF16 for dense perception and Transformer operations, with FP32 recurrent state and adaptation reductions in the current implementation. Batch independent streams, keep the workspace bounded, cache frozen input features during replay, and group internal work by selected depth. Compile stable shapes after measuring the eager implementation.

The present refinement implementation uses dynamic row selection and a Python loop. It is functional batched code, not a claim of an optimized CUDA scheduler. H100/A100 latency must include perception, state update, refinement, learning contention and action application.

On two H100s, start with one homogeneous training pool; assign CPU processes to environments and storage. Use the second GPU for synchronized training when scaling is measured, or for concurrent collection/evaluation. A100 machines can collect data and run independent seeds. Mixed H100/A100 synchronization is not assumed to improve throughput.

## 8. Application and scientific contribution

The first application is a **digital experimental operator**: receive an objective, observe instruments or a visual workspace, choose measurements and interventions, wait for delayed results while continuing useful work, revise predictions, and retain what was learned.

Two complementary evaluations serve that application:

1. The existing instrument/thermal processes measure continual control, delayed feedback, adaptation and elapsed-time behavior under controlled conditions.
2. [DiscoveryWorld](https://github.com/allenai/discoveryworld) is the selected external task suite for visual scientific interaction and experiment sequences. Its adapter and training integration remain to be built. Preserve its official protocol and scores; an independently timed extension is a separately reported setting.

These task suites keep the same core architecture. Tool schemas and output heads differ. Neither benchmark alone covers general multimodal agency, lifetime learning and real-time execution.

The research mechanism is **jointly training ongoing inference and parameter adaptation through interruptions and subsequent actions**. The model should learn when useful partial computation can continue, when new observation deserves more computation, and when learning from an outcome improves later decisions. Pretrained perception supplies input competence; it is not the claimed contribution.

Compare a recurrent policy, a restarted workspace, persistent inference without local learning, fixed learning/compute schedules, and the joint model with the same perception and resource accounting. Include strong task-native agents. A paper contribution requires improvement attributable to the proposed state/training mechanism; modular composition alone does not establish it.

## 9. Code boundary

| Component | Code / state |
| --- | --- |
| Frozen image/text towers, projection, masks, feature cache interface | Implemented in [cdm/multimodal.py](cdm/multimodal.py) |
| Multimodal assimilation and direct workspace attention | Implemented in the same module and [cdm/core.py](cdm/core.py); tested with small real Siglip towers |
| Persistent core, process predictor, activity/action heads, prediction adapters | Existing implementation and structured-process experiments |
| Correct option-value policy gradient | Fixed in [experiments/lifetime/train.py](experiments/lifetime/train.py); detached centering weights |
| Full pretrained-weight multimodal training | Not run in this update |
| Long-term retrieval, prediction extension, tool/subgoal heads, action adapters and consolidation | Specified above; implementation remains |
| Autonomous digital experimentation | Application target; not a completed result |

The next integration is a trajectory/environment adapter that connects multimodal events to the lifetime learner, followed by joint training. A new encoder implementation does not require redesigning the persistent core.
