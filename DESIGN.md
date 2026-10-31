# Continual Decision Model: first multimodal architecture

**Selected on 19 October; pretrained reuse extended on 20 October 2026.** CDM is a persistent, goal-conditioned model that learns actions and the use of observation, computation and experience. This document fixes the first integrated architecture. [STATUS.md](STATUS.md) records implementation and measured behavior; [TRAINING.md](TRAINING.md) defines optimization.

The main path is **pretrained perception → recurrent state ↔ persistent Transformer workspace → action and activity policies**, with action-conditioned prediction, episodic memory and parameter learning connected to the same state. A frozen vision-language model supplies semantic context on demand; an offline teacher supplies training trajectories. The continual core remains responsible for state, actions and activity choices.

## 1. Concrete model

| Part | Selected structure | Role |
| --- | --- | --- |
| Image and short-text perception | Frozen **SigLIP 2 Base, patch16, 224px**, with its paired text tower; trainable projections to width 512 | Reuse pretrained representations of images and instructions |
| Long instructions and semantic context | Frozen **Qwen3.5-2B**, retaining its native vision-language processing and language decoder; a learned reader compresses final hidden states to 8 context tokens | Reuse pretrained language and visual knowledge when the current task needs it |
| Structured observations | Two-layer MLP, width 512; explicit value-presence fields | Encode numeric readings, task fields, executed actions and tool results |
| Time | MLP over log(1 + elapsed time), observation delay and action age | Distinguish fresh and delayed observations |
| Fast state | One **GRU**, width 512, preceded by attention from its previous state to event tokens | Incorporate each new event without clearing unfinished work |
| Ongoing inference | **8 persistent slots × 512**, refined by **one shared Transformer block**, 8 heads, MLP width 2048 | Continue inference using the current state, observation tokens and recent records |
| Prediction | Action-conditioned MLP with rank-8 instance adapters; extend its outputs to future features, measurements, reward and duration | Compare consequences and learn from returned results |
| External action | Goal-conditioned, task-specific action head | Produce control or a typed tool request from the freshest state |
| Internal activity | Factorized policy over observation requests and internal work; target and budget heads as the corresponding operations are added | Choose sensing, refinement, retrieval, learning or waiting |
| Long-term experience | Timestamped episode store, learned retrieval key, top-8 retrieved records, 32 recent working records | Revisit useful experiences beyond the current window |
| Continued learning | Instance prediction/action adapters; replay and distillation into shared prediction/action parameters | Improve future behavior and make useful repeated inference cheaper |

The dimensions define the width-512 reference configuration. Loading a released CDM checkpoint instead retains its original configuration, including width and action schema; the input projections and semantic reader match that width. A width-1024 checkpoint is not copied into a width-512 model. Comparisons use the same initialization and dimensions. The machine-readable choices are in [configs/cdm_multimodal_v1.json](configs/cdm_multimodal_v1.json).

The encoder checkpoint is pinned to revision **75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2**. Its FixRes architecture uses the Transformers SiglipModel class. Images produce a 14×14 patch grid; spatial pooling retains a 4×4 grid before projection. Text uses the checkpoint's 64-token context. Longer instructions and image/text interpretation use Qwen3.5-2B's native processor. SigLIP supplies a short-text perception baseline, while Qwen retains its pretrained language decoder. The initial semantic input budget is 4,096 tokens per call; longer histories are selected from retained observations and summaries.

SigLIP 2 uses a vision Transformer. A CNN is not required in this model. The existing convolutional maze planner remains an experimental comparison. The selected recurrence is a GRU; a matrix recurrence is a later matched-backbone comparison, not another unresolved prerequisite.

~~~mermaid
flowchart TD
    E["Images, instructions, readings, tool results"] --> P["Pretrained towers and input projections"]
    P --> H["Fast recurrent state"]
    P --> Z["Persistent Transformer workspace"]
    E --> V["On-demand vision-language model"]
    V --> C["Learned reader and cached context"]
    C --> H
    C --> Z
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

## 2. Pretrained competence and its connection

Reuse three kinds of learned parameters:

- **CDM core:** load a released lifetime checkpoint, including its observation projector, GRU, workspace, predictor and action/activity heads. With structured observations alone, the new wrapper reproduces the original state and readouts. A zero-initialized input gain introduces new multimodal information without changing the initial fast update. Supplying new tokens to workspace attention can still change refinement; train that connection on the task.
- **Online semantic model:** Qwen3.5-2B supplies native image/text fusion and language generation. Its 2,048-wide final hidden states pass through a learned projection and cross-attention reader conditioned on the current CDM state. The reader emits eight tokens at the core's width. Keep the pretrained tokenizer, visual positions, modality metadata and decoder together.
- **Offline teacher:** Qwen3.6-35B-A3B generates candidate action/experiment trajectories for imitation and distillation, scored by returned task outcomes. It is outside the fast action path. The collection and distillation pipeline remains to be implemented.

The pretrained model and the CDM state use different learned coordinates. The reader learns their relationship from action and outcome losses. It does not assume that arbitrary hidden vectors from different models are interchangeable. Reading hidden states encodes the supplied sequence; generating and then encoding additional reasoning or a proposed plan is a separate computation.

Compress semantic context once when a result arrives. Cache these eight tokens, their source observation interval and the reader version. Update their age features without running the semantic model or attending over its entire input again. During training, recompute trainable projections; cache the frozen model's features. At deployment, a trained reader can cache its compressed output.

A completed semantic result informs the current recurrent state and workspace. It does not replace the current state with the state from when the job began. Its age is measured from source capture, including the time spent computing it. The fast policy can continue between completed semantic calls. The code separates encoding, compression, age updates and assimilation; an asynchronous worker and a learned call/target policy are still integration work.

The native language decoder can answer explicit image/text requests. Expressing CDM's ongoing goals and experience through it additionally needs a context serializer or a trained state-to-decoder connection. General tool schemas, world dynamics and local learning objectives remain task-trained CDM components. Audio and specialized perception are added when an application supplies data and actions that use them.

Model revisions, sources and the reuse comparison are in [ref/PRETRAINED_COMPONENTS.md](ref/PRETRAINED_COMPONENTS.md).

## 3. State and event contract

The running model retains

\[
S_t=(h_t,Z_t,M_t,D_t,\eta_t,q_t,g_t),
\]

where \(h_t\) is fast context, \(Z_t\) is unfinished inference, \(M_t\) is working memory, \(D_t\) is accumulated experience, \(\eta_t\) is instance-owned adaptive parameters and optimizer state, \(q_t\) is pending work, and \(g_t\) is the continuing objective and current subgoal.

An event carries its source and type, capture time, arrival time, payload, and related request/action identifier. The model consumes events in arrival order and retains capture time. A returned measurement is attached to the action and conditions that produced it. An imagined observation has its own type and does not replace a received measurement.

The event log stores exact identifiers and media references. The neural input contains encoded content, timing and task-relevant relations. Arbitrary request identifiers are not treated as meaningful real numbers.

In the implemented frontend, numeric and timing tokens are always present; image and text tokens may be absent or masked per stream. Numeric schemas include explicit indicators for missing readings. Token masks prevent absent inputs from affecting attention. Frozen encoder features can be cached by asset and encoder revision; projected features must be recomputed while their projection is being trained.

## 4. Observation, inference and action

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

## 5. Prediction and planning

The implemented process predictor estimates the next scalar reading's mean and variance for each candidate action. Its first layer has an instance-owned rank-8 adapter. Action scores can read these predictions. The numeric distance-to-setpoint term is specific to process tasks and is disabled for unrelated tool tasks.

The integrated prediction extension uses a width-512 latent transition initialized from \([h_t,\operatorname{pool}(Z_t)]\). A residual two-layer MLP conditions each transition on an action embedding and elapsed time. Separate readouts predict frozen observation features, numeric measurements, task reward and operation duration. Future feature targets come from the frozen perception towers; prediction targets are tied to actual returned observations.

Short imagined action sequences use repeated transitions of this same model. Their outcomes inform the action head or supply training examples for a faster policy. The first implementation does not require pixel generation or a large language model in every control step.

## 6. Proactive activity and goals

The model receives a continuing objective. It can then initiate an observation, experiment, tool call or internal calculation without another instruction.

External action and internal work use separate heads. An activity has an operation, target and budget. Available operations are:

- request a sensor reading, image, document or tool result;
- refine the workspace or compare candidate actions;
- retrieve earlier experience;
- update an instance adapter from received outcomes;
- rehearse experience or distill useful computation;
- wait until a chosen reconsideration time or an arriving event.

The [joint scientific-task model](experiments/continual_experiment/README.md)
trains measurement and experiment choices together with refinement, retrieval
from a bounded received-experience bank, and fast/slow prediction updates.
General tool targets, learned subgoals and reconsideration time remain design
extensions. The comparison measures actual activity use and task benefit
separately.

A subgoal is a task-grounded target with an estimated completion value and duration. In digital experimentation, candidates include measuring a variable, testing a candidate condition, acquiring a required object or repeating a measurement. The policy selects and revises them using downstream task return. Free-form goal descriptions alone are not an objective.

Value, prediction uncertainty, expected duration, progress and pending operations provide computational self-assessment. Routine states can take a short path; difficult or informative states can receive more computation. Quiet periods can use replay or wait. Useful continuity does not require full GPU utilization.

## 7. Memory, learning and consolidation

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

### Implemented experimental memory

The [shared scientific-task model](experiments/continual_experiment/EXPERIMENTAL_MEMORY.md)
stores raw input–outcome ledgers from completed tasks. At a new task it
reconstructs a response predictor from each ledger using the current model and
four learned functional updates. Observed returns then determine how much to
use each historical predictor, alongside a predictor fitted only to the current
task. This preserves the meaning of the stored measurements when shared model
weights change during training.

The primary implementation retains six task ledgers with up to 32 observations
each. Writes replace the oldest task; repeated relations can occupy multiple
slots. Hidden task identities are not supplied. Likelihood training updates the
response model and applicability reader; the action-credit comparison additionally
trains that reader from subsequent task costs. The same core weights serve
physics and chemistry, with separate h/z states for each interface.

The completed primary comparison establishes cheaper recovery of returning
chemical relationships, together with additional cost on new relationships.
Fixed evidence reading and neural-state resets preserve the recovery benefit.
This is an implemented route for reuse of experimental knowledge; autonomous
archive allocation and neural consolidation remain separate research targets.

### Candidate: function-preserving transfer

This candidate is not implemented and is excluded from the current eight-run
[joint-activity comparison](experiments/continual_experiment/README.md).
In [the current model](cdm/continual_experiment.py), prediction is
$\hat y(x)=p_\theta(x)+x(F+S)$, with fast and slow adapter matrices $F,S$.
LEARN and CONSOLIDATE use the same replay residual and update either matrix
with a different learned step size. Both gradients vanish at zero residual.
Fast weights decay by $\rho=0.99$ on each received event; slow weights do not.

An alternative CONSOLIDATE would redistribute an already learned correction:

$$
S' = S+\alpha F,\qquad F'=(1-\alpha)F,\qquad 0\leq\alpha\leq1.
$$

The sum is unchanged, preserving all current predictions and their contribution
to action scores at fixed input and shared state. With fixed shared parameters,
no further learning, and the same subsequent input sequence, after $k$ received
events the prediction difference from leaving the weights unchanged is

$$
\Delta\hat y_k(x)=\alpha(1-\rho^k)xF.
$$

Thus transfer changes retention even when replay already fits. It adds no
representational capacity: writing the same proportions into the two stores
at acquisition could produce the same state. Its purpose is to let later
evidence determine persistence. The retention curve for a transferred increment,
$\alpha+(1-\alpha)\rho^k$, generally differs from a single exponential decay.

A scalar example separates retention from indiscriminate preservation. Let
$p=0,x=1,F=1,S=0$, with the current target equal to 1. For an illustrative
decay $\rho=1/2$, one later event yields prediction 0.5 without transfer and
1 after full transfer. Squared error is 0.25 versus 0 if the target stays 1,
but 0.25 versus 1 if it changes to 0. Current zero-residual gradient updates
cannot distinguish these outcomes; future usefulness determines the choice.

The activity and value heads currently see total replay error, which cannot
distinguish $(F=u,S=0)$ from $(F=0,S=u)$ when other state is identical.
A compact additional input is per-output **slow-only replay error**:

$$
E_{\mathrm{slow}}=\frac1N\sum_{i=1}^{N}
\left[p_\theta(x_i)+x_iS-y_i\right]^{\odot2}.
$$

Together with total error, it exposes dependence on temporary corrections
using received outcomes alone. It does not establish that those corrections
will remain useful. The learning objective must credit later prediction and
task returns against current computation cost and interference after changes.
Since immediate predictions are preserved, training needs delayed outcomes
within its differentiation horizon or a value estimate that carries their
credit across truncation; lower current replay error cannot train this choice.
Task gains also require the retained predictions to improve action selection.

Related foundations are [differentiable plasticity (Miconi et al., 2018)](https://proceedings.mlr.press/v80/miconi18a.html)
and [interacting memory timescales (Benna and Fusi, 2016)](https://www.nature.com/articles/nn.4401).

## 8. Training and GPU execution

[TRAINING.md](TRAINING.md) specifies one integrated progression: load reusable core/perception/semantic weights, train their new connections on trajectories, and adapt action/prediction heads to the task; jointly train action and activity over persistent lifetimes; differentiate through selected local learning steps; consolidate on mixed old and recent experience.

Use BF16 for dense perception and Transformer operations, with FP32 recurrent state and adaptation reductions in the current implementation. Batch independent streams, keep the workspace bounded, cache frozen input features during replay, and group internal work by selected depth. Compile stable shapes after measuring the eager implementation.

The present refinement implementation uses dynamic row selection and a Python loop. It is functional batched code, not a claim of an optimized CUDA scheduler. H100/A100 latency must include perception, state update, refinement, learning contention and action application.

On two H100s, first separate core training from semantic feature/trajectory production. Cache frozen features for replay so semantic inference does not recur in every gradient step. Run the offline teacher in a dedicated collection phase or allocate memory separately from the online model; MoE activation count is not its total weight storage. A100 machines can collect trajectories and run independent comparisons or seeds. Measure throughput before combining heterogeneous devices into a synchronous training group. On GPU, Qwen's DeltaNet path uses the compatible flash-linear-attention and causal-conv1d kernels; the CPU test uses the PyTorch implementation.

## 9. Application and scientific contribution

The first application is a **digital experimental operator**: receive an objective, observe instruments or a visual workspace, choose measurements and interventions, wait for delayed results while continuing useful work, revise predictions, and retain what was learned.

Two complementary evaluations serve that application:

1. The existing instrument/thermal processes measure continual control, delayed feedback, adaptation and elapsed-time behavior under controlled conditions.
2. [DiscoveryWorld](https://github.com/allenai/discoveryworld) is the selected external task suite for visual scientific interaction and experiment sequences. Its native adapter, dynamic task policy, trajectory trainer and on-policy fine-tuning path are implemented in [the task experiment](experiments/discoveryworld/README.md). Preserve its official protocol and scores; an independently timed extension is a separately reported setting.

These task suites keep the same core architecture. Tool schemas and output heads differ. Neither benchmark alone covers general multimodal agency, lifetime learning and real-time execution.

The research mechanism is **jointly training ongoing inference and parameter adaptation through interruptions and subsequent actions**. The model should learn when useful partial computation can continue, when new observation deserves more computation, and when learning from an outcome improves later decisions. Pretrained perception supplies input competence; it is not the claimed contribution.

Compare a recurrent policy, a restarted workspace, persistent inference without local learning, fixed learning/compute schedules, and the joint model with the same perception and resource accounting. Include strong task-native agents. A paper contribution requires improvement attributable to the proposed state/training mechanism; modular composition alone does not establish it.

## 10. Code boundary

| Component | Code / state |
| --- | --- |
| Frozen image/text towers, projection, masks, feature cache interface | Implemented in [cdm/multimodal.py](cdm/multimodal.py) |
| Multimodal assimilation and direct workspace attention | Implemented in the same module and [cdm/core.py](cdm/core.py); tested with small real Siglip towers |
| Exact loading of released CDM core configurations and weights | Implemented in MultimodalCore.from_lifetime_checkpoint; source behavior tested on structured-only inputs |
| Frozen native vision-language model, trainable latent reader, bounded semantic cache and native generation interface | Implemented in [cdm/semantic.py](cdm/semantic.py); actual small Qwen3.5 modules tested with image/text inputs |
| Persistent core, process predictor, activity/action heads, prediction adapters | Existing implementation and structured-process experiments |
| Correct option-value policy gradient | Fixed in [experiments/lifetime/train.py](experiments/lifetime/train.py); detached centering weights |
| Dynamic scientific actions and received-result prediction | Implemented in [cdm/digital.py](cdm/digital.py), using the existing GRU/workspace and a public-observation object registry |
| Scientific-task trajectory training and fresh-interaction policy updates | Implemented in [experiments/discoveryworld/](experiments/discoveryworld/README.md); task execution is evaluated separately from imitation accuracy |
| Scientific-task activity choice, bounded experience retrieval and fast/slow prediction updates | Jointly trained in [continual_experiment](experiments/continual_experiment/README.md); eight trainings and paired native task-stream comparisons |
| Functional response updates and experimental-memory applicability | Implemented in [response_replay.py](cdm/response_replay.py) and [experimental_memory.py](cdm/experimental_memory.py); shared-core training and new/return native comparisons in [the study](experiments/continual_experiment/EXPERIMENTAL_MEMORY.md) |
| Observations arriving during internal work or one pending tool request | Implemented by `ContinualSession` in [cdm/continual_experiment.py](cdm/continual_experiment.py); the native scientific study uses synchronous tool execution |
| Full pretrained-weight multimodal training and teacher distillation | Not run in this update |
| Asynchronous semantic execution and learned semantic-call policy | Interfaces defined above; worker and policy training remain |
| General long-term retrieval, latent world prediction, learned subgoals and skill consolidation | Specified above; beyond the implemented bounded memory and prediction adapters |
| Autonomous digital experimentation | Shared-core policies execute native Reactor Lab and Combinatorial Chemistry tasks from reset, using structured observations and numerical tools. [Current results](experiments/continual_experiment/EXPERIMENTAL_MEMORY.md) distinguish operation in both domains from transfer between them. |

The scientific-task path now learns from returned physical and chemical
experiments. Within-task response fitting and archived observations both improve
specific native comparisons. Stable improvement over strong references, useful
autonomous computation and broad multimodal operation remain research goals.
Physical relation fitting continues to use the numerical model.
