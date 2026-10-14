# Continual Decision Model

**CDM studies how a model can maintain an ongoing understanding of the world, choose its next activity, and learn from what happens.**

A robot keeps moving while a model computes. An experiment keeps running while a hypothesis is revised. A delayed observation may change which action matters next. Intelligence in these settings depends on carrying understanding forward and spending time well.

The Continual Decision Model is a research project on persistent, proactive, real-time intelligence. Its central question is:

> Can a trainable model learn both what to do in the world and how to allocate its own observation, thought, memory, and learning over time?

**Project stage:** research foundation and first architecture proposal; implementation and experimental results are forthcoming.

## The model

A CDM maintains an evolving internal state across observations, decisions, and interactions. It receives asynchronous information, updates its understanding, selects activities, and incorporates their consequences. A conversation is one event within this longer process.

**Continual** describes continuity of state, activity, and adaptation across time. **Decision** includes external actions and internal choices: inspecting a signal, refining a hypothesis, recalling an experience, running an experiment, revising a plan, or waiting for useful evidence. **Model** places these capabilities inside the trainable computation that updates state and selects behavior.

Proactivity is central to the design. Goals, uncertainty, anticipated outcomes, and available resources should influence what the model initiates next. The same process that chooses an external action should learn when additional observation or computation is worth its cost.

## What continuity makes possible

| Capability | What it means in operation |
| --- | --- |
| Persistent understanding | Maintain beliefs, goals, unresolved questions, and pending actions as new events arrive. |
| Multimodal perception | Integrate text, images, video, audio, telemetry, and other sensors with their timing and context. |
| Self-directed activity | Choose the next observation, inference, experiment, or action without a new instruction for every step. |
| Latent computation | Develop internal representations and predictions; use language when communication serves the task. |
| Timely decisions | Produce useful actions within the environment's deadlines while allocating deeper computation selectively. |
| Continual learning | Turn feedback into better predictions, transferable skills, and more effective future decisions. |
| Self-assessment | Estimate uncertainty, expected progress, and the value of further computation or information. |
| Generalization | Reuse mechanisms and knowledge across tasks, modalities, goals, and unfamiliar conditions. |

Consider an instrument investigating a material. The model tracks an uncertain response curve, chooses a measurement, and uses the waiting interval to compare explanations. A new reading arrives; it updates its beliefs, changes the next experiment, and records the result. Later, a related material requires fewer measurements because earlier experience has improved its strategy. This sequence connects perception, thought, action, and learning in one research process.

## Research direction

The first design combines a persistent recurrent core, a compact attention workspace, variable internal computation, and learned activity selection. Prediction, action, and value heads share the evolving state. Selective memory and parameter adaptation extend what the model can retain and learn.

Three distinctions guide the work:

- **State and learning:** carrying information through a sequence, retaining an experience, and updating model parameters have different effects and need separate measurements.
- **External time and internal computation:** the world advances while the model thinks. More computation is useful only when its benefit justifies its delay.
- **Local success and transfer:** an effective controller becomes evidence for a broader model only when its mechanisms remain useful across new tasks and conditions.

The research programme compares alternative mechanisms under shared information, interaction, and compute budgets. Architecture choices will follow measured improvements in decision quality, adaptation, and responsiveness.

## Applications

| Setting | The opportunity |
| --- | --- |
| Robotics and embodied systems | Combine perception, manipulation, navigation, and task revision during continuous interaction. |
| Mobile and electromechanical systems | Adapt decisions to changing dynamics, sensor timing, and operating conditions. |
| Autonomous science | Select informative experiments, manage delayed outcomes, and refine explanations over long investigations. |
| Edge intelligence | Sustain useful local decisions within limited memory, compute, and energy budgets. |
| Digital environments | Maintain projects, investigate open questions, and respond to evolving information over time. |

These settings share a requirement: the model must decide while events are still unfolding.

## Read the project

- [Research programme](RESEARCH.md): the scientific questions, hypotheses, and evidence that would establish progress.
- [First technical design](DESIGN.md): architecture, time semantics, data, training, optimization, and evaluation.
- [Research foundations and positioning](ref/README.md): six literature reviews, closest-work comparisons, architectural alternatives, experimental decisions, and a source-checked bibliography.

CDM is an independent model research project. [Eidolve](https://github.com/Cherish0925/eidolve) is a related project and a potential integration environment for persistent knowledge, evidence, and memory. The CDM model and its experiments will have explicit interfaces so they can also operate in other environments.
