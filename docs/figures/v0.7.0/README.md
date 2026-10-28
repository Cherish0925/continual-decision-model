# CDM v0.7.0 architecture figures

These figures describe release `v0.7.0`, commit
`67c1d8f41d24d0e1572de224adffa5897f177fc0`. The main diagram follows the
original action reader with the transferred numerical experiment selector.
This is the configuration reported at 23/24 completed Reactor Lab instances.

## Figures

1. **Implemented architecture and offline training.**
   [SVG](cdm_v0.7_architecture.svg) · [PNG](cdm_v0.7_architecture.png)
2. **Persistent core and separate prediction-feedback configuration.**
   [SVG](cdm_v0.7_core.svg) · [PNG](cdm_v0.7_core.png)
3. **Related architectures: CDM, I2A, Step-DAD and EvoBelief.**
   [SVG](cdm_related_architectures.svg) · [PNG](cdm_related_architectures.png)

[Three-page vector PDF](cdm_v0.7_architecture_comparison.pdf)

### Figure 1 caption

**CDM v0.7.0 in DiscoveryWorld Reactor Lab.** Frozen SigLIP 2 features and
public numerical observations enter a persistent GRU and Transformer workspace.
The action reader scores public candidates using the neural state and explicit
numerical features. A separate numerical model maintains nine affine hypotheses
from observed calibrations. A learned selector predicts the success and cost of
executing each candidate experiment. The experiment executor retains the chosen
plan through setting changes and local probing, and records unsuccessful choices
within the instance. The public action interface dispatches ordinary policy
actions or numerical-dialogue operations, and new observations update both the
neural and numerical state. Blue modules have fixed weights during this study;
green identifies the selector trained in v0.7.0; orange identifies explicit
state and numerical procedures. The dashed connection transfers offline-learned
selector weights. Validation branches do not update weights.

The numerical-only selector receives ten features per candidate plan and has
317,314 allocated parameters. Its state input is zeroed by `PlanValue.forward`;
the state branch remains allocated. This does not remove the neural state from
the ordinary action reader. The figure aggregates feature preprocessing and
execution control; it does not depict every Python call.

### Figure 2 caption

**One observation update of the reference-reader core.** Attention reads the
event tokens using the previous GRU state as its query. Normalization, an input
bridge and the source event MLP produce the GRU input. Eight persistent slots
then undergo self-attention, cross-attention to the current state and event
tokens, and an MLP, with residual additions and final slot normalization.
The reference path applies this shared block once per observation. The action
reader receives the normalized fast state concatenated with the normalized mean
of the slots. Measurement-record embeddings are part of the event tokens.
The separate predictive-parent configuration forecasts candidate-action
consequences and feeds the top eight candidates' forecast tokens into two extra
workspace updates. These cycles revise predictions; they are not successive
simulated world transitions. This predictive path is not active in the transferred
reference-reader configuration shown in Figure 1.

### Figure 3 caption

**Selected computational mechanisms in related architectures.** CDM combines
persistent neural state, explicit numerical hypotheses and a learned experiment
selector. I2A encodes model-generated rollouts as additional policy context.
Step-DAD periodically adapts a design policy as experimental observations
accumulate. EvoBelief maintains and revises external world rules while its
language model stays frozen. EvoBelief's actor proposes actions without reading
the validated rule text; the rules affect subsequent prediction and planning.
The drawings are schematic redraws, not copies of the authors' figures and not
performance rankings. Color in this comparison identifies component roles;
it does not imply that every blue component in every paper was frozen in training.

## Implementation sources

All links below are pinned to the illustrated release.

| Figure element | Source |
| --- | --- |
| SigLIP 2 features and 4-by-4 image pooling | [`features.py`](https://github.com/Cherish0925/continual-decision-model/blob/v0.7.0/experiments/discoveryworld/features.py) |
| Token projection, attention ingest, state update and candidate scores | [`cdm/digital.py`](https://github.com/Cherish0925/continual-decision-model/blob/v0.7.0/cdm/digital.py) |
| GRU, slot attention, context attention, MLP and state readout | [`cdm/core.py`](https://github.com/Cherish0925/continual-decision-model/blob/v0.7.0/cdm/core.py) |
| 125-dimensional numerical action features and residual reader | [`belief_actions.py`](https://github.com/Cherish0925/continual-decision-model/blob/v0.7.0/experiments/discoveryworld/belief_actions.py) |
| Ten plan features, selector and persistent executor | [`experimental_plans.py`](https://github.com/Cherish0925/continual-decision-model/blob/v0.7.0/experiments/discoveryworld/experimental_plans.py) |
| Selector objectives, state ablation and offline optimization | [`train_plans.py`](https://github.com/Cherish0925/continual-decision-model/blob/v0.7.0/experiments/discoveryworld/train_plans.py) |
| Separate prediction-feedback computation | [`predictive_actions.py`](https://github.com/Cherish0925/continual-decision-model/blob/v0.7.0/experiments/discoveryworld/predictive_actions.py) |
| Protocol and reported outcomes | [`PREDICTION.md`](https://github.com/Cherish0925/continual-decision-model/blob/v0.7.0/experiments/discoveryworld/PREDICTION.md), [`RESULTS.md`](https://github.com/Cherish0925/continual-decision-model/blob/v0.7.0/experiments/discoveryworld/RESULTS.md) |

The persistent state changes during evaluation; neural weight updates do not
occur in this configuration. The scientific-task path has a fixed workspace
update count. It does not activate the process-control experiments' adaptive
activity policy or online neural adapters. State and failed-plan records are
reset between these benchmark instances. The input includes exact public
numerical fields, not only image and text embeddings. Output is a structured
native action, not a generated natural-language answer.

## References

1. Weber et al. **Imagination-Augmented Agents for Deep Reinforcement Learning.**
   NeurIPS 2017. <https://arxiv.org/abs/1707.06203>.
2. Hedman et al. **Step-DAD: Semi-Amortized Policy-Based Bayesian Experimental
   Design.** ICML 2025, PMLR 267:22904-22923.
   <https://proceedings.mlr.press/v267/hedman25a.html>.
3. Wang et al. **EvoBelief: Falsifiable World Models for Self-Improving LLM
   Agents.** 2026 preprint. <https://arxiv.org/abs/2610.12789>.
   The comparison uses the method and Appendix B-C description of its structured
   world model, actor and prediction/planning interfaces.

## Reproduce the figures

```bash
python -m pip install matplotlib numpy
python docs/figures/v0.7.0/render.py
```

The renderer creates SVG files with editable text, 180-dpi PNG previews and one
three-page vector PDF with embedded fonts. The figures describe the pinned
implementation rather than a proposed future architecture.
