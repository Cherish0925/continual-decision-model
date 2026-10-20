# Pretrained components for the first multimodal CDM

Selection updated: **20 October 2026**. This note supplements the literature collection; it does not change the reported size or reading depth of that collection.

## Reuse map

| Reused component | Use in CDM | Run frequency | Learned connection |
| --- | --- | --- | --- |
| Released CDM lifetime core | Persistent state, prediction and action/activity policies | Every decision event | New perception/semantic paths train around the original core |
| SigLIP 2 Base FixRes | Image patches and short-text representations | When the corresponding input changes | Modality projections and event attention |
| Qwen3.5-2B | Native image/text semantics and language generation | When the task needs semantic work | Eight context tokens from a state-conditioned cross-attention reader |
| Qwen3.6-35B-A3B | Offline action/experiment trajectory generation | Training-data collection phases | Action imitation and outcome-grounded distillation |

The active decision model contains the fast core and the first two pretrained input paths. The teacher is a data-production role. Reuse the native semantic model's processor, visual positions, fusion and decoder together; an isolated language or vision block does not preserve all of those learned functions.

## Reusing CDM's trained weights

Release [v0.4.0](https://github.com/Cherish0925/continual-decision-model/releases/tag/v0.4.0) contains 35 second-round models and their evaluations. The checkpoint loader in [cdm/multimodal.py](../cdm/multimodal.py) reads the saved configuration and loads all core tensors strictly. Source dimensions and task/action meanings are retained. It does not resize weights to the separate width-512 reference configuration.

Structured-only input reproduces the original state and readouts. When new inputs are supplied, the zero-initialized input gain preserves the initial fast state update; additional workspace tokens still need task training. Reusing a checkpoint provides a trained starting point, not a measured transfer result.

## Selected input model

**Google SigLIP 2 Base, patch16, 224px, FixRes** supplies a pretrained image tower and a paired short-text tower. Freeze both for the first CDM training run; learn width-512 projections and the persistent core.

| Property | Selected value |
| --- | --- |
| Checkpoint | [google/siglip2-base-patch16-224](https://huggingface.co/google/siglip2-base-patch16-224) |
| Revision | `75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2` |
| License | Apache-2.0, as stated by the publisher |
| Image input | 224×224, 16×16 patches |
| CDM image representation | 14×14 encoder patches pooled spatially to 4×4, then projected |
| Text input | At most 64 tokens per encoded event, using the checkpoint processor |
| Implementation | `AutoModel` resolves this FixRes checkpoint to `SiglipModel` |

Sources: [publisher model card](https://huggingface.co/google/siglip2-base-patch16-224), [checkpoint configuration](https://huggingface.co/google/siglip2-base-patch16-224/blob/75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2/config.json), [SigLIP 2 documentation](https://huggingface.co/docs/transformers/model_doc/siglip2), [Siglip API](https://huggingface.co/docs/transformers/v4.57.1/en/model_doc/siglip), and [paper](https://arxiv.org/abs/2502.14786).

This is an engineering selection: one checkpoint supplies aligned image/text representations and an established inference interface. The CDM implementation retains spatial and text tokens so the workspace can attend to them directly. It does not use image classification labels as its observation state.

The publisher reports the combined checkpoint at roughly 0.4B parameters. The paired text vocabulary contributes substantial storage; the entire checkpoint is not an 86M-parameter vision-only model. Freezing it avoids its optimizer states and backward pass, while its forward cost still belongs in end-to-end latency.

The text tower is not a generative language model. The frontend does not itself supply scientific reasoning, dialogue, tool use or online learning. Those functions belong to the trained CDM core and task/output modules. Long instructions use the semantic model below; audio requires another encoder; the fast visual path handles timestamped frames.

## Selected semantic model

**Qwen3.5-2B** is the initial online semantic model. Its released model includes a vision encoder, a causal language model, a native multimodal processor and a language output head. The dense language model uses Gated DeltaNet and full-attention layers; it is distinct from the larger MoE teacher.

| Property | Selected value |
| --- | --- |
| Checkpoint | [Qwen/Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B) |
| Revision | 15852e8c16360a2fea060d615a32b45270f8a8fc |
| License | Apache-2.0 |
| Hidden width / language layers | 2,048 / 24 |
| Initial CDM input budget | 4,096 tokens per call |
| Connection | Final hidden states → learned projection and cross-attention → 8 tokens at the CDM core width |
| Deployment cache | Compress once per completed input; refresh age features between calls |

Sources: [publisher model card](https://huggingface.co/Qwen/Qwen3.5-2B), [pinned configuration](https://huggingface.co/Qwen/Qwen3.5-2B/blob/15852e8c16360a2fea060d615a32b45270f8a8fc/config.json), and [Transformers implementation documentation](https://huggingface.co/docs/transformers/model_doc/qwen3_5).

The implementation reads the final layer from the base model, avoiding vocabulary logits when only context is needed. A separate generation method retains the pretrained decoder. Encoding an input does not run a generated multi-step reasoning trace. The reader learns from CDM action/outcome objectives and respects padding and unavailable results. Processor outputs include modality token types; keep them when passing image/text inputs to the model.

This selection supplies useful pretrained semantics within the available hardware budget. Its value for CDM must be measured against the same core with SigLIP alone and a direct pretrained-model agent. The implementation is tested with Transformers 5.4.0; GPU kernel performance remains to be measured with compatible DeltaNet kernels.

## Selected offline teacher

**[Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B)**, revision **995ad96eacd98c81ed38be0c5b274b04031597b0**, is selected for offline trajectory generation and distillation. The publisher releases it under Apache-2.0. Its MoE architecture has substantially more stored weights than the parameters activated per token; reserve memory for the full weights and inference state.

Use the teacher to propose actions, observations and experiment sequences, then execute them and learn from task outcomes. Keep its inference outside the fast control path. Teacher collection and distillation are planned training work; this change implements the online semantic connection.

## Selected external application suite

[DiscoveryWorld](https://github.com/allenai/discoveryworld) provides scientific interaction tasks, agent examples, an API and released interaction data. Use its official task scores and data protocol for the external application comparison. The public API is step-based; do not interpret a standard run as an independently timed environment. A wall-clock extension must be named and reported separately.

The CDM task adapter is not implemented in this update. Its job is to map allowed visual/text observations, task-defined actions and returned outcomes to the common event schema. Internal simulator state can be used for scoring where the benchmark permits it, but is not an additional policy input.

## Attribution and comparisons

The encoder, GRU, attention, low-rank adapters, actor–critic optimization and replay are established methods. The CDM experiment concerns the training of persistent computation, activity choice and adaptation over a changing lifetime. Give all comparison models the same input encoder when measuring that contribution. Relevant prior work is indexed in [COMPARISON.md](COMPARISON.md), [INTERNAL_COMPUTATION.md](INTERNAL_COMPUTATION.md) and [CONTINUAL_LEARNING.md](CONTINUAL_LEARNING.md).
