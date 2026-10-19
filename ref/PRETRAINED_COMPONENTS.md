# Pretrained components for the first multimodal CDM

Selection date: **19 October 2026**. This note supplements the literature collection; it does not change the reported size or reading depth of that collection.

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

The text tower is not a generative language model. The frontend does not itself supply scientific reasoning, dialogue, tool use or online learning. Those functions belong to the trained CDM core and task/output modules. Long documents require ordered chunks; audio requires another encoder; video is a sequence of timestamped visual events.

## Selected external application suite

[DiscoveryWorld](https://github.com/allenai/discoveryworld) provides scientific interaction tasks, agent examples, an API and released interaction data. Use its official task scores and data protocol for the external application comparison. The public API is step-based; do not interpret a standard run as an independently timed environment. A wall-clock extension must be named and reported separately.

The CDM task adapter is not implemented in this update. Its job is to map allowed visual/text observations, task-defined actions and returned outcomes to the common event schema. Internal simulator state can be used for scoring where the benchmark permits it, but is not an additional policy input.

## Attribution and comparisons

The encoder, GRU, attention, low-rank adapters, actor–critic optimization and replay are established methods. The CDM experiment concerns the training of persistent computation, activity choice and adaptation over a changing lifetime. Give all comparison models the same input encoder when measuring that contribution. Relevant prior work is indexed in [COMPARISON.md](COMPARISON.md), [INTERNAL_COMPUTATION.md](INTERNAL_COMPUTATION.md) and [CONTINUAL_LEARNING.md](CONTINUAL_LEARNING.md).
