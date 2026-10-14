# CDM: Research Foundations and Positioning

**Literature cutoff: 14 October 2026.**

Continual Decision Model (CDM) studies trainable intelligence that maintains state, chooses activities, acts in an evolving environment, and improves through experience. This reference collection connects that objective to the literature and identifies the comparisons needed to establish a contribution.

The central distinction is between **the capability we want**, **the mechanism we propose**, and **the evidence we have**. A broad objective gives the project direction. A specific mechanism makes an experiment possible. Experimental results determine what a paper can claim.

## Start here

| Question | Document |
| --- | --- |
| What exactly are we trying to build and understand? | [Research positioning](POSITIONING.md) |
| Which existing methods most directly overlap with our ideas? | [Closest-work comparison](COMPARISON.md) |
| What would distinguish a useful new mechanism experimentally? | [Experiments and research decisions](EXPERIMENTS.md) |
| Which sources were checked, and how deeply? | [Bibliography](BIBLIOGRAPHY.md) and [search method](SEARCH_METHOD.md) |

## The intellectual map

| Research thread | Central question | Review |
| --- | --- | --- |
| Sequence architectures and memory | How can finite state retain and expose useful information? | [Sequence models](SEQUENCE_MODELS.md) |
| Internal computation and real time | How can a model develop a representation and decide when computation is worthwhile? | [Internal computation](INTERNAL_COMPUTATION.md) |
| Planning, metareasoning, and revision | How should an agent choose computations and reuse work when circumstances change? | [Planning](PLANNING.md) |
| Continual and meta-learning | How does experience alter state, memory, update rules, and lasting competence? | [Continual learning](CONTINUAL_LEARNING.md) |
| World models and embodied interaction | How do predictions, multimodal state, and actions support sustained behavior? | [World models](WORLD_MODELS.md) |
| Proactivity, exploration, and agent systems | How are activities and goals initiated, and where does their implementation reside? | [Agency](AGENCY.md) |

## What the survey changes

Attention plus recurrence, dynamic computation depth, persistent memory, learned planning budgets, and incremental reuse of computation all have substantial precedents. Recent models also combine short-term computational state with longer-term memory. Their existence is useful: CDM can begin with stronger foundations and more demanding comparisons.

One concrete hypothesis concerns **how ongoing latent computation incorporates new evidence while the model continues to act**. Its potential contribution lies in a specified update mechanism, training procedure, and demonstrated benefit. The experiment programme compares this route with shared recurrent state, explicit learned planning, and adaptive parameter memory. Proactive information acquisition, lasting learning, and transfer remain central to the broader programme.

The initial implementation in [DESIGN.md](../DESIGN.md) supplies a starting experiment. It does not fix the final backbone or establish novelty. The comparison and experiment documents here refine what that experiment must resolve.

## Using the references

[papers.json](papers.json) contains structured records: mechanism, demonstrated scope, relevance to CDM, reading priority, and the location of the inspected evidence. [references.bib](references.bib) provides citation records for the same collection. The bibliography lists the source editions used; preprint dates and later publication dates can differ.

Each review separates reported findings from proposed CDM experiments. A source marked **abstract** supports a limited description of the reported method and evaluation. **Full text** means the cited sections were inspected, not that an independent reproduction has been completed. Source code availability is not treated as a reproduced result.

This is a broad, curated research map with explicit coverage and reading depth. New work should be added when it changes a design decision, a baseline, or the meaning of a proposed claim.
