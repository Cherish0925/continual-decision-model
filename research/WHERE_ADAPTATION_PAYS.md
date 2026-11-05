# Where online adaptation pays, and where it cannot

Written 5 November 2026. The same architecture now has two measured outcomes
that point in opposite directions. This note states what separates them, what
that implies about the current benchmark choice, and the comparison that would
settle the project's actual claim.

## Two settings, opposite results

**Process lifetimes.** Trained through the sequence of learning steps the agent
actually takes, the instance's own updates carry half of performance: 0.153 ±
0.003 with the updates against 0.307 when they are discarded at matched
compute, holding on three held-out families and against a real clock
([results](../experiments/lifetime/RESULTS.md#second-round-training-through-deployment)).

**DiscoveryWorld native policies.** Across the 32-condition input and
initialization study, summing successful executions over all cells
([table](../experiments/discoveryworld/FULL_SUITE.md#input-state-and-decoding-completed-comparison)):

| Factor | | |
| --- | ---: | ---: |
| Policy | CDM 24 | GRU 39 |
| Observation features | lexical 47 | frozen SigLIP text 16 |
| Action history | 0 steps 32 | 8 steps 31 |
| Core initialization | inherited 31 | random 32 |

Two seeds per cell on sixteen instances, so no single cell is decisive. The
pattern across cells is not: the architecture, the inherited core, the
pretrained text encoder and the explicit feedback input are each null or
negative. These are not four separate puzzles. They are one fact.

## What separates the two settings

In the process families the per-episode unknown is the instrument's response
surface. It is continuous, identifiable from a few dozen readings, and it
determines the optimal action directly. Feedback arrives every tick. Adaptation
pays because the thing to be learned is learnable within the episode from the
episode's own data, and because the agent can already act.

In DiscoveryWorld the binding constraint is upstream of any of that. Before an
agent can benefit from a measurement it must know that a soil sample is carried
to an instrument, that the instrument is activated, that its readout is read.
That knowledge is not identifiable from sixteen thousand recorded actions; it
is world knowledge and procedural competence.

The published reference makes the size of that constraint explicit. Zero-shot
GPT-4o scores 29/120 with ReAct, 20/120 with Plan+Execute, 23/120 with
Hypothesizer ([Jansen et al., NeurIPS 2024, Table 4](https://papers.nips.cc/paper_files/paper/2024/file/13836f251823945316ae067350a5c366-Paper-Datasets_and_Benchmarks_Track.pdf)).
That 24% comes from pretraining, not from any adaptation mechanism, and all
three variants score 0/15 on Reactor Lab. A 35.9 M-parameter model reading
lexical features, trained on 150 demonstrations, is not short of an
architecture. It is short of everything GPT-4o brings before the first action.

This explains every entry in the factor table at once. The inherited core
encodes instrument response dynamics, which the bottleneck does not use, so it
is worth nothing. The semantic encoder needs more data than 16,015 actions to
beat features that key directly on object names, so it is worth less than
nothing. Explicit feedback cannot help an agent that does not know what to do
instead. And CDM's additional machinery — workspace refinement, instance
adapters, outcome prediction — adds optimization burden against a constraint it
does not address, so the plain recurrent policy does better.

## What the table does and does not say about the selected architecture

Two entries bear directly on choices recorded in [DESIGN.md](../DESIGN.md), and
they must be read at their actual scope.

**Reusing a released CDM core gives nothing here** (31 against 32). The cores
were trained on instrument response dynamics under a different action schema;
only the core weights transfer, while input projections and action heads are
new. For a task whose constraint is procedural competence there is nothing for
those weights to carry. This is evidence against inheritance as a default for a
new action schema, not against inheritance within a task family, where the
process-lifetime results already show it working.

**The frozen encoder arm used SigLIP text features on trajectories containing
no images.** It therefore tests a frozen semantic text encoder at this data
scale, where it loses to lexical features 16 to 47. It says nothing about
SigLIP as an image encoder, which is the role DESIGN.md actually assigns it.
The usable conclusion is narrower and still worth acting on: with 16,015
training actions, a 768-dimensional frozen text embedding is a worse input than
features that key directly on object names. Pretrained representations need
either more data or a task where their structure is the relevant one.

## The consequence for the benchmark choice

DiscoveryWorld was selected to test proactive scientific behaviour. Its
measured bottleneck at the available data scale is pretrained competence. A
correct implementation of this project's mechanism would still score near zero
end to end. Continuing the 32-condition protocol into matched continuations and
autonomous interaction spends compute on a configuration where all four factors
are already null, against a ceiling set by something the architecture does not
contain.

That is a benchmark–hypothesis mismatch, not an implementation failure. The
engineering of the last two weeks is sound: the replay is exact across 49,304
decisions, the interventions are correct, the records are honest about scope.
The error was in what the measurements were asked to decide.

## The claim this project can actually settle

CDM's claim is about which observation or experiment to perform next, and how a
returned result changes the next decision. That claim is separable from
execution, and five of the eight themes have exactly that structure: Plant
Nutrients, Archaeology Dating, Proteomics, Combinatorial Chemistry and Reactor
Lab each require running an experiment, reading a result and revising a
hypothesis.

The comparison that settles the claim holds the executor fixed and varies only
the part CDM owns:

- a competent executor handles navigation, object handling and instrument
  operation — the frozen vision-language model already selected in
  [DESIGN.md](../DESIGN.md), or the existing teacher;
- the executor alone chooses which experiment to run next, by its own
  reasoning, as the control;
- the same executor with CDM selecting the experiment and holding the belief
  across results, as the treatment;
- both report completions, number of native actions, and wall-clock decision
  latency on the same instances.

EvoBelief reports GPT-5.4 completing 10/12 and 7/12 Reactor Lab and 9/12 and
10/12 Plant Nutrients instances under a stated action and call budget
([Wang et al.](https://arxiv.org/abs/2610.12789)), which gives an external
reference on two of the five themes at a matched protocol.

This is a decomposition, not a concession, provided both the decomposed and the
end-to-end numbers are reported. It measures the mechanism where the mechanism
applies, against a control that differs in exactly one component.

## What the two settings together are worth

A result that holds everywhere is rarely true. The pair here is more useful than
either alone, and it is the shape of a complete argument:

> Training a model through its own deployed update sequence makes online
> adaptation pay when the episode's unknown is identifiable from the episode's
> own observations and the agent can already act. Where the binding constraint
> is instead pretrained world knowledge or procedural competence, the same
> mechanism contributes nothing, and a plain recurrent policy does as well or
> better.

The first clause is established on process lifetimes with matched-compute
controls. The second is established by the factor table above and calibrated by
the published GPT-4o scores. Delimiting a mechanism is a result; the negative
half is not a failure to be repaired before publication.

## Recommended sequence

1. Stop the 32-condition continuation. Record the factor totals as its outcome.
2. Build the fixed executor interface and run the three-arm comparison above on
   the five identification themes.
3. Keep the process families as the controlled setting for the positive result
   and for any further mechanism work.
4. Do not add components to the native policy. Nothing in the evidence suggests
   a missing component; the evidence suggests a missing precondition.
