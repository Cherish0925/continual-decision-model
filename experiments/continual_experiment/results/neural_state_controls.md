# Recurrent state and workspace across task transitions

This follow-up separates the GRU state h from the Transformer workspace z. Each single-state condition resets the other neural state, outcome replay and fast/slow adapters at task boundaries. The previous h-and-z-only and full-reset conditions provide the other two cells of the comparison. Model weights, ordered instances, initial simulator randomness, observations, tools and action budgets are unchanged.

The new controls execute 736 native tasks using both released v0.10 full models. The original task cohort is reused to explain the previous state effect; these are not independent confirmation results for a new method.

| Repetition | Carried state | All tasks: completed | All: actions | Initial later | First switch | Switch later | First return | Return later |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 201 | Reset | 179/184 | 197.12 | 196.56 | 187.25 | 199.79 | 212.95 | 199.36 |
| 201 | Carry h | 179/184 | 148.58 | 114.32 | 207.10 | 117.67 | 236.15 | 126.25 |
| 201 | Carry z | 179/184 | 197.12 | 196.56 | 187.25 | 199.79 | 212.95 | 199.36 |
| 201 | Carry h and z | 179/184 | 148.58 | 114.32 | 207.10 | 117.67 | 236.15 | 126.25 |
| 202 | Reset | 178/184 | 201.35 | 199.39 | 191.30 | 205.49 | 216.25 | 206.02 |
| 202 | Carry h | 174/184 | 193.80 | 157.51 | 235.00 | 188.10 | 248.85 | 194.20 |
| 202 | Carry z | 178/184 | 201.35 | 199.39 | 191.30 | 205.49 | 216.25 | 206.02 |
| 202 | Carry h and z | 174/184 | 193.80 | 157.51 | 235.00 | 188.10 | 248.85 | 194.20 |

Action means include failed tasks. Initial later and switch later denote subsequent fresh instances sharing the current relation. First return denotes the first fresh instance after an intervening relation; return later denotes subsequent instances in that returning segment.

For both trained models, h-only and h-and-z carry match on every instance in completion, native action count, measurement count and test count. z-only and full reset also match instance by instance on all four outcomes. Thus the observed cross-task effects follow the retained recurrent state. This does not imply that the workspace is unused within a task or that its internal representations are identical.

The trade-off persists across transitions. Carrying h reduces the mean cost of successive same-relation tasks, but increases first-switch and first-return costs. On first-return tasks, repetition 201 completes 19/20 with either h carry or reset; repetition 202 completes 16/20 with h carry and 18/20 with reset. Neither selective reset gives a free improvement that preserves all continuity benefits and removes all transition costs.

## Paired transition effects

Differences are candidate minus reset. Intervals resample complete ordered streams and retain all instances in each sampled stream. The two training repetitions share the cohort and are reported separately.

| Repetition | Carried state | Initial later | First switch | First return | Return later |
| --- | --- | ---: | ---: | ---: | ---: |
| 201 | h | -82.24 [-117.09, -47.73] | +19.85 [-2.30, +47.05] | +23.20 [+1.10, +48.00] | -73.11 [-104.57, -43.16] |
| 201 | z | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] |
| 202 | h | -41.88 [-58.28, -24.53] | +43.70 [+4.05, +85.80] | +32.60 [+3.25, +65.75] | -11.82 [-27.05, +1.64] |
| 202 | z | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] |

## Representation and inference

The 1,024-dimensional GRU state receives public operation events, including task-local object indices, tool/quantity identifiers, returned measurements, execution feedback and query identity. The eight 1,024-dimensional workspace slots repeatedly attend to that GRU state and the current object–quantity evidence ledger. Their slots are not assigned to particular objects. Action and activity readouts concatenate the normalized GRU vector with the normalized mean workspace vector. Additional REFINE activity updates z alone; the GRU update has no direct input from z or the complete evidence ledger.

The released model receives a new initial observation at a task boundary. The initial event has a distinct pattern, so a boundary is implicitly observable, although there is no dedicated boundary input or declared separation between instance-specific and transferable information. Object slots retain their roles: slots 0 and 1 are initial supports, and slots 2 and 3 are targets; the objects and their numerical values change. Reusing these role indices is not itself an implementation error. Obsolete instance context and excessive preference for the recent relation remain hypotheses; these interventions do not identify their encoded content.

## Next training comparison

A concrete candidate is to let the existing GRU update read the current public evidence and the preceding workspace result. Add zero-initialized projections of an evidence summary and the workspace summary to the received-event embedding before the GRU. Initialization exactly preserves the old policy; training can then use the existing GRU reset/update gates to retain useful prior information and revise current-instance content. Workspace computations can affect the recurrent state at the next received observation, without counting an internal computation as a second observation.

Train this feedback connection on continuous streams with fresh objects and coefficients, and propagate future task rewards and received-outcome prediction losses across task boundaries. Any object relabeling must consistently change events, evidence, candidate actions, known/query masks and records; start with permutations within the existing support and target roles. Compare against an equally trained original recurrence on independent development streams, reporting same-relation continuation, first switch, first return, completion and total action cost together. Hidden relation identities and the known A–B–A evaluation schedule are not inputs. This is a proposed experiment, not a demonstrated repair or a novelty claim.

All 20 untreated first episodes per condition reproduce the original completion, action, measurement and test counts. Replay and adapter state are reset in every selective neural condition; they can still form and be used within each task.

[Per-instance outcomes](neural_state_instances.csv) · [Phase summaries and paired contrasts](neural_state_summary.json) · [Previous state-group control](state_controls.md)

Reproduce the new runs with `python -m experiments.continual_experiment.state_controls --state-modes h_only z_only --output runs/continual-experiment/neural_state_controls`. Then run `python -m experiments.continual_experiment.summarize_neural_states --input runs/continual-experiment/neural_state_controls --previous runs/continual-experiment/state_controls_v010/full`. Both original checkpoints and their initial RNG records are required.
