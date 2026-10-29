# Native confirmation results

The method and settings were fixed after the separate 30-instance development
cohort. Confirmation uses 20 ordered six-instance streams, 120 new instances in
total, the same saved initial environment randomness and the same native
execution tools as the continual-physics comparison. All runs completed without
execution errors; task failures remain in every operation and timing average.

| Method | Completed / 120 | Mean native actions | Decision computation / instance | Total time / instance |
|---|---:|---:|---:|---:|
| Bayesian calibration, task reset | 117 | 133.34 | 0.516 s | 6.782 s |
| Bayesian calibration, forgetting memory | 118 | 98.41 | 0.396 s | 5.110 s |
| Frozen physical-belief parent, repeat 1 | 117 | 106.68 | — | 5.287 s |
| Frozen physical-belief parent, repeat 2 | 119 | 102.84 | — | 5.245 s |

Retaining physical relationships reduces operations by 34.93 per instance
(95% paired whole-stream bootstrap interval: 25.14–44.78 fewer operations).
The completion difference is one instance. This establishes useful memory in
this classical reference on these ordered streams; it does not establish a
neural continual-learning advantage.

The memory reference uses 8.27 and 4.43 fewer mean operations than the two
frozen parents. Both corresponding confidence intervals cross zero. It
completes one more instance than the first parent and one fewer than the
second. The current evidence does not establish overall superiority to the
strongest parent.

The reference has synchronous decision execution, no learned activity policy,
and no artificial result-delivery ticks. Timings are observed CPU wall times,
not a claim that waiting and internal computation have been matched. Decision
computation covers ledger construction, posterior sampling and action choice;
total time also includes bootstrap, environment execution and final memory
storage. Parent phase timings were not recorded and are left blank.

[Per-instance values](results/confirmation/instances.csv) and
[whole-stream paired comparisons](results/confirmation/summary.json) include
return-after-interference comparisons. The first task after a relationship
returns uses 121.65 actions with reset and 67.45 with memory. The exact intervals
and both return positions are in the machine-readable results. On first return,
mean measurement requests fall from 9.55 to 5.60 and setting-test requests from
4.80 to 2.35. The first tested relationship matches the actual task relationship
in 7/20 reset instances and 18/20 memory instances. Relationship identity is
used only for this post-execution analysis and is never a model input.

The two remaining memory-reference failures concern first encounters with a
radiation relationship: one initial task and one task switch. Neither has an
earlier matching relationship in its own stream. These
remain failures of this reference, including its approximate input-noise model
and myopic measurement planner. No parameters were changed after confirmation.
