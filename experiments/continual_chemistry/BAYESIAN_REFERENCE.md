# Finite-family chemical experiment reference

This independent reference uses the public simulator's reaction function as a
prior. It does not receive an instance's solution, seed, hidden recipe or task
score. Its assumptions are stronger than those of the learned response model;
the comparison measures the remaining gap to model-based experiment design
with a known response family.

Hypotheses are the 55 or 161 primitive integer ingredient proportions through
six total parts, with a uniform prior. For a query mixture q and hypothesis h,
the cosine similarity determines an intrinsic rust endpoint R: similarities
**at least** 0.99, 0.66 and 0.33 give endpoints 0, 1 and 2, respectively;
otherwise R=3. Normalizing proportions before computing the cosine is
equivalent to normalizing their count vectors. With current rust r, the actual
return is Y=min(r,R). Each real experiment removes hypotheses inconsistent
with its before/query/after record; unchanged rust therefore remains censored
feedback. An empty posterior is reported as a model-family mismatch, without
consulting the hidden recipe.

The known response family is consistent with all 10,610 training returns.
The public executor charges a recipe

`6 + total parts + nonzero components + 2 × (a preceding experiment exists)`

operations, followed by four exit operations when the key is clean. The formula
matches all 10,610 recorded recipe costs exactly. Actual native operation
counts, including preparation and termination, determine reported results.

The one-experiment rule maximizes posterior completion probability divided by
expected cost. The two-experiment rule enumerates possible first returns and
chooses a second experiment separately in each nonterminal branch. It maximizes
two-experiment completion probability divided by expected cost, including
termination cost. The finite fractional objective is solved by successive
maximizations of probability minus the current ratio times cost. The policy
then executes only its first experiment and replans after receiving reality.
This is a limited-horizon Bayesian reference, not a globally optimal policy.
Both methods use identical hypotheses, public histories, tools and candidate
constraints; no entropy coefficient or development-tuned threshold is added.

| Cohort | Method | Completed | Mean native operations | Mean inference seconds |
| --- | --- | ---: | ---: | ---: |
| Original 30 development instances | One experiment | 30/30 | 87.33 | 0.00474 |
| Original 30 development instances | Two experiments | 30/30 | 93.33 | 0.01920 |
| Separate 60-instance cohort | One experiment | 60/60 | 98.60 | 0.01015 |
| Separate 60-instance cohort | Two experiments | 60/60 | 101.40 | 0.03975 |

After the development run, both unchanged methods were declared as additional
comparisons on the already evaluated 60-instance learned-model cohort. They
use the same saved initial environment states and 1,000-operation budget.
They were not part of the original six-condition declaration, and neither
method was tuned on the confirmation results. The 60 instances represent 22
named recipe groups, not 60 independent chemical laws.

Two-experiment planning adds 2.80 native operations on average compared with
the matched one-experiment rule; its 22-group paired 95% interval is
[−3.98, 9.28]. It also uses more decision computation. Thus this comparison
does not establish a benefit from additional planning. Both methods provide a
much stronger reference than fixed recipe enumeration, which uses 367.38
operations on this cohort. The learned joint models use 143.35 and 278.37;
their weaker prior information must accompany this comparison. All complete
60/60 tasks. Inference timing is actual concurrent CPU execution, not a
standardized throughput benchmark.

[Development records](results/bayesian_development.json),
[confirmation comparisons](results/bayesian_confirmation.json) and
[120 confirmation executions](results/bayesian_confirmation_instances.csv)
preserve both methods, all outcomes, computation time and paired group
intervals. The reference reads the official response-family definition from
the study specification; it does not import instance attributes from
DiscoveryWorld's hidden world objects.

```bash
python -m experiments.continual_chemistry.bayesian_reference --check-data
PYTHONHASHSEED=0 python -m experiments.continual_chemistry.bayesian_reference \
  --cohort development --output runs/continual_physics/chemistry_bayesian/development
PYTHONHASHSEED=0 python -m experiments.continual_chemistry.bayesian_reference \
  --cohort confirmation --output runs/continual_physics/chemistry_bayesian/confirmation
```
