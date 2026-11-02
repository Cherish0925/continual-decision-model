# Experimental memory: results, corrections and next comparison

**2 November 2026. Source revision: `aecc237ecb191685d7e3b4f9f80565a78fdc9976`.** This synthesis uses the released native records, the November 1 analytical revision and the current implementation. It accompanies the [model and research direction](CONTINUAL_AGENT.md). No new model training or native task execution is reported here.

The useful result is that retained experimental observations can substantially reduce work when a relationship returns. The unresolved question is how to obtain that benefit without expensive inappropriate reuse, and how to extend it beyond a known response family. Continued-state architecture, learned memory reading and autonomous computation have not yet established separate advantages in these comparisons.

## 1. Recomputed native results

The [calculation script](analysis/v014_reanalysis.py) reads the released CSVs, checks paired task identities and derives the following values. Its [JSON output](analysis/v014_summary.json) includes source hashes, difficulty/exposure breakdowns and paired stream bootstrap intervals.

| Primary cohort: 240 chemical tasks per condition | Mean operations, run 911 | Mean operations, run 912 |
| --- | ---: | ---: |
| Frozen initial model | 78.22 | 84.30 |
| Six ledgers, fixed reader | 77.65 | 81.98 |
| Twelve ledgers, same fixed reader | 69.77 | 72.43 |
| Twelve ledgers: new tasks only | 95.99 | 100.73 |
| Initial model: new tasks only | 79.17 | 84.80 |
| Twelve ledgers: returning tasks only | 43.55 | 44.12 |
| Initial model: returning tasks only | 77.28 | 83.79 |

Every condition in this table completes all its tasks. Relative to the initial model, twelve ledgers save **10.80% / 14.08%** overall. Candidate-minus-initial differences are −8.45 / −11.87 operations; 95% paired stream bootstrap intervals are [−17.12, −0.12] / [−19.87, −4.81]. These intervals resample the twenty complete streams 4,000 times. They describe this cohort, not uncertainty over all possible training runs.

Two distinct comparisons must remain separate:

- Six versus twelve ledgers uses the same weights and reader. It measures a capacity change.
- Twelve-ledger continuation versus the frozen initial model changes both weights and memory. It does not establish that continuation training is necessary. **The frozen initial model plus twelve ledgers and fixed weighting is the missing comparison.**

Capacity was explored after the primary results. On the second, 120-task cohort, twelve versus six ledgers gives **69.97 versus 78.02** and **75.02 versus 85.38** operations, with every task completed. This supports the capacity effect on another cohort; it does not supply the missing initial-model comparison. The two training runs share test tasks and are not twice as many independent test instances.

These findings update the earlier conclusion that the project has no useful cross-task memory. They do not establish durable parameter learning, recovery through neural state alone, or superiority to independently trained memory/meta-RL methods.

## 2. What the Bayesian calculation does and does not establish

The corrected analytical replay constructs a prior from the **320 training episodes**, rather than the test cohorts. It counts sorted nonzero recipe proportions separately by difficulty and assumes ingredient labels are exchangeable. This gives 19 target proportions for Normal and 46 for Challenge. The version reporting the numbers below places zero mass outside the observed proportion structures.

| Primary cohort, mean operation cost | Normal | Challenge |
| --- | ---: | ---: |
| Twelve-ledger CDM, run 911; native records | 58.98 | 80.56 |
| Twelve-ledger CDM, run 912; native records | 63.11 | 81.74 |
| Reported Bayesian replay, no cross-task memory | 69.4 | 93.9 |
| Reported Bayesian replay, with memory | 60.0 | 79.9 |

The last two rows are **retained analytical replay results**, not new native executions and not recomputed by the script accompanying this document. The replay uses the official cosine response, deterministic outcome censoring, operation-cost accounting and a specified memory prior. Normal uses an exact finite dynamic program; Challenge uses an information heuristic. The companion script independently reconstructs the training prior and solves the smaller action example in the next section.

The replay assumes a 0.55 return probability, shared uniformly among known earlier target identities, with the fresh branch excluding those identities. The current training generator is different: it samples distinct recipe groups, forces the first two contexts to be new, and eventually forces returns when its supply of contexts is exhausted. Episode-frequency weighting is also different from uniform recipe-group sampling. A training-derived prior avoids choosing it from the test answers; it does not automatically equal the generator's predictive distribution.

The reference additionally knows the response family and identifies past latent targets, while the learned model reconstructs forecasts from observations. Its mean is not a universal lower bound for the learned policy. In Challenge, heuristic expected cost is an attainable cost under that reference model, not the optimum. Similar means therefore do not imply that only a few operations of improvement remain. A claim of a performance ceiling would require a justified lower bound under matched information, actions and task generation.

Whether a relationship has returned may itself require an experiment. A nonzero new-task penalty can therefore be part of an optimal tradeoff under a specified prior; it is not automatically a memory bug. A matched model and prior are needed to separate that identification cost from avoidable choices. Do not train toward zero new-task overhead by rewarding a task identity that the deployed policy cannot observe.

For the next reference implementation, expose both the true training-generator prior and a fitted prior, include probability for unfamiliar structures, and vary return assumptions on development streams. Count the cost of identifying a past relationship, or mark exact historical identities as extra information. Preserve the official task results while testing other response laws in a separately named extension.

## 3. Recomputed example: an experiment changes later observability

Consider remembered targets A=(1,1,1) and B=(1,3,0), each with probability 0.275. The remaining 0.45 is distributed over the other seventeen Normal targets according to the training-derived prior. Initial rust is three. There are 55 primitive candidate recipes with at most six total parts. Each row fixes the first action and follows the exact optimal continuation for this finite model.

| First action | Expected total cost | Immediate success probability | Information about the target, nats |
| --- | ---: | ---: | ---: |
| A=(1,1,1) | 79.86 | 0.275 | 0.588 |
| B=(1,3,0) | 51.36 | 0.275 | 1.242 |
| Optimal first action (1,4,0) | 51.06 | 0.275 | 1.268 |

Costs include fourteen preparation operations and four exit operations. The [script](analysis/v014_reanalysis.py) enumerates all candidate actions and posterior subsets; it makes no approximation to this particular finite problem.

**Trying the better remembered recipe is only 0.30 operations worse than optimal.** The large cost belongs to trying A, which makes less useful distinctions before changing the apparatus state. This is a counterexample to immediate-progress selection, not a demonstration that memory generally causes the large penalty.

In the actual task, observed rust is `min(previous rust, latent response)`. A reduction to rust one can hide distinctions that would have remained visible at rust three. Future value must therefore depend on both the updated belief and the changed apparatus. Information gain alone need not minimize task cost.

The current trained policy is **not** the bare immediate-progress rule. `ReactionExperimentPolicy.experiment_logits` combines `4 * log(progress / cost)` with a learned residual, and the residual is trained with task-return feedback. The finite example does not establish which action that trained policy takes. Use its released action traces, or a matched execution, before attributing the example's failure to it.

## 4. Corrections that matter for the next implementation

| Issue | Correct interpretation or implementation |
| --- | --- |
| Information about archive identity | With no archives there is only one fresh component, so identity information is zero even when observations distinguish unfamiliar relationships. Represent alternatives within the fresh component and compute information about the relationship. |
| Entropy under noisy observations | Predictive entropy includes observation noise. Subtract expected conditional entropy to obtain information gain. The deterministic special case is not a general formula. |
| The former `a3_bayes_frontier.py` baseline | A full-cohort no-memory mean was subtracted from conditional-subset costs. This creates nonzero apparent gains even with memory disabled. Compare the same tasks and subset weights; the identical-policy difference must be zero. The former frontier and “16.6 to 11.7” claim are superseded. |
| Sweeping one prior scalar | The best scalar setting in a limited sweep does not establish the best possible reader, representation or action policy. Keep scalar tuning separate from method comparison. |
| “The actor has no task feedback” | Incorrect for the current code. The selector residual receives task-return learning; the direct-credit variant also differentiates through mixed memory predictions while detaching individual experts. |
| “The selector cannot see collapse probability” | The residual receives four marginal outcome probabilities. For known rust, an immediate collapse probability is derivable from them. An informative comparison would hold these marginals fixed while changing posterior partitions and future action opportunities. |
| Resetting neural state | A reset between apparatuses tests cross-task h/z carry. It does not remove within-task recurrence or show that an MLP is sufficient. Keep h/z unchanged in the first selector comparison. |
| Caching archive predictions | Their current implementation conditions on h/z and predictor weights. A cache made once per task can become stale as either changes. Cache only invariant components, or train and compare the changed model explicitly. |
| Same successful recipe implies same relationship | Not generally true. Nearby targets or multi-solution laws can share a solution and differ on other queries. Retain distributions or distinguishable records instead of unconditional merging. |
| Archive count implies return probability | Eviction and duplicate records change which old laws are available. Normalize component mass separately from particle/slot count; learn compatibility with available memory rather than interpreting global recurrence as available recall. |
| No gain from a greedy information rule | This can reject that rule. It does not reject future-cost selection, learned observation models, or a richer predictive belief. |
| Equal first-observation entropy | It does not match sequential identifiability, achievable success, costs or planning difficulty across generators. Measure these properties separately. |

The shared-solution caveat has a simple calculation: targets (1,3,0) and (1,4,0) are both solved by query (1,4,0), but query (2,1,0) has latent levels one and two respectively. The second target is outside the nineteen-target training prior. This is a reason to preserve distinctions in broader task families, not a reported failure in the existing cohort.

## 5. The next comparison and its interpretation

Use four initial conditions with the same prediction model, action tools and within-task h/z:

1. Frozen initial policy without cross-task ledgers.
2. The same frozen policy with twelve ledgers and fixed weighting.
3. That model with a relationship-information selector, leaving the predictor unchanged.
4. A trained belief/value selector, with its training interactions and compute reported.

Retain the existing twelve-ledger continuation as the strongest released candidate. Run the information-only selector as a baseline, not a mandatory intermediate architecture. These comparisons decide whether memory continuation is needed and whether selection can improve on a strong initialization. They should not become another long sequence of official-family tuning rounds.

The principal experiment then trains on varied response mechanisms, changed familiar laws, unfamiliar laws and task returns, with recurrence and delays varied independently. Include independently trained recurrent meta-RL, latent-context meta-RL and a relevant episodic-memory method. The [direction document](CONTINUAL_AGENT.md) specifies their common model interfaces, training targets, timed application and consolidation study; [the focused comparison](../ref/BELIEF_AND_COMPUTATION.md) identifies the nearest prior methods.

Different outcomes lead to useful decisions:

- If adding memory to the frozen initial policy matches the continuation, use that simpler initialization and redirect training to unfamiliar relationships.
- If learning changes prediction but not task cost, inspect which action rankings change and their later outcomes before adding further predictor complexity.
- If a marginal selector matches a hypothesis-set selector, retain the simpler action representation for that setting.
- If known-family planning remains strongest, identify its information advantage and test the learned method where the family must be inferred. Do not weaken the reference.
- If archive recall helps but consolidation does not, retain recall as an achieved capability and train shared updates on later-use tasks. Do not rename recall as consolidated knowledge.

## Reproduction

```bash
python research/analysis/v014_reanalysis.py --output research/analysis/v014_summary.json
```

This standard-library command reproduces the native-record tables, paired intervals, training-prior counts and finite action example. Full native execution uses the existing [release instructions](../experiments/discoveryworld/RELEASE.md#experimental-memory-v0140). The reported full Bayesian replay means remain secondary calculations until its complete corrected runner and per-task outputs are committed and checked under the common execution protocol.
