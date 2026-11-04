# DiscoveryWorld: all eight themes

The [published benchmark](https://allenai.github.io/discoveryworld/) crosses eight themes with Easy, Normal and Challenge difficulty and API seeds 0–4. The [official protocol](https://github.com/allenai/discoveryworld/blob/fd591323920be0d3786ef350955de1945aa571e5/agents/README.md) allows 100 actions per Easy instance and 1,000 per Normal or Challenge instance. The 120 instances exclude the tutorial and the ten small-skills tests.

| Theme | What the agent must do across this theme |
| --- | --- |
| Reactor Lab | Measure crystals and use their properties to tune reactors. |
| Combinatorial Chemistry | Mix ingredients, observe the outcome and revise the recipe. |
| Archaeology Dating | Compare artifacts and use dating measurements to identify their chronology. |
| Plant Nutrients | Vary nutrient conditions and observe the plants' response over time. |
| Lost in Translation | Infer unfamiliar words from interactions and act on a request. |
| Space Sick | Investigate what is making the colonists ill and how to prevent it. |
| Proteomics | Compare protein profiles and identify the outlying species. |
| It's (not) Rocket Science! | Use planetary and vehicle measurements to set launch parameters. |

**Environment result.** Each instance loaded, returned its public observation, accepted `DISCOVERY_FEED_GET_UPDATES` and advanced exactly one tick. The [120 records](results/full_suite_environment.csv) give the action and initial/final clock readings for two builds of upstream revision `fd591323920be0d3786ef350955de1945aa571e5`: unchanged on Python 3.11.17, and on Python 3.12.14 with [two integer bounds corrected](python312_rocket.patch). Without the patch, Python 3.12 loaded the other 105 instances but could not create the 15 rocket worlds because `randint` received floats for a whole-number mass range. No policy was scored in these two runs.

## Run

From the CDM checkout, list the 120 cases without loading the simulator:

```sh
python -m experiments.discoveryworld.full_suite list
```

For a Python 3.12 environment with CDM installed, use an editable simulator checkout so the compatibility change applies to the code being run:

```sh
git clone https://github.com/allenai/discoveryworld.git ../discoveryworld
git -C ../discoveryworld checkout fd591323920be0d3786ef350955de1945aa571e5
git -C ../discoveryworld apply "$PWD/experiments/discoveryworld/python312_rocket.patch"
python -m pip install -e ../discoveryworld
python -m experiments.discoveryworld.full_suite probe --workers 4 --output runs/dw-probe
```

To score a trained native-action policy, supply one factory and one configuration for all cases. For example, the existing digital policy factory accepts a JSON object with `checkpoint`, `feature_cache`, and `device`; supply `encoder_path` when needed for the chosen features.

```sh
python -m experiments.discoveryworld.full_suite run --observation png \
  --policy experiments.discoveryworld.evaluate:create --policy-config model.json \
  --policy-id native-policy-run-1 --output runs/dw-policy
```

Each case starts with fresh agent state. The agent receives the public UI and action outcomes; `png` also supplies images. The runner records native actions, task success (the simulator's `completedSuccessfully`) and normalized progress for all 24 theme/difficulty combinations. It saves a policy's submitted scientific notes separately: explanatory knowledge needs [its own scoring procedure](https://github.com/allenai/discoveryworld/blob/fd591323920be0d3786ef350955de1945aa571e5/scripts/knowledgeScorer.py). A missing answer or a run error is not a measured score. To compare two policies with the same initial simulator random streams, pass the first scored output directory as `--rng-from`.

## Published comparisons

[Jansen et al., NeurIPS 2024, Table 4](https://papers.nips.cc/paper_files/paper/2024/file/13836f251823945316ae067350a5c366-Paper-Datasets_and_Benchmarks_Track.pdf) evaluated three ways of running GPT-4o, each zero-shot with a fresh episode for every instance. The table below counts successful completions. Each of the 24 published theme/difficulty completion fractions has five trials; the counts are summed before rounding.

| GPT-4o agent | Input | Easy / 40 | Normal / 40 | Challenge / 40 | Total / 120 |
| --- | --- | ---: | ---: | ---: | ---: |
| ReAct | text | 15 | 7 | 7 | 29 (24.2%) |
| Plan+Execute | text | 7 | 7 | 6 | 20 (16.7%) |
| Hypothesizer | text and images | 11 | 9 | 3 | 23 (19.2%) |

All three scored 0/15 on Reactor Lab. Finishing only Reactor Lab and Combinatorial Chemistry perfectly would already yield 30/120, one above the best of these three overall. This is why a cross-domain comparison needs the other six themes reported separately. Keep completion, procedural progress and explanatory knowledge as separate measures.

[Wang et al., EvoBelief, Table 3 and Appendix E](https://arxiv.org/abs/2610.12789) offer a different comparison: GPT-5.4, two themes, twelve fresh instances per theme and difficulty, at most 320 environment actions and 1,650 model calls per instance. EvoBelief completed 10/12 Normal and 7/12 Challenge Reactor Lab instances; in Plant Nutrients it completed 9/12 and 10/12. These are four groups of twelve, not scores on the official 120. A direct CDM comparison needs the same instances, observations, tools and action limits. Report training exposure, native action counts and end-to-end decision latency alongside the scores.

## The model comparison

The [v0.17.0 native evaluator](../continual_experiment/evaluate_shared_tasks.py) uses separate physics and chemistry action interfaces. Its checkpoint is not a drop-in checkpoint for the native-action factory above. The next training target is a single policy over public actions and their arguments: select a visible object, read or use it, choose a dialogue option, move, or wait. Use the shared core to predict how candidate experiments would change the next observation and to choose an action. Keep the published evaluation instances out of training; changing seeds alone does not guarantee a new law.

Freeze the resulting weights and evaluate all 120 cases, including failures. Compare with the [published language-agent results](https://github.com/allenai/discoveryworld/tree/fd591323920be0d3786ef350955de1945aa571e5/data) using the corresponding observation access and action budget. Record per-decision time as well as scientific results if claiming a speed advantage. To study continual learning, separately compare carrying or clearing experience on matched streams of returning problems; do not mix those scores into the fresh-start benchmark.

## Native policy training

The [training protocol](native_learning.json) now compares a shared CDM with a recurrent policy using the same received observations, candidate actions, source core, trajectories and epoch budget. Each condition is trained from two source cores. The recurrent policy uses less internal computation; this comparison matches training exposure rather than FLOPs.

Two separate questions determine the data split:

| Question | Training and development | Final evaluation |
| --- | --- | --- |
| Can one weight set act across the eight themes? | All themes; training seeds 1000–1003, development seed 2000 | All 120 official cases, reported by theme and difficulty |
| Can that policy transfer to an untrained kind of problem? | Exclude **Plant Nutrients** and **Lost in Translation** from both training and development | Report these two themes separately from the other six |

There are 96 planned training episodes and 24 development episodes before theme exclusion. These are instance splits, not evidence of disjoint scientific laws. The [response-table analysis](../continual_experiment/FUNCTION_BELIEF.md#response-functions-in-training-and-test) shows why changing seeds is insufficient. The theme comparison excludes those themes from this policy's adaptation; it does not establish absence of related material in pretrained encoders or language teachers.

The eight-theme condition is task-trained. Its official-case score must retain
that training exposure when compared with the published zero-shot LLM agents.

For Easy Combinatorial Chemistry, the [upstream generator](https://github.com/allenai/discoveryworld/blob/fd591323920be0d3786ef350955de1945aa571e5/discoveryworld/scenarios/storage_shed.py) maps the four training seeds to four distinct successful substances. All five official Easy cases and the development seed use targets in that set. This is overlap in the generated targets, not in full trajectories; it neither implies that the teacher found each target nor that the student memorized it. A score here measures independent execution on new instances, not discovery of a previously unseen target.

The native action interface covers public objects, navigation, instruments and dialogue without a scenario-specific solver. The teacher may propose actions using previously observed object handles; the simulator decides whether those objects are currently accessible. The learned policy enumerates accessible interactions and navigation to observed objects. Out-of-reach teacher proposals are recorded as failures, not silently replaced with different objects. The teacher sees received feedback and a compact, model-authored notebook. Task scorecards, generator parameters and solutions are excluded from its input. Current collection uses text and structured observations; it does not supply images. Its successes are teacher results, not CDM results.

```sh
python -m experiments.discoveryworld.collect_suite \
  --model data/cache/teacher --model-source PROVIDER/MODEL \
  --endpoint http://127.0.0.1:8000/v1 --served-model teacher \
  --workers 16 --seeds 1000 1001 1002 1003 --output runs/dw-native/train
python -m experiments.discoveryworld.collect_suite \
  --model data/cache/teacher --model-source PROVIDER/MODEL \
  --endpoint http://127.0.0.1:8000/v1 --served-model teacher \
  --workers 16 --seeds 2000 --output runs/dw-native/development
```

The local server must support structured chat responses and the checkpoint's chat template. Collection records the teacher identity, prompt version, actual actions and returned outcomes. Unfinished or malformed teacher generations remain visible in the records. `--resume` replays a saved prefix and requires identical public feedback before appending; initial RNG snapshots alone do not guarantee whole-world replay in every theme. A changed prefix requires a fresh episode, not continuation with mismatched observations.

This collection uses the instruction-tuned GPT-OSS-20B checkpoint distributed
as [`unsloth/gpt-oss-20b`](https://huggingface.co/unsloth/gpt-oss-20b). Download it
to `data/cache/gpt-oss-20b`. The tested inference environment is separate from
the training environment: vLLM 0.33.0 with its PyTorch 2.14.0 and Transformers
5.20.2 dependencies. One tested H100 server command is:

```sh
CUDA_VISIBLE_DEVICES=0 vllm serve data/cache/gpt-oss-20b \
  --host 127.0.0.1 --port 8000 --served-model-name teacher \
  --max-model-len 24576 --max-num-seqs 32 \
  --gpu-memory-utilization 0.65 --enforce-eager
```

Use `--max-new-tokens 2048 --max-input-tokens 22000` in the collection commands.
The teacher retains received numerical measurements separately from its own
notebook. Object-argument definitions are shared within the output schema, so
observing more objects does not repeat their handles for every operation.

The learned native policy imports the continual core of the released inherited-continuation models and trains a shared dynamic action head. It does not import the specialist chemistry and physics controllers. The first implementation has **55,360,574 parameters**, excluding its frozen SigLIP feature encoder. Numeric measurements remain tied to their observed objects. Sequence training carries state through the complete episode. The declared comparison imitates valid actions from successful teacher episodes; unsuccessful episodes still supply received-outcome training. Failed or unavailable actions remain in history without imitation labels. Predicting outcomes uses only actions actually executed and represented by the policy. Teacher task completion is used for offline label selection, never supplied as a policy input.

After collection and trajectory selection, the complete eight-run comparison is launched with:

```sh
python -m experiments.discoveryworld.train_suite \
  --train-trajectories runs/dw-native/train \
  --validation-trajectories runs/dw-native/development \
  --source-root runs/continual_physics/joint_belief_20261103 \
  --encoder-path data/cache/siglip2 --feature-cache runs/dw-native/features \
  --output runs/dw-native/models --gpus 0 1 --wait-for-collections --evaluate
```

The queue waits for the declared collection attempts to end, retains nonempty traces, and records supervision coverage by theme. The declared endpoint is the eighth epoch's `last.pt`, not the best final-test score. Training accuracy and outcome loss describe imitation and prediction; task ability requires subsequent native execution. With `--evaluate`, each final endpoint executes all 120 official cases. The full-suite runner loads each weight set once per worker, resets task state between instances and records decision time separately from simulator time.

The [initial pipeline result](results/native_pipeline.json) used incomplete teacher traces for one training epoch. Of 741 training transitions, 385 supplied an action to imitate. On 342 development transitions the policy matched 52 of 156 labeled actions. Executing the policy independently at development seed 2001 gave **0/8** Easy completions: seven themes had zero procedural progress, and Reactor Lab reached 0.18. The eight formal training runs are a separate comparison.

In this comparison, the [trainer](train.py) learns from the recorded actions and their observed consequences. During [independent execution](evaluate.py), observations change the policy's recurrent state and measurement history, but the weights remain fixed. The optional actor–critic stage below uses the student's own actions. It runs separately from the eight-run imitation comparison.

### Hosted demonstrations and teacher-free learning

[DeepSeek-V4.1-Flash-1017](https://api-docs.deepseek.com/news/news261017/) can be selected as a demonstration teacher. The hosted and local collectors present the same public observations, action arguments, and measurement history. The hosted collector uses the official `deepseek-flash` alias, validates each returned JSON action against the displayed choices, and records the service's model identifier, token use, and finish reason. The input-token cap is applied to the count returned by the service; an over-budget call is recorded without executing its action. Install the `hosted-teacher` extra and provide `DEEPSEEK_API_KEY` in the process environment. For example:

```sh
python -m experiments.discoveryworld.collect_suite --deepseek-flash \
  --themes "Archaeology Dating" "Proteomics" --difficulties Easy --seeds 3011 \
  --max-steps 60 --workers 2 --output runs/native_teachers/flash
```

For a paired teacher comparison, run the local teacher on the same cases and cap first, then add `--rng-from runs/native_teachers/local` to the hosted command. Use separate output directories. Compare tasks completed and actions used, then test whether the resulting demonstrations help the **student acting alone**. No hosted-teacher task score is reported here.

The [on-policy trainer](finetune.py) samples an action, executes it, and updates the shared policy using subsequent changes in simulator task progress. One policy, critic, and optimizer span the entire schedule; episode-specific state resets at each new world. A compatible native checkpoint can start the run without further demonstrations:

```sh
python -m experiments.discoveryworld.finetune --checkpoint "$NATIVE_POLICY" \
  --native-suite all --seeds 3008 3009 --device cuda:0 --encoder-path data/cache/siglip2 \
  --feature-cache runs/native_rl/features --output runs/native_rl/warm
```

`--native-suite holdout` omits Plant Nutrients and Lost in Translation; start from the corresponding six-theme checkpoint for a transfer comparison. To learn without *any* demonstrations or pretrained weights, replace the checkpoint argument with `--random-native-width 32`: this initializes the policy and critic randomly and uses fixed lexical and numeric features. The [eight-theme CPU run](results/native_actor_cold_start.json) records all case totals and the change in action weights. It tests learning directly from interactions; it does not measure a trained model's benchmark performance.

An experiment generator would add a different capability. For example, let a proposer find two candidate laws that agree on the measurements so far; let the solver choose the experiment that separates them. The training world supplies the outcome. Compare this learned curriculum against a fixed one by how quickly the next policy learns new laws at the same interaction budget. The current trainer does not change its curriculum or learning rule.
