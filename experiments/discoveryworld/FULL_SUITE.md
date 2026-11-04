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

When replay cannot reproduce an interrupted world, `--resume --restart-interrupted`
keeps completed cases and moves unfinished attempts under `interrupted/` before
starting fresh trajectories. Their records and collection costs remain available;
the old and new observations are never joined into one episode.

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

[DeepSeek-V4.1-Flash-1017](https://api-docs.deepseek.com/news/news261017/) can be selected as a demonstration teacher through DeepSeek or [Alibaba Cloud Bailian](https://help.aliyun.com/zh/model-studio/deepseek-api). Both collectors present the same public observations, action arguments, and measurement history. The native experiment environment is required; hosted requests use Python's HTTP client without an additional provider SDK. Set `DEEPSEEK_API_KEY` for the direct API; set `DASHSCOPE_API_KEY` and your `DASHSCOPE_BASE_URL` for Bailian. Credentials are read from the process environment and excluded from collection metadata. The direct API uses `deepseek-flash`; Bailian defaults to `deepseek-v4.1-flash-1017` and uses its `enable_thinking` option.

```sh
python -m experiments.discoveryworld.collect_suite --deepseek-flash \
  --themes "Archaeology Dating" "Proteomics" --difficulties Easy --seeds 3011 \
  --max-steps 60 --workers 2 --output runs/native_teachers/flash
```

For parallel Bailian collection across all themes and difficulties:

```sh
python -m experiments.discoveryworld.collect_suite --bailian-flash \
  --seeds 1000 1001 1002 1003 --workers 48 --reasoning-effort high \
  --max-input-tokens 65536 --max-new-tokens 32768 --action-corrections 2 \
  --output runs/native_teachers/science
```

Each returned action must match the public interface. `--action-corrections 2` returns a malformed proposal to the teacher up to twice, without executing an action or supplying a solution. Every response and its token use are retained. Transport retries handle rate limits, transient service errors and closed connections. The provider-reported input count enforces the input-token budget; a call beyond it does not execute an action. Concurrent CPU environments advance independently. A disconnected simulator is recorded as an environment error with its existing trajectory preserved; other cases continue. This is distinct from an unsuccessful teacher policy.

For a paired teacher comparison, run the local teacher on the same cases and cap first, then add `--rng-from runs/native_teachers/local` to the hosted command. On the first eight Easy training cases, Flash completed **7/8**, compared with **1/8** for the local teacher. Both had a 100-action limit and shared initial random streams, but their model and generation budgets differed: 16,384 versus 2,048 output tokens per call. These are demonstration results, not student scores or a compute-matched teacher comparison. [Per-case records](results/flash_demonstrations.json).

The [on-policy trainer](finetune.py) samples an action, executes it, and updates the shared policy using subsequent changes in simulator task progress. One policy, critic, and optimizer span the entire schedule; episode-specific state resets at each new world. A compatible native checkpoint can start the run without further demonstrations:

```sh
python -m experiments.discoveryworld.finetune --checkpoint "$NATIVE_POLICY" \
  --native-suite all --seeds 3008 3009 --device cuda:0 --encoder-path data/cache/siglip2 \
  --feature-cache runs/native_rl/features --output runs/native_rl/warm
```

`--native-suite holdout` omits Plant Nutrients and Lost in Translation; start from the corresponding six-theme checkpoint for a transfer comparison. To learn without *any* demonstrations or pretrained weights, replace the checkpoint argument with `--random-native-width 32`: this initializes the policy and critic randomly and uses fixed lexical and numeric features. The [eight-theme CPU run](results/native_actor_cold_start.json) records all case totals and the change in action weights. It tests learning directly from interactions; it does not measure a trained model's benchmark performance.

The batched trainer runs multiple native environments around one shared policy and optimizer. Simulator subprocesses exchange public observations and actions without importing Torch or loading CUDA; the training process owns GPU computation. `--small-skills-first` begins with the ten official basic-operation tasks, using environment rewards alone. For example:

```sh
python -m experiments.discoveryworld.finetune --random-native-width 1024 \
  --native-suite all --small-skills-first --seeds 1000 1001 1002 1003 \
  --workers 8 --sequence-length 8 --threads 1 --mode cdm --random-seed 301 \
  --device cuda:0 --lr 0.0001 --feature-cache runs/native_rl/features \
  --output runs/native_rl/autonomous-cdm
```

The initial checkpoint is saved before the first action. Pass that checkpoint with `--checkpoint` and `--mode gru` to train the recurrent reference from the same weights. Each scheduled case has the same maximum action budget; actual interactions can differ when a task terminates early. Record those interactions and compute separately. This structured-input policy has **55,504,931 parameters**, including parameters inactive in GRU mode. It uses fixed lexical/numeric features, no pretrained encoder, and one fixed workspace refinement per action. Activity scheduling and cross-instance hidden-state retention are separate capabilities.

An experiment generator would add a different capability. For example, let a proposer find two candidate laws that agree on the measurements so far; let the solver choose the experiment that separates them. The training world supplies the outcome. Compare this learned curriculum against a fixed one by how quickly the next policy learns new laws at the same interaction budget. The current trainer does not change its curriculum or learning rule.

### Separating teacher and architecture effects

The running comparison has three routes, each with CDM and a recurrent policy without the workspace (GRU):

| Route | Demonstrations | Subsequent learning |
| --- | --- | --- |
| Autonomous | None; random initialization and fixed lexical/numeric inputs | Student actions and returned task-progress rewards |
| Basic operations | The same fixed set of ten official small-skills tasks | The same student interaction curriculum |
| Complete solutions | Basic skills and eight-theme scientific demonstrations | The same student interaction curriculum; reported separately |

The basic-operation collection completed **65/80** training and **15/20** development cases. It contains 2,457 and 728 actual operations respectively. It teaches navigation, object use, dialogue and observation; these demonstrations still guide action selection within those elementary tasks. No eight-theme solution enters this route. Both students completed eight supervised epochs on the same files, with 1,176 represented valid action labels from successful training episodes. Failed trajectories retain their observed outcomes. Independent execution after this fixed endpoint completed **0/8 Easy development cases** for each student (seed 2001). This fits basic operations but has not yet taught scientific task completion. The per-theme results are included in the same record.

To collect basic-operation demonstrations, use `collect_suite --bailian-flash --small-skills --seeds 4000 4001 4002 4003 4004 4005 4006 4007`; collect development with seeds 5000 and 5001. Then run:

```sh
python -m experiments.discoveryworld.train_demonstrations \
  --initial-checkpoint runs/native_rl/autonomous-cdm/initial.pt \
  --train-collections runs/native_teachers/skills \
  --development-collections runs/native_teachers/skills-development \
  --scope basic_operations --mode cdm --epochs 8 --seed 301 --workers 8 \
  --learning-seeds 1000 1001 1002 1003 --device cuda:0 \
  --feature-cache runs/native_rl/features --output runs/native_rl/basic-cdm \
  --wait-for-collections --evaluate
```

Use the same files and initial checkpoint for GRU. `--scope basic_operations` rejects scientific solution trajectories. For the complete-solution comparison, choose `--scope complete_solutions` and include scientific collections. The pipeline snapshots completed data, trains eight imitation epochs, learns from the student's actions, and optionally evaluates both final endpoints on all 120 official cases with no teacher. Checkpoints are chosen by the declared endpoint, not test performance. `full_suite probe` also saves initial simulator random streams; `--evaluation-rng-from` reuses them across models.

The first pair of eight-epoch basic-operation fits is complete; autonomous and post-imitation interaction learning are in progress. Native-library faults interrupted several Torch 2.11.0+cu128 attempts. Their records and checkpoints are retained. The four current trainings share a separate Torch 2.14.0+cu130 / NumPy 2.3.5 runtime, the same starting weights and equivalent lexical features, with independent CPU simulator workers. Earlier attempts are separate from this comparison. The final reinforcement and official-case scores remain pending. The complete-solution pair waits for its separate scientific collection and uses the same runtime. These are one-initialization feasibility runs, not an established architecture result.

Within each route, match source weights, received observations, action access, scheduled cases and maximum interaction budgets. Report actual student interactions, teacher operations, API tokens and compute separately. The pure-learning route preserves the question of learning without a teacher. The basic-operation route asks whether instruction helps the student reach useful experiments. The complete-solution route measures distillation. The [DiscoveryWorld training settings](https://arxiv.org/html/2406.06769v2#A3.S2) permit training on separate instances, but such scores are task-trained and must retain that distinction from the published zero-shot agents. Different seeds may still share scientific laws, as documented above.

Reserve pairs of unseen laws that give the same initial readings but require different final decisions. In each pair, make both laws equally likely and offer a measurement whose outcomes distinguish them. With a single final choice, a policy that ignores all distinguishing observations can succeed on at most half the pair on average. Compare completion and measurement cost across the four conditions. This tests what the student does with a new observation, even if both architectures learned the same instrument skills.
