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

Each case starts with fresh agent state. The agent receives the public UI and action outcomes; `png` also supplies images. The runner records native actions, task success (the simulator's `completedSuccessfully`) and normalized progress for all 24 theme/difficulty combinations. It saves a policy's submitted scientific notes separately: explanatory knowledge needs [its own scoring procedure](https://github.com/allenai/discoveryworld/blob/fd591323920be0d3786ef350955de1945aa571e5/scripts/knowledgeScorer.py). A missing answer or a run error is not a measured score.

For paired runs, use the same `--object-rng-seed 73105` for every policy. This
sets otherwise unseeded native generators before world construction and the
initial tick, while preserving seeds explicitly set by the simulator. It
changes no task rule or policy input. The value is recorded with the run.
Post-reset snapshots (`--rng-from`) alone are insufficient for every theme:
Space Sick can create different initial objects before those states are
restored. Earlier partial evaluations retain that error. Five Space Sick
instances now reproduce initial observations, RNG states and the next native
action across independent resets. Full-suite comparisons using the new
initialization are separate from earlier scored runs.

## Published comparisons

[Jansen et al., NeurIPS 2024, Table 4](https://papers.nips.cc/paper_files/paper/2024/file/13836f251823945316ae067350a5c366-Paper-Datasets_and_Benchmarks_Track.pdf) evaluated three ways of running GPT-4o, each zero-shot with a fresh episode for every instance. The table below counts successful completions. Each of the 24 published theme/difficulty completion fractions has five trials; the counts are summed before rounding.

| GPT-4o agent | Input | Easy / 40 | Normal / 40 | Challenge / 40 | Total / 120 |
| --- | --- | ---: | ---: | ---: | ---: |
| ReAct | text | 15 | 7 | 7 | 29 (24.2%) |
| Plan+Execute | text | 7 | 7 | 6 | 20 (16.7%) |
| Hypothesizer | text and images | 11 | 9 | 3 | 23 (19.2%) |

All three scored 0/15 on Reactor Lab. Reactor Lab and Combinatorial Chemistry together account for 30 of the 120 official instances. This is a count of benchmark instances, **not a measured CDM score of 30/120**. Earlier CDM results of [30/30 on chemistry development instances](../continual_chemistry/README.md#forecasting-the-next-two-experiments) use a separate single-theme set and a specialized execution interface. They cannot be substituted for the corresponding official cases. Report all eight themes, and keep completion, procedural progress and explanatory knowledge as separate measures.

[Wang et al., EvoBelief, Table 3 and Appendix E](https://arxiv.org/abs/2610.12789) offer a different comparison: GPT-5.4, two themes, twelve fresh instances per theme and difficulty, at most 320 environment actions and 1,650 model calls per instance. EvoBelief completed 10/12 Normal and 7/12 Challenge Reactor Lab instances; in Plant Nutrients it completed 9/12 and 10/12. These are four groups of twelve, not scores on the official 120. A direct CDM comparison needs the same instances, observations, tools and action limits. Report training exposure, native action counts and end-to-end decision latency alongside the scores.

## The model comparison

The [v0.17.0 native evaluator](../continual_experiment/evaluate_shared_tasks.py) uses separate physics and chemistry action interfaces. Its checkpoint is not a drop-in checkpoint for the native-action factory above. The next training target is a single policy over public actions and their arguments: select a visible object, read or use it, choose a dialogue option, move, or wait. Use the shared core to predict how candidate experiments would change the next observation and to choose an action. Keep the published evaluation instances out of training; changing seeds alone does not guarantee a new law.

Freeze the resulting weights and evaluate all 120 cases, including failures. Compare with the [published language-agent results](https://github.com/allenai/discoveryworld/tree/fd591323920be0d3786ef350955de1945aa571e5/data) using the corresponding observation access and action budget. Record per-decision time as well as scientific results if claiming a speed advantage. To study continual learning, separately compare carrying or clearing experience on matched streams of returning problems; do not mix those scores into the fresh-start benchmark.

## Native policy training

### Completed autonomous and basic-operation comparison

On 5 November, the four declared cloud conditions completed their full
136-episode curriculum and all 120 official evaluation cases. Each endpoint
uses seed 301; these are four conditions, not four independent seed repeats.
Observations use fixed lexical/numeric features without a pretrained encoder.
The policy parameters remain fixed during evaluation.

| Initialization and policy | Training actions | Optimizer updates | Completed official cases |
| --- | ---: | ---: | ---: |
| Autonomous CDM | 71,200 | 1,117 | 0/120 |
| Autonomous GRU | 71,200 | 1,117 | 0/120 |
| Basic-operation CDM | 69,144 | 1,090 | 0/120 |
| Basic-operation GRU | 70,141 | 1,107 | 0/120 |

The completed evaluation has no runtime errors. Earlier attempts with
post-reset RNG replay remain separate. The basic-operation models completed
nine and eight training episodes, including the small-skills curriculum, but
did not complete the scientific evaluation. These training successes do not
establish transfer to the eight scientific themes.

Repeated greedy actions motivated an additional fixed-weight comparison on
eight new Easy development instances, seed 3001. Each endpoint ran once with
greedy decoding and once with temperature-1 sampling, using policy seed 301.
All eight combinations scored 0/8. Sampling increased mean progress in some
conditions, without solving a task. Thus changing the execution rule alone
does not resolve the current competence gap. Progress is the simulator's
reported final score, which may include progress already present at reset.

[Every case, training count and action summary](results/native_actor_comparison.json)
is available. The native model and record archives on
[v0.18.0](https://github.com/Cherish0925/continual-decision-model/releases/tag/v0.18.0)
retain all four endpoints. The record archive includes the evaluation source
used for the sampling comparison. After extracting both native archives,
`python -m experiments.discoveryworld.compare_native_learning --runs native_actor --out comparison.json`
recreates the summary. A model's `policy.json` can be passed to `full_suite run`
with `--object-rng-seed 73105` for the official-case protocol.

The fixed snapshot of 150 training demonstrations (16,015 teacher actions)
and 38 development demonstrations has completed eight imitation epochs for
both policies. Each scored **0/8** on new Easy development instances at seed
3002. CDM's action imitation accuracy was 30.8% on training observations and
29.4% on development observations; GRU's was 31.4% and 29.0%. Only 3,812 training
actions had valid imitation labels from successful episodes. Both models
repeated actions despite received feedback: in Reactor Lab, each submitted the
same unsuccessful OPEN action 99 times. These results show inadequate learned
execution; they do not establish a single causal explanation or a teacher defect.
[Training curves and all development cases](results/native_demonstration_snapshot.json)
retain the unsuccessful outcomes. This imitation-only snapshot is separate
from the final complete-data pipeline, which awaits the remaining scientific
demonstrations and evaluates imitation and reinforcement endpoints separately.

### Learning from executed actions and student failures

The next comparison changes the information available to learning and the states
on which the student receives instruction. It retains the native action interface.

Previously, the recurrent update received the observation, objects and measurement
records. The chosen native action was not explicitly fed back to that state.
During stochastic training, two different actions can therefore lead to the same
generic failure observation without identifying which command failed. The new
optional `--action-history 8` input keeps the executed command, observed object
descriptions or dialogue option, location change, and returned feedback for the
last eight actions. Each record is encoded separately. It is visible only at the
following decision; expert proposals and scorecard values do not enter it.
This is an implementation hypothesis being tested, not an established explanation
of the entire performance failure.

The [public history representation](../../cdm/digital.py) is shared by imitation,
on-policy training, prefix replay and independent execution. With history disabled,
previous checkpoint inputs and caches remain unchanged. The two current H100
trainings use the fixed 150 training / 38 development trajectories, the original
initial weights, eight epochs and the same losses as the completed demonstration
reference. They change only this feedback input. Evaluation uses final weights
from the declared epoch, including new Easy instances.

[Recovery collection](collect_recovery.py) executes eight actions from the frozen
completed-demonstration CDM, then lets Flash attempt the remaining task using the
actual public history. The collection covers all eight themes at Easy and Normal
difficulty, with training seeds 1100–1101 and caps of 100/500 actions. One Rocket
Easy world at seed 1100 fails during simulator construction; it is retained as an
environment error. Seed 1102 also fails its construction check. Seed 1103 creates
successfully and supplies the replacement, chosen before any policy evaluation.

Every transition records its actual controller. Student actions receive outcome
supervision but no imitation label, even if the teacher later completes the task.
Only valid teacher actions in completed successful episodes receive imitation
labels. A real collected chemistry episode was prepared through the training
path: all eight student actions retained their outcomes and received zero imitation
labels; 29 teacher actions supplied valid imitation labels.

```sh
python -m experiments.discoveryworld.collect_recovery \
  --checkpoint runs/student/last.pt --feature-cache runs/native-features \
  --output runs/recovery --seeds 1100 1101 --difficulties Easy Normal \
  --student-steps 8 --max-steps 500 --workers 32 --provider bailian
```

This is assisted collection. The queued follow-up trains CDM and GRU with and
without action history for four additional epochs on the same augmented data,
then evaluates them without a teacher from the initial states of seeds 3003–3004.
The 16 development worlds were created successfully before training. These scores
are separate from the official 120 cases and from teacher recovery success.

The broader [input and initialization protocol](native_input_comparison.json)
contains 32 trainings: two input representations, two core initializations, two
history settings, two policy types, and two training seeds. Inherited and random
cores have the same architecture and action grid; only core weights are inherited,
and native input projections and action heads are newly initialized. The frozen
semantic input uses SigLIP2 text features; these trajectories contain no images.
All conditions receive the same 150/38 trajectories and optimizer-window budget.
CDM and GRU have different internal computation costs. The existing
[`train_suite`](train_suite.py) accepts this protocol and generates the matched
random initializations. Cloud preparation and training are scheduled; a result
table will be added after independent execution finishes.

### What the four frozen policies actually did

All four ended at 0/120. Reading the [released action traces](https://github.com/Cherish0925/continual-decision-model/releases/tag/v0.18.0) identifies two different failures. Here a repeated action has the same verb **and arguments** in two adjacent steps of the same case. An accepted command means the native API returned success, not that the scientific task advanced.

| Official greedy policy | Cases using one identical action from start to finish | Repeated after a failed command / failed commands with a following step |
| --- | ---: | ---: |
| Autonomous CDM | 120/120 | 72,900/72,900 |
| Autonomous GRU | 80/120 | 50,445/50,445 |
| Basic-operation CDM | 35/120 | 23,628/23,628 |
| Basic-operation GRU | 25/120 | 23,723/23,784 |

The autonomous CDM issued `USE` on all 84,000 steps; within each case the object arguments never changed. The basic-operation CDM made 60,321 accepted moves. Of 60,236 adjacent pairs of accepted moves, 60,046 went in opposite directions. Its only other accepted commands were three object teleports. It did learn something on the training distribution: of nine training completions, six were elementary skills and three were scientific instances (two Archaeology Dating, one Space Sick). None transferred to a full official completion. On eight separate Easy development cases, sampling instead of taking the highest-scoring action broke the repetition but all four policies still scored 0/8.

The [on-policy action loss](parallel_learning.py) uses subsequent changes in native task progress. In the autonomous CDM, 35 of 1,117 updates had a nonzero **sum** of the collected rewards; this is not a count of individual rewarding actions. The batched actor–critic run does not train the outcome predictor on each attempted action. The [supervised snapshot](results/native_demonstration_snapshot.json) is a different experiment: 16,015 training actions, 3,812 valid imitation labels from successful episodes, and 7/47 correct changes of expert action after repeated runs on validation. Exact action agreement is not an upper bound on task completion: another action can also be valid.

To separate execution from long-horizon planning, hand control to a frozen policy within a successful experiment on development instances. To train recovery, collect continuations from states the student actually reaches after a failed interaction or an alternating walk. Keep the attempted command, bound objects, returned result, and the consequences of trying an alternative. Predicting whether an action will be accepted supplies a learning signal; acceptance alone would reward the unproductive walking above. Keep the action interface fixed when comparing random and inherited initialization, and select changes on new development instances.

#### An executed handoff comparison

The four v0.18.0 checkpoints were also tested on **12 new Reactor Lab Normal instances, seeds 5100–5111**. The public-history specialist completed all twelve in 214–225 native steps. Replay its history until immediately before its last action, then let a frozen student take up to 32 actions. All handoff states had 0.92 task progress and the specialist's final action was available among nine public dialogue choices. A uniform policy starts from the same states. No model was trained on these histories.

| Frozen policy after the assisted prefix | First choice in all 12 states | Completions within 32 steps |
| --- | --- | ---: |
| Autonomous CDM | Decrease frequency by 1 Hz | 6/12 |
| Same weights, reset neural state at handoff | Decrease frequency by 1 Hz | 6/12 |
| Autonomous GRU | Exit dialogue | 0/12 |
| Basic-operation CDM | Exit dialogue | 0/12 |
| Basic-operation GRU | Increase frequency by 1 Hz | 5/12 |
| Uniform public-action sampling | Varied | 0/12 |

Every autonomous-CDM suffix repeats the same decrement; every basic-operation-GRU suffix repeats the same increment. Their completions can therefore be reproduced by fixed-direction adjustments from these specialist-prepared states. Resetting the autonomous CDM's GRU and workspace, while retaining the same public object/measurement histories, leaves **all twelve action sequences and completions unchanged**. This intervention concerns these handoff states, not every use of the architecture. Assisted completion is not fresh-start competence and does not establish a learned scientific strategy.

The useful next training distinction is a pair of otherwise similar public situations that require opposite actions. For frequency control, collect real states above and below the same target; for failed interactions, collect the student's actual failure state and a successful continuation from it. Train the observation–action–returned-result connection on those cases, then move the handoff earlier until the student can complete new episodes independently. Retain the fixed-direction and independently trained recurrent policies as comparisons. Do not fit to these twelve development histories or the official 120 cases.

The native simulator can depend on earlier worlds in the same interpreter even when their initial seeds match. Each reference and continuation above therefore starts in a fresh process with fixed hash and NumPy seeds, and every forced step must match the reference's public-observation hash and returned action result. During the forced prefix the unused action readout is omitted; GRU, workspace and adapter states were checked to equal full policy execution on replayed public events for both policy modes. The original attempts with mismatched histories are excluded from the final matched comparison.

[Complete action analysis and handoff outcomes](results/native_action_behavior.json) · [Public reference actions and observation hashes](results/native_handoff_references.json.gz).

```bash
python -m experiments.discoveryworld.native_handoff prepare --root runs/native-handoff --seeds 5100 5101 5102 5103 5104 5105 5106 5107 5108 5109 5110 5111
python -m experiments.discoveryworld.native_handoff run --root runs/native-handoff --models native_actor --label autonomous-cdm --seeds 5100 5101 5102 5103 5104 5105 5106 5107 5108 5109 5110 5111
python -m experiments.discoveryworld.native_action_behavior --runs native_actor --handoffs runs/native-handoff --out native-action-behavior.json
```

Repeat the second command for `autonomous-gru`, `basic_operations-cdm`, `basic_operations-gru`, `uniform`, and `autonomous-cdm-reset` before combining results. Use the pinned DiscoveryWorld dependency. This comparison used CPU Torch 2.11.0; it did not change the running server jobs or any published checkpoint. Source archives: `native-actor-records.tar.gz`, SHA-256 `a45e108ab5b3fa7370126e01c9cc6385cd03995fcadb163be2c540d522a4c77d`; `native-actor-models.tar.gz`, SHA-256 `619864109c639cc8e6e575f18315f0c88425f8cee125999e161e7fc5dfb7d8b9`.

### Earlier inherited-core protocol

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

Use the same files and initial checkpoint for GRU. `--scope basic_operations` rejects scientific solution trajectories. For the complete-solution comparison, choose `--scope complete_solutions` and include scientific collections. The pipeline snapshots completed data, trains eight imitation epochs, learns from the student's actions, and optionally evaluates both final endpoints on all 120 official cases with no teacher. Checkpoints are chosen by the declared endpoint, not test performance. Use the same `--evaluation-object-rng-seed 73105` across endpoints to reproduce native initialization.

Early local interaction-learning attempts suffered native-library faults under
Torch 2.11.0+cu128 and were retried in a separate Torch 2.14.0+cu130 environment.
Their records and checkpoints remain separate from the completed cloud
comparison above. The four cloud endpoints finished training and each scored
0/120. The complete-solution imitation-plus-reinforcement pair still awaits its
full scientific collection. These are one-initialization feasibility runs,
not an established architecture advantage.

Within each route, match source weights, received observations, action access, scheduled cases and maximum interaction budgets. Report actual student interactions, teacher operations, API tokens and compute separately. The pure-learning route preserves the question of learning without a teacher. The basic-operation route asks whether instruction helps the student reach useful experiments. The complete-solution route measures distillation. The [DiscoveryWorld training settings](https://arxiv.org/html/2406.06769v2#A3.S2) permit training on separate instances, but such scores are task-trained and must retain that distinction from the published zero-shot agents. Different seeds may still share scientific laws, as documented above.

Reserve pairs of unseen laws that give the same initial readings but require different final decisions. In each pair, make both laws equally likely and offer a measurement whose outcomes distinguish them. With a single final choice, a policy that ignores all distinguishing observations can succeed on at most half the pair on average. Compare completion and measurement cost across the four conditions. This tests what the student does with a new observation, even if both architectures learned the same instrument skills.
