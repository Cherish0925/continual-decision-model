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

## The model comparison

The [v0.17.0 native evaluator](../continual_experiment/evaluate_shared_tasks.py) uses separate physics and chemistry action interfaces. Its checkpoint is not a drop-in checkpoint for the native-action factory above. The next training target is a single policy over public actions and their arguments: select a visible object, read or use it, choose a dialogue option, move, or wait. Use the shared core to predict how candidate experiments would change the next observation and to choose an action. Keep the published evaluation instances out of training; changing seeds alone does not guarantee a new law.

Freeze the resulting weights and evaluate all 120 cases, including failures. Compare with the [published language-agent results](https://github.com/allenai/discoveryworld/tree/fd591323920be0d3786ef350955de1945aa571e5/data) using the corresponding observation access and action budget. Record per-decision time as well as scientific results if claiming a speed advantage. To study continual learning, separately compare carrying or clearing experience on matched streams of returning problems; do not mix those scores into the fresh-start benchmark.
