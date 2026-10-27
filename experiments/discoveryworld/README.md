# CDM on digital experiments

The next campaign trains a model to choose experiments, continue useful inference as observations arrive, and retain what it learns. It extends the existing GRU/Transformer core and its deployed-update training. The main architecture remains in [DESIGN.md](../../DESIGN.md).

## Application and comparison settings

Use **Reactor Lab**, **Combinatorial Chemistry**, and **Plant Nutrients** as the initial application families. They provide instrument use and quantitative relations, multistep interventions, and delayed experimental outcomes. Expand to the remaining official discovery topics after the same model can learn and act across these families.

Two settings answer different questions:

| Setting | Purpose | Environment clock |
| --- | --- | --- |
| Official DiscoveryWorld | Scientific task completion, procedural progress, and discovery knowledge | One native action followed by one native world tick |
| Continuing process experiments, then a separately named digital extension | Decisions during computation, delayed returns, returning conditions and learning between requests | Environment advances independently of model inference |

Do not relabel the standard DiscoveryWorld run as real-time. Its API advances only when the caller invokes tick. Keep official scores comparable, and report any continuing-life or independently timed extension separately.

## Implemented task connection

[adapter.py](adapter.py) runs the actual DiscoveryWorld API. It returns the public UI, images, documented actions, teleport locations and observation times. An action produces a fresh observation after the world tick, a result, task scores, an observed-progress training reward, and separate terminal/time-limit flags. The acting policy receives public observations; detailed scorecards and hidden scientific rules are not its inputs.

[collect.py](collect.py) records these transitions from a local policy or an action file. PNG images are stored once per content hash. Each row retains the observation, action, returned result, next observation, task scores and timing needed by a trajectory learner. State after a time limit can be bootstrapped; it is not marked as a completed task.

The simulator is pinned to **fd591323920be0d3786ef350955de1945aa571e5**. Two integration tests run its real Reactor Lab scenario and cover world advancement, current observations, episode replacement and image persistence. No model weights or GPUs are needed for these tests.

This adapter supplies the environment and trajectory interface. Task policies and multimodal trainers connect through the stateful policy factory below.

## Model work that directly enables training

| Existing component | Required integration |
| --- | --- |
| Frozen perception and semantic model | Encode actual task observations; preserve exact instrument readings, units and object references in experience records |
| GRU and persistent Transformer workspace | Retain state through each task; supply new observations between single refinement blocks |
| Process-control action head | Add a task head for operation type, observed-object arguments, direction/location arguments and dialog choices |
| Scalar process predictor | Keep the existing process branch; add prediction of received tool results, measured quantities or frozen observation features for digital tasks |
| Activity policy | Add semantic-call and retrieval choices, using task return and measured operation duration |
| Recent experience and instance adapters | Connect retained episodes to retrieval and local updates; train on later use, including returning conditions |

Use a shared operation/object representation or a factorized action head. Object UUIDs identify observed objects; their integer magnitudes have no physical meaning. The same decoder must support a changing set of visible objects. A fixed table of 81 process-control actions cannot express this interface.

The width-1024 released core can initialize compatible state modules. New task heads need training. Compare its transfer with a random-core initialization using the same dimensions and trajectories before attributing a benefit to the old process weights. Both remain the same architectural family.

Keep numeric observations available alongside compressed semantic context. Eight latent tokens are a bounded semantic connection; they are not a replacement for exact experimental records.

## Data and optimization

1. **Collect executable trajectories.** Use the local pretrained teacher and task-native agents on training instances. Retain successful and unsuccessful experiments with their received results. Use useful actions for imitation; use both successes and failures for outcome learning. Record teacher/model revision and collection cost.
2. **Train the actual action path.** Freeze pretrained perception and semantics initially. Train the new connections, task heads and persistent core on action imitation plus action-conditioned outcome prediction. Include immediate action and deeper refinement readouts. Evaluate by executing the resulting policy in the simulator.
3. **Train action and activity together.** Continue with recurrent actor–critic learning on fresh interactions. Optimize task return with measured observation/computation costs. Preserve state across training windows and reconstruct the local parameter state that existed at each collected decision.
4. **Train learning on its later use.** Replay the instance updates actually taken, on then-available observations, and optimize predictions/actions after those updates. Include situations that recur after intervening tasks. The future-loss and return horizon must reach those recurrences; increasing an unroll length alone does not fix a discount that removes their value.
5. **Train and measure quiet-period work.** Replay old and recent received experience, then distill useful prediction/action behavior. Initially keep state-producing shared parameters fixed during live local adaptation. Recompute state when a later consolidation phase changes those parameters.

Stages initialize one continuing model. Begin the first complete observation–action–outcome–gradient–new-action loop immediately, then improve it; do not require a collection of isolated modules to be perfected first.

Once a policy completes tasks, train conditional activity choices from that checkpoint.
Use matched continuations from the same model/environment state to compare acting
now, observing again, refining the workspace, and taking a local learning step.
Include both states where another observation helps and states where it is redundant.
Targets use subsequent task outcomes and operation costs; a shorter continuation
that fails the task is not automatically preferable. The existing lifetime
[trainer](../lifetime/train.py) and [runtime](../lifetime/runtime.py) already contain
activity-value learning and matched continuation rollouts. Adapt these mechanisms
to the native scientific-task actions rather than duplicating the learning code.

An initial activity-head fit can use a fixed successful core. Continue with joint
updates of the recurrent core, workspace, trainable connections and action/activity
heads so that the state can represent what those choices need. Keep the pretrained
perception frozen initially. Return to earlier conditions after intervening tasks,
retaining instance updates and experience, so the optimization includes their later
use. Task completion, activity selection and retained learning belong to the same
training run.

Split before generating training windows. Reserve official evaluation instances from training. Different random seeds do not necessarily define different scientific rules: Reactor Lab selects a relevant dimension with seed modulo 5, and Plant Nutrients Challenge selects a logical form with seed modulo 4. Record rule families and parameter combinations for dataset splitting, without exposing those labels to the policy. Report new-instance and held-out-rule performance separately. Public benchmark logs are not a training set for the same scored instances.

Repeated conditions also need distinct treatments. Repeating an identical seeded world measures exact-context return. Recovering a scientific relation with changed objects or appearance measures a stronger kind of retention and needs a separately defined generator. Use the existing process lifetimes for controlled returning dynamics while that extension is built.

## The research comparison

The central mechanism is training the model through the sequence it actually executes: observation, partial inference, action, returned outcome, retrieval and local learning. Three contributions matter:

- **Useful continuation:** after a new observation, continuing the existing workspace improves subsequent actions relative to restarting or leaving it unchanged at matched work.
- **Useful retention:** after intervening tasks, retained experience or learned parameters reduce the observations and interactions needed to recover performance.
- **Useful initiative:** learned observation/computation/learning choices improve task return relative to fixed schedules after including their time and resource costs.

Start with a directly acting pretrained model, a recurrent policy with the same frozen inputs and training data, and the integrated CDM. Use paired changes to memory, refinement and retained updates to locate each contribution. Include independently trained strong alternatives; interventions on one checkpoint are not a substitute for them.

Report task completion and procedural progress, recovery after change and return, and observation-to-applied-action latency with its tail. Report discovery-knowledge scores under the official procedure when that evaluator is available. An alternative evaluator is a separate result. Preserve failed runs and count collection, teacher inference and training in the compute budget.

## Execution on two H100s

Use the available two H100s as the baseline resource. Additional A100s increase independent collection and replication; their availability does not block this campaign.

- Initially allocate one H100 to teacher trajectories and frozen features, and the other to the continual core and task heads.
- Cache frozen features during replay; recompute trainable projections/readers. Batch by input length and refinement depth. Keep recurrent state and adaptation reductions in FP32 while using BF16 dense computation.
- Use CPU workers for independent worlds. The native API renders and writes multiple PNG images per observation, so measure simulation/rendering separately from model time. Do not copy the process simulator's 1,024-stream or 100-million-transition settings into this renderer without a throughput measurement.
- Once data and collection are ready, allocate both GPUs to the integrated model and its strongest comparison, then replicate the selected comparison over at least three training seeds. Larger parameter counts follow a task/data bottleneck study.

The first campaign result is an executed scientific task with learned actions and measured held-out behavior, followed by the matched continuation/learning comparison. Installation, test counts and an idle GPU are not that result.

## Run

Install the pinned simulator in the research environment:

~~~bash
python -m pip install "discoveryworld @ git+https://github.com/allenai/discoveryworld.git@fd591323920be0d3786ef350955de1945aa571e5"
python -m unittest discover -s tests -p 'test_discoveryworld_adapter.py' -v
~~~

For multimodal training, use the repository's multimodal extra in that environment. It pins Transformers 5.4.0 and requires a compatible Torch/Torchvision pair. Use the server's existing model weights and released CDM checkpoints where applicable.

~~~bash
python -m pip install -e '.[multimodal]'
~~~

Replay the included native measurement sequence to collect observations and returned outcomes:

~~~bash
python -m experiments.discoveryworld.collect --scenario "Reactor Lab" \
  --difficulty Normal --seed 100 --max-steps 4 \
  --actions experiments/discoveryworld/fixtures/reactor_measurement_seed100.jsonl \
  --output runs/discoveryworld/replay-100
~~~

The four actions use object references from the public observation of this specific instance. They acquire a thermometer and measure a crystal. On the pinned simulator they returned 32.22 degrees Celsius. This exercises a real tool/measurement transition and image recording; it is a scripted connection example and does not complete the scientific task.

Each input line is a native action dictionary, for example:

~~~json
{"action": "ROTATE_DIRECTION", "arg1": "north"}
~~~

For a live local policy, supply its factory as module:factory and optional JSON configuration:

~~~bash
python -m experiments.discoveryworld.collect --scenario "Reactor Lab" \
  --difficulty Normal --seed 100 --policy your_local_policy:create \
  --policy-config path/to/model.json --output runs/discoveryworld/policy-100
~~~

The factory receives that configuration and returns a callable. The callable receives the public observation plus action descriptions and teleport locations, and returns a native action dictionary or dialog choice. Its recurrent state remains in that callable across decisions. This is the connection point for the server's teacher and the trained CDM policy; no hosted model API is required.

Read trajectory.jsonl for transitions, images/ for assets, run.json for the episode configuration, and summary.json for task results. Use a new output directory per run.

## Observation throughput

[observation.py](observation.py) uses the native world and UI renderers while avoiding
the two debug-frame writes per observation. Its PNG mode preserves both image keys
and decoded pixels, with lighter lossless PNG compression. The image transport is
larger; this mode is intended for local collection. RGB mode returns HWC uint8
arrays directly to a local image processor and skips PNG/base64 encoding.

Use `--observation-mode png` with the collection command, or construct
`DiscoveryWorldEnv(output, observation_mode="png")`. Existing native observation
behavior remains available. For direct model integration:

~~~python
from experiments.discoveryworld.observation import get_observation

observation = get_observation(api)            # native-compatible PNG data URLs
observation = get_observation(api, rgb=True)  # arrays in vision["rgb_no_grid"] and vision["rgb_with_grid"]
~~~

Call only one reader per environment decision: reading the native UI consumes a
queued tool message. Save trajectory images separately when collecting training
data. In RGB mode, send the array directly to the existing image processor; the
PNG trajectory writer expects PNG mode.

On this CPU execution environment, 24 observation reads per mode in Reactor Lab
Normal seed 100 gave the following medians. These timings exclude world ticks,
model inference, training, and trajectory serialization.

| Reader | Time per observation | Relative speed |
| --- | ---: | ---: |
| Native API | 206.8 ms | 1.00× |
| In-memory PNG | 66.8 ms | 3.10× |
| In-memory RGB | 58.6 ms | 3.53× |

[The recorded run](benchmarks/observation_cpu_20261027.json) also includes payload
sizes. RGB arrays use more transport memory and should stay inside the local
inference process or use shared memory when collected by separate workers.
The native-simulator comparison checks both image views pixel for pixel, public
UI, action results and step counters through the four-action instrument sequence.
This is an observation-reader speedup; whole-training throughput depends on the
remaining simulator, inference and training time.

~~~bash
python -m unittest discover -s tests -p 'test_discoveryworld_observation.py' -v
python -m experiments.discoveryworld.benchmark_observation --steps 24 \
  --output runs/discoveryworld/observation-performance.json
~~~

## Sources and continuation

- [Official project and evaluation definitions](https://allenai.github.io/discoveryworld/)
- [Pinned API](https://github.com/allenai/discoveryworld/blob/fd591323920be0d3786ef350955de1945aa571e5/discoveryworld/DiscoveryWorldAPI.py)
- [Reactor Lab generator](https://github.com/allenai/discoveryworld/blob/fd591323920be0d3786ef350955de1945aa571e5/discoveryworld/scenarios/reactor_lab.py)
- [Plant Nutrients Challenge generator](https://github.com/allenai/discoveryworld/blob/fd591323920be0d3786ef350955de1945aa571e5/discoveryworld/scenarios/plant_growing_challenge.py)
- [Previous CDM results](../lifetime/RESULTS.md#second-round-training-through-deployment)

