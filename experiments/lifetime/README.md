# Lifetime: one model that keeps acting, measuring, computing and learning

A single model runs each instrument for a whole lifetime. It keeps one state throughout, chooses at every tick how to act, whether to request a precise measurement, and how to spend the compute until the next tick: on refinements of its latent workspace or on a learning step that updates its own adapter. These activities have simulated durations; control is produced every simulator tick. This is the integrated campaign of [EXPERIMENTS.md](../../EXPERIMENTS.md#primary-comparisons): the same model under settings that switch its learned activities and its learning on and off.

## Task

[`cdm/instrument.py`](../../cdm/instrument.py) simulates a fleet of instruments in parallel on the GPU. Each has two actuators with nine levels each (81 actions) and an output that relaxes toward a peaked response of the actuator positions, as when aligning a beam onto a detector:

\[
r(v)=a\,\exp\!\Bigl(-\frac{\lVert v-c\rVert^2}{2w^2}\Bigr)+b+\delta_t,\qquad
y_{t+1}=y_t+\frac{r(v_t)-y_t}{\tau}+\epsilon_t .
\]

| Quantity | Range |
| --- | --- |
| Height \(a\), peak position \(c\), width \(w\), floor \(b\) | 0.8–1.5; [−0.7, 0.7]²; 0.25–0.6; −0.3–0.1 |
| Time constant \(\tau\), actuator backlash | 2–8 ticks; up to 0.15 |
| Drift \(\delta_t\) | random walk, 0.004 per tick, reset when the condition changes |
| Cheap reading | every tick, noise 0.25 |
| Precise measurement | on request, one at a time, noise 0.02, returns the output at the request time after 3 ticks |

A lifetime holds four operating conditions, each with its own response and dynamics. The active condition changes with probability 1/400 per tick; one change in five brings a condition never seen before, the others return to an earlier one. Nothing announces a change. Requests start with probability 1/40 per idle tick and stay active for 60 ticks; a setpoint is the response at an action where the peak is visible. Between requests the instrument is idle.

Reward per tick is minus the absolute tracking error while a request is active, minus 0.002 per refinement, 0.01 per precise measurement and 0.02 per learning step.

| Reference policy | Tracking error while a request is active |
| --- | ---: |
| Random actions | 0.308 |
| Centre action | 0.353 |
| Knows the condition and drift, picks the action whose response is closest to the setpoint | 0.112 |

The last reference ignores the relaxation and the backlash, so a policy that anticipates them can do better; it is a reference point, not a bound.

## Model

[`cdm/core.py`](../../cdm/core.py), one network for every setting:

| Part | Content |
| --- | --- |
| Fast state | Event features (readings, setpoint, request flags, previous control and activities, pending measurement, time since the last measurement, last prediction error) → MLP → GRU |
| Workspace | 8 latent slots; a refinement applies one shared Transformer block: slot self-attention, attention to the fast state and the 32 most recent experience records, MLP |
| Predictor | Mean and log-variance of the next reading for every action, from the fast state, the slot mean and learned action features. Its first layer carries an instance-owned rank-8 adapter |
| Control | Scores every action from the shared trunk, the action's features and the action's own predicted reading relative to the setpoint, with weights shared across actions, minus a learned multiple of the action's predicted distance from the setpoint |
| Activities | Precise measurement (yes, no); compute (0, 1, 2 or 4 refinements, or a learning step) |
| Value | Critic over the shared trunk |

The main campaign uses width 1,024 and 35.9 M parameters.

### Time

Control is produced every tick from the fast state, the workspace completed so far and the predictions under the current adapter. The compute between one tick and the next goes to the chosen option. Refinements are read from the next tick on. A learning step samples 32 of the instance's last 64 ticks (state features, action, following reading), takes one normalised gradient step of size 0.05 on the adapter, and occupies the compute for two ticks: during the second the instance can neither refine nor start another step, and the updated adapter is used from the tick after it completes. A precise measurement arrives three ticks after the request while the instrument keeps moving.

## Training

Recurrent PPO over lifetimes ([`train.py`](train.py), [`runtime.py`](runtime.py)). 1,024 streams run lifetimes of 16,384 ticks with staggered starts. Fast state, workspace, experience records, adapters and learning buffers persist across 128-tick training windows; only the end of a lifetime resets them, and only that end is terminal for the value targets, which otherwise bootstrap from the next decision state.

For each window: four epochs over minibatches of 64 streams; backpropagation through the fast state over the window from its stored starting state, with each stream reset at its collected lifetime boundaries; every tick's refinement recomputed from the stored workspace before it, with workspace gradients extending to the next tick only; γ = 0.995, λ = 0.95, clip 0.2, entropy 0.003, rewards scaled by 0.1 for the critic, Adam at 3·10⁻⁴. The shared predictor is trained on the readings that followed each action (Gaussian likelihood, weight 0.5). Policy probabilities use the predictions cached during collection, which include the instance's adapter at that tick; PPO does not differentiate through those cached predictions.

| Setting | Measurement | Compute | Predictions |
| --- | --- | --- | --- |
| `full` | learned | learned: refinements or learning | yes |
| `fixed_learn` | learned | a learning step every 8 ticks; refinements learned otherwise | yes |
| `frozen` | learned | learned refinements; no learning | yes |
| `world_model` | every 8 ticks | a learning step every 8 ticks, one refinement otherwise | yes |
| `recurrent` | learned | none | not supplied, no prediction loss |

All settings share the network and the training procedure; the recurrent policy leaves the workspace and predictor unused. A sixth setting, `full_meta`, adds a loss that trains the shared network through the learning step itself: from the adapter each stream held when the window began, one step on the first half of the window, then the prediction loss on the second half, with gradients through the step (weight 0.5, `--meta 0.5`).

Each setting has five seeds of 2·10⁷ interactions, 10⁸ per setting.

## Evaluation

[`evaluate.py`](evaluate.py) runs each trained model through one uninterrupted lifetime per instrument on newly sampled instruments, with nothing reset. It reports tracking error, reward, response time to a new setpoint, prediction error and activity rates. Tracking error and condition-switch statistics use the request and condition under which the action was applied; the following observation describes the next decision. Response time counts executed controls, starting at one. Requests that expire without reaching the tolerance are assigned their duration; unfinished requests without a first hit at the end of evaluation are counted separately. Partial final lifetime bins use their actual number of samples.

`--also frozen` repeats the evaluation with learning disabled. This measures the combined effect of removing learning and changing subsequent computation choices and trajectories; it does not hold the activity schedule fixed. [`latency.py`](latency.py) measures service time and response time relative to scheduled starts, including accumulated lateness, with a learner on a second CUDA stream. It repeatedly uses fixed inputs and discards adapter updates, so it measures GPU contention rather than an evolving instrument lifetime. Control against an independently advancing environment remains a separate experiment.

The corrections in [issue #1](https://github.com/Cherish0925/continual-decision-model/issues/1) change PPO state reconstruction at resets and evaluation accounting. Previously trained checkpoints can be reevaluated with corrected metrics; their training still used the earlier reset handling. Compare them separately from models trained after the correction.

## Run

```bash
python experiments/lifetime/launch.py --output-root runs/lifetime/main --seeds 0 1 2 3 4 \
    --variants full fixed_learn frozen world_model recurrent "full_meta=full --meta 0.5" -- \
    --streams 1024 --width 1024 --hidden 1024 --interactions 2e7 --lifetime 16384
python experiments/lifetime/evaluate.py --model runs/lifetime/main/full/seed0/model.pt --also frozen \
    --devices 512 --ticks 16384 --output runs/lifetime/eval/full_seed0.json      # one file per model
python experiments/lifetime/report.py --evaluations runs/lifetime/eval/*.json --runs runs/lifetime/main \
    --output experiments/lifetime/results
python experiments/lifetime/probe.py --model runs/lifetime/main/full/seed0/model.pt \
    --output experiments/lifetime/results/probe.json
python experiments/lifetime/latency.py --model runs/lifetime/main/full/seed0/model.pt \
    --output experiments/lifetime/results/latency.json
```

