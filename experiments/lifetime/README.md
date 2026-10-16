# Lifetime: one model that keeps acting, measuring, computing and learning

A single model runs each process for a whole lifetime. It keeps one state throughout, chooses at every tick how to act, whether to request a precise measurement, and how to spend the compute until the next tick: on refinements of its latent workspace or on a learning step that updates its own adapter. These activities have simulated durations and control is produced every tick; [`realtime.py`](realtime.py) also runs the controller against processes that advance on their own wall clock. This is the integrated campaign of [EXPERIMENTS.md](../../EXPERIMENTS.md#primary-comparisons): the same model under settings that switch its learned activities and its learning on and off, then one set of weights across two process families.

## Processes

[`cdm/instrument.py`](../../cdm/instrument.py) holds what every family shares (`ProcessFleet`): a lifetime has four operating conditions; the active one changes with probability 1/400 per tick, one change in five brings a condition never seen before and the others return to an earlier one, and nothing announces a change. Requests start with probability 1/40 per idle tick and stay active for 60 ticks; a setpoint is the steady output of one of the actions under the current condition. A cheap reading arrives every tick with noise 0.25; a precise measurement can be requested, one at a time, and returns the output at the request time after three ticks with noise 0.02. Two actuators with nine levels each give 81 actions. Every family advances in parallel on the GPU.

**Instruments** (`InstrumentFleet`) relax toward a peaked response of the actuator positions, as when aligning a beam onto a detector; the actuators move with backlash:

\[
r(v)=a\,\exp\!\Bigl(-\frac{\lVert v-c\rVert^2}{2w^2}\Bigr)+b+\delta_t,\qquad
y_{t+1}=y_t+\frac{r(v_t)-y_t}{\tau}+\epsilon_t .
\]

| Quantity | Range |
| --- | --- |
| Height \(a\), peak position \(c\), width \(w\), floor \(b\) | 0.8–1.5; [−0.7, 0.7]²; 0.25–0.6; −0.3–0.1 |
| Time constant \(\tau\), actuator backlash | 2–8 ticks; up to 0.15 |
| Drift \(\delta_t\) | random walk, 0.004 per tick, reset when the condition changes |

Setpoints come from actions where the peak is visible.

**Thermal processes** ([`cdm/thermal.py`](../../cdm/thermal.py), `ThermalFleet`) heat a block that exchanges heat with a body; the body loses heat to its surroundings, faster while coolant flows, and its temperature is the output:

\[
C_1\dot T_1=\eta h^\gamma-k_{12}(T_1-T_2)-k_1(T_1-T_a),\qquad
C_2\dot T_2=k_{12}(T_1-T_2)-k_2(1+\kappa c)(T_2-T_a).
\]

The first actuator sets the heater command \(h\), which acts after a transport delay of up to two ticks; the second sets the coolant flow \(c\). A condition draws \(k_{12}\in[0.1,0.3]\), \(k_1\in[0.02,0.06]\), \(k_2\in[0.06,0.15]\), \(\kappa\in[0.5,2]\), \(\gamma\in[0.5,2]\), \(T_a\in[-0.4,0.2]\), block and body time constants of 1–3 and 3–8 ticks, the delay, and a steady rise of 0.9–1.6 at full heat without coolant, which fixes \(\eta\). The ambient drifts like the instrument offset. From ambient, full heat reaches 90 % of its rise in a median of 37 ticks.

`MixedFleet` runs both families in fixed blocks of streams; the model is not told which family a stream belongs to.

Reward per tick is minus the absolute tracking error while a request is active, minus 0.002 per refinement, 0.01 per precise measurement and 0.02 per learning step.

[`references.py`](references.py) gives two reference policies per family ([results](results/references.json)): random actions, and a controller that knows the true steady response and takes the action whose steady output is closest to the setpoint. The second ignores the dynamics, so a policy that anticipates relaxation, backlash or delay can do better; it is a reference point, not a bound.

| Family | Random actions | True-steady greedy |
| --- | ---: | ---: |
| Instrument | 0.305 | 0.115 |
| Thermal | 0.269 | 0.091 |
| Instrument, time constant 8–16, width 0.15–0.25 | 0.389 | 0.224 |
| Instrument, cheap noise 0.4 | 0.305 | 0.115 |
| Instrument, a change every 150 ticks, half of them new | 0.311 | 0.113 |

## Model

[`cdm/core.py`](../../cdm/core.py), one network for every setting:

| Part | Content |
| --- | --- |
| Fast state | Event features (readings, setpoint, request flags, previous control and activities, pending measurement, time since the last measurement, last prediction error) → MLP → GRU |
| Workspace | 8 latent slots; a refinement applies one shared Transformer block (slot self-attention, attention to the fast state and the 32 most recent experience records, MLP) and normalises each slot, so the workspace keeps one scale over any number of refinements |
| Predictor | Mean and log-variance of the next reading for every action, from the fast state, the slot mean and learned action features. Its first layer carries an instance-owned rank-8 adapter |
| Control | Scores every action from the shared trunk, the action's features and the action's own predicted reading relative to the setpoint, with weights shared across actions, minus a learned multiple of the action's predicted distance from the setpoint |
| Activities | Precise measurement (yes, no); compute (0, 1, 2 or 4 refinements, or a learning step) |
| Value | Critic over the shared trunk |

The campaign uses width 1,024 and 35.9 M parameters.

### Time

Control is produced every tick from the fast state, the workspace completed so far and the predictions under the current adapter. The compute between one tick and the next goes to the chosen option. Refinements are read from the next tick on. A learning step samples 32 of the instance's last 64 ticks (state features, action, following reading), takes one normalised gradient step of size 0.05 on the adapter, and occupies the compute for two ticks: during the second the instance can neither refine nor start another step, and the updated adapter is used from the tick after it completes. A precise measurement arrives three ticks after the request while the process keeps moving.

[`realtime.py`](realtime.py) removes the simulated tick: the fleet steps on a wall-clock schedule in its own thread and CUDA stream and holds the previous action when no new one has arrived; the actor publishes each action as soon as the control path finishes, refinements follow on its stream, and learning steps run on a learner thread and third stream whose adapter is installed when it returns. Because nothing on the control path waits, the workspace reads experience records one tick older than in the simulated runtime.

## Training

Recurrent PPO over lifetimes ([`train.py`](train.py), [`runtime.py`](runtime.py)). 1,024 streams run lifetimes of 16,384 ticks with staggered starts. Fast state, workspace, experience records, adapters and learning buffers persist across 128-tick training windows; only the end of a lifetime resets them, and only that end is terminal for the value targets, which otherwise bootstrap from the next decision state.

For each window: four epochs over minibatches of 64 streams; backpropagation through the fast state over the window from its stored starting state, with each stream reset at its collected lifetime boundaries; every tick's refinement recomputed from the stored workspace before it, with workspace gradients extending to the next tick only; γ = 0.995, λ = 0.95, clip 0.2, entropy 0.003, rewards scaled by 0.1 for the critic, Adam at 3·10⁻⁴. The shared predictor is trained on the readings that followed each action (Gaussian likelihood, weight 0.5). Policy probabilities use the predictions cached during collection, which include the instance's adapter at that tick; PPO does not differentiate through those cached predictions. Each window logs its collection and update time; each run records the source revision.

| Setting | Measurement | Compute | Predictions |
| --- | --- | --- | --- |
| `recurrent` | learned | none | not supplied, no prediction loss |
| `predict_only` | learned | none | yes |
| `frozen` | learned | learned refinements; no learning | yes |
| `fixed_learn` | learned | a learning step every 8 ticks; refinements learned otherwise | yes |
| `full` | learned | learned: refinements or learning | yes |
| `world_model` | every 8 ticks | a learning step every 8 ticks, one refinement otherwise | yes |
| `full_meta` | as `full`, with a loss through the learning step | | yes |

Read top to bottom, each setting adds one ability: world prediction, learned workspace computation, online learning on a fixed schedule, and the choice of when to learn; `world_model` against `fixed_learn` separates learned from fixed allocation of observation and compute. `full_meta` trains the shared network through the learning step itself: from the adapter each stream held when the window began, one step on the first half of the window, then the prediction loss on the second half, with gradients through the step (weight 0.5, `--meta 0.5`). All settings share the network and the training procedure; settings that do not refine or predict leave those parts unused. Each setting has five seeds of 2·10⁷ interactions, 10⁸ per setting.

The two-family experiment trains `full`, `frozen` and `recurrent` on mixed lifetimes, half instrument and half thermal (`--family mixed`), and `full` on thermal lifetimes alone (`--family thermal`), three seeds each.

## Evaluation

[`evaluate.py`](evaluate.py) runs each trained model through one uninterrupted lifetime per process on newly sampled processes, with nothing reset. It reports tracking error, reward, response time to a new setpoint, prediction error with the instance adapter and, on the same states, of the shared predictor alone, and activity rates; tracking error, prediction error, learning and refinement as functions of time since the last condition change, separately for new and returning conditions; and error across the lifetime. Tracking error and condition statistics use the request and condition under which the action was applied. Response time counts executed controls, starting at one; requests that expire without reaching the tolerance are assigned their duration, and requests still open when the evaluation ends are counted separately.

Each learning model is also evaluated under other settings with `--also`, every setting starting from the same random state so the comparisons use common random numbers:

- `discard`: chosen learning steps keep their time and cost but their adapter updates are thrown away; this isolates what the updates contribute.
- `frozen`: learning is switched off and its compute goes to the remaining options, so this measures removing learning together with the change of computation. A model trained on fixed schedules has untrained activity heads and is evaluated with `discard` only.

`--family` evaluates a model on either family; `--instrument` changes fields of the process specification, for example `time_constant=8,16 width=0.15,0.25` for dynamics outside the training ranges. [`report.py`](report.py) aggregates seeds, converts refinements and learning steps into measured GPU time per instance and tick, and gives per-seed paired differences between settings.

[`probe.py`](probe.py) `readout` compares the policy with a greedy readout of the model's own predictions, with learning on and off. `decisions` follows one trajectory and compares the adapted predictor and the shared predictor alone with the noise-free next output of every action (`expected_next`): residual error for the action taken and over all actions, how often their greedy choices differ, and the one-step regret and steady-response gap of each choice. [`latency.py`](latency.py) times the control path, a refinement and a learning step for several numbers of instances, and response time from scheduled starts with a learner on a second stream; it repeats fixed inputs, so it measures contention for the GPU. [`realtime.py`](realtime.py) measures control of the advancing processes: response time from published observation to published action, steps run on a held action, tracking error, and refinement and learning throughput.

The corrections of [issue #1](https://github.com/Cherish0925/continual-decision-model/issues/1) changed PPO state reconstruction at lifetime resets and evaluation accounting. Checkpoints trained before them are re-evaluated with the corrected metrics and reported separately ([results/prefix_trainer](results/prefix_trainer)).

## Run

```bash
python experiments/lifetime/launch.py --output-root runs/lifetime/c4 --seeds 0 1 2 3 4 \
    --variants full "full_meta=full --meta 0.5" frozen fixed_learn world_model predict_only recurrent -- \
    --streams 1024 --width 1024 --hidden 1024 --interactions 2e7 --lifetime 16384
python experiments/lifetime/launch.py --output-root runs/lifetime/x1 --seeds 0 1 2 \
    --variants "mixed_full=full --family mixed" "mixed_frozen=frozen --family mixed" \
    "mixed_recurrent=recurrent --family mixed" "thermal_full=full --family thermal" -- \
    --streams 1024 --width 1024 --hidden 1024 --interactions 2e7 --lifetime 16384
python experiments/lifetime/evaluate.py --model runs/lifetime/c4/full/seed0/model.pt --also frozen discard \
    --devices 512 --ticks 16384 --output runs/lifetime/eval/full_seed0.json      # one file per model
python experiments/lifetime/report.py --evaluations runs/lifetime/eval/*.json --runs runs/lifetime/c4 \
    --latency experiments/lifetime/results/latency.json --output experiments/lifetime/results/main
python experiments/lifetime/probe.py --model runs/lifetime/c4/full/seed0/model.pt --output experiments/lifetime/results/probe.json
python experiments/lifetime/latency.py --model runs/lifetime/c4/full/seed0/model.pt --output experiments/lifetime/results/latency.json
python experiments/lifetime/realtime.py --model runs/lifetime/c4/full/seed0/model.pt --output experiments/lifetime/results/realtime.json
python experiments/lifetime/references.py --output experiments/lifetime/results/references.json \
    --family slow_narrow instrument "time_constant=8,16 width=0.15,0.25"
```

`launch.py` starts a run on a GPU only when enough memory is free and skips runs that have finished or are in progress.
