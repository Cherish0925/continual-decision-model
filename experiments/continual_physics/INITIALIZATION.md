# Why the initial continuous runtime changed the inherited policy

The first continual-physics initialization preserved every inherited model
tensor, but did not preserve the computation performed with those tensors.
The initial models completed 63/120 and 74/120 native tasks, compared with
117/120 and 119/120 for their respective parents. Training the first model
subsequently increased completion to 94/120. The initial loss of capability
therefore cannot be attributed entirely to the new training objective.

We replayed six existing public prefixes, using the same received measurements,
initial neural state, physical prior and executed feedback. No simulator was
run and no new evaluation instances were selected for training. This identifies
local computational causes; it does not assign a percentage of the full
completion-rate difference to each cause.

## Two changes already alter real decisions

The initial activity head gives ACT a bias of 2 and all other activities a
score of zero. While a tool result is pending, ACT is unavailable. Greedy
selection resolves the remaining tie in index order and selects REFINE rather
than WAIT. A two-tick delivery delay therefore adds two workspace updates on
the previous observation after every request. The original parent updates its
workspace once per newly received observation. No learned computation benefit
is needed for this change to occur.

Separately, the new observation builder clears the parent's failed-relation
flag and action mask. The inherited policy was trained to switch experiments
after that flag is set. Clearing it permits repeated tests of the same relation
without new measurements. Excluding previously failed frequency intervals from
the numerical proposal does not by itself train the inherited action network
to choose when such retries are useful.

| Initialization | Native world | Decision, zero-based | Parent | Original wrapper | WAIT during pending | WAIT + inherited failed-relation contract |
|---|---:|---:|---:|---:|---:|---:|
| 231 | 42000 | 5 | 20 | 20 | 20 | 20 |
| 231 | 42802 | 7 | 9 | 20 | 20 | 9 |
| 231 | 43303 | 3 | 12 | 0 | 12 | 12 |
| 232 | 43303 | 1 | 2 | 12 | 2 | 2 |
| 232 | 43804 | 1 | 5 | 3 | 5 | 5 |
| 232 | 43604 | 4 | 12 | 21 | 21 | 12 |

Actions 0–19 select an object and instrument; actions 20–28 test a relation's
predicted setting. In world 42802, action 9 measures support object 2 with the
microscope, whereas action 20 retests the density relation. Both recorded
policies received exactly the same 20 failed setting observations before this
decision. World 43604 similarly has an identical failed-setting sequence.
In worlds 43303 and 43804 the early divergence occurs before any failed test,
isolating the extra workspace computation from failure handling.

The inherited GRU states remain identical on these prefixes. The two extra
refinements change the workspace state and its action readout. With WAIT and
the original candidate inputs, inherited action logits match the parent
exactly. The random prediction and physical-memory layers do not explain
these initial changes: their action contributions are zero, and these runs
select neither retrieval nor online learning at initialization.

## A third semantic change

The original candidate mean column describes the Gaussian prediction. The
wrapper replaces it with a frequency proposal, including control-grid rounding
and exclusion of failed intervals. Even a zero physical correction head does
not make those quantities identical. In the examples above this change does
not independently explain the first discrete action divergence, but it breaks
the claim that the inherited input representation is unchanged.

The new runtime also carries neural state between tasks while the parent
comparison resets neural state. That is a separate experimental choice. The
prefix comparisons start at the first task with equal initial state and do not
attribute cross-task differences to the two mechanisms above.

## Optional inherited-interface candidate

`ContinualPhysicsConfig(inherit_parent_interface=True)` defines a new candidate:

- Initial pending activity prefers WAIT over the other zero-score activities;
  ACT remains preferred when an external action is available.
- Existing candidate fields keep their original interpretation. New physical
  corrections and failure-conditioned weights enter action scores through a
  separately zero-initialized residual.
- A test executes the corrected Gaussian mean through the original numerical
  tool. It does not replace that action's meaning with an exclusion-based search
  proposal. The original failed-relation field and mask remain in effect.

This mode retains the parent's restriction that a failed relation needs new
relevant evidence before being tested again. It is an explicit alternative
research setting, not a permanent requirement for CDM. Giving the model freer
retry actions requires a corresponding representation and training signal.
Any eventual task improvement from the candidate would include this action-set
change and could not be described as a pure parameter-initialization effect.
The mode does not reset neural state at task boundaries; state-reset controls
remain separate.

All 27 replayed decisions in the six prefixes have exactly equal candidate and
parent action logits and masks. Candidate save/load preserves every tensor and
the pending-WAIT preference. The existing checkpoint format without the new
option retains its original behavior. The normal training loader accepts the
candidate, and both execution environments maintain the same failed-relation
contract when this option is present. One CPU window of four requests completed
with finite gradients and no optimizer step. Nine existing core/native tests
passed. No candidate training campaign or native performance evaluation has
been run.

The four ongoing native-cost training jobs and their frozen source remain
unchanged.

## Reproduce the prefix comparison

```bash
python -m experiments.continual_physics.compare_initial_prefix \
  --parent runs/physical_belief/joint_20261029/carry_seed221/policy.pt \
  --initial runs/continual_physics/joint_20261029/full_seed231/initial.pt \
  --records \
    runs/continual_physics/native_confirmation/parent_seed221/confirmation_00/episode000_seed42000/summary.json \
    runs/continual_physics/native_confirmation/parent_seed221/confirmation_08/episode000_seed42802/summary.json \
    runs/continual_physics/native_confirmation/parent_seed221/confirmation_13/episode000_seed43303/summary.json \
  --output runs/continual_physics/initial_prefix/seed231
```

For initialization 232 use `joint_repeat_20261029/carry_seed222/policy.pt`,
`full_seed232/initial.pt`, and parent-222 first-task records for worlds 43303,
43804 and 43604. The script writes the explicit candidate checkpoint, received
public feedback sample and every prefix comparison. Compact results are in
[initial_prefix.csv](results/initial_prefix.csv); the full computation comparisons
are in [initial_prefix.json](results/initial_prefix.json).
