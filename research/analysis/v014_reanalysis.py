"""Recompute selected v0.14 results and a finite experimental-design example.

Standard library only. Run from any directory:
    python research/analysis/v014_reanalysis.py --output research/analysis/v014_summary.json

This reads published records; it does not train or execute a native task. The
dynamic program is exact only for its stated finite, deterministic model.
"""

import argparse
import collections
import csv
import hashlib
import json
import math
import random
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS = Path("experiments/continual_experiment/results")
TRAIN = Path("experiments/continual_chemistry/results/exploration_data.json")


def recipes(n, limit=6):
    def partition(total, count):
        if count == 1:
            yield (total,)
        else:
            for amount in range(total, -1, -1):
                for tail in partition(total - amount, count - 1):
                    yield (amount, *tail)

    return [q for total in range(1, limit + 1)
            for q in partition(total, n) if math.gcd(*q) == 1]


def response(q, h):
    cosine = sum(a * b for a, b in zip(q, h)) / math.sqrt(
        sum(a * a for a in q) * sum(b * b for b in h))
    return 0 if cosine >= .99 else 1 if cosine >= .66 else 2 if cosine >= .33 else 3


def action_cost(q, first):
    return 6 + sum(q) + sum(x > 0 for x in q) + (0 if first else 2)


def entropy(probabilities):
    return -sum(p * math.log(p) for p in probabilities if p > 0)


def training_prior(episodes, difficulty):
    n = {"Normal": 3, "Challenge": 4}[difficulty]
    selected = [e for e in episodes if e["split"] == "train" and e["difficulty"] == difficulty]
    counts = collections.Counter()
    for e in selected:
        parts = dict(item.rsplit(":", 1) for item in e["recipe_group"].split("|"))
        counts[tuple(sorted(int(v) for v in parts.values() if int(v)))] += 1
    placements = collections.defaultdict(list)
    for q in recipes(n):
        placements[tuple(sorted(x for x in q if x))].append(q)
    weights = {q: count / len(selected) / len(placements[structure])
               for structure, count in counts.items() for q in placements[structure]}
    support = sorted(weights)
    assert math.isclose(sum(weights.values()), 1)
    return support, [weights[q] for q in support], {
        "episodes": len(selected), "support_size": len(support),
        "structure_counts": {str(k): v for k, v in sorted(counts.items())},
    }


class FinitePlanner:
    """Posterior subsets suffice for deterministic responses and a fixed prior."""

    def __init__(self, hypotheses, weights):
        self.weights = weights
        self.actions = recipes(len(hypotheses[0]))
        self.parts = {}
        for a, q in enumerate(self.actions):
            for rust in (1, 2, 3):
                outcomes = collections.defaultdict(int)
                for i, h in enumerate(hypotheses):
                    outcomes[min(rust, response(q, h))] |= 1 << i
                self.parts[a, rust] = sorted(outcomes.items())
        self.order = sorted(range(len(self.actions)), key=lambda a: action_cost(self.actions[a], True))

    @lru_cache(None)
    def mass(self, mask):
        return sum(w for i, w in enumerate(self.weights) if mask & (1 << i))

    @lru_cache(None)
    def value(self, mask, rust, first):
        if rust == 0:
            return 4.0, None  # Native exit action.
        best, best_action = math.inf, None
        for a in self.order:
            value = action_cost(self.actions[a], first)
            if value >= best:
                break
            parts = [(o, mask & subset) for o, subset in self.parts[a, rust] if mask & subset]
            if parts == [(rust, mask)]:
                continue  # No state change or information; positive cost makes repetition dominated.
            for o, subset in parts:
                value += self.mass(subset) / self.mass(mask) * self.value(subset, o, False)[0]
                if value >= best:
                    break
            if value < best:
                best, best_action = value, a
        return best, best_action

    def forced_first(self, q, mask):
        a = self.actions.index(q)
        return action_cost(q, True) + sum(
            self.mass(mask & subset) / self.mass(mask) * self.value(mask & subset, o, False)[0]
            for o, subset in self.parts[a, 3] if mask & subset)


def action_example(support, base):
    remembered = [(1, 1, 1), (1, 3, 0)]
    indices = [support.index(q) for q in remembered]
    remaining = sum(w for i, w in enumerate(base) if i not in indices)
    weights = [.275 if i in indices else .45 * w / remaining for i, w in enumerate(base)]
    planner = FinitePlanner(support, weights)
    mask = (1 << len(support)) - 1
    optimum, a = planner.value(mask, 3, True)
    rows = []
    for q in dict.fromkeys([*remembered, planner.actions[a]]):
        outcomes = collections.defaultdict(float)
        for h, w in zip(support, weights):
            outcomes[response(q, h)] += w
        conditional_entropy = 0
        groups = [[i for i in range(len(support)) if i not in indices], *[[i] for i in indices]]
        for group in groups:
            mass = sum(weights[i] for i in group)
            conditional = collections.defaultdict(float)
            for i in group:
                conditional[response(q, support[i])] += weights[i] / mass
            conditional_entropy += mass * entropy(conditional.values())
        rows.append({
            "first_action": q, "total_expected_cost": planner.forced_first(q, mask) + 14,
            "success_probability": outcomes[0], "relationship_information_nats": entropy(outcomes.values()),
            "component_identity_information_nats": entropy(outcomes.values()) - conditional_entropy,
            "immediate_progress_per_cost": sum(p * (3 - o) for o, p in outcomes.items()) / action_cost(q, True),
        })
    return {"return_probability": .55, "available_actions": len(planner.actions),
            "preparation_cost": 14, "exit_cost": 4,
            "optimum_total_expected_cost": optimum + 14, "actions": rows}


def read_rows(path):
    with (ROOT / path).open() as f:
        return [r for r in csv.DictReader(f) if r["task"] == "chemistry"]


def aggregate(rows):
    return {"tasks": len(rows),
            "completed": sum(r["completed_successfully"].lower() == "true" for r in rows),
            "mean_operations": sum(int(r["native_actions"]) for r in rows) / len(rows)}


def paired_difference(candidate, reference):
    key = lambda r: (r["stream"], r["episode_index"])
    lookup = {key(r): r for r in reference}
    assert len(lookup) == len(reference) == len(candidate)
    assert set(lookup) == {key(r) for r in candidate}
    by_stream = collections.defaultdict(list)
    for row in candidate:
        ref = lookup[key(row)]
        assert all(row[f] == ref[f] for f in ("seed", "difficulty", "exposure"))
        by_stream[row["stream"]].append(int(row["native_actions"]) - int(ref["native_actions"]))
    streams = list(by_stream.values())
    rng = random.Random(20261102)
    samples = []
    for _ in range(4000):
        draws = [rng.choice(streams) for _ in streams]
        samples.append(sum(sum(x) for x in draws) / sum(len(x) for x in draws))
    samples.sort()
    delta = sum(sum(x) for x in streams) / len(candidate)
    return {"candidate_minus_reference": delta,
            "saving_percent": -100 * delta / aggregate(reference)["mean_operations"],
            "paired_stream_bootstrap_95_percent": [samples[99], samples[3899]],
            "streams": len(streams), "bootstrap_replicates": 4000, "bootstrap_seed": 20261102}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    paths = [TRAIN, *[RESULTS / name for name in (
        "experimental_memory_instances.csv", "memory_capacity_native_instances.csv",
        "memory_capacity_confirmation_instances.csv")]]
    episodes = json.loads((ROOT / TRAIN).read_text())["episodes"]
    priors = {d: training_prior(episodes, d) for d in ("Normal", "Challenge")}
    original, capacity, confirmation = [read_rows(p) for p in paths[1:]]
    output = {
        "source_revision": "aecc237ecb191685d7e3b4f9f80565a78fdc9976",
        "scope": "Published CSV aggregation and finite-model calculation; no new training or native executions.",
        "source_sha256": {str(p): hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in paths},
        "training_prior": {d: values[2] for d, values in priors.items()},
        "prior_assumptions": "Episode-frequency structure prior; exchangeable ingredient labels; zero mass outside observed structures.",
        "native_records": {}, "primary_paired": {}, "confirmation_paired": {},
    }
    for label, rows in (("primary_original", original), ("primary_capacity", capacity), ("confirmation", confirmation)):
        selected = sorted({r["condition"] for r in rows if r["condition"].startswith(("initial", "capacity"))})
        output["native_records"][label] = {}
        for condition in selected:
            subset = [r for r in rows if r["condition"] == condition]
            output["native_records"][label][condition] = {
                "all": aggregate(subset), **{v: aggregate([r for r in subset if r[k] == v])
                for k, values in (("difficulty", ("Normal", "Challenge")), ("exposure", ("new", "return"))) for v in values}}
    for run in (911, 912):
        select = lambda rows, c: [r for r in rows if r["condition"] == c]
        output["primary_paired"][str(run)] = paired_difference(
            select(capacity, f"capacity12_{run}"), select(original, f"initial{run}"))
        output["confirmation_paired"][str(run)] = paired_difference(
            select(confirmation, f"capacity12_{run}"), select(confirmation, f"capacity6_{run}"))
    output["finite_action_example"] = action_example(*priors["Normal"][:2])
    output["identity_counterexample"] = {
        "targets": [[1, 3, 0], [1, 4, 0]], "common_successful_action": [1, 4, 0],
        "common_action_levels": [response((1, 4, 0), h) for h in ((1, 3, 0), (1, 4, 0))],
        "distinguishing_action": [2, 1, 0],
        "distinguishing_levels": [response((2, 1, 0), h) for h in ((1, 3, 0), (1, 4, 0))],
        "scope": "Broader target family; (1,4,0) is not in the training-derived target support.",
    }
    rendered = json.dumps(output, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
