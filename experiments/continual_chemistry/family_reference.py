"""Finite, known-family response belief for a strong model-based comparison.

The reference knows the five generator formulas and a fixed parameter grid.
It sees only chosen, censored responses, never the current law or its identity.
Continuous parameters are approximated; the likelihood floor accounts for grid
error. The two-step rule uses an explicit approximate continuation cost and is
not an optimality bound.
"""
from functools import lru_cache
import random

import numpy as np
import torch

from .recipes import primitive_recipes
from .response_families import TRAIN_FAMILIES, response_levels, sample_law


@lru_cache(maxsize=2)
def function_bank(n):
    rng = random.Random(1849 + n)
    recipes = primitive_recipes(n)
    queries = torch.tensor(recipes)
    predictions, masses = [], []
    for family in TRAIN_FAMILIES:
        group = []
        for target in recipes:
            if family == 'cosine':
                variants = [{}]
            elif family == 'metric':
                variants = [dict(metric=[10 ** rng.uniform(-.9, .9) for _ in range(n)]) for _ in range(4)]
            elif family == 'interaction':
                variants = [dict(interaction=scale) for scale in (1., 2.5, 4.)]
            elif family == 'inhibition':
                variants = [dict(inhibitor=i, tolerance=tol) for i in range(n) for tol in (.04, .11, .18)]
            else:
                variants = [dict(second_target=list(rng.choice(recipes))) for _ in range(4)]
            for options in variants:
                law = {**sample_law(rng, n, (family,)), 'target': list(target), **options}
                group.append(response_levels(queries, law).numpy().astype(np.uint8))
        predictions.extend(group)
        masses.extend([1. / (len(TRAIN_FAMILIES) * len(group))] * len(group))
    return np.stack(predictions), np.asarray(masses), np.asarray(recipes)


class FamilyBelief:
    def __init__(self, n, reuse=.45, capacity=12, floor=.005, shortlist=12, selection='two_step'):
        self.levels, self.fresh, self.recipes = function_bank(n)
        self.reuse, self.capacity, self.floor, self.shortlist = reuse, capacity, floor, shortlist
        self.selection = selection
        self.archives = []
        self.posterior = self.fresh.copy()
        self.observations = 0

    def start(self):
        if self.observations:
            self.archives.append(self.posterior.copy())
            self.archives = self.archives[-self.capacity:] if self.capacity else []
        self.posterior = self.fresh.copy()
        if self.archives:
            self.posterior = (1 - self.reuse) * self.fresh + self.reuse * np.mean(self.archives, axis=0)
        self.observations = 0

    def probabilities(self, rust):
        observed = np.minimum(self.levels, rust)
        result = np.stack([(self.posterior[:, None] * (observed == y)).sum(0)
                           for y in range(rust + 1)], -1)
        return (1 - self.floor) * result + self.floor / (rust + 1)

    def receive(self, action, before, after):
        likelihood = (1 - self.floor) * (np.minimum(self.levels[:, action], before) == after)
        likelihood = likelihood + self.floor / (before + 1)
        self.posterior *= likelihood
        self.posterior /= self.posterior.sum()
        self.observations += 1

    def choose(self, rust, valid):
        p = self.probabilities(rust)
        cost = 6 + self.recipes.sum(-1) + (self.recipes > 0).sum(-1) + 2 * (self.observations > 0)
        entropy = -(p * np.log(p)).sum(-1)
        # Each conditional outcome has the same assumed error floor, so its
        # conditional entropy is constant across actions for this rust level.
        direct = np.where(valid, p[:, 0] / cost, -np.inf)
        info = np.where(valid, entropy / cost, -np.inf)
        if self.selection == 'greedy':
            return int(direct.argmax())
        if self.selection == 'information':
            k = rust + 1
            noise = np.full(k, self.floor / k); noise[0] += 1 - self.floor
            conditional_entropy = -(noise * np.log(noise)).sum()
            information = (entropy - conditional_entropy).clip(0)
            # Compare immediate completion and information per operation. One
            # fixed coefficient is declared before inspecting these results.
            return int(np.where(valid, (p[:, 0] + .5 * information) / cost, -np.inf).argmax())
        chosen = []
        for ranking in (np.argsort(-direct), np.argsort(-info)):
            added = 0
            for a in ranking:
                if valid[a] and a not in chosen:
                    chosen.append(int(a)); added += 1
                if added >= self.shortlist // 2:
                    break
        success = (self.levels == 0).astype(float)
        values = []
        for a in chosen:
            q = float(cost[a] + 4 * p[a, 0])
            for y in range(1, rust + 1):
                likelihood = (1 - self.floor) * (np.minimum(self.levels[:, a], rust) == y)
                posterior = self.posterior * (likelihood + self.floor / (rust + 1))
                posterior /= posterior.sum()
                next_success = (1 - self.floor) * posterior @ success + self.floor / (y + 1)
                next_cost = 8 + self.recipes.sum(-1) + (self.recipes > 0).sum(-1)
                feasible = valid.copy(); feasible[a] = False
                # A repeated unsuccessful deterministic experiment cannot help.
                continuation = np.where(feasible, next_cost / next_success.clip(1e-8), np.inf).min()
                q += p[a, y] * (min(float(continuation), 1000.) + 4.)
            values.append(q)
        return chosen[int(np.argmin(values))]
