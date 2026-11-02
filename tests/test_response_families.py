import random
import torch

from experiments.continual_chemistry.active_environment import ActiveChemicalEnvironment
from experiments.continual_chemistry.response_families import TRAIN_FAMILIES, response_levels, sample_law, sample_lifetime
from experiments.continual_chemistry.recipes import primitive_recipes


def test_every_law_has_an_available_solution_and_respects_censoring():
    for n in (3, 4):
        queries = torch.tensor(primitive_recipes(n))
        for family in (*TRAIN_FAMILIES, 'interaction_inhibition'):
            law = sample_law(random.Random(23), n, (family,))
            response = response_levels(queries, law)
            assert response.min() == 0
            assert response.max() <= 3
            for rust in range(4):
                observed = torch.minimum(response, torch.tensor(rust))
                assert observed.max() <= rust
            if family == 'cosine':
                target = torch.tensor(law['target']).expand_as(queries)
                expected = ActiveChemicalEnvironment.response(queries, target, torch.full((len(queries),), 3))
                torch.testing.assert_close(response, expected)


def test_returns_preserve_the_law_and_changed_tasks_are_new_identities():
    stream = sample_lifetime(random.Random(81), TRAIN_FAMILIES, length=100)
    seen = {}
    for task in stream:
        if task['exposure'] == 'return':
            assert task['law'] == seen[task['identity']]
        else:
            assert task['identity'] not in seen
        seen[task['identity']] = task['law']
    assert {t['exposure'] for t in stream} == {'new', 'return', 'changed'}
