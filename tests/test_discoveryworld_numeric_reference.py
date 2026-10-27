"""Numerical reference boundaries using the recorded public calibration prefix."""
from copy import deepcopy
from dataclasses import asdict
import gzip
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from experiments.discoveryworld.calibration import CalibrationLog
from experiments.discoveryworld.numeric_reference import (
    QUANTITIES, affine_fit, evaluate_samples, extract, quantity_vector,
)


def samples():
    fixture = Path(__file__).parent / 'fixtures/discoveryworld_calibration.json.gz'
    log = CalibrationLog()
    for row in json.loads(gzip.decompress(fixture.read_bytes()))['transitions']:
        log.append_transition(row)
    return [{'seed': 310, 'source': str(fixture),
             'decision_sequence': e.opportunity.decision_sequence,
             'support_count': len(e.opportunity.support),
             'query': asdict(e.opportunity.query),
             'support': [asdict(s) for s in e.opportunity.support],
             'evaluation_only_target': asdict(e.label)} for e in log.query_examples]


class NumericReferenceTests(unittest.TestCase):
    def test_actual_query_keeps_condition_boundary_and_units(self):
        query = samples()[0]['query']
        old = quantity_vector(query)
        newer = deepcopy(query['features']['radiation_usv_h'][0])
        newer.update(mean=newer['mean'] + 100., last_sequence=newer['last_sequence'] + 1)
        newer['conditions']['agent_location'] = {'x': 99, 'y': 99}
        query['features']['radiation_usv_h'] = [*query['features']['radiation_usv_h'], newer]
        new = quantity_vector(query)
        np.testing.assert_array_equal(new-old, [0., 0., 100., 0., 0., 0., 0., 0., 0.])
        self.assertEqual(len(query['features']['radiation_usv_h']), 2)
        self.assertEqual(newer['unit'], 'uSv/h')
        query['features']['density_g_cm3'] = []
        with self.assertRaisesRegex(ValueError, 'must be observed'):
            quantity_vector(query)

    def test_two_anchor_interpolation_is_not_identification(self):
        pred, fits, mse = affine_fit([[0., 0.], [1., 1.]], [0., 10.], [2., 7.])
        np.testing.assert_allclose(fits, [[0., 0.], [10., 10.]])
        np.testing.assert_allclose(mse, [0., 0.])
        np.testing.assert_allclose(pred, [20., 70.])
        _, _, third_mse = affine_fit([[0., 0.], [1., 1.], [2., 7.]], [0., 10., 20.], [3., 2.])
        self.assertAlmostEqual(third_mse[0], 0.)
        self.assertGreater(third_mse[1], 1.)

    def test_entire_held_instance_labels_cannot_select_its_hyperparameters(self):
        raw = samples()
        # Other folds are controlled counterfactual copies, not model results.
        for seed, offset in [(311, 500.), (312, -700.)]:
            other = deepcopy(raw[:2])
            for row in other:
                row['seed'] = seed
                row['evaluation_only_target']['frequency_hz'] += offset
                for support in row['support']:
                    support['label']['frequency_hz'] += offset
            raw.extend(other)
        before = [r for r in evaluate_samples(raw) if r['seed'] == 310]
        changed = deepcopy(raw)
        for row in changed:
            if row['seed'] == 310:
                row['evaluation_only_target']['frequency_hz'] += 90000.
        after = [r for r in evaluate_samples(changed) if r['seed'] == 310]
        self.assertEqual(len(before), 28)
        for first, second in zip(before, after):
            self.assertEqual(first['prediction_hz'], second['prediction_hz'])
            self.assertEqual(first['settings'], second['settings'])
            self.assertEqual(first['support_predictions_hz'], second['support_predictions_hz'])
            self.assertNotEqual(first['target_hz'], second['target_hz'])

    def test_nontraining_source_is_rejected_before_reading_events(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'run.json').write_text(json.dumps({'split': 'heldout_rule'}))
            with self.assertRaisesRegex(ValueError, 'training trajectories only'):
                extract(root/'trajectory.jsonl')


if __name__ == '__main__':
    unittest.main()
