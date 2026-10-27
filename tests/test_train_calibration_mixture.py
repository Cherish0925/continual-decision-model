from copy import deepcopy
from dataclasses import asdict
import gzip
import json
from pathlib import Path
import unittest

import torch

from experiments.discoveryworld.calibration import CalibrationLog
from experiments.discoveryworld.calibration_mixture import AffineMixture
from experiments.discoveryworld.train_calibration_mixture import compact_sample, fit_normalization, tensors, evaluate


class MixtureTrainingTests(unittest.TestCase):
    def test_real_prefix_future_loss_and_same_compute_control(self):
        torch.set_num_threads(1)
        fixture = Path(__file__).parent / "fixtures/discoveryworld_calibration.json.gz"
        log = CalibrationLog()
        for row in json.loads(gzip.decompress(fixture.read_bytes()))["transitions"]:
            log.append_transition(row)
        samples = []
        for example in log.query_examples:
            opportunity = example.opportunity
            samples.append(compact_sample({"query": asdict(opportunity.query),
                "support": [asdict(row) for row in opportunity.support],
                "decision_sequence": opportunity.decision_sequence,
                "evaluation_only_target": asdict(example.label)}, ("Reactor Lab", "Normal", 310)))
        self.assertEqual([len(row["support_hz"]) for row in samples], [2, 3])
        model = AffineMixture(fit_normalization(samples))
        before = model.prior_mean.detach().clone()
        optimizer = torch.optim.Adam(model.parameters(), lr=.001)
        losses = []
        for row in samples:
            X, y, x, target = tensors(row, "cpu")
            losses.append(model.predictive_nll(x, target, model.fit_posterior(X, y)).mean())
        torch.stack(losses).mean().backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
        optimizer.step()
        self.assertFalse(torch.equal(model.prior_mean, before))
        compared = evaluate(model, samples)
        self.assertEqual(compared["retained"]["posterior_computations"], 2)
        self.assertEqual(compared["computed_discarded"]["posterior_computations"], 2)
        self.assertLess(compared["retained"]["metrics"]["all"]["known_support_mae_hz"],
                        compared["computed_discarded"]["metrics"]["all"]["known_support_mae_hz"])
        # A future result changes its target, never the already frozen input.
        original = samples[0]
        changed = deepcopy(original)
        changed["target_hz"] += 1000
        for first, second in zip(tensors(original, "cpu")[:3], tensors(changed, "cpu")[:3]):
            self.assertTrue(torch.equal(first, second))


if __name__ == "__main__":
    unittest.main()
