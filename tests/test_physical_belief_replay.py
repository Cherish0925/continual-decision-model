import json

import torch

from experiments.physical_belief.replay_failures import PublicFailureMask, replay


def bootstrap():
    return {"public_object_order": [{"slot": i, "public_handle": i} for i in range(4)],
            "known_support_slots": [0, 1]}


def test_failure_mask_reopens_only_changed_relevant_evidence():
    ledger = PublicFailureMask(bootstrap())

    def measure(obj, value):
        ledger.receive({"action_index": obj * 5, "query_index": 2, "success": False,
                        "result": {"measurements": [{"kind": "instrument", "target_handle": obj,
                                     "quantity": "density_g_cm3", "value": value}]}})

    def fail():
        ledger.receive({"action_index": 20, "query_index": 2, "success": False, "result": {}})

    measure(0, 1.)
    fail()
    measure(3, 2.)  # The as-yet unknown next target cannot revise this prediction.
    assert bool(ledger.failed[0])
    measure(0, 1.)  # Repeating the same support value leaves the fit unchanged.
    assert bool(ledger.failed[0])
    measure(0, 2.)
    assert not bool(ledger.failed[0])
    fail()
    ledger.receive({"action_index": 21, "query_index": 2, "success": True, "result": {}})
    assert not bool(ledger.failed.any())
    assert bool(ledger.known[2])


def test_future_success_is_scoring_only_and_unobserved_failures_are_excluded(tmp_path):
    location = tmp_path / "stream" / "episode000" / "summary.json"
    location.parent.mkdir(parents=True)
    request = {"kind": "test", "target_handle": 2, "reactor_handle": 12}
    prediction = {"means": [50. + 50. * q for q in range(9)], "std": [20.] * 9,
                  "weights": [1. / 9] * 9}
    failure = {"action_index": 20, "query_index": 2, "success": False, "decision": 0,
               "physical_prediction": prediction,
               "result": {"request": request, "ending": "local_experiment_exhausted",
                          "attempts": [{"frequency_hz": 50., "activated": False}]}}
    success = {"action_index": 21, "query_index": 2, "success": True, "decision": 1,
               "result": {"request": request,
                          "attempts": [{"frequency_hz": 100., "activated": True}]}}
    episode = {"stream": "stream", "episode_index": 0, "seed": 1,
               "bootstrap": bootstrap(), "decisions": [failure, success]}

    def run():
        location.write_text(json.dumps(episode))
        return replay(tmp_path, 2.)

    first = run()
    success["result"]["attempts"][0]["frequency_hz"] = 200.
    second = run()
    for a, b in zip(first["records"], second["records"]):
        assert a["posterior_weights"] == b["posterior_weights"]
        assert a["posterior_means_hz"] == b["posterior_means_hz"]
    assert first["methods"]["unchanged"]["mean_future_success_interval_nll"] != second["methods"]["unchanged"]["mean_future_success_interval_nll"]

    episode["decisions"].pop()
    without_future = run()
    assert without_future["failed_boundaries"] == 1
    assert without_future["methods"]["exclude_observed_history"]["scored_boundaries"] == 0
    assert all(r["future_success_interval_log_probability"] is None for r in without_future["records"])

    failure["result"]["ending"] = "native_budget_exhausted"
    unbound = run()
    assert unbound["failed_boundaries"] == 0
    assert unbound["skipped"]["not_complete_setting_failure"] == 1
