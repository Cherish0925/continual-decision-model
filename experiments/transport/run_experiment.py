#!/usr/bin/env python3
"""CPU numerical experiment on polynomial transport and finite-step reconstruction.

Run: python run_experiment.py --output-dir runs/transport-local
NumPy alone is required.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import platform
import sys
import time
from pathlib import Path

# One thread makes these small-matrix timing measurements less variable.
for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(key, "1")
import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"
REPORT = ROOT / "RESULTS.md"
CONFIG = {
    "dimension": 16,
    "training_seeds": [1101, 2202, 3303, 4404, 5505],
    "train_samples_per_seed": 1800,
    "finite_progress_train_samples_per_seed": 1800,
    "validation_streams": 3,
    "validation_ticks": 64,
    "test_streams_per_regime": 6,
    "test_ticks": 128,
    "diagnostic_samples_per_regime": 240,
    "train_condition_number_range": [2.0, 24.0],
    "transport_polynomial_degree": 2,
    "ridge": 1e-7,
    "validation_damping_candidates": [0.0, 0.25, 0.5, 1.0],
    "finite_progress_train_depths": [1, 2, 4, 8, 16],
    "finite_progress_unseen_depths": [3, 6, 12],
    "main_budget_dense_matrix_vector_products": 5,
    "master_validation_seed": 818181,
    "master_test_seed": 919191,
    "master_diagnostic_seed": 717171,
}
REGIMES = {
    "stationary_control": dict(drift=0.0, rotate=0.0, kappa=(4.0, 20.0)),
    "slow_change_id": dict(drift=0.08, rotate=0.015, kappa=(4.0, 20.0)),
    "fast_target_ood": dict(drift=0.8, rotate=0.015, kappa=(4.0, 20.0)),
    "curvature_ood": dict(drift=0.08, rotate=0.22, kappa=(40.0, 80.0)),
    "abrupt_joint_ood": dict(drift=0.8, rotate=0.22, kappa=(40.0, 80.0), abrupt=True),
}


def rng_for(namespace, index, seed=0):
    return np.random.default_rng(np.random.SeedSequence([namespace, index, seed]))


def spd(rng, kappa, n):
    U, _ = np.linalg.qr(rng.normal(size=(n, n)))
    lam = np.geomspace(1.0, kappa, n)
    return (U * lam) @ U.T, U, lam


def context(kappa, depth=None):
    # Bounds fixed from training specification, never estimated on test.
    u = np.clip(np.log(kappa / 2.0) / np.log(12.0), 0.0, 1.0)
    if depth is None:
        return np.array([1.0, u, u * u])
    v = np.clip(np.log(depth) / np.log(16.0), 0.0, 1.0)
    return np.array([1.0, u, u * u, v, v * v, u * v])


def basis_vectors(H, q, L):
    v0 = q / L
    v1 = v0 - H @ v0 / L
    v2 = v1 - H @ v1 / L
    return np.column_stack((v0, v1, v2))


def predict_transport(H, q, L, W, depth=None):
    return basis_vectors(H, q, L) @ (W @ context(L, depth))


def fit_model(seed, finite=False):
    ns = 302 if finite else 301
    rng = rng_for(ns, seed)
    n = CONFIG["dimension"]
    samples = CONFIG["finite_progress_train_samples_per_seed" if finite else "train_samples_per_seed"]
    features, targets = [], []
    for _ in range(samples):
        kappa = np.exp(rng.uniform(np.log(2), np.log(24)))
        H, U, lam = spd(rng, kappa, n)
        if finite:
            q = rng.normal(size=n)
            depth = int(rng.choice(CONFIG["finite_progress_train_depths"]))
            gain = (1.0 - (1.0 - lam / kappa) ** depth) / lam
            target = U @ (gain * (U.T @ q))
        else:
            # Paired context changes, evaluated at an unfinished prior iterate.
            x0 = rng.normal(size=n)
            H0, _, _ = spd(rng, np.exp(rng.uniform(np.log(2), np.log(24))), n)
            H0 = 0.97 * H + 0.03 * H0
            z0 = x0 + 0.2 * rng.normal(size=n)
            x1 = x0 + rng.uniform(0.03, 0.2) * rng.normal(size=n)
            q = H @ x1 - H0 @ x0 - (H - H0) @ z0
            target = U @ ((U.T @ q) / lam)
            depth = None
        # Relative vector fit: prevents large targets alone dominating learning.
        scale = max(np.linalg.norm(target), 1e-10)
        B = basis_vectors(H, q, kappa)
        phi = context(kappa, depth)
        features.append(np.einsum("ij,k->ijk", B, phi).reshape(n, -1) / scale)
        targets.append(target / scale)
    A, y = np.concatenate(features), np.concatenate(targets)
    gram = A.T @ A
    reg = CONFIG["ridge"] * np.trace(gram) / len(gram)
    coeff = np.linalg.solve(gram + reg * np.eye(len(gram)), A.T @ y)
    W = coeff.reshape(3, -1)
    return W, float(np.mean((A @ coeff - y) ** 2))


def stream(namespace, index, settings, ticks):
    rng = rng_for(namespace, index)
    n = CONFIG["dimension"]
    lo, hi = settings["kappa"]
    kappa0 = np.exp(rng.uniform(np.log(lo), np.log(hi)))
    _, U, _ = spd(rng, kappa0, n)
    x = rng.normal(size=n)
    Hs, bs, xs, Ls = [], [], [], []
    for t in range(ticks + 1):
        if t:
            if settings.get("abrupt"):
                if t % 12 == 0:
                    x = rng.normal(size=n)
                if t % 20 == 0:
                    _, U, _ = spd(rng, kappa0, n)
            else:
                drift = settings["drift"]
                x = np.sqrt(1.0 - min(drift, 0.999) ** 2) * x + drift * rng.normal(size=n)
                i, j = rng.choice(n, 2, replace=False)
                angle = settings["rotate"] * rng.normal()
                c, s = np.cos(angle), np.sin(angle)
                ui, uj = U[:, i].copy(), U[:, j].copy()
                U[:, i], U[:, j] = c * ui + s * uj, -s * ui + c * uj
        amplitude = 0.0 if settings["drift"] == 0 else 0.12
        L = np.clip(kappa0 * np.exp(amplitude * np.sin(0.07 * t)), lo, hi)
        lam = np.geomspace(1.0, L, n)
        H = (U * lam) @ U.T
        Hs.append(H)
        bs.append(H @ x)
        xs.append(x.copy())
        Ls.append(L)
    return tuple(np.asarray(v) for v in (Hs, bs, xs, Ls))


def gd(H, b, z, L, steps):
    for _ in range(steps):
        z = z + (b - H @ z) / L
    return z


def cg(H, b, z, iterations=4, jacobi=False):
    # One H @ z plus one H @ p per iteration: at most five dense matvecs.
    r = b - H @ z
    invdiag = 1.0 / np.diag(H) if jacobi else np.ones_like(r)
    u = invdiag * r
    p = u.copy()
    ru = r @ u
    for _ in range(iterations):
        if ru < 1e-28:
            break
        Hp = H @ p
        alpha = ru / (p @ Hp)
        z = z + alpha * p
        r = r - alpha * Hp
        unew = invdiag * r
        runew = r @ unew
        p = unew + (runew / ru) * p
        u, ru = unew, runew
    return z


METHODS = [
    "stale_3_ticks_gd5", "restart_gd5", "warm_gd1", "warm_gd5", "warm_gd_latency",
    "warm_nesterov5", "warm_cg4", "warm_jacobi_cg4", "learned_transport",
    "learned_current_residual", "readout_only_transport",
    "scalar_transport", "analytic_previous_inverse", "analytic_current_inverse",
    "exact_current_solution",
]


def run_method(data, method, W=None, damping=1.0, return_trace=False, latency_steps=5):
    Hs, bs, xs, Ls = data
    n = Hs.shape[-1]
    z = np.zeros(n)
    # Initial residual can be formed without a matvec because z == 0.
    cached_r = bs[0].copy()
    trace = []
    t0 = time.perf_counter_ns()
    for t in range(1, len(Hs)):
        H, b, L = Hs[t], bs[t], Ls[t]
        carried_z = None
        if method == "stale_3_ticks_gd5":
            j = max(0, t - 3)
            z = gd(Hs[j], bs[j], z, Ls[j], 5)
        elif method == "restart_gd5":
            z = gd(H, b, np.zeros(n), L, 5)
        elif method.startswith("warm_gd"):
            steps = latency_steps if method == "warm_gd_latency" else int(method.removeprefix("warm_gd"))
            z = gd(H, b, z, L, steps)
        elif method == "warm_nesterov5":
            y = z.copy()
            beta = (np.sqrt(L) - 1) / (np.sqrt(L) + 1)
            for _ in range(5):
                newz = y + (b - H @ y) / L
                y = newz + beta * (newz - z)
                z = newz
        elif method in ("warm_cg4", "warm_jacobi_cg4"):
            z = cg(H, b, z, jacobi=(method == "warm_jacobi_cg4"))
        elif method in ("learned_transport", "learned_current_residual", "readout_only_transport", "scalar_transport", "analytic_previous_inverse", "analytic_current_inverse"):
            r = b - H @ z
            q = r - cached_r
            if method == "learned_current_residual":
                # Same learned coefficients and work, but correct all current residual.
                q = r
            if method in ("learned_transport", "learned_current_residual", "readout_only_transport"):
                displacement = damping * predict_transport(H, q, L, W)
            elif method == "scalar_transport":
                displacement = q / L
            elif method == "analytic_previous_inverse":
                displacement = np.linalg.solve(Hs[t - 1], q)
            else:
                displacement = np.linalg.solve(H, q)
            zbar = z + displacement
            rbar = b - H @ zbar
            if method == "readout_only_transport":
                # Emit the transported decision, but persist only a one-step GD state.
                carried_z = z + r / L
                cached_r = r - H @ r / L
            z = zbar + rbar / L
            # Fifth learned-method matvec: maintain the old-world residual at output z.
            if carried_z is None:
                cached_r = rbar - H @ rbar / L
        elif method == "exact_current_solution":
            z = np.linalg.solve(H, b)
        else:
            raise ValueError(method)
        trace.append(z.copy())
        if carried_z is not None:
            z = carried_z
    elapsed = (time.perf_counter_ns() - t0) / 1e9
    Z = np.asarray(trace)
    delta = Z - xs[1:]
    gaps = 0.5 * np.einsum("ti,tij,tj->t", delta, Hs[1:], delta)
    zero_gaps = 0.5 * np.einsum("ti,tij,tj->t", xs[1:], Hs[1:], xs[1:])
    residual = bs[1:] - np.einsum("tij,tj->ti", Hs[1:], Z)
    values = {
        "normalized_gap": float(gaps.sum() / zero_gaps.sum()),
        "mean_gap": float(gaps.mean()),
        "relative_residual_rms": float(np.sqrt((residual ** 2).sum() / (bs[1:] ** 2).sum())),
        "cpu_seconds": elapsed,
        "microseconds_per_tick": 1e6 * elapsed / len(Z),
        "max_state_norm": float(np.linalg.norm(Z, axis=1).max()),
    }
    if not np.isfinite(list(values.values())).all():
        raise FloatingPointError(f"Nonfinite result: {method}")
    return (values, Z, gaps) if return_trace else values


def write_csv(path, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    if path.name == "finite_progress_raw.csv":
        path.with_suffix(".csv.gz").write_bytes(gzip.compress(path.read_bytes(), mtime=0))
        path.unlink()


def calibrate_latency(validation, W, damping, seed):
    # Freeze the GD budget using validation timing alone. No test scores enter.
    rows = []
    candidate_times = [run_method(validation[0], "learned_transport", W, damping)["microseconds_per_tick"] for _ in range(9)]
    target = float(np.median(candidate_times))
    for steps in range(1, 17):
        measured = [run_method(validation[0], f"warm_gd{steps}")["microseconds_per_tick"] for _ in range(5)]
        rows.append(dict(training_seed=seed, gd_steps=steps, gd_cpu_us_median=float(np.median(measured)),
                         candidate_cpu_us_median=target))
    eligible = [r for r in rows if r["gd_cpu_us_median"] <= target]
    selected = max(eligible, key=lambda r: r["gd_steps"])["gd_steps"] if eligible else 1
    return selected, rows


def diagnostic(W_inverse, W_progress, seed):
    rows = []
    regimes = {
        "seen_depth_id": ([1, 2, 4, 8, 16], (2, 24)),
        "unseen_depth_id": ([3, 6, 12], (2, 24)),
        "unseen_depth_curvature_ood": ([3, 6, 12], (40, 80)),
    }
    n = CONFIG["dimension"]
    for ridx, (regime, (depths, bounds)) in enumerate(regimes.items()):
        rng = rng_for(CONFIG["master_diagnostic_seed"], ridx)
        for i in range(CONFIG["diagnostic_samples_per_regime"]):
            L = np.exp(rng.uniform(np.log(bounds[0]), np.log(bounds[1])))
            H, U, lam = spd(rng, L, n)
            b0, db = rng.normal(size=(2, n))
            k = int(rng.choice(depths))
            zold = gd(H, b0, np.zeros(n), L, k)
            target = gd(H, b0 + db, np.zeros(n), L, k)
            exact_delta = U @ (((1 - (1 - lam / L) ** k) / lam) * (U.T @ db))
            assert np.allclose(target - zold, exact_delta, atol=1e-11)
            norm = max(np.linalg.norm(exact_delta), 1e-12)
            variants = (
                "unchanged_state", "warm_gd3", "learned_inverse_transport",
                "learned_finite_progress_transport", "exact_equilibrium_transport",
                "exact_finite_progress_tangent",
            )
            for method in variants:
                start = time.perf_counter_ns()
                if method == "unchanged_state":
                    estimate = zold
                elif method == "warm_gd3":
                    estimate = gd(H, b0 + db, zold, L, 3)
                elif method == "learned_inverse_transport":
                    estimate = zold + predict_transport(H, db, L, W_inverse)
                elif method == "learned_finite_progress_transport":
                    estimate = zold + predict_transport(H, db, L, W_progress, k)
                elif method == "exact_equilibrium_transport":
                    estimate = zold + np.linalg.solve(H, db)
                else:
                    # Actual tangent computation, charged k matvecs, not free oracle lookup.
                    estimate = zold + gd(H, db, np.zeros(n), L, k)
                elapsed = (time.perf_counter_ns() - start) / 1e3
                # Record objective too: closeness to a finite iterate is not task superiority.
                xstar = np.linalg.solve(H, b0 + db)
                delta = estimate - xstar
                target_delta = target - xstar
                gap = 0.5 * delta @ H @ delta
                target_gap = 0.5 * target_delta @ H @ target_delta
                rows.append(dict(training_seed=seed, regime=regime, sample=i, depth=k,
                                 method=method, kappa=L,
                                 relative_finite_state_error=float(np.linalg.norm(estimate - target) / norm),
                                 objective_gap=float(gap), target_objective_gap=float(target_gap),
                                 cpu_microseconds=elapsed))
    return rows


def sanity(W):
    data = stream(9888, 0, REGIMES["slow_change_id"], 30)
    Hs, bs, xs, Ls = data
    assert np.all(np.linalg.eigvalsh(Hs) > 0.999999)
    assert np.allclose(np.einsum("tij,tj->ti", Hs, xs), bs)
    # Prefix invariance: future observations must have no effect on earlier outputs.
    _, full, gaps = run_method(data, "learned_transport", W, return_trace=True)
    prefix = tuple(x[:16] for x in data)
    _, short, _ = run_method(prefix, "learned_transport", W, return_trace=True)
    assert np.array_equal(full[:15], short)
    H, b, x, z = Hs[1], bs[1], xs[1], full[0]
    f = lambda v: 0.5 * v @ H @ v - b @ v
    assert np.isclose(f(z) - f(x), gaps[0], atol=1e-10)
    assert run_method(data, "exact_current_solution")["normalized_gap"] < 1e-25
    # No innovation gives exactly zero learned transport.
    assert np.array_equal(predict_transport(H, np.zeros_like(b), Ls[1], W), np.zeros_like(b))
    # Fixed H exact transport preserves the same residual (finite-state distinction).
    db = np.ones_like(b) * 0.1
    transported = z + np.linalg.solve(H, db)
    assert np.allclose(b - H @ z, b + db - H @ transported)


def summarize(main_rows, diag_rows, models):
    def average(rows, field):
        vals = np.array([r[field] for r in rows])
        return float(vals.mean()), float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
    aggregated = []
    for regime in REGIMES:
        for method in METHODS:
            subset = [r for r in main_rows if r["regime"] == regime and r["method"] == method]
            mean, sd = average(subset, "normalized_gap")
            times = [r["microseconds_per_tick"] for r in subset]
            aggregated.append(dict(regime=regime, method=method, normalized_gap_mean=mean,
                                   across_stream_seed_sd=sd, cpu_us_median=float(np.median(times))))
    write_csv(OUT / "main_summary.csv", aggregated)
    diag_summary = []
    for regime in sorted(set(r["regime"] for r in diag_rows)):
        for method in sorted(set(r["method"] for r in diag_rows)):
            subset = [r for r in diag_rows if r["regime"] == regime and r["method"] == method]
            err, sd = average(subset, "relative_finite_state_error")
            gap, _ = average(subset, "objective_gap")
            diag_summary.append(dict(regime=regime, method=method, relative_finite_state_error_mean=err,
                                     relative_finite_state_error_sd=sd, mean_objective_gap=gap))
    write_csv(OUT / "finite_progress_summary.csv", diag_summary)

    # Paired stream comparisons: average model seeds first, bootstrap independent streams.
    paired = []
    bootstrap_rng = np.random.default_rng(666001)
    for regime in REGIMES:
        for baseline in ("warm_gd5", "warm_nesterov5", "warm_cg4", "warm_jacobi_cg4"):
            ratios, seed_ratios = [], []
            for stream_id in range(CONFIG["test_streams_per_regime"]):
                cr = [r["normalized_gap"] for r in main_rows if r["regime"] == regime and r["stream"] == stream_id and r["method"] == "learned_transport"]
                br = [r["normalized_gap"] for r in main_rows if r["regime"] == regime and r["stream"] == stream_id and r["method"] == baseline]
                ratios.append(np.mean(cr) / np.mean(br))
            for seed in CONFIG["training_seeds"]:
                cr = [r["normalized_gap"] for r in main_rows if r["regime"] == regime and r["training_seed"] == seed and r["method"] == "learned_transport"]
                br = [r["normalized_gap"] for r in main_rows if r["regime"] == regime and r["training_seed"] == seed and r["method"] == baseline]
                seed_ratios.append(np.mean(cr) / np.mean(br))
            samples = np.asarray(ratios)[bootstrap_rng.integers(0, len(ratios), (5000, len(ratios)))].mean(axis=1)
            paired.append(dict(regime=regime, baseline=baseline,
                               mean_paired_gap_ratio=float(np.mean(ratios)),
                               stream_bootstrap_ci_low=float(np.quantile(samples, 0.025)),
                               stream_bootstrap_ci_high=float(np.quantile(samples, 0.975)),
                               ratio_across_seed_min=float(min(seed_ratios)),
                               ratio_across_seed_max=float(max(seed_ratios))))
    write_csv(OUT / "paired_comparisons.csv", paired)

    target_gap = float(np.mean([
        r["target_objective_gap"] for r in diag_rows
        if r["regime"] == "unseen_depth_id"
        and r["method"] == "exact_finite_progress_tangent"
    ]))
    objective_lookup = {
        r["method"]: r["mean_objective_gap"] for r in diag_summary
        if r["regime"] == "unseen_depth_id"
    }
    lines = [
        "# Numerical transport experiment", "",
        "A five-seed CPU experiment on changing 16-dimensional quadratic objectives. "
        "The fitted polynomial correction underperforms the stronger solvers tested. "
        "A separate finite-step reconstruction task explains why copying unfinished "
        "computation can preserve its optimization error.", "",
        "## Problem and fitted models", "",
        "For f_t(z) = 1/2 z^T H_t z − b_t^T z, emit a decision vector at each simulated tick. "
        "H_t is symmetric positive definite. The correction uses the context innovation "
        "q_t = (b_t − H_t z) − r_previous, with the cached residual evaluated at the same prior output. "
        "Apply a learned degree-two inverse polynomial to q_t, then one gradient correction. "
        "All methods receive current H, b, and the same spectral bounds.", "",
        "The inverse model has nine fitted parameters. A separate 18-parameter model also "
        "conditions on interruption depth and learns the change in a same-depth gradient iterate. "
        "Both use polynomial approximation and prediction-correction methods.", "",
        "## Training and evaluation", "",
        f"Five independent training seeds; {CONFIG['train_samples_per_seed']} paired samples "
        f"per inverse model and {CONFIG['finite_progress_train_samples_per_seed']} per finite-step model. "
        "Damping is selected on three validation streams. Six independent test streams per regime "
        "have 128 ticks each. Training, validation, main tests, and reconstruction tests use "
        "separate random-seed namespaces. Parameters are frozen before test.", "",
        "The main score is cumulative excess quadratic objective divided by the cumulative "
        "excess objective of always emitting zero; lower is better. Exact solutions supply labels "
        "and scores and are separately reported as numerical reference methods.", "",
        "Selected damping by seed: " + ", ".join(
            f"{m['seed']}: {m['damping']}" for m in models) + ".", "",
        "Validation-calibrated gradient iterations at approximately the candidate CPU latency: "
        + ", ".join(f"{m['seed']}: {m['latency_gd_steps']}" for m in models)
        + ". This iteration budget depends on machine timing.", "",
        "## Main results", "",
        "| Regime | Learned transport | Current-residual control | Readout-only control | Warm GD (5) | Latency GD | Nesterov (5) | CG (4) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for regime in REGIMES:
        values = [next(r["normalized_gap_mean"] for r in aggregated
                       if r["regime"] == regime and r["method"] == method)
                  for method in ["learned_transport", "learned_current_residual",
                                 "readout_only_transport", "warm_gd5", "warm_gd_latency",
                                 "warm_nesterov5", "warm_cg4"]]
        lines.append("| " + regime + " | " + " | ".join(f"{v:.6g}" for v in values) + " |")
    lines += [
        "", "The correction loses to Nesterov, CG, the same-weights current-residual control, "
        "and validation-latency-calibrated GD in all five regimes. It beats ordinary GD5 only "
        "under fast target changes. Persisting the correction helps relative to readout-only "
        "correction during changes, but does not overcome the stronger methods.", "",
        "Ratios below are means of per-stream ratios after averaging training seeds. "
        "They differ from ratios of the aggregate means above. Bootstrap intervals resample "
        "the six independent test streams; repeated deterministic baseline seeds are not "
        "independent replications.", "",
        "| Regime | Learned / CG gap | Stream-bootstrap 95% interval |",
        "|---|---:|---:|",
    ]
    for row in paired:
        if row["baseline"] == "warm_cg4":
            lines.append(
                f"| {row['regime']} | {row['mean_paired_gap_ratio']:.3g} | "
                f"[{row['stream_bootstrap_ci_low']:.3g}, {row['stream_bootstrap_ci_high']:.3g}] |")
    lines += [
        "", "## Finite-step state reconstruction", "",
        "For fixed H and target change Δb, the change in the k-step gradient trajectory is "
        "(I − (I − H/L)^k) H^{-1} Δb. This differs from equilibrium displacement H^{-1} Δb. "
        "The depth-conditioned model trains on k = 1, 2, 4, 8, 16 and tests on unseen "
        "k = 3, 6, 12. Relative state error divides reconstruction distance by the exact "
        "state displacement.", "",
        "| Test distribution | Learned finite-step error | Learned inverse error | Exact inverse error | Exact finite tangent error |",
        "|---|---:|---:|---:|---:|",
    ]
    for regime in sorted(set(r["regime"] for r in diag_summary)):
        values = [next(r["relative_finite_state_error_mean"] for r in diag_summary
                       if r["regime"] == regime and r["method"] == method)
                  for method in ["learned_finite_progress_transport", "learned_inverse_transport",
                                 "exact_equilibrium_transport", "exact_finite_progress_tangent"]]
        lines.append("| " + regime + " | " + " | ".join(f"{v:.6g}" for v in values) + " |")
    lines += [
        "", f"On unseen in-distribution depths, the same-depth reference itself has mean "
        f"objective gap **{target_gap:.6g}**. Exact reconstruction has zero state error "
        "while retaining that gap. The learned finite-step update has mean gap "
        f"{objective_lookup['learned_finite_progress_transport']:.6g}, compared with "
        f"{objective_lookup['learned_inverse_transport']:.6g} for the learned inverse and "
        f"{objective_lookup['exact_equilibrium_transport']:.6g} for exact equilibrium transport.", "",
        "The different rankings follow from the target: reproducing an unfinished iterate "
        "and improving its objective are different tasks. This experiment corrects the "
        "training choice. Neural training "
        "will optimize subsequent task outcomes, with state reconstruction as an auxiliary comparison. "
        "The exact finite tangent uses k gradient steps and is reported as a variable-work reference.", "",
        "## Computation and timing", "",
        "The candidate uses five H-vector products per tick: current residual (1), polynomial "
        "basis (2), corrected-state residual (1), and output residual cache (1). GD5 and "
        "Nesterov5 use five; CG4 uses an initial residual and at most four search-direction products. "
        "Coefficient generation and vector operations add overhead. Equal product counts "
        "therefore require a separate latency comparison.", "",
        "The current-residual control applies the same fitted polynomial to the whole residual. "
        "The readout-only control emits the correction but retains its uncorrected one-step "
        "state. Both use five products. Exact-solve references have a different cost.", "",
        "CPU timings include each method loop and trace copies; data generation, offline training, "
        "and scoring are separate. The simulator advances once per tick, independently of CPU "
        "microseconds. The stale baseline uses observations three ticks old by construction.", "",
        "| Method | Median CPU microseconds / tick |", "|---|---:|",
    ]
    for method in METHODS:
        times = [r["microseconds_per_tick"] for r in main_rows if r["method"] == method]
        lines.append(f"| {method} | {np.median(times):.3f} |")
    lines += [
        "", "## Interpretation and next experiment", "",
        "In stationary conditions, context innovation is zero and the candidate spends its "
        "five-product budget on only one corrective step. Under changes, its retained update "
        "still loses to the stronger solver controls. This polynomial construction is not "
        "selected as the neural CDM core.", "",
        "The problems are small, fully observed quadratics with supplied spectral bounds. "
        "Streams are exogenous and the stored iterate equals the action vector. The next "
        "experiment uses a distinct learned latent workspace, nonlinear recurrence, "
        "action-dependent observations, and measured inference time.", "",
        "Current-residual, readout-only, and latency controls were added after inspecting "
        "the initial result; candidate parameters, hyperparameters, and test streams remained "
        "unchanged. The finite-step test uses a fixed zero seed. Five trained models and "
        "six test streams per regime define the replication structure.", "",
        "## Files", "",
        "The results directory contains main_raw.csv, main_summary.csv, paired_comparisons.csv, "
        "finite_progress_raw.csv.gz, finite_progress_summary.csv, validation.csv, "
        "latency_calibration.csv, frozen_models_seed_*.npz, metadata.json, and config.json. "
        "The implementation checks causal prefixes, residual/objective identities, "
        "zero innovation, finite-step sensitivities, and exact solutions.", "",
    ]
    REPORT.write_text("\n".join(lines))


def main(argv=None):
    global OUT, REPORT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/transport"),
                        help="Run directory containing RESULTS.md and results/")
    args = parser.parse_args(argv)
    output_root = args.output_dir.resolve()
    OUT = output_root / "results"
    REPORT = output_root / "RESULTS.md"
    started = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "config.json").write_text(json.dumps(dict(CONFIG, regimes=REGIMES), indent=2))
    validation = [stream(CONFIG["master_validation_seed"], i, REGIMES["slow_change_id"], CONFIG["validation_ticks"]) for i in range(CONFIG["validation_streams"])]
    tests = {(regime, i): stream(CONFIG["master_test_seed"] + 100 * ridx, i, settings, CONFIG["test_ticks"])
             for ridx, (regime, settings) in enumerate(REGIMES.items())
             for i in range(CONFIG["test_streams_per_regime"])}
    models, validation_rows, main_rows, diag_rows, latency_rows = [], [], [], [], []
    for seed in CONFIG["training_seeds"]:
        W, train_mse = fit_model(seed)
        Wfinite, finite_mse = fit_model(seed, finite=True)
        sanity(W)
        candidates = []
        for damping in CONFIG["validation_damping_candidates"]:
            scores = [run_method(d, "learned_transport", W, damping)["normalized_gap"] for d in validation]
            score = float(np.mean(scores))
            validation_rows.append(dict(training_seed=seed, damping=damping, mean_normalized_gap=score))
            candidates.append((score, damping))
        _, damping = min(candidates)
        latency_steps, measurements = calibrate_latency(validation, W, damping, seed)
        latency_rows.extend(measurements)
        np.savez(OUT / f"frozen_models_seed_{seed}.npz", inverse_weights=W, finite_progress_weights=Wfinite,
                 damping=damping, latency_gd_steps=latency_steps, train_mse=train_mse, finite_progress_train_mse=finite_mse)
        models.append(dict(seed=seed, damping=damping, latency_gd_steps=latency_steps, train_mse=train_mse, finite_progress_train_mse=finite_mse))
        print(f"seed={seed} damping={damping} inverse_mse={train_mse:.6g} finite_mse={finite_mse:.6g}", flush=True)
        # Deterministic shuffled execution order reduces systematic timing-order bias.
        order_rng = rng_for(5858, seed)
        for (regime, stream_id), data in tests.items():
            for method in order_rng.permutation(METHODS):
                values = run_method(data, method, W, damping, latency_steps=latency_steps)
                main_rows.append(dict(training_seed=seed, regime=regime, stream=stream_id, method=method, **values))
        diag_rows.extend(diagnostic(W, Wfinite, seed))
    write_csv(OUT / "main_raw.csv", main_rows)
    write_csv(OUT / "finite_progress_raw.csv", diag_rows)
    write_csv(OUT / "validation.csv", validation_rows)
    write_csv(OUT / "latency_calibration.csv", latency_rows)
    summarize(main_rows, diag_rows, models)
    metadata = dict(python=sys.version, numpy=np.__version__, platform=platform.platform(),
                    processor=platform.processor(), total_wall_seconds=time.perf_counter() - started,
                    thread_environment={k: os.environ.get(k) for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")},
                    main_rows=len(main_rows), diagnostic_rows=len(diag_rows), model_runs=models,
                    sanity_assertions="passed", hardware_scope="CPU NumPy only")
    (OUT / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
