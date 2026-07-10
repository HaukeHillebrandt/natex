"""Task 3: vectorized Bernoulli LLR kernel — parity with the phase-1 loop,
boundary suprema (audit item 21), bracketed Newton (item 22), and the
homogeneous-treatment fast path in lord3_scan."""

import time

import numpy as np
import pytest
from scipy.special import expit

import natex.rdd.lord3 as lord3_mod
from natex.data.spec import Dataset, DatasetSpec
from natex.data.synthetic import make_synthetic
from natex.rdd.lord3 import lord3_scan
from natex.scan.statistics import (
    bernoulli_llr_all_splits,
    bernoulli_llr_all_splits_reference,
    fit_log_odds_offset,
    fit_log_odds_offsets,
    masked_offset_log_lik,
    offset_log_lik,
)


def _mixed_case(seed: int, k: int, m: int):
    rng = np.random.default_rng(seed)
    eta = rng.normal(size=k)
    t = rng.binomial(1, expit(eta)).astype(float)
    G = np.asarray(rng.random((k, m)) < 0.5)
    return t, eta, G


def test_vectorized_matches_reference():
    rng = np.random.default_rng(0)
    k = 24
    eta = rng.normal(size=k)
    t = rng.binomial(1, expit(eta)).astype(float)
    assert 0 < t.sum() < k  # seed sanity: both classes present
    G_rand = np.asarray(rng.random((k, 40)) < 0.5)
    # Hand-built pure-group columns: one side all-1, the other all-0 (boundary
    # suprema, audit item 21) — plus its complement.
    sharp = (t == 1.0)[:, None]
    # Near-degenerate columns: a single point on one side.
    single = np.zeros((k, 2), dtype=bool)
    single[0, 0] = True
    single[1:, 1] = True
    # Fully degenerate columns must be exactly 0.0 in both implementations.
    degenerate = np.column_stack([np.ones(k, bool), np.zeros(k, bool)])
    G = np.column_stack([G_rand, sharp, ~sharp, single, degenerate])
    new = bernoulli_llr_all_splits(t, eta, G)
    ref = bernoulli_llr_all_splits_reference(t, eta, G)
    assert new.shape == ref.shape == (G.shape[1],)
    assert np.allclose(new, ref, atol=1e-8)


def test_fit_log_odds_offsets_matches_scalar_per_column():
    t, eta, G = _mixed_case(seed=7, k=20, m=15)
    M = np.column_stack([G, (t == 1.0), (t == 0.0)])  # includes pure columns
    thetas = fit_log_odds_offsets(t, eta, M)
    lls = masked_offset_log_lik(thetas, t, eta, M)
    for j in range(M.shape[1]):
        mask = M[:, j]
        theta_j = fit_log_odds_offset(t[mask], eta[mask])
        if np.isinf(theta_j):
            assert thetas[j] == theta_j
        else:
            assert abs(thetas[j] - theta_j) < 1e-8
        assert lls[j] == pytest.approx(offset_log_lik(theta_j, t[mask], eta[mask]), abs=1e-8)


def test_masked_offset_log_lik_boundary_conventions():
    eta = np.array([-0.3, 0.2, 1.0])
    t = np.array([1.0, 0.0, 1.0])
    M = np.column_stack(
        [
            np.array([True, False, True]),  # masked t all ones
            np.array([False, True, False]),  # masked t all zeros
            np.array([True, True, False]),  # mixed
            np.zeros(3, dtype=bool),  # empty column
        ]
    )
    at_pos = masked_offset_log_lik(np.array([np.inf] * 4), t, eta, M)
    assert at_pos[0] == 0.0  # matching pure column: boundary supremum
    assert at_pos[1] == -np.inf
    assert at_pos[2] == -np.inf
    assert at_pos[3] == 0.0  # vacuous
    at_neg = masked_offset_log_lik(np.array([-np.inf] * 4), t, eta, M)
    assert at_neg[0] == -np.inf
    assert at_neg[1] == 0.0
    assert at_neg[2] == -np.inf
    assert at_neg[3] == 0.0


def test_llr_nonnegative_and_degenerate_zero():
    t, eta, G_rand = _mixed_case(seed=3, k=20, m=30)
    G = np.column_stack([G_rand, np.ones(20, bool), np.zeros(20, bool)])
    out = bernoulli_llr_all_splits(t, eta, G)
    assert np.all(out >= 0.0)
    assert out[-2] == 0.0  # all-True column: exactly zero, not merely tiny
    assert out[-1] == 0.0


def test_pure_group_boundary_supremum_finite():
    rng = np.random.default_rng(4)
    k = 30
    eta = 0.5 * rng.normal(size=k)
    g = np.r_[np.ones(15, bool), np.zeros(15, bool)]
    t = g.astype(float)  # sharp split: t == g exactly
    mixed = np.asarray(rng.random((k, 25)) < 0.5)
    # Guard: no mixed column may replicate the sharp split or its complement.
    assert not any(np.array_equal(mixed[:, j], g) or np.array_equal(mixed[:, j], ~g)
                   for j in range(mixed.shape[1]))
    out = bernoulli_llr_all_splits(t, eta, np.column_stack([g, mixed]))
    assert np.isfinite(out[0])
    assert np.all(out[0] > out[1:])  # boundary supremum beats every mixed split


def test_homogeneous_neighborhood_scores_zero():
    rng = np.random.default_rng(5)
    k = 16
    eta = rng.normal(size=k)  # mixed eta
    t = np.ones(k)
    G = np.asarray(rng.random((k, 12)) < 0.5)
    assert np.all(bernoulli_llr_all_splits(t, eta, G) == 0.0)
    assert np.all(bernoulli_llr_all_splits_reference(t, eta, G) == 0.0)


def _constant_region_dataset() -> Dataset:
    """1-D forcing variable; T identically 0 below z=0.5, mixed above."""
    import pandas as pd

    rng = np.random.default_rng(11)
    n = 120
    z = np.linspace(0.0, 1.0, n)
    t = np.where(z < 0.5, 0.0, rng.binomial(1, 0.5, size=n).astype(float))
    df = pd.DataFrame({"z": z, "t": t})
    spec = DatasetSpec(treatment="t", outcome=None, forcing=["z"], covariates=["z"])
    return Dataset(df, spec)


def test_scan_fast_path_identical_discoveries(monkeypatch):
    ds = _constant_region_dataset()
    res_fast = lord3_scan(ds, k=10)
    assert res_fast.model == "bernoulli"
    # The constant-T region must actually trigger the fast path (skipped centers).
    assert len(res_fast.discoveries) < ds.n
    monkeypatch.setattr(lord3_mod, "bernoulli_llr_all_splits", bernoulli_llr_all_splits_reference)
    res_ref = lord3_scan(ds, k=10)
    # Compare per center: exact-tie llrs may legally reorder between kernels
    # (stable sort keys differing at ~1e-16), but every center must appear in
    # both runs with the same llr and split.
    by_center_fast = {d.center_index: d for d in res_fast.discoveries}
    by_center_ref = {d.center_index: d for d in res_ref.discoveries}
    assert by_center_fast.keys() == by_center_ref.keys()
    for c, a in by_center_fast.items():
        b = by_center_ref[c]
        assert a.llr == pytest.approx(b.llr, abs=1e-8)
        assert np.array_equal(a.group1, b.group1)


def test_scan_end_to_end_parity(monkeypatch):
    ds, _ = make_synthetic(n=600, zeta=3.0, kind="binary", rng=np.random.default_rng(2))
    res_new = lord3_scan(ds, k=30)
    monkeypatch.setattr(lord3_mod, "bernoulli_llr_all_splits", bernoulli_llr_all_splits_reference)
    res_ref = lord3_scan(ds, k=30)
    top_new = [(d.center_index, d.llr) for d in res_new.top(5)]
    top_ref = [(d.center_index, d.llr) for d in res_ref.top(5)]
    assert [c for c, _ in top_new] == [c for c, _ in top_ref]
    assert np.allclose([v for _, v in top_new], [v for _, v in top_ref], atol=1e-8)


def test_perf_guard_200_neighborhoods():
    rng = np.random.default_rng(6)
    k, m = 50, 49
    cases = []
    for _ in range(200):
        eta = rng.normal(size=k)
        t = rng.binomial(1, expit(eta)).astype(float)
        G = np.asarray(rng.random((k, m)) < 0.5)
        cases.append((t, eta, G))
    start = time.perf_counter()
    for t, eta, G in cases:
        bernoulli_llr_all_splits(t, eta, G)
    assert time.perf_counter() - start < 5.0
