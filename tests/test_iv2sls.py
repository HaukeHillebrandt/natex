"""Tests for the general k-instrument HC1 2SLS estimator (phase 5, task 1)."""

import numpy as np
import pytest

from natex.estimate.iv2sls import iv_2sls


def _endog_dgp(n, rng, tau=1.0, endog=0.6, k=1, strength=0.8, phi=0.0):
    """Endogenous-treatment DGP: u confounds T and y; Z are valid instruments
    unless phi != 0 plants the LAST instrument directly in y (exclusion violator)."""
    Z = rng.normal(size=(n, k))
    u = rng.normal(size=n)
    T = strength * Z.sum(axis=1) + u + 0.5 * rng.normal(size=n)
    y = tau * T + endog * u + 0.5 * rng.normal(size=n) + phi * Z[:, -1]
    return y, T, Z


def test_just_identified_matches_analytic_iv_ratio():
    rng = np.random.default_rng(0)
    y, T, Z = _endog_dgp(200, rng)
    est = iv_2sls(y, T, Z)
    z = Z[:, 0]
    ratio = np.cov(z, y)[0, 1] / np.cov(z, T)[0, 1]
    assert est.method == "2sls"
    assert est.tau == pytest.approx(ratio, abs=1e-10)
    assert est.j_stat is None
    assert est.j_p is None
    assert est.j_df == 0
    assert est.n_used == 200


def test_consistency_vs_biased_ols():
    # Calibrated across seeds 1..5: |tau_2sls - 1| in [0.002, 0.012]; OLS bias
    # in [0.306, 0.317] (plim 0.6/1.89 ~ 0.317). Pinned seed 1 with margin.
    rng = np.random.default_rng(1)
    y, T, Z = _endog_dgp(4000, rng, tau=1.0, endog=0.6)
    est = iv_2sls(y, T, Z)
    ols = np.linalg.lstsq(np.c_[np.ones(y.size), T], y, rcond=None)[0][1]
    assert abs(est.tau - 1.0) < 0.1
    assert ols - 1.0 > 0.2
    assert est.se > 0.0


def test_overidentified_valid_instruments_j_not_rejected():
    # Calibrated across seeds 2,12,22,32,42: j_p_valid in [0.08, 0.88];
    # planted violator j_p < 1e-116 at every seed. Pinned seed 2.
    rng = np.random.default_rng(2)
    y, T, Z = _endog_dgp(2000, rng, k=3)
    est = iv_2sls(y, T, Z)
    assert est.j_df == 2
    assert est.j_stat is not None and est.j_stat >= 0.0
    assert est.j_p > 0.01


def test_planted_exclusion_violator_drives_j_rejection():
    rng = np.random.default_rng(2)
    y, T, Z = _endog_dgp(2000, rng, k=3, phi=0.8)
    est = iv_2sls(y, T, Z)
    assert est.j_p < 0.01


def test_hc1_ci_coverage_under_heteroskedasticity():
    # sigma_i proportional to |z_i|; nominal 95% CI, loose >= 90/100 bound.
    n, covered = 800, 0
    for seed in range(100):
        rng = np.random.default_rng(1000 + seed)
        z = rng.normal(size=n)
        u = rng.normal(size=n)
        T = 0.8 * z + u + 0.5 * rng.normal(size=n)
        y = 1.0 * T + 0.6 * u + np.abs(z) * rng.normal(size=n)
        est = iv_2sls(y, T, z[:, None])
        if est.ci[0] <= 1.0 <= est.ci[1]:
            covered += 1
    assert covered >= 90  # observed at seeds 1000..1099: 98/100


def test_controls_partialling_restores_consistency():
    # w shifts both T and y and is correlated with z, so z is only a valid
    # instrument CONDITIONAL on w. plim of the no-controls IV estimate is
    # ~1.76 here (bias ~0.76); with the control, consistency is restored.
    # Calibrated seeds 3,13,23,33,43: |tau_ctrl-1| <= 0.020, no-control bias
    # in [0.750, 0.771]. Pinned seed 3.
    rng = np.random.default_rng(3)
    n = 4000
    w = rng.normal(size=n)
    z = 0.7 * w + 0.7 * rng.normal(size=n)
    T = 0.7 * z + w + 0.7 * rng.normal(size=n)
    y = 1.0 * T + 1.5 * w + 0.7 * rng.normal(size=n)
    with_ctrl = iv_2sls(y, T, z[:, None], controls=w[:, None])
    without = iv_2sls(y, T, z[:, None])
    assert abs(with_ctrl.tau - 1.0) < 0.1
    assert abs(without.tau - 1.0) > 0.2


def test_all_nan_outcome_gives_nan_never_zero():
    rng = np.random.default_rng(4)
    _, T, Z = _endog_dgp(50, rng)
    est = iv_2sls(np.full(50, np.nan), T, Z)
    assert np.isnan(est.tau) and np.isnan(est.se)
    assert np.isnan(est.ci[0]) and np.isnan(est.ci[1])
    assert est.weak_instrument is True
    assert est.n_used == 0
    assert est.extras["n_dropped"] == 50


def test_constant_instrument_is_rank_deficient_nan():
    rng = np.random.default_rng(5)
    y, T, _ = _endog_dgp(200, rng)
    est = iv_2sls(y, T, np.ones((200, 1)))
    assert np.isnan(est.tau)
    assert not est.tau == 0.0
    assert est.extras.get("rank_deficient") is True
    assert est.weak_instrument is True


def test_empty_instrument_list_gives_nan():
    rng = np.random.default_rng(6)
    y, T, _ = _endog_dgp(50, rng)
    est = iv_2sls(y, T, np.empty((50, 0)))
    assert np.isnan(est.tau)
    assert est.weak_instrument is True


def test_nonfinite_rows_dropped_and_counted():
    rng = np.random.default_rng(7)
    y, T, Z = _endog_dgp(500, rng)
    y = y.copy()
    T = T.copy()
    y[:3] = np.nan
    T[3] = np.inf
    Z[4, 0] = np.nan
    est = iv_2sls(y, T, Z)
    assert est.n_used == 495
    assert est.extras["n_dropped"] == 5
    assert np.isfinite(est.tau)


def test_first_stage_f_strong_design():
    # Calibrated seeds 8,18,28,38,48: F in [946, 1094], partial_r2 in
    # [0.325, 0.371]. Pinned seed 8.
    rng = np.random.default_rng(8)
    y, T, Z = _endog_dgp(2000, rng, strength=0.8)
    est = iv_2sls(y, T, Z)
    assert est.first_stage_F > 100.0
    assert est.weak_instrument is False
    # T = 0.8 z + u + 0.5 e: population partial R^2 = 0.64/1.89 ~ 0.34
    assert 0.2 < est.partial_r2 < 0.5


def test_issue_11_duplicated_instrument_collapses_to_just_identified():
    # [z, z] (and [z, 2z]) carry ONE instrument's information: the rank-1
    # moment space makes the just-identified moment hold exactly, so Hansen J
    # was a structurally guaranteed pass (J ~ 0, df = 1) and the first-stage
    # F was halved by the nominal k. The effective rank must drive J, F and
    # the AR set instead; tau itself is unchanged (same projection space).
    rng = np.random.default_rng(0)
    y, T, Z = _endog_dgp(300, rng)
    base = iv_2sls(y, T, Z)
    for dup_z in (np.c_[Z, Z], np.c_[Z, 2.0 * Z]):
        dup = iv_2sls(y, T, dup_z)
        assert dup.tau == pytest.approx(base.tau, rel=1e-8)
        assert dup.j_stat is None and dup.j_p is None and dup.j_df == 0
        assert dup.extras["rank_deficient"] is True
        assert dup.extras["k_effective"] == 1
        assert dup.first_stage_F == pytest.approx(base.first_stage_F, rel=1e-6)
        assert dup.ar_kind == base.ar_kind == "interval"
        assert dup.ar_ci == pytest.approx(base.ar_ci, rel=1e-8)


def test_issue_11_instrument_collinear_with_controls_is_nan_with_reason():
    # After partialling on [1, controls] the instrument block is identically
    # zero: no identifying variation at all -> NaN (never 0.0), flagged.
    rng = np.random.default_rng(1)
    n = 300
    w = rng.normal(size=n)
    u = rng.normal(size=n)
    T = 0.8 * w + u + 0.5 * rng.normal(size=n)
    y = T + 0.6 * u + 0.5 * rng.normal(size=n)
    est = iv_2sls(y, T, (2.0 * w)[:, None], controls=w[:, None])
    assert np.isnan(est.tau) and np.isnan(est.se)
    assert est.extras["reason"] == "instruments collinear with controls"
    assert est.extras["rank_deficient"] is True
    assert est.extras["k_effective"] == 0
    assert est.weak_instrument is True


def test_first_stage_f_pure_noise_instrument():
    # Instrument independent of T: F ~ chi2_1; calibrated across seeds
    # 9,19,29,39,49: F in [0.012, 1.99]. Pinned seed 9 with margin.
    rng = np.random.default_rng(9)
    n = 1000
    u = rng.normal(size=n)
    T = u + 0.5 * rng.normal(size=n)
    y = 1.0 * T + 0.6 * u + 0.5 * rng.normal(size=n)
    z = rng.normal(size=n)
    est = iv_2sls(y, T, z[:, None])
    assert est.first_stage_F < 5.0
    assert est.weak_instrument is True
