"""Tests for closed-form Anderson-Rubin/Fieller confidence sets (phase 5, task 2).

Calibration evidence for stochastic assertions is recorded inline per test
(>= 5 seeds run during implementation; one seed pinned with margin).
"""

import numpy as np
import pytest
from scipy import stats

from natex.data.synthetic import make_synthetic
from natex.estimate.iv2sls import ar_confidence_set, iv_2sls
from natex.estimate.local2sls import local_2sls, wald_estimate
from natex.rdd.lord3 import lord3_scan


def _endog_dgp(n, rng, tau=1.0, endog=0.6, strength=0.8):
    """Homoskedastic endogenous-treatment DGP with a single instrument."""
    z = rng.normal(size=n)
    v = rng.normal(size=n)
    e = endog * v + np.sqrt(1.0 - endog**2) * rng.normal(size=n)
    T = strength * z + v
    y = tau * T + e
    return y, T, z


def _homoskedastic_first_stage_f(T, z):
    """Classical (homoskedastic) first-stage F of z after an intercept, k=1."""
    zt = z - z.mean()
    tt = T - T.mean()
    n = T.size
    tpt = (zt @ tt) ** 2 / (zt @ zt)
    return float(tpt / ((tt @ tt - tpt) / (n - 2)))


def _covers(kind, interval, rays, tau):
    if kind == "interval":
        return interval[0] <= tau <= interval[1]
    if kind == "disjoint":
        return tau <= rays[0][1] or tau >= rays[1][0]
    return kind == "unbounded"


# ---------------------------------------------------------------- Fieller k=1


def test_fieller_hand_case_k1():
    # 6-point dataset, intercept only (q=1), k=1, dof=4. The AR quadratic is
    # hand-solved here from the scalar Fieller construction -- an independent
    # algebraic route from the implementation's projection form.
    z = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
    T = np.array([0.1, -0.2, 0.0, 1.1, 0.9, 1.3])
    y = np.array([0.0, 0.1, -0.1, 2.0, 1.8, 2.2])
    n, dof = 6, 4
    f_crit = float(stats.f.ppf(0.95, 1, dof))
    c = f_crit / dof
    zt, tt, yt = z - z.mean(), T - T.mean(), y - y.mean()
    szz, szt, szy = zt @ zt, zt @ tt, zt @ yt
    stt, syy, syt = tt @ tt, yt @ yt, yt @ tt
    a = (1 + c) * szt**2 / szz - c * stt
    b = -2.0 * ((1 + c) * szy * szt / szz - c * syt)
    cc = (1 + c) * szy**2 / szz - c * syy
    disc = b**2 - 4 * a * cc
    assert a > 0 and disc > 0  # strong hand case: a bounded Fieller interval
    r1 = (-b - np.sqrt(disc)) / (2 * a)
    r2 = (-b + np.sqrt(disc)) / (2 * a)

    ar = ar_confidence_set(y, T, z[:, None])
    assert ar.kind == "interval"
    assert ar.f_crit == pytest.approx(f_crit, abs=1e-12)
    assert ar.interval[0] == pytest.approx(min(r1, r2), abs=1e-8)
    assert ar.interval[1] == pytest.approx(max(r1, r2), abs=1e-8)
    assert ar.rays is None
    # k=1: the 2SLS/IV point zeroes the single moment, so AR(tau_hat) == 0.
    assert ar.ar_at_2sls == pytest.approx(0.0, abs=1e-10)
    assert n == z.size


# ------------------------------------------------------- strong-instrument


def test_strong_instrument_interval_close_to_wald():
    # Calibrated across seeds 10,20,30,40,50 (n=4000, strength 0.8): endpoint
    # offsets / Wald half-width all <= 0.05. Pinned seed 10 with margin 0.15.
    rng = np.random.default_rng(10)
    y, T, z = _endog_dgp(4000, rng)
    est = iv_2sls(y, T, z[:, None])
    assert est.ar_kind == "interval"
    lo, hi = est.ar_ci
    wlo, whi = est.ci
    hw = (whi - wlo) / 2.0
    assert abs(lo - wlo) <= 0.15 * hw
    assert abs(hi - whi) <= 0.15 * hw


# --------------------------------------------------------- weak-instrument


def test_weak_instrument_never_a_fabricated_interval():
    # Zero first stage (strength 0): calibrated seeds 7,17,27,37,47 all give
    # homoskedastic F < F_crit and kind in {unbounded, disjoint}. Pinned seed 7.
    rng = np.random.default_rng(7)
    y, T, z = _endog_dgp(300, rng, strength=0.0)
    assert _homoskedastic_first_stage_f(T, z) < stats.f.ppf(0.95, 1, 298)
    est = iv_2sls(y, T, z[:, None])
    assert est.ar_kind in {"unbounded", "disjoint"}
    assert est.ar_ci is None
    ar = ar_confidence_set(y, T, z[:, None])
    assert ar.kind == est.ar_kind
    assert ar.interval is None
    if ar.kind == "disjoint":
        assert ar.rays is not None
        assert est.extras["ar_rays"] == ar.rays


def test_boundedness_iff_first_stage_f_exceeds_crit():
    # Classical identity: the AR set is bounded iff the homoskedastic
    # first-stage F exceeds F_crit(1, n-2). Exact algebra -- assert the iff at
    # every point of a strength sweep straddling the threshold.
    rng = np.random.default_rng(21)
    n = 200
    z = rng.normal(size=n)
    v = rng.normal(size=n)
    e = 0.6 * v + 0.8 * rng.normal(size=n)
    f_crit = float(stats.f.ppf(0.95, 1, n - 2))
    bounded_seen, unbounded_seen = False, False
    for strength in (0.0, 0.02, 0.05, 0.1, 0.15, 0.25, 0.5):
        T = strength * z + v
        y = 1.0 * T + e
        ar = ar_confidence_set(y, T, z[:, None])
        bounded = ar.kind == "interval"
        assert bounded == (_homoskedastic_first_stage_f(T, z) > f_crit)
        bounded_seen |= bounded
        unbounded_seen |= not bounded
    assert bounded_seen and unbounded_seen  # the sweep straddles the threshold


# ------------------------------------------------------------------ empty set


def test_empty_set_reachable_with_invalid_second_instrument():
    # k=2: z1 strong and valid, z2 irrelevant but planted directly in y --
    # the model is rejected at every tau. Calibrated seeds 11,111,211,311,411:
    # kind == "empty" at all five. Pinned seed 11.
    rng = np.random.default_rng(11)
    n = 400
    z1 = rng.normal(size=n)
    z2 = rng.normal(size=n)
    u = rng.normal(size=n)
    T = 0.9 * z1 + u + 0.5 * rng.normal(size=n)
    y = 1.0 * T + 0.6 * u + 0.3 * rng.normal(size=n) + 1.5 * z2
    ar = ar_confidence_set(y, T, np.c_[z1, z2])
    assert ar.kind == "empty"
    assert ar.interval is None and ar.rays is None
    est = iv_2sls(y, T, np.c_[z1, z2])
    assert est.ar_kind == "empty"
    assert est.ar_ci is None


def test_k1_never_empty():
    # Structural impossibility: for k=1 the IV point always satisfies
    # g(tau_hat) <= 0, so the set is never empty at any strength.
    for seed in range(5):
        rng = np.random.default_rng(seed)
        for strength in (0.0, 0.05, 0.3):
            y, T, z = _endog_dgp(120, rng, strength=strength)
            ar = ar_confidence_set(y, T, z[:, None])
            assert ar.kind != "empty"


# ------------------------------------------------------------------- coverage


def test_weak_instrument_coverage_ar_beats_wald():
    # 200 replications at concentration mu^2 ~= 4 (n=250, pi=sqrt(4/250)),
    # endogeneity 0.95, true tau=1. AR is exact under homoskedastic normal
    # errors; the Wald CI undercovers under weak identification with strong
    # endogeneity. Calibrated across seed bases 20000/30000/40000/50000/60000:
    # AR covers 187-195/200, Wald 170-185/200 (Wald < AR at every base; at
    # endogeneity 0.6 Wald does NOT undercover -- 189/200 -- hence 0.95 here).
    # Pinned base 20000: AR 188/200, Wald 173/200.
    n, tau = 250, 1.0
    pi = np.sqrt(4.0 / n)
    endog = 0.95
    ar_covered = wald_covered = 0
    for seed in range(200):
        rng = np.random.default_rng(20000 + seed)
        z = rng.normal(size=n)
        v = rng.normal(size=n)
        e = endog * v + np.sqrt(1.0 - endog**2) * rng.normal(size=n)
        T = pi * z + v
        y = tau * T + e
        est = iv_2sls(y, T, z[:, None])
        rays = est.extras.get("ar_rays")
        if _covers(est.ar_kind, est.ar_ci, rays, tau):
            ar_covered += 1
        if est.ci[0] <= tau <= est.ci[1]:
            wald_covered += 1
    assert ar_covered >= 184  # >= 92% (target 95%; observed 188)
    assert wald_covered <= ar_covered - 10  # Wald materially undercovers (obs. 173)


# ------------------------------------------------------------ local_2sls wire


def _discovered(seed=0):
    ds, _ = make_synthetic(n=2000, zeta=4.0, tau=2.0, kind="binary", rng=np.random.default_rng(seed))
    res = lord3_scan(ds, k=60, rng=np.random.default_rng(seed + 1))
    return ds, res.discoveries[0]


def test_local_2sls_fills_finite_ar_ci_overlapping_wald():
    ds, d = _discovered()
    est = local_2sls(ds, d)
    assert est.ar_kind == "interval"
    assert est.ar_ci is not None
    lo, hi = est.ar_ci
    assert np.isfinite(lo) and np.isfinite(hi) and lo < hi
    # overlaps the 2SLS Wald CI
    assert max(lo, est.ci[0]) <= min(hi, est.ci[1])


def test_wald_estimate_leaves_ar_defaults():
    ds, d = _discovered(seed=3)
    est = wald_estimate(ds, d)
    assert est.ar_ci is None
    assert est.ar_kind is None


def test_iv_2sls_nan_path_leaves_ar_defaults():
    rng = np.random.default_rng(4)
    _, T, z = _endog_dgp(50, rng)
    est = iv_2sls(np.full(50, np.nan), T, z[:, None])
    assert np.isnan(est.tau)
    assert est.ar_ci is None
    assert est.ar_kind is None


def test_issue_11_ar_set_duplicated_instrument_is_exact_fieller():
    # k_eff = 1: f_crit and dof must use the effective rank measured after
    # partialling, so the set from [z, z] is the k = 1 Fieller set — not an
    # F(2, n-3)-parametrized (anti-conservative) deformation of it.
    rng = np.random.default_rng(13)
    y, T, z = _endog_dgp(300, rng)
    one = ar_confidence_set(y, T, z[:, None])
    dup = ar_confidence_set(y, T, np.c_[z, z])
    assert one.f_crit == pytest.approx(float(stats.f.ppf(0.95, 1, 298)), abs=1e-12)
    assert dup.f_crit == one.f_crit
    assert dup.kind == one.kind == "interval"
    assert dup.interval == pytest.approx(one.interval, rel=1e-10)


def test_issue_11_ar_set_control_collinear_instruments_raise():
    rng = np.random.default_rng(14)
    y, T, _ = _endog_dgp(100, rng)
    w = rng.normal(size=100)
    with pytest.raises(ValueError, match="collinear"):
        ar_confidence_set(y, T, (3.0 * w)[:, None], controls=w[:, None])


def test_ar_confidence_set_input_validation():
    rng = np.random.default_rng(5)
    y, T, z = _endog_dgp(20, rng)
    with pytest.raises(ValueError, match="instrument"):
        ar_confidence_set(y, T, np.empty((20, 0)))
    with pytest.raises(ValueError, match="underdetermined"):
        ar_confidence_set(y[:2], T[:2], z[:2, None])
