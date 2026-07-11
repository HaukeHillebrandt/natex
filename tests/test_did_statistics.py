"""Tests for the window-restricted DiD LLR kernels (phase 3, task 3).

Covers: WindowStats [T0-W, T0) / [T0, T0+W) convention, the double-beta LLR
(Eq 6.9) against hand-computed scalars and the harmonic-mean identity
(Eq 6.11), the audit-15 corrected profiled single-Delta GLR against a
scipy brute-force maximization, the Bernoulli window LLR (audit item 19)
against the scalar phase-1 reference, and the shared properties: window
restriction (exact equality), LLR >= 0, permutation invariance, empty /
degenerate columns exactly 0.0.
"""

import numpy as np
import pytest
from scipy.optimize import minimize

from natex.did.statistics import (
    WindowStats,
    bernoulli_window_llr_masks,
    double_beta_llr_masks,
    double_beta_q,
    single_delta_llr,
    single_delta_stats,
    window_stats,
    working_residuals,
)
from natex.scan.statistics import fit_log_odds_offset, offset_log_lik

# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

# 6 in-window records: pre = {-3, -2, -1}, post = {0, 1, 2} for T0=0, W=3.
T6 = np.array([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0])
R6 = np.array([0.5, -0.2, 0.1, 1.0, 1.4, 0.8])
S6 = np.array([1.0, 0.5, 2.0, 1.0, 0.25, 0.5])  # sigma^2 -> w = [1, 2, .5, 1, 4, 2]


def ws6() -> WindowStats:
    return window_stats(T6, R6, S6, T0=0.0, W=3.0)


def extended_with_outside(t, r, sigma2):
    """Append out-of-window records with huge residuals (window T0=0, W=3)."""
    t_x = np.concatenate([t, [-100.0, 3.0, 57.0]])
    r_x = np.concatenate([r, [1e6, -1e6, 4e5]])
    s_x = np.concatenate([sigma2, [1e-4, 1e-4, 1e-4]])
    return t_x, r_x, s_x


# ---------------------------------------------------------------------------
# WindowStats convention
# ---------------------------------------------------------------------------


def test_window_stats_boundary_convention():
    t = np.array([-3.0 - 1e-9, -3.0, -0.5, 0.0, 2.9, 3.0])
    r = np.ones(6)
    s2 = np.full(6, 2.0)
    ws = window_stats(t, r, s2, T0=0.0, W=3.0)
    np.testing.assert_array_equal(ws.in_window, [False, True, True, True, True, False])
    np.testing.assert_array_equal(ws.g1, [False, False, False, True, True, False])
    # c, b zeroed outside the window; w = 1/sigma2 inside.
    np.testing.assert_array_equal(ws.b, [0.0, 0.5, 0.5, 0.5, 0.5, 0.0])
    np.testing.assert_array_equal(ws.c, [0.0, 0.5, 0.5, 0.5, 0.5, 0.0])
    assert not np.any(ws.g1 & ~ws.in_window)


def test_window_stats_validation():
    with pytest.raises(ValueError):
        window_stats(T6, R6, S6, T0=0.0, W=0.0)
    bad = S6.copy()
    bad[2] = 0.0
    with pytest.raises(ValueError):
        window_stats(T6, R6, bad, T0=0.0, W=3.0)
    with pytest.raises(ValueError):
        window_stats(T6, R6[:-1], S6, T0=0.0, W=3.0)


# ---------------------------------------------------------------------------
# double-beta (Eq 6.9)
# ---------------------------------------------------------------------------


def test_double_beta_hand_computed():
    ws = ws6()
    # subset {0, 1, 3, 4}: pre = records 0,1; post = records 3,4 (w = 1,2,1,4).
    m = np.zeros(6, dtype=bool)
    m[[0, 1, 3, 4]] = True
    M = np.column_stack([np.ones(6, dtype=bool), m])
    got = double_beta_llr_masks(ws, M)

    # hand calculation, subset column: C1 = 1*1.0 + 4*1.4, B1 = 5; C0 = 1*0.5 + 2*(-0.2), B0 = 3.
    c1, b1 = 1.0 * 1.0 + 4.0 * 1.4, 1.0 + 4.0
    c0, b0 = 1.0 * 0.5 + 2.0 * (-0.2), 1.0 + 2.0
    expect_sub = c1**2 / (2 * b1) + c0**2 / (2 * b0) - (c1 + c0) ** 2 / (2 * (b1 + b0))
    # full column: post w,r = (1,1.0),(4,1.4),(2,0.8); pre = (1,0.5),(2,-0.2),(0.5,0.1).
    fc1, fb1 = 1.0 * 1.0 + 4.0 * 1.4 + 2.0 * 0.8, 7.0
    fc0, fb0 = 1.0 * 0.5 + 2.0 * (-0.2) + 0.5 * 0.1, 3.5
    expect_full = fc1**2 / (2 * fb1) + fc0**2 / (2 * fb0) - (fc1 + fc0) ** 2 / (2 * (fb1 + fb0))
    np.testing.assert_allclose(got, [expect_full, expect_sub], rtol=0, atol=1e-12)

    # algebraic identity (Eq 6.11): LLR = HM(B1, B0) * ((q1 - q0)/2)^2.
    q1, q2 = double_beta_q(ws, M)
    hm = np.array([2 * fb1 * fb0 / (fb1 + fb0), 2 * b1 * b0 / (b1 + b0)])
    np.testing.assert_allclose(got, hm * ((q1 - q2) / 2.0) ** 2, rtol=0, atol=1e-12)


def test_double_beta_degenerate_columns_exactly_zero():
    ws = ws6()
    pre_only = T6 < 0.0
    post_only = T6 >= 0.0
    empty = np.zeros(6, dtype=bool)
    M = np.column_stack([pre_only, post_only, empty])
    got = double_beta_llr_masks(ws, M)
    assert got.shape == (3,)
    assert np.all(got == 0.0)


def test_double_beta_q_nan_on_empty_sides():
    ws = ws6()
    pre_only = T6 < 0.0
    M = np.column_stack([np.ones(6, dtype=bool), pre_only, np.zeros(6, dtype=bool)])
    q1, q2 = double_beta_q(ws, M)
    # full column: precision-weighted means.
    np.testing.assert_allclose(q1[0], (1.0 + 5.6 + 1.6) / 7.0, atol=1e-12)
    np.testing.assert_allclose(q2[0], (0.5 - 0.4 + 0.05) / 3.5, atol=1e-12)
    assert np.isnan(q1[1]) and not np.isnan(q2[1])  # pre-only: empty post side
    assert np.isnan(q1[2]) and np.isnan(q2[2])  # empty subset: both NaN, never 0


# ---------------------------------------------------------------------------
# single-Delta (audit item 15)
# ---------------------------------------------------------------------------


def brute_force_glr(t, r, sigma2, pid, n_prof):
    """Exact Gaussian GLR maximized numerically over (Delta, mu_1..mu_P)."""
    w = 1.0 / sigma2
    d = np.where(t >= 0.0, 1.0, -1.0)
    rbar = np.array(
        [np.sum(w[pid == i] * r[pid == i]) / np.sum(w[pid == i]) for i in range(n_prof)]
    )
    nll0 = 0.5 * float(np.sum(w * (r - rbar[pid]) ** 2))

    def nll1(params):
        delta, mu = params[0], np.asarray(params[1:])
        resid = r - mu[pid] - d * delta
        return 0.5 * float(np.sum(w * resid**2))

    res = minimize(nll1, x0=np.concatenate([[0.0], rbar]), method="BFGS")
    assert res.fun <= nll0 + 1e-12
    return nll0 - res.fun


def test_single_delta_matches_bruteforce_and_beats_unprofiled():
    """Audit-15 regression: profiled GLR == numeric optimum; thesis-printed
    (mu frozen at the H0 MLE, B = sum w) statistic is strictly smaller under
    unbalanced pre/post precision (the 4.74-vs-5.33 class of counterexample).

    Calibration (plan policy): across seeds 0..11 on this fixed design the
    brute-force parity error is <= 2e-13 (atol 1e-6 pinned with margin) and
    the profiled-minus-unprofiled gap is >= 0.023 (seed 7: 0.75)."""
    rng = np.random.default_rng(7)
    pid = np.repeat(np.arange(3), 4)
    # unbalanced pre/post precision inside each profile -> delta_bar != 0.
    t = np.array([-3.0, -2.0, -1.0, 1.0, -1.0, 0.0, 1.0, 2.0, -2.0, -1.0, 0.0, 1.0])
    sigma2 = np.array([0.5, 1.0, 2.0, 0.25, 1.0, 0.5, 0.4, 2.0, 0.3, 3.0, 1.0, 0.6])
    r = rng.normal(0.0, 1.0, 12) + np.where(t >= 0.0, 0.8, -0.8)

    ws = window_stats(t, r, sigma2, T0=0.0, W=4.0)
    C, B = single_delta_stats(ws, pid)
    assert C.shape == B.shape == (3,)
    llr = single_delta_llr(float(C.sum()), float(B.sum()))

    brute = brute_force_glr(t, r, sigma2, pid, 3)
    assert brute > 0.0
    np.testing.assert_allclose(llr, brute, rtol=0, atol=1e-6)

    # the UNprofiled statistic: same C-tilde, but B = full precision mass sum w.
    w = 1.0 / sigma2
    d = np.where(t >= 0.0, 1.0, -1.0)
    rbar_map = {i: np.sum(w[pid == i] * r[pid == i]) / np.sum(w[pid == i]) for i in range(3)}
    rbar = np.array([rbar_map[i] for i in pid])
    c_unprof = float(np.sum(w * d * (r - rbar)))
    np.testing.assert_allclose(c_unprof, C.sum(), atol=1e-10)  # C coincides; only B differs
    llr_unprof = c_unprof**2 / (2.0 * float(w.sum()))
    assert llr_unprof < llr - 1e-6
    # delta_bar != 0 for at least one profile => B_tilde strictly below sum w.
    assert float(B.sum()) < float(w.sum()) - 1e-9


def test_single_delta_balanced_precision_equals_unprofiled():
    """With delta_bar = 0 in every profile the correction is inert."""
    t = np.array([-2.0, -1.0, 0.0, 1.0, -2.0, -1.0, 0.0, 1.0])
    sigma2 = np.array([0.5, 2.0, 0.5, 2.0, 1.0, 0.25, 1.0, 0.25])  # mirrored pre/post
    r = np.array([0.3, -0.6, 1.1, 0.2, -0.4, 0.9, 0.5, -0.1])
    pid = np.repeat([0, 1], 4)
    ws = window_stats(t, r, sigma2, T0=0.0, W=2.0)
    C, B = single_delta_stats(ws, pid)
    np.testing.assert_allclose(B.sum(), (1.0 / sigma2).sum(), atol=1e-12)
    np.testing.assert_allclose(
        single_delta_llr(float(C.sum()), float(B.sum())),
        brute_force_glr(t, r, sigma2, pid, 2),
        rtol=0,
        atol=1e-6,
    )


def test_single_delta_both_signs():
    """Negating all residuals: identical LLR, Delta_hat = C/B negated."""
    ws_pos = ws6()
    ws_neg = window_stats(T6, -R6, S6, T0=0.0, W=3.0)
    pid = np.array([0, 1, 0, 1, 0, 1])
    Cp, Bp = single_delta_stats(ws_pos, pid)
    Cn, Bn = single_delta_stats(ws_neg, pid)
    np.testing.assert_array_equal(Bp, Bn)
    np.testing.assert_array_equal(Cn, -Cp)
    np.testing.assert_array_equal(
        single_delta_llr(Cp.sum(), Bp.sum()), single_delta_llr(Cn.sum(), Bn.sum())
    )
    assert Cn.sum() / Bn.sum() == -(Cp.sum() / Bp.sum())


def test_single_delta_degenerate_profiles_zeroed():
    # profile 0: one record only; profile 1: post-side only; profile 2: healthy.
    t = np.array([1.0, 0.5, 1.5, -1.0, 1.0])
    r = np.array([3.0, 2.0, -1.0, 0.5, 0.7])
    s2 = np.ones(5)
    pid = np.array([0, 1, 1, 2, 2])
    ws = window_stats(t, r, s2, T0=0.0, W=2.0)
    C, B = single_delta_stats(ws, pid)
    assert C[0] == 0.0 and B[0] == 0.0
    assert C[1] == 0.0 and B[1] == 0.0
    assert B[2] > 0.0


def test_single_delta_llr_guards():
    assert single_delta_llr(0.0, 0.0) == 0.0
    assert single_delta_llr(2.0, -1.0) == 0.0
    out = single_delta_llr(np.array([2.0, 1.0, 0.0]), np.array([4.0, 0.0, 4.0]))
    np.testing.assert_array_equal(out, [0.5, 0.0, 0.0])


# ---------------------------------------------------------------------------
# Bernoulli window LLR (audit item 19)
# ---------------------------------------------------------------------------


def _sup(t, eta):
    return offset_log_lik(fit_log_odds_offset(t, eta), t, eta)


def bern_reference(theta, eta, ws, M):
    """Scalar per-column reference from the phase-1 primitives."""
    g0 = ws.in_window & ~ws.g1
    out = np.zeros(M.shape[1])
    for j in range(M.shape[1]):
        s1 = M[:, j] & ws.g1
        s0 = M[:, j] & g0
        sw = s1 | s0
        if not s1.any() or not s0.any():
            out[j] = 0.0
            continue
        ll1 = _sup(theta[s1], eta[s1]) + _sup(theta[s0], eta[s0])
        out[j] = max(ll1 - _sup(theta[sw], eta[sw]), 0.0)
    return out


def bern_fixture():
    rng = np.random.default_rng(3)
    t = np.arange(-6.0, 6.0)  # 12 records, all inside T0=0, W=6
    theta = np.array([0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0], dtype=float)
    eta = rng.normal(0.0, 0.7, size=12)
    ws = window_stats(t, r=theta - 0.5, sigma2=np.full(12, 0.25), T0=0.0, W=6.0)
    post = t >= 0.0
    cols = [np.ones(12, dtype=bool)]
    for _ in range(6):
        cols.append(rng.random(12) < 0.6)
    pure_post = np.zeros(12, dtype=bool)  # post side all theta == 1: boundary supremum
    pure_post[[0, 1, 2, 6, 7, 10]] = True
    assert np.all(theta[pure_post & post] == 1.0)
    cols.append(pure_post)
    cols.append(~post)  # degenerate: pre-side only
    cols.append(np.zeros(12, dtype=bool))  # empty
    return theta, eta, ws, np.column_stack(cols)


def test_bernoulli_parity_with_scalar_reference():
    theta, eta, ws, M = bern_fixture()
    got = bernoulli_window_llr_masks(theta, eta, ws, M)
    ref = bern_reference(theta, eta, ws, M)
    np.testing.assert_allclose(got, ref, rtol=0, atol=1e-8)
    assert np.all(np.isfinite(got))
    assert np.all(got >= 0.0)
    assert got[-2] == 0.0 and got[-1] == 0.0  # degenerate + empty: exactly 0.0
    assert np.isfinite(got[-3]) and got[-3] >= 0.0  # pure post side: boundary supremum


# ---------------------------------------------------------------------------
# shared properties: window restriction, LLR >= 0, permutation invariance
# ---------------------------------------------------------------------------


def test_window_restriction_exact_equality():
    """Out-of-window records with huge residuals change nothing, exactly."""
    rng = np.random.default_rng(11)
    ws = ws6()
    M = np.column_stack([rng.random(6) < 0.7 for _ in range(8)] + [np.ones(6, dtype=bool)])
    pid = np.array([0, 1, 0, 1, 0, 1])
    theta_b = np.array([0, 1, 1, 0, 1, 1], dtype=float)
    eta = rng.normal(0.0, 0.5, size=6)

    t_x, r_x, s_x = extended_with_outside(T6, R6, S6)
    ws_x = window_stats(t_x, r_x, s_x, T0=0.0, W=3.0)
    M_x = np.vstack([M, np.ones((3, M.shape[1]), dtype=bool)])  # outsiders in every subset
    pid_x = np.concatenate([pid, [0, 1, 0]])
    theta_x = np.concatenate([theta_b, [1.0, 1.0, 0.0]])
    eta_x = np.concatenate([eta, [5.0, -5.0, 0.0]])

    np.testing.assert_array_equal(double_beta_llr_masks(ws, M), double_beta_llr_masks(ws_x, M_x))
    C, B = single_delta_stats(ws, pid)
    C_x, B_x = single_delta_stats(ws_x, pid_x)
    np.testing.assert_array_equal(C, C_x)
    np.testing.assert_array_equal(B, B_x)
    np.testing.assert_array_equal(
        bernoulli_window_llr_masks(theta_b, eta, ws, M),
        bernoulli_window_llr_masks(theta_x, eta_x, ws_x, M_x),
    )


def test_llr_nonnegative_and_permutation_invariant():
    rng = np.random.default_rng(23)
    n, s_cols = 40, 25
    t = rng.uniform(-5.0, 5.0, size=n)  # some records fall outside T0=0.5, W=3
    r = rng.normal(0.0, 1.5, size=n)
    sigma2 = rng.lognormal(0.0, 0.5, size=n)
    pid = rng.integers(0, 4, size=n)
    theta_b = rng.integers(0, 2, size=n).astype(float)
    eta = rng.normal(0.0, 0.8, size=n)
    M = rng.random((n, s_cols)) < 0.5
    M[:, 0] = False  # empty subset column
    ws = window_stats(t, r, sigma2, T0=0.5, W=3.0)

    db = double_beta_llr_masks(ws, M)
    C, B = single_delta_stats(ws, pid)
    sd = single_delta_llr(C, B)
    bl = bernoulli_window_llr_masks(theta_b, eta, ws, M)
    for vals in (db, sd, bl):
        assert np.all(vals >= 0.0)
        assert np.all(np.isfinite(vals))
    assert db[0] == 0.0 and bl[0] == 0.0

    perm = rng.permutation(n)
    ws_p = window_stats(t[perm], r[perm], sigma2[perm], T0=0.5, W=3.0)
    np.testing.assert_allclose(
        double_beta_llr_masks(ws_p, M[perm]), db, rtol=1e-12, atol=1e-12
    )
    C_p, B_p = single_delta_stats(ws_p, pid[perm])
    np.testing.assert_allclose(C_p, C, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(B_p, B, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        bernoulli_window_llr_masks(theta_b[perm], eta[perm], ws_p, M[perm]),
        bl,
        rtol=0,
        atol=1e-8,
    )


# ---------------------------------------------------------------------------
# working residuals (Bernoulli priority ordering only)
# ---------------------------------------------------------------------------


def test_working_residuals():
    theta = np.array([0.0, 1.0, 1.0])
    p_hat = np.array([0.25, 0.5, 0.9])
    r, s2 = working_residuals(theta, p_hat)
    np.testing.assert_array_equal(r, theta - p_hat)  # exact definition
    np.testing.assert_allclose(r, [-0.25, 0.5, 0.1], atol=1e-15)
    np.testing.assert_allclose(s2, [0.1875, 0.25, 0.09], atol=1e-15)
