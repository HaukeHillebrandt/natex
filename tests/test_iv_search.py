"""Tests for Belloni plug-in Lasso instrument selection (phase 5, task 4).

Statistical-test policy: every stochastic assertion below was calibrated
across >= 5 seeds during implementation; the pinned seed and the observed
ranges are recorded in a comment next to each threshold.
"""

import inspect

import numpy as np
import pytest

from natex.data.synthetic_iv import make_iv_synthetic
from natex.iv.search import select_instruments


def _standardized_single_column():
    """z in {-1,+1} balanced (centered, z_i^2 == 1) and d centered with
    mean-square exactly 1, so the first-iteration loading is psi = 1."""
    z = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0])
    d = np.array([1.3, -0.4, 0.2, -1.1, 0.7, -0.9, 0.5, -0.3])
    d = d - d.mean()
    d = d / np.sqrt(np.mean(d**2))
    return z, d


def test_analytic_soft_threshold_pins_sklearn_alpha_mapping():
    # Single instrument, psi = 1: pi_lasso = soft(z'd, lam/2) / (z'z).
    z, d = _standardized_single_column()
    s = float(z @ d)
    assert abs(s) > 0.5  # construction sanity: the signal is not degenerate
    lam = abs(s)  # threshold lam/2 = |s|/2 < |s| -> interior solution
    res = select_instruments(d, z[:, None], lam=lam, max_iter=1)
    expected = np.sign(s) * (abs(s) - lam / 2.0) / float(z @ z)
    assert res.loadings[0] == pytest.approx(1.0, abs=1e-12)
    assert res.pi_lasso[0] == pytest.approx(expected, abs=1e-8)
    assert res.lam == lam
    assert res.n_iter == 1
    assert res.selected == ["z1"]


def test_analytic_soft_threshold_kills_coefficient_beyond_threshold():
    z, d = _standardized_single_column()
    s = float(z @ d)
    res = select_instruments(d, z[:, None], lam=2.5 * abs(s), max_iter=1)
    assert res.selected == []
    assert np.all(res.pi_lasso == 0.0)
    assert np.all(res.pi_post == 0.0)
    assert np.isnan(res.first_stage_F)
    assert res.weak is True


def test_recovers_top_strong_instruments_on_bcch_dgp():
    # Calibrated across seeds 0..7: selection always a SUBSET of the true
    # support (false positives 0/8), first_stage_F in [38.5, 65.3], and
    # {z1, z2} always in; z3 additionally in at 6/8 seeds (Lasso shrinkage
    # under the rho=0.5 Toeplitz design absorbs z3 into z2/z4 at seeds 0, 3).
    # Pinned seed 2: selected == [z1, z2, z3], F = 62.3.
    data = make_iv_synthetic(n=500, p=50, s=5, mu2=180.0, rng=np.random.default_rng(2))
    pool = data.df[data.pool_names].to_numpy()
    res = select_instruments(data.df["T"].to_numpy(), pool, pool_names=data.pool_names)
    assert {"z1", "z2", "z3"} <= set(res.selected)
    false_positives = set(res.selected) - set(data.true_support)
    assert len(false_positives) <= 2
    assert res.first_stage_F > 10.0
    assert res.weak is False
    assert 0.0 < res.partial_r2 < 1.0
    assert res.lam > 0.0


def test_null_pool_yields_honest_empty_selection():
    # mu2 = 0 gives an exact pure-noise pool (pi = 0). Calibrated seeds 0..4:
    # selection was empty at 5/5 seeds; the gate only requires >= 4/5.
    empties = 0
    for seed in range(5):
        data = make_iv_synthetic(n=500, p=50, s=5, mu2=0.0, rng=np.random.default_rng(seed))
        pool = data.df[data.pool_names].to_numpy()
        res = select_instruments(data.df["T"].to_numpy(), pool, pool_names=data.pool_names)
        if res.selected == []:
            empties += 1
            assert np.isnan(res.first_stage_F)
            assert np.isnan(res.partial_r2)
            assert res.weak is True
            assert np.all(res.pi_lasso == 0.0)
            assert np.all(res.pi_post == 0.0)
    assert empties >= 4


def _confounded_pool(seed, n=400):
    """A control x drives both T and pool column z1; z1 has no signal for T
    once x is partialled out (pure FWL demonstration)."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    z_spur = x + 0.1 * rng.normal(size=n)
    noise = rng.normal(size=(n, 4))
    t_vec = x + rng.normal(size=n)
    return t_vec, np.c_[z_spur, noise], x


def test_controls_partialling_blocks_spurious_column():
    # Calibrated across seeds 0..4: with controls z1 is never selected;
    # without controls z1 is always selected. Pinned seed 0.
    t_vec, pool, x = _confounded_pool(seed=0)
    with_controls = select_instruments(t_vec, pool, controls=x)
    without_controls = select_instruments(t_vec, pool)
    assert "z1" not in with_controls.selected
    assert "z1" in without_controls.selected


def test_plugin_is_rng_free_and_bitwise_deterministic():
    data = make_iv_synthetic(n=300, p=20, s=3, mu2=120.0, rng=np.random.default_rng(1))
    pool = data.df[data.pool_names].to_numpy()
    t_vec = data.df["T"].to_numpy()
    a = select_instruments(t_vec, pool, pool_names=data.pool_names)
    b = select_instruments(t_vec, pool, pool_names=data.pool_names)
    assert a.selected == b.selected
    assert np.array_equal(a.pi_lasso, b.pi_lasso)
    assert np.array_equal(a.pi_post, b.pi_post)
    assert np.array_equal(a.loadings, b.loadings)
    assert a.lam == b.lam
    assert a.first_stage_F == b.first_stage_F
    assert a.n_iter == b.n_iter


def test_explicit_float_lambda_is_rng_free_and_bitwise_deterministic():
    data = make_iv_synthetic(n=300, p=20, s=3, mu2=120.0, rng=np.random.default_rng(1))
    pool = data.df[data.pool_names].to_numpy()
    t_vec = data.df["T"].to_numpy()
    a = select_instruments(t_vec, pool, pool_names=data.pool_names, lam=50.0)
    b = select_instruments(t_vec, pool, pool_names=data.pool_names, lam=50.0)
    assert a.selected == b.selected
    assert np.array_equal(a.pi_lasso, b.pi_lasso)
    assert np.array_equal(a.loadings, b.loadings)
    assert a.lam == 50.0 == b.lam


def test_cv_without_rng_raises():
    data = make_iv_synthetic(n=300, p=20, s=3, mu2=120.0, rng=np.random.default_rng(1))
    pool = data.df[data.pool_names].to_numpy()
    with pytest.raises(ValueError, match="Generator"):
        select_instruments(data.df["T"].to_numpy(), pool, lam="cv")


def test_cv_with_rng_runs_and_reports_lambda():
    data = make_iv_synthetic(n=300, p=20, s=3, mu2=120.0, rng=np.random.default_rng(1))
    pool = data.df[data.pool_names].to_numpy()
    res = select_instruments(
        data.df["T"].to_numpy(), pool, pool_names=data.pool_names,
        lam="cv", rng=np.random.default_rng(0),
    )
    assert np.isfinite(res.lam) and res.lam > 0.0
    assert "z1" in res.selected  # CV selects at least the strongest true instrument
    assert res.extras["lam_source"] == "cv"


def test_zero_variance_column_excluded_with_diagnostic():
    data = make_iv_synthetic(n=300, p=10, s=3, mu2=120.0, rng=np.random.default_rng(2))
    pool = data.df[data.pool_names].to_numpy().copy()
    pool[:, 4] = 7.0  # z5 (a null column) becomes constant
    res = select_instruments(data.df["T"].to_numpy(), pool, pool_names=data.pool_names)
    assert "z5" in res.extras["dropped_zero_variance"]
    assert "z5" not in res.selected
    assert np.isfinite(res.loadings).all()
    assert res.pi_lasso[4] == 0.0
    assert res.pi_post[4] == 0.0
    assert "z1" in res.selected  # the strong instruments are unaffected


def test_explicit_lambda_support_size_monotone():
    data = make_iv_synthetic(n=500, p=50, s=5, mu2=180.0, rng=np.random.default_rng(3))
    pool = data.df[data.pool_names].to_numpy()
    t_vec = data.df["T"].to_numpy()
    sizes = [
        len(select_instruments(t_vec, pool, pool_names=data.pool_names, lam=lam).selected)
        for lam in (0.1, 100.0, 1e6)
    ]
    assert sizes[0] >= 45  # tiny lambda keeps (almost) everything
    assert sizes[2] == 0  # huge lambda kills everything
    assert sizes[0] >= sizes[1] >= sizes[2]


def test_signature_never_receives_outcome():
    # Discovery-honesty analog: selection cannot even receive y.
    params = inspect.signature(select_instruments).parameters
    assert "y" not in params
    assert "outcome" not in params


def test_invalid_arguments_raise():
    z, d = _standardized_single_column()
    pool = z[:, None]
    with pytest.raises(ValueError):
        select_instruments(d, pool, lam="bogus")
    with pytest.raises(ValueError):
        select_instruments(d, pool, lam=-1.0)
    with pytest.raises(ValueError):
        select_instruments(d, pool, max_iter=0)
    with pytest.raises(ValueError):
        select_instruments(d, pool, c=0.0)
    with pytest.raises(ValueError):
        select_instruments(d, pool, gamma=1.5)
    with pytest.raises(ValueError):
        select_instruments(d, pool, pool_names=["a", "b"])
