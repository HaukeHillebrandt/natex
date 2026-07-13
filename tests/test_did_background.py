"""Tests for the DiD treatment background model (phase 3, task 2)."""

import numpy as np
import pytest

from natex.did.background import DiDBackground, fit_did_background
from natex.did.panel import CategoricalPanel


def make_panel(
    codes: np.ndarray,
    t: np.ndarray,
    theta: np.ndarray,
    unit: np.ndarray | None = None,
) -> CategoricalPanel:
    """Hand-rolled CategoricalPanel (bypasses Dataset/build_panel)."""
    t = np.asarray(t, dtype=float)
    codes = np.asarray(codes, dtype=np.int64).reshape(t.size, -1)
    m = codes.shape[1]
    if unit is None:
        unit = np.zeros(t.size, dtype=np.int64)
    unit = np.asarray(unit, dtype=np.int64)
    return CategoricalPanel(
        codes=codes,
        dim_names=[f"d{j}" for j in range(m)],
        dim_values=[np.arange(int(codes[:, j].max()) + 1) for j in range(m)],
        t=t,
        theta=np.asarray(theta, dtype=float),
        y=None,
        unit=unit,
        unit_values=np.arange(int(unit.max()) + 1),
    )


def planted_panel(seed: int = 0) -> tuple[CategoricalPanel, np.ndarray]:
    """6 units x 30 periods; theta = unit intercept + linear trend + noise, no jump."""
    rng = np.random.default_rng(seed)
    n_units, n_periods = 6, 30
    unit = np.repeat(np.arange(n_units), n_periods)
    t = np.tile(np.arange(n_periods, dtype=float), n_units)
    intercepts = rng.normal(0.0, 2.0, size=n_units)
    theta = intercepts[unit] + 0.5 * t + rng.normal(0.0, 0.1, size=unit.size)
    codes = (unit % 3).astype(np.int64)  # 3 covariate profiles
    return make_panel(codes, t, theta, unit=unit), unit


# ---------------------------------------------------------------------------
# normal path: unit effects + polynomial time
# ---------------------------------------------------------------------------


def test_normal_absorbs_unit_effects_and_trend():
    panel, unit = planted_panel(seed=1)
    bg = fit_did_background(panel, model="normal", degree=1, unit_effects=True)
    assert isinstance(bg, DiDBackground)
    assert bg.kind == "normal"
    assert bg.eta is None
    assert bg.r is not None and bg.sigma2 is not None
    assert bg.fitted.shape == bg.r.shape == bg.sigma2.shape == (panel.n,)
    np.testing.assert_allclose(bg.r, panel.theta - bg.fitted)
    # Thresholds calibrated over seeds 0..11: worst per-unit |mean| ~ 7e-15 and
    # worst |slope| ~ 2e-16 (OLS residuals are orthogonal to the design), so
    # 0.05 carries an enormous margin.
    # Per-unit residual means ~ 0 (intercepts absorbed).
    for u in range(6):
        assert abs(float(bg.r[unit == u].mean())) < 0.05
    # No remaining time trend (slope of r on t ~ 0).
    ts = panel.t - panel.t.mean()
    slope = float(ts @ bg.r) / float(ts @ ts)
    assert abs(slope) < 0.05


def test_rank_deficient_design_does_not_raise():
    # Full set of unit dummies + polynomial time: Eq 6.4's rank deficiency is
    # handled by the minimum-norm lstsq/pinv solution, never an error.
    panel, _ = planted_panel(seed=2)
    bg = fit_did_background(panel, model="normal", degree=2, unit_effects=True)
    assert np.all(np.isfinite(bg.fitted))
    assert np.all(np.isfinite(bg.r))


# ---------------------------------------------------------------------------
# per-profile variance with data-scaled shrinkage (audit item 24)
# ---------------------------------------------------------------------------


def shrinkage_panel(seed: int = 3) -> tuple[CategoricalPanel, np.ndarray]:
    """Profile 0 has exactly 2 records (low noise); profiles 1, 2 have 40 each."""
    rng = np.random.default_rng(seed)
    codes = np.concatenate([[0, 0], np.repeat([1, 2], 40)]).astype(np.int64)
    n = codes.size
    t = np.arange(n, dtype=float)
    theta = rng.normal(0.0, 1.0, size=n)
    theta[:2] = 0.01 * rng.normal(size=2)  # tiny-variance 2-record profile
    return make_panel(codes, t, theta), codes


def test_two_record_profile_is_shrunk_toward_global():
    panel, codes = shrinkage_panel()
    bg = fit_did_background(panel, model="normal", degree=1, unit_effects=True)
    r = bg.r
    raw = float(np.var(r[codes == 0]))
    glob = float(np.var(r))
    assert raw != pytest.approx(glob)
    lo, hi = min(raw, glob), max(raw, glob)
    got = bg.sigma2[codes == 0]
    assert np.all(got > lo) and np.all(got < hi)  # strictly between raw and global
    # Both records of one profile share one variance.
    assert got[0] == got[1]


def test_zero_variance_profile_gets_scaled_floor():
    # Two identical records (same t, theta, profile) => residuals identical =>
    # raw profile variance exactly 0. With shrink=0.0 (no shrinkage at all) the
    # data-scaled floor 1e-12 * global variance must still make sigma2 > 0.
    rng = np.random.default_rng(4)
    codes = np.concatenate([[0, 0], np.repeat([1, 2], 40)]).astype(np.int64)
    n = codes.size
    t = np.concatenate([[5.0, 5.0], np.arange(n - 2, dtype=float)])
    theta = rng.normal(0.0, 1.0, size=n)
    theta[1] = theta[0]
    panel = make_panel(codes, t, theta)
    bg = fit_did_background(panel, model="normal", degree=1, unit_effects=True, shrink=0.0)
    assert float(np.var(bg.r[codes == 0])) == 0.0
    glob = float(np.var(bg.r))
    got = bg.sigma2[codes == 0]
    assert np.all(got > 0.0)
    np.testing.assert_allclose(got, 1e-12 * glob)


def test_shrink_one_pins_sigma2_at_global():
    panel, _ = shrinkage_panel(seed=5)
    bg = fit_did_background(panel, model="normal", degree=1, unit_effects=True, shrink=1.0)
    glob = float(np.var(bg.r))
    np.testing.assert_allclose(bg.sigma2, np.full(panel.n, glob))


# ---------------------------------------------------------------------------
# model="auto" dispatch and the Bernoulli path
# ---------------------------------------------------------------------------


def binary_panel(seed: int = 6) -> CategoricalPanel:
    rng = np.random.default_rng(seed)
    n_units, n_periods = 4, 25
    unit = np.repeat(np.arange(n_units), n_periods)
    t = np.tile(np.arange(n_periods, dtype=float), n_units)
    p = 1.0 / (1.0 + np.exp(-(0.5 * unit - 1.0 + 0.05 * t)))
    theta = (rng.random(unit.size) < p).astype(float)
    return make_panel((unit % 2).astype(np.int64), t, theta, unit=unit)


def test_auto_dispatches_bernoulli_for_binary_theta():
    panel = binary_panel()
    bg = fit_did_background(panel, model="auto")
    assert bg.kind == "bernoulli"
    assert bg.r is None and bg.sigma2 is None
    assert bg.eta is not None and bg.eta.shape == (panel.n,)
    assert np.all(np.isfinite(bg.eta))
    assert np.all((bg.fitted > 0.0) & (bg.fitted < 1.0))


def test_auto_dispatches_normal_for_continuous_theta():
    panel, _ = planted_panel(seed=7)
    bg = fit_did_background(panel, model="auto")
    assert bg.kind == "normal"
    assert bg.eta is None
    assert bg.r is not None and bg.sigma2 is not None


def test_unknown_model_raises():
    panel, _ = planted_panel(seed=8)
    with pytest.raises(ValueError, match="model"):
        fit_did_background(panel, model="poisson")


# ---------------------------------------------------------------------------
# determinism
# ---------------------------------------------------------------------------


def test_normal_path_deterministic():
    panel, _ = planted_panel(seed=9)
    a = fit_did_background(panel, model="normal")
    b = fit_did_background(panel, model="normal")
    np.testing.assert_array_equal(a.fitted, b.fitted)
    np.testing.assert_array_equal(a.r, b.r)
    np.testing.assert_array_equal(a.sigma2, b.sigma2)


def test_bernoulli_path_deterministic():
    panel = binary_panel(seed=10)
    a = fit_did_background(panel, model="bernoulli")
    b = fit_did_background(panel, model="bernoulli")
    np.testing.assert_array_equal(a.fitted, b.fitted)
    np.testing.assert_array_equal(a.eta, b.eta)


def test_issue_5_no_iprint_optimize_warning_from_bernoulli_background():
    """Issue #5, DiD fit site: the sklearn/scipy 'iprint' OptimizeWarning must
    be suppressed in fit_did_background's logistic fit as well (it fires per
    replica in the panel randomization test)."""
    import warnings

    panel = binary_panel()
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        fit_did_background(panel, model="bernoulli")
    assert not [w for w in rec if "iprint" in str(w.message)]
