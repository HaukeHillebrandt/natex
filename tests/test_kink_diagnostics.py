"""Paper-prescribed diagnostic battery for known-cutoff kink designs.

Covers the validation battery of IZA DP 18313: bandwidth/donut sensitivity
grids and placebo kinks (Fig. A3), event-study kink contrasts (Fig. A2 D),
covariate kink regressions and the binned pre/post density-difference kink
test (Fig. A4, Table A3).

Stochastic calibration (seed ranges observed during implementation are
recorded in each test's docstring per the repo statistical-test policy).
"""

import numpy as np
import pytest

from natex.kink import (
    covariate_kinks,
    density_kink_difference,
    event_study_kinks,
    placebo_kinks,
    sensitivity_grid,
)


def _grid(n_side: int = 30) -> np.ndarray:
    left = np.linspace(-1.0, -0.02, n_side)
    right = np.linspace(0.02, 1.0, n_side)
    return np.r_[left, right]


def _kinked_density_sample(rng, n: int, slope: float) -> np.ndarray:
    """Accept-reject draws on [-1, 1] from f(x) proportional to 1 + slope*max(x, 0)."""
    out = np.empty(0)
    bound = 1.0 + slope
    while out.size < n:
        proposal = rng.uniform(-1.0, 1.0, 4 * n)
        accept = rng.uniform(0.0, bound, 4 * n) <= 1.0 + slope * np.maximum(proposal, 0.0)
        out = np.r_[out, proposal[accept]]
    return out[:n]


def test_sensitivity_grid_recovers_tau_at_every_bandwidth_and_donut():
    v = _grid(40)
    b = 0.3 * v + 0.9 * np.maximum(v, 0.0)
    y = 2.0 * b + 0.1 * v

    grid = sensitivity_grid(
        y,
        v,
        bandwidths=(0.6, 0.8, 1.0),
        donuts=(0.0, 0.1),
        policy_kink=0.9,
        kernel="uniform",
    )

    assert len(grid) == 6
    combos = {(est.extras["bandwidth"], est.extras["donut"]) for est in grid}
    assert combos == {(h, d) for h in (0.6, 0.8, 1.0) for d in (0.0, 0.1)}
    for est in grid:
        assert est.tau == pytest.approx(2.0, abs=1e-9)


def test_sensitivity_grid_rejects_invalid_combinations_and_mismatched_designs():
    v = _grid(10)
    y = v.copy()
    with pytest.raises(ValueError, match="bandwidth/donut"):
        sensitivity_grid(y, v, bandwidths=(0.5, 1.0), donuts=(0.6,), policy_kink=0.9)
    with pytest.raises(ValueError, match="policy_kink_change"):
        sensitivity_grid(y, v, bandwidths=(1.0,), policy_kink_change=0.9)
    post = np.repeat([0, 1], v.size // 2)
    with pytest.raises(ValueError, match="policy_kink "):
        sensitivity_grid(y, v, bandwidths=(1.0,), post=post, policy_kink=0.9)


def test_placebo_kinks_flag_the_true_cutoff_and_not_the_shifted_ones():
    """Calibration (seeds 0-9, n=6000, sd=0.3, cutoffs +/-0.5, bandwidth
    0.4): placebo p-values span [0.11, 0.97], all > 0.05; true-cutoff p
    <= 5.3e-4 in 10/10 seeds with estimates in [0.47, 0.91] around the true
    0.6. Seed 5 pinned (placebo p 0.90/0.80, true p 1.4e-7, estimate
    0.70)."""
    rng = np.random.default_rng(5)
    x = rng.uniform(-1.0, 1.0, 6000)
    y = 0.6 * np.maximum(x, 0.0) + 0.2 * x + rng.normal(0.0, 0.3, x.size)

    placebo = placebo_kinks(y, x, (-0.5, 0.5), bandwidth=0.4)
    true = placebo_kinks(y, x, (0.0,), bandwidth=0.4)

    assert [p.cutoff for p in placebo.placebos] == [-0.5, 0.5]
    assert placebo.n_evaluated == 2
    assert placebo.n_significant == 0
    assert placebo.empirical_size == 0.0
    for row in placebo.placebos:
        assert row.p_value > 0.05
    assert true.n_significant == 1
    assert true.empirical_size == 1.0
    assert true.placebos[0].estimate == pytest.approx(0.6, abs=0.2)
    assert true.placebos[0].p_value < 1e-5


def test_placebo_kinks_report_unevaluable_cutoffs_as_nan_never_zero():
    x = _grid(20)
    y = 0.4 * x
    grid = placebo_kinks(y, x, (5.0,), bandwidth=0.4)
    row = grid.placebos[0]
    assert np.isnan(row.estimate) and np.isnan(row.p_value)
    assert row.reason is not None
    assert grid.n_evaluated == 0
    assert np.isnan(grid.empirical_size)


def test_covariate_kinks_detect_a_kinked_covariate_and_pass_a_smooth_one():
    """Calibration (seeds 0-9, n=4000, sd=0.2): kinked-covariate p <= 1.6e-77
    with estimates in [0.72, 0.85]; smooth-covariate p spans [0.14, 0.85]
    with |estimate| <= 0.057. Seed 0 pinned (smooth p 0.61, kinked p
    3e-77)."""
    rng = np.random.default_rng(0)
    x = rng.uniform(-1.0, 1.0, 4000)
    smooth = 0.5 * x + rng.normal(0.0, 0.2, x.size)
    kinked = 0.8 * np.maximum(x, 0.0) + rng.normal(0.0, 0.2, x.size)

    rows = covariate_kinks({"smooth": smooth, "kinked": kinked}, x, bandwidth=0.8)

    by_name = {row.name: row for row in rows}
    assert set(by_name) == {"smooth", "kinked"}
    assert by_name["kinked"].estimate == pytest.approx(0.8, abs=0.15)
    assert by_name["kinked"].p_value < 1e-8
    assert abs(by_name["smooth"].estimate) < 0.15
    assert by_name["smooth"].p_value > 0.05


def test_covariate_dik_cancels_a_time_stable_covariate_kink_exactly():
    v0 = _grid(30)
    v = np.tile(v0, 2)
    post = np.repeat([False, True], v0.size)
    stable = 0.7 * np.maximum(v, 0.0) + 0.1 * v

    rows = covariate_kinks({"stable": stable}, v, post=post, bandwidth=1.0, kernel="uniform")

    assert rows[0].estimate == pytest.approx(0.0, abs=1e-10)


def test_covariate_kinks_require_a_named_mapping():
    v = _grid(10)
    with pytest.raises(ValueError, match="mapping"):
        covariate_kinks(v, v, bandwidth=1.0)


def test_event_study_kinks_recover_per_period_contrasts_exactly():
    v0 = _grid(30)
    v = np.tile(v0, 3)
    period = np.repeat([2008, 2009, 2010], v0.size)
    kink_by_period = np.where(period == 2010, 0.9, 0.2)
    y = kink_by_period * np.maximum(v, 0.0) + 0.4 * v + 0.3 * (period - 2008)

    study = event_study_kinks(y, v, period, base_period=2009, bandwidth=1.0, kernel="uniform")

    assert study.base_period == 2009
    assert [k.period for k in study.kinks] == [2008, 2010]
    pre, post = study.kinks
    assert pre.estimate == pytest.approx(0.0, abs=1e-10)
    assert post.estimate == pytest.approx(0.7, abs=1e-10)
    # Exact data: zero contrast with zero SE has an undefined t statistic
    # (NaN, never a fabricated significance), a nonzero one is exact.
    assert np.isnan(pre.p_value)
    assert post.p_value == 0.0
    assert post.ci == pytest.approx((0.7, 0.7), abs=1e-10)


def test_event_study_requires_the_base_period_to_exist():
    v = _grid(10)
    period = np.repeat([0, 1], v.size // 2)
    with pytest.raises(ValueError, match="base_period"):
        event_study_kinks(v, v, period, base_period=7, bandwidth=1.0)


def test_density_kink_difference_is_null_for_a_stable_density():
    """Calibration (seeds 0-9, n=20000/period, defaults n_bins=80, degree=2):
    null p spans [0.096, 0.996]; seed 1 pinned (p=0.83). The paper's Table
    A3 spec (degree=13) over-rejects under the null (4/10 seeds < 0.05),
    which is why degree=2 is the default; the override must still run."""
    rng = np.random.default_rng(1)
    pre = rng.uniform(-1.0, 1.0, 20000)
    post_draws = rng.uniform(-1.0, 1.0, 20000)
    running = np.r_[pre, post_draws]
    post = np.repeat([False, True], 20000)

    result = density_kink_difference(running, post, bandwidth=1.0)

    assert result.n_bins == 80
    assert result.degree == 2
    assert result.n_pre == 20000 and result.n_post == 20000
    assert result.bin_centers.size == 80
    assert np.isfinite(result.density_difference).all()
    assert result.p_value > 0.05

    paper_spec = density_kink_difference(running, post, bandwidth=1.0, degree=13)
    assert paper_spec.degree == 13
    assert np.isfinite(paper_spec.estimate) and np.isfinite(paper_spec.se)


def test_density_kink_difference_detects_a_post_period_density_kink():
    """Calibration (seeds 0-9, slope=3.0, n=20000/period, defaults): p <=
    1.3e-3 in 10/10 seeds with estimates in [0.52, 1.07] around the analytic
    kink 3/3.5 = 0.857; seed 0 pinned (p=1e-10, estimate 1.00)."""
    rng = np.random.default_rng(0)
    pre = rng.uniform(-1.0, 1.0, 20000)
    post_draws = _kinked_density_sample(rng, 20000, slope=3.0)
    running = np.r_[pre, post_draws]
    post = np.repeat([False, True], 20000)

    result = density_kink_difference(running, post, bandwidth=1.0)

    assert result.estimate == pytest.approx(3.0 / 3.5, abs=0.3)
    assert result.estimate > 0.0
    assert result.p_value < 1e-8


def test_density_kink_difference_validates_bins_and_reports_empty_windows():
    x = _grid(20)
    post = np.repeat([0, 1], x.size // 2)
    with pytest.raises(ValueError, match="even"):
        density_kink_difference(x, post, bandwidth=1.0, n_bins=39)
    with pytest.raises(ValueError, match="residual"):
        density_kink_difference(x, post, bandwidth=1.0, n_bins=8, degree=3)

    far_pre = np.r_[np.full(20, 5.0), _grid(10)]
    far_post = np.r_[np.zeros(20, dtype=bool), np.ones(20, dtype=bool)]
    result = density_kink_difference(far_pre, far_post, bandwidth=1.0, n_bins=8, degree=1)
    assert result.n_pre == 0
    assert np.isnan(result.estimate) and np.isnan(result.p_value)
    assert "pre" in result.reason
