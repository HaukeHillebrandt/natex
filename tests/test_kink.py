"""Known-cutoff regression-kink and difference-in-kinks estimators."""

import numpy as np
import pytest

from natex.kink import difference_in_kinks, regression_kink


def _grid(n_side: int = 30) -> np.ndarray:
    left = np.linspace(-1.0, -0.02, n_side)
    right = np.linspace(0.02, 1.0, n_side)
    return np.r_[left, right]


def _policy(v: np.ndarray, base_slope: float, kink: float) -> np.ndarray:
    return base_slope * v + kink * np.maximum(v, 0.0)


def test_sharp_rkd_exact_slope_ratio_in_original_units():
    v = 100.0 * _grid()
    kink = -0.4
    tau = 2.5
    b = _policy(v, base_slope=0.7, kink=kink)
    y = tau * b + 3.0 + 0.2 * v

    est = regression_kink(
        y,
        v,
        policy_kink=kink,
        cutoff=0.0,
        bandwidth=100.0,
        kernel="uniform",
    )

    assert est.method == "sharp_rkd"
    assert est.reduced_form == pytest.approx(tau * kink, abs=1e-10)
    assert est.first_stage == pytest.approx(kink, abs=1e-12)
    assert est.tau == pytest.approx(tau, abs=1e-10)
    assert est.n_by_cell == {"left": 30, "right": 30}
    assert est.weak_first_stage is False


def test_fuzzy_rkd_exact_first_stage_and_fieller_point_set():
    v = _grid()
    b = _policy(v, base_slope=0.4, kink=0.8)
    y = 1.75 * b - 0.3 * v

    est = regression_kink(
        y,
        v,
        treatment=b,
        bandwidth=1.0,
        kernel="uniform",
    )

    assert est.method == "fuzzy_rkd"
    assert est.tau == pytest.approx(1.75, abs=1e-10)
    assert est.first_stage == pytest.approx(0.8, abs=1e-10)
    assert np.isinf(est.first_stage_F)
    assert est.weak_first_stage is False
    assert est.fieller_kind == "interval"
    assert est.fieller_ci == pytest.approx((1.75, 1.75), abs=1e-7)


def test_sharp_dik_cancels_a_time_invariant_confounding_kink():
    v0 = _grid(40)
    v = np.tile(v0, 2)
    post = np.repeat([False, True], v0.size)
    kink_pre, kink_post = 0.3, 1.1
    b = np.where(
        post,
        _policy(v, base_slope=-0.2, kink=kink_post),
        _policy(v, base_slope=0.4, kink=kink_pre),
    )
    tau = 2.0
    stable_bias_kink = 0.9 * np.maximum(v, 0.0)
    y = tau * b + stable_bias_kink + 0.5 * v + post.astype(float)

    est = difference_in_kinks(
        y,
        v,
        post,
        policy_kink_change=kink_post - kink_pre,
        bandwidth=1.0,
        kernel="uniform",
    )
    post_rkd = regression_kink(
        y[post],
        v[post],
        policy_kink=kink_post,
        bandwidth=1.0,
        kernel="uniform",
    )

    assert est.method == "sharp_dik"
    assert est.tau == pytest.approx(tau, abs=1e-10)
    assert est.reduced_form == pytest.approx(tau * (kink_post - kink_pre), abs=1e-10)
    assert abs(post_rkd.tau - tau) > 0.5
    assert est.n_by_cell == {
        "pre_left": 40,
        "pre_right": 40,
        "post_left": 40,
        "post_right": 40,
    }


def test_fuzzy_dik_is_the_ratio_of_post_minus_pre_slope_kinks():
    v0 = _grid(35)
    v = np.tile(v0, 2)
    post = np.repeat([False, True], v0.size)
    # Both sides change over time; there is no clean, fixed-slope side.
    left_slope = np.where(post, -0.1, 0.5)
    kink = np.where(post, 1.4, 0.2)
    b = left_slope * v + kink * np.maximum(v, 0.0)
    y = -1.25 * b + 0.8 * np.maximum(v, 0.0) + np.where(post, 0.2 * v, -0.4 * v)

    est = difference_in_kinks(
        y,
        v,
        post,
        treatment=b,
        bandwidth=1.0,
        kernel="uniform",
    )

    assert est.method == "fuzzy_dik"
    assert est.first_stage == pytest.approx(1.2, abs=1e-10)
    assert est.tau == pytest.approx(-1.25, abs=1e-10)
    assert est.extras["first_stage_kinks"] == pytest.approx({"pre": 0.2, "post": 1.4})


def test_numeric_covariates_are_adjusted_without_changing_the_target():
    v = _grid(50)
    z = np.exp(v)
    b = _policy(v, base_slope=0.2, kink=0.7)
    y = 3.0 * b + 4.0 * z

    unadjusted = regression_kink(
        y, v, policy_kink=0.7, bandwidth=1.0, kernel="uniform"
    )
    adjusted = regression_kink(
        y,
        v,
        policy_kink=0.7,
        bandwidth=1.0,
        kernel="uniform",
        covariates=z,
    )

    assert abs(unadjusted.tau - 3.0) > 0.2
    assert adjusted.tau == pytest.approx(3.0, abs=1e-10)


def test_nonfinite_rows_are_dropped_only_when_used_and_counted():
    v = _grid(20)
    b = _policy(v, base_slope=0.3, kink=0.9)
    y = 2.0 * b
    z = np.ones(v.size)
    y[0] = np.nan
    b[1] = np.inf
    z[2] = np.nan

    sharp = regression_kink(
        y, v, policy_kink=0.9, bandwidth=1.0, kernel="uniform"
    )
    fuzzy = regression_kink(
        y, v, treatment=b, bandwidth=1.0, kernel="uniform", covariates=z
    )

    assert sharp.n_used == v.size - 1
    assert sharp.extras["n_dropped_nonfinite"] == 1
    assert fuzzy.n_used == v.size - 3
    assert fuzzy.extras["n_dropped_nonfinite"] == 3


def test_cluster_robust_dik_reports_cluster_count():
    rng = np.random.default_rng(4)
    n_units = 100
    unit = np.repeat(np.arange(n_units), 2)
    post = np.tile([False, True], n_units)
    v = np.repeat(rng.uniform(-1.0, 1.0, n_units), 2)
    b = 0.3 * v + np.where(post, 1.0, 0.2) * np.maximum(v, 0.0)
    unit_shock = np.repeat(rng.normal(scale=0.2, size=n_units), 2)
    y = 1.5 * b + 0.4 * v + unit_shock + rng.normal(scale=0.05, size=v.size)

    est = difference_in_kinks(
        y,
        v,
        post,
        policy_kink_change=0.8,
        bandwidth=1.0,
        clusters=unit,
    )

    assert est.extras["inference"] == "CR1"
    assert est.extras["n_clusters"] == n_units
    assert np.isfinite(est.se) and est.se > 0.0


def test_zero_fuzzy_first_stage_is_nan_never_zero_and_flagged_weak():
    v = _grid(30)
    treatment = 0.4 * v
    y = np.maximum(v, 0.0)

    est = regression_kink(
        y,
        v,
        treatment=treatment,
        bandwidth=1.0,
        kernel="uniform",
    )

    assert abs(est.first_stage) < 1e-12
    assert np.isnan(est.tau) and est.tau != 0.0
    assert est.weak_first_stage is True
    assert est.fieller_kind in {"empty", "unbounded", "disjoint"}


def test_underdetermined_cell_returns_nan_with_reason():
    v = np.array([-0.5, -0.25, 0.25, 0.5])
    y = v.copy()
    est = regression_kink(
        y,
        v,
        policy_kink=1.0,
        bandwidth=1.0,
        degree=2,
        kernel="uniform",
    )
    assert np.isnan(est.tau)
    assert "underdetermined" in est.extras["reason"]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"bandwidth": 0.0}, "bandwidth"),
        ({"bandwidth": 1.0, "degree": 0}, "degree"),
        ({"bandwidth": 1.0, "kernel": "gaussian"}, "kernel"),
        ({"bandwidth": 1.0, "donut": 1.0}, "donut"),
        ({"bandwidth": 1.0, "policy_kink": 0.0}, "policy_kink"),
    ],
)
def test_rkd_rejects_invalid_design_arguments(kwargs, message):
    v = _grid(5)
    y = v.copy()
    policy = kwargs.pop("policy_kink", 1.0)
    with pytest.raises(ValueError, match=message):
        regression_kink(y, v, policy_kink=policy, **kwargs)


def test_exactly_one_sharp_or_fuzzy_first_stage_must_be_supplied():
    v = _grid(5)
    y = v.copy()
    with pytest.raises(ValueError, match="exactly one"):
        regression_kink(y, v, bandwidth=1.0)
    with pytest.raises(ValueError, match="exactly one"):
        regression_kink(y, v, treatment=v, policy_kink=1.0, bandwidth=1.0)


def test_dik_requires_both_pre_and_post_rows():
    v = _grid(5)
    with pytest.raises(ValueError, match="pre and post"):
        difference_in_kinks(
            v,
            v,
            np.ones(v.size, dtype=bool),
            policy_kink_change=1.0,
            bandwidth=1.0,
        )
