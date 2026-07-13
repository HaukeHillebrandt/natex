"""Known-cutoff regression-kink and difference-in-kinks estimators."""

import numpy as np
import pytest
from scipy import stats

from natex.kink import difference_in_kinks, regression_kink
from natex.kink.estimate import _fieller


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

    tiny_units = regression_kink(
        1e-200 * y,
        v,
        policy_kink=1e-200 * kink,
        cutoff=0.0,
        bandwidth=100.0,
        kernel="uniform",
    )
    assert tiny_units.tau == pytest.approx(tau, rel=1e-10)
    assert tiny_units.se == 0.0

    extreme_confidence = regression_kink(
        y,
        v,
        policy_kink=kink,
        cutoff=0.0,
        bandwidth=100.0,
        kernel="uniform",
        alpha=1e-16,
    )
    assert np.isfinite(extreme_confidence.extras["critical_value"])
    assert np.isfinite(extreme_confidence.ci).all()


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


def test_nonfinite_numeric_cluster_rows_are_dropped_and_counted():
    v = _grid(20)
    y = 0.9 * np.maximum(v, 0.0) + 0.2 * v
    clusters = np.arange(v.size, dtype=float)
    clusters[0] = np.inf

    est = regression_kink(
        y,
        v,
        policy_kink=0.9,
        bandwidth=1.0,
        kernel="uniform",
        clusters=clusters,
    )

    assert est.n_used == v.size - 1
    assert est.extras["n_dropped_nonfinite"] == 1


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


def test_cluster_robust_inference_requires_two_clusters_in_every_cell():
    v = _grid(5)
    clusters = np.r_[np.zeros(5), [0, 1, 0, 1, 0]]
    y = 0.7 * np.maximum(v, 0.0) + 0.2 * v

    est = regression_kink(
        y,
        v,
        policy_kink=0.7,
        bandwidth=1.0,
        kernel="uniform",
        clusters=clusters,
    )

    assert np.isnan(est.tau)
    assert est.extras["n_clusters_by_cell"] == {"left": 1, "right": 2}
    assert "two clusters in every cell" in est.extras["reason"]


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

    running = np.tile(v, 2)
    post = np.repeat([0.0, 1.0], v.size)
    post[0] = np.inf
    y = 0.5 * running + np.tile([0.2, 0.8], v.size) * np.maximum(running, 0.0)
    est = difference_in_kinks(
        y,
        running,
        post,
        policy_kink_change=0.6,
        bandwidth=1.0,
        kernel="uniform",
    )
    assert est.n_used == running.size - 1
    assert est.extras["n_dropped_nonfinite"] == 1


def test_fuzzy_fieller_and_first_stage_are_invariant_to_running_units():
    rng = np.random.default_rng(813)
    x = rng.uniform(-1.0, 1.0, 2000)
    treatment = 0.4 * x + 0.8 * np.maximum(x, 0.0) + rng.normal(0.0, 0.3, x.size)
    y = 1.7 * treatment + 0.2 * x + rng.normal(0.0, 0.5, x.size)
    base = regression_kink(
        y, x, treatment=treatment, bandwidth=0.8, kernel="epanechnikov"
    )

    for factor in (1e3, 1e6, 1e12):
        scaled = regression_kink(
            y,
            factor * x,
            treatment=treatment,
            bandwidth=factor * 0.8,
            kernel="epanechnikov",
        )
        assert scaled.tau == pytest.approx(base.tau, rel=1e-11)
        assert scaled.se == pytest.approx(base.se, rel=1e-11)
        assert scaled.first_stage_F == pytest.approx(base.first_stage_F, rel=1e-10)
        assert scaled.fieller_kind == base.fieller_kind == "interval"
        assert scaled.fieller_ci == pytest.approx(base.fieller_ci, rel=1e-10)


def test_fuzzy_fieller_is_equivariant_to_outcome_and_treatment_units():
    rng = np.random.default_rng(813)
    x = rng.uniform(-1.0, 1.0, 2000)
    treatment = 0.4 * x + 0.8 * np.maximum(x, 0.0) + rng.normal(0.0, 0.3, x.size)
    y = 1.7 * treatment + 0.2 * x + rng.normal(0.0, 0.5, x.size)
    base = regression_kink(
        y, x, treatment=treatment, bandwidth=0.8, kernel="epanechnikov"
    )

    for outcome_factor, treatment_factor in ((1e6, 1.0), (1.0, 1e-6)):
        scaled = regression_kink(
            outcome_factor * y,
            x,
            treatment=treatment_factor * treatment,
            bandwidth=0.8,
            kernel="epanechnikov",
        )
        ratio_factor = outcome_factor / treatment_factor
        assert scaled.tau == pytest.approx(ratio_factor * base.tau, rel=1e-11)
        assert scaled.se == pytest.approx(ratio_factor * base.se, rel=1e-11)
        assert scaled.first_stage_F == pytest.approx(base.first_stage_F, rel=1e-10)
        assert scaled.fieller_kind == base.fieller_kind == "interval"
        assert scaled.fieller_ci == pytest.approx(
            tuple(ratio_factor * endpoint for endpoint in base.fieller_ci),
            rel=1e-10,
        )

    for common_factor in (1e-200, 1e155):
        scaled = regression_kink(
            common_factor * y,
            x,
            treatment=common_factor * treatment,
            bandwidth=0.8,
            kernel="epanechnikov",
        )
        assert scaled.tau == pytest.approx(base.tau, rel=1e-10)
        assert scaled.se == pytest.approx(base.se, rel=1e-10)
        assert scaled.first_stage_F == pytest.approx(base.first_stage_F, rel=1e-10)
        assert scaled.fieller_kind == base.fieller_kind == "interval"
        assert scaled.fieller_ci == pytest.approx(base.fieller_ci, rel=1e-10)


def test_exact_tiny_but_nonzero_fuzzy_kink_is_not_canonicalized_away():
    x = _grid(30)
    treatment = 5e-13 * np.maximum(x, 0.0)
    y = 2.0 * treatment + 0.1 * x
    est = regression_kink(
        y, x, treatment=treatment, bandwidth=1.0, kernel="uniform"
    )
    assert est.first_stage == pytest.approx(5e-13, rel=1e-10)
    assert np.isinf(est.first_stage_F)
    # The identifying kink is roughly 1e-11 of the common slope, so the
    # remaining tolerance reflects float64 representation of the generated y.
    assert est.tau == pytest.approx(2.0, rel=1e-4)


def test_covariate_adjustment_is_invariant_to_covariate_units():
    x = _grid(50)
    z = np.exp(x)
    treatment = _policy(x, base_slope=0.2, kink=0.7)
    y = 3.0 * treatment + 4.0 * z
    base = regression_kink(
        y,
        x,
        policy_kink=0.7,
        bandwidth=1.0,
        kernel="uniform",
        covariates=z,
    )
    scaled = regression_kink(
        y,
        x,
        policy_kink=0.7,
        bandwidth=1.0,
        kernel="uniform",
        covariates=1e-16 * z,
    )
    shifted = regression_kink(
        y,
        x,
        policy_kink=0.7,
        bandwidth=1.0,
        kernel="uniform",
        covariates=z + 4e13,
    )
    assert scaled.tau == pytest.approx(base.tau, abs=1e-9)
    assert scaled.extras["n_covariates"] == 1
    assert shifted.tau == pytest.approx(base.tau, abs=0.05)
    assert shifted.extras["n_covariates"] == 1


def test_hc1_se_scales_with_outcome_units_instead_of_collapsing_to_zero():
    rng = np.random.default_rng(14)
    x = rng.uniform(-1.0, 1.0, 1000)
    y = 0.6 * np.maximum(x, 0.0) + 0.2 * x + rng.normal(0.0, 0.3, x.size)
    base = regression_kink(y, x, policy_kink=0.6, bandwidth=0.8)
    scaled = regression_kink(1e-13 * y, x, policy_kink=0.6, bandwidth=0.8)
    shifted = regression_kink(y + 1e13, x, policy_kink=0.6, bandwidth=0.8)
    assert scaled.tau == pytest.approx(1e-13 * base.tau, rel=1e-10, abs=1e-25)
    assert scaled.se == pytest.approx(1e-13 * base.se, rel=1e-10, abs=1e-25)
    assert scaled.se > 0.0
    assert shifted.tau == pytest.approx(base.tau, rel=1e-2)
    assert shifted.se == pytest.approx(base.se, rel=1e-2)


def test_fuzzy_ratio_se_uses_stable_combined_influence_scores():
    rng = np.random.default_rng(123)
    x = rng.uniform(-1.0, 1.0, 4000)
    raw = 0.4 * x + 0.8 * np.maximum(x, 0.0) + rng.normal(0.0, 0.3, x.size)
    treatment = 1e8 * raw
    y = treatment + 0.2 * x + rng.normal(0.0, 0.5, x.size)
    fuzzy = regression_kink(y, x, treatment=treatment, bandwidth=0.8)
    transformed = regression_kink(
        y - fuzzy.tau * treatment,
        x,
        policy_kink=fuzzy.first_stage,
        bandwidth=0.8,
    )
    oracle_se = transformed.reduced_form_se / abs(fuzzy.first_stage)
    assert fuzzy.se == pytest.approx(oracle_se, rel=1e-8)
    assert fuzzy.se > 0.0


def test_clustered_intervals_use_cluster_degrees_of_freedom():
    rng = np.random.default_rng(15)
    clusters = np.repeat(np.arange(4), 50)
    x = rng.uniform(-1.0, 1.0, clusters.size)
    y = 0.8 * np.maximum(x, 0.0) + rng.normal(size=x.size)
    est = regression_kink(
        y,
        x,
        policy_kink=0.8,
        bandwidth=1.0,
        kernel="uniform",
        clusters=clusters,
    )
    critical = (est.ci[1] - est.tau) / est.se
    assert critical == pytest.approx(stats.t.ppf(0.975, 3), rel=1e-10)
    assert est.extras["critical_df"] == 3


def test_clustered_fuzzy_fieller_uses_the_same_t_critical_value():
    rng = np.random.default_rng(0)
    clusters = np.repeat(np.arange(4), 100)
    x = rng.uniform(-1.0, 1.0, clusters.size)
    treatment = (
        0.2 * x
        + 2.0 * np.maximum(x, 0.0)
        + rng.normal(0.0, 0.1, x.size)
    )
    y = 1.5 * treatment + 0.1 * x + rng.normal(0.0, 0.1, x.size)
    est = regression_kink(
        y,
        x,
        treatment=treatment,
        bandwidth=1.0,
        kernel="uniform",
        clusters=clusters,
    )

    assert est.fieller_kind == "interval"
    critical_squared = stats.t.ppf(0.975, 3) ** 2
    var_outcome = est.reduced_form_se**2
    var_first_stage = est.first_stage_se**2
    covariance = est.extras["outcome_first_stage_covariance"]
    for endpoint in est.fieller_ci:
        equation = (est.reduced_form - endpoint * est.first_stage) ** 2
        equation -= critical_squared * (
            var_outcome
            + endpoint**2 * var_first_stage
            - 2.0 * endpoint * covariance
        )
        assert equation == pytest.approx(0.0, abs=1e-12)


def test_fieller_handles_exact_zero_numerator_and_denominator_limits():
    critical = stats.norm.ppf(0.975)

    strong_zero_numerator = _fieller(0.0, 2.0, 0.0, 0.1, 0.0, critical)
    denominator_zero_numerator_excludes_zero = _fieller(
        1.0, 0.0, 0.1, 0.0, 0.0, critical
    )
    denominator_zero_numerator_includes_zero = _fieller(
        0.1, 0.0, 0.1, 0.0, 0.0, critical
    )
    both_exactly_zero = _fieller(0.0, 0.0, 0.0, 0.0, 0.0, critical)
    tiny_strong_zero_numerator = _fieller(
        0.0, 2e-200, 0.0, 0.0, 0.0, critical
    )
    tiny_nonzero_numerator_zero_denominator = _fieller(
        1e-200, 0.0, 0.0, 0.0, 0.0, critical
    )

    assert strong_zero_numerator[:2] == ("interval", (0.0, 0.0))
    assert denominator_zero_numerator_excludes_zero[0] == "empty"
    assert denominator_zero_numerator_includes_zero[0] == "unbounded"
    assert both_exactly_zero[0] == "unbounded"
    assert tiny_strong_zero_numerator[:2] == ("interval", (0.0, 0.0))
    assert tiny_nonzero_numerator_zero_denominator[0] == "empty"


def test_saturated_cell_is_reported_as_insufficient_for_inference():
    x = np.r_[[-0.8, -0.2], np.linspace(0.1, 1.0, 10)]
    y = 0.4 * x + np.maximum(x, 0.0)
    est = regression_kink(
        y, x, policy_kink=1.0, bandwidth=1.0, kernel="uniform"
    )
    assert np.isnan(est.tau)
    assert "residual degrees of freedom" in est.extras["reason"]


def test_zero_weight_kernel_endpoint_does_not_supply_residual_degrees_of_freedom():
    x = np.array([-1.0, -0.8, -0.2, 0.2, 0.6, 0.8])
    y = 0.4 * x + np.maximum(x, 0.0)
    est = regression_kink(
        y, x, policy_kink=1.0, bandwidth=1.0, kernel="triangular"
    )

    assert est.n_by_cell["left"] == 2
    assert est.extras["n_outside_bandwidth"] == 0
    assert est.extras["n_zero_weight_excluded"] == 1
    assert np.isnan(est.tau)
    assert "residual degrees of freedom" in est.extras["reason"]
