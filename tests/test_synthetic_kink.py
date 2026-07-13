"""Seeded synthetic RKD/DiK data and analytic estimand oracles."""

import numpy as np
import pytest

from natex.data.synthetic_kink import make_dik_synthetic, make_rkd_synthetic
from natex.kink import difference_in_kinks, regression_kink


_RKD_FINITE_SCALARS = (
    "tau",
    "cutoff",
    "policy_intercept",
    "policy_left_slope",
    "policy_kink",
    "bias_kink",
    "outcome_left_slope",
    "level_jump",
    "treatment_noise",
    "outcome_noise",
)

_DIK_FINITE_SCALARS = (
    "tau",
    "cutoff",
    "policy_intercept",
    "policy_left_slope_pre",
    "policy_left_slope_post",
    "policy_kink_pre",
    "policy_kink_post",
    "bias_kink_pre",
    "bias_kink_post",
    "outcome_left_slope_pre",
    "outcome_left_slope_post",
    "level_jump",
    "treatment_noise",
    "outcome_noise",
)


def test_rkd_synthetic_is_deterministic_and_wired_as_a_dataset():
    a, truth_a = make_rkd_synthetic(n=200, rng=np.random.default_rng(7))
    b, truth_b = make_rkd_synthetic(n=200, rng=np.random.default_rng(7))

    assert a.df.equals(b.df)
    assert truth_a == truth_b
    assert a.spec.treatment == "policy"
    assert a.spec.outcome == "y"
    assert a.spec.forcing == ["running"]
    assert a.spec.covariates == ["running"]
    assert a.spec.time is None


def test_dik_synthetic_is_balanced_and_carries_period_role():
    data, truth = make_dik_synthetic(n=202, rng=np.random.default_rng(8))

    counts = data.df["post"].value_counts().to_dict()
    assert counts == {0: 101, 1: 101}
    assert data.spec.time == "post"
    assert truth.policy_kink_change == pytest.approx(
        truth.policy_kink_post - truth.policy_kink_pre
    )
    assert truth.expected_dik == pytest.approx(truth.tau)


@pytest.mark.parametrize("fuzzy", [False, True])
def test_rkd_synthetic_recovers_analytic_estimand(fuzzy):
    # Calibrated at seeds 100..109 with this n/h: local-linear errors span
    # [-0.104, 0.122] sharp and [-0.176, 0.151] fuzzy; 19/20 Wald CIs cover.
    data, truth = make_rkd_synthetic(
        n=8000,
        fuzzy=fuzzy,
        bias_kink=0.25,
        outcome_noise=0.35,
        rng=np.random.default_rng(20 + fuzzy),
    )
    kwargs = (
        {"treatment": data.T}
        if fuzzy
        else {"policy_kink": truth.policy_kink}
    )
    est = regression_kink(
        data.y,
        data.df["running"].to_numpy(),
        bandwidth=0.65,
        degree=1,
        **kwargs,
    )

    assert abs(est.tau - truth.expected_rkd) < 0.2
    assert est.ci[0] < truth.expected_rkd < est.ci[1]
    assert est.weak_first_stage is False


@pytest.mark.parametrize("fuzzy", [False, True])
def test_dik_synthetic_cancels_stable_bias_kink(fuzzy):
    # Calibrated at seeds 200..209 with this n/h: local-linear errors span
    # [-0.139, 0.257] sharp and [-0.172, 0.284] fuzzy; all 20 CIs cover.
    data, truth = make_dik_synthetic(
        n=12000,
        fuzzy=fuzzy,
        bias_kink_pre=0.8,
        bias_kink_post=0.8,
        outcome_noise=0.4,
        rng=np.random.default_rng(30 + fuzzy),
    )
    kwargs = (
        {"treatment": data.T}
        if fuzzy
        else {"policy_kink_change": truth.policy_kink_change}
    )
    est = difference_in_kinks(
        data.y,
        data.df["running"].to_numpy(),
        data.df["post"].to_numpy(),
        bandwidth=0.7,
        degree=1,
        **kwargs,
    )

    assert abs(est.tau - truth.expected_dik) < 0.3
    assert est.ci[0] < truth.expected_dik < est.ci[1]
    assert abs(truth.expected_rkd_post - truth.tau) > 0.5


def test_dik_oracle_exposes_parallel_kink_trends_violation_bias():
    _, truth = make_dik_synthetic(
        n=100,
        tau=1.5,
        policy_kink_pre=0.2,
        policy_kink_post=1.0,
        bias_kink_pre=0.4,
        bias_kink_post=0.8,
        rng=np.random.default_rng(9),
    )
    assert truth.expected_dik == pytest.approx(1.5 + (0.8 - 0.4) / (1.0 - 0.2))


def test_fuzzy_data_adds_policy_noise_but_preserves_schedule_kink():
    sharp, truth = make_rkd_synthetic(
        n=500, fuzzy=False, rng=np.random.default_rng(10)
    )
    fuzzy, _ = make_rkd_synthetic(
        n=500, fuzzy=True, rng=np.random.default_rng(10)
    )
    running = sharp.df["running"].to_numpy()
    schedule = 0.5 + truth.policy_left_slope * running + truth.policy_kink * np.maximum(
        running - truth.cutoff, 0.0
    )

    assert not np.array_equal(fuzzy.T, sharp.T)
    assert np.std(fuzzy.T - schedule) > 0.1


@pytest.mark.parametrize("maker", [make_rkd_synthetic, make_dik_synthetic])
def test_kink_synthetic_requires_an_explicit_rng(maker):
    with pytest.raises(ValueError, match="Generator"):
        maker(n=100)


@pytest.mark.parametrize("maker", [make_rkd_synthetic, make_dik_synthetic])
def test_kink_synthetic_rejects_the_wrong_rng_type(maker):
    with pytest.raises(TypeError, match="rng must be a numpy Generator"):
        maker(n=100, rng="seed")


@pytest.mark.parametrize("name", _RKD_FINITE_SCALARS)
@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf], ids=["nan", "inf", "neg_inf"])
def test_rkd_synthetic_requires_every_structural_scalar_to_be_finite(name, bad):
    with pytest.raises(ValueError, match=name):
        make_rkd_synthetic(
            n=100,
            rng=np.random.default_rng(0),
            **{name: bad},
        )


@pytest.mark.parametrize("name", _DIK_FINITE_SCALARS)
@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf], ids=["nan", "inf", "neg_inf"])
def test_dik_synthetic_requires_every_structural_scalar_to_be_finite(name, bad):
    with pytest.raises(ValueError, match=name):
        make_dik_synthetic(
            n=100,
            rng=np.random.default_rng(0),
            **{name: bad},
        )


@pytest.mark.parametrize(
    ("policy_kink_pre", "policy_kink_post", "post_rkd_defined"),
    [(0.0, 1.0, True), (1.0, 0.0, False)],
)
def test_dik_synthetic_allows_a_zero_kink_in_either_period(
    policy_kink_pre, policy_kink_post, post_rkd_defined
):
    _, truth = make_dik_synthetic(
        n=100,
        policy_kink_pre=policy_kink_pre,
        policy_kink_post=policy_kink_post,
        rng=np.random.default_rng(0),
    )

    assert truth.policy_kink_change == policy_kink_post - policy_kink_pre
    assert bool(np.isfinite(truth.expected_rkd_post)) is post_rkd_defined


@pytest.mark.parametrize(
    ("maker", "kwargs", "message"),
    [
        (make_rkd_synthetic, {"n": 5}, "n"),
        (make_rkd_synthetic, {"n": 100, "policy_kink": 0.0}, "policy_kink"),
        (
            make_dik_synthetic,
            {"n": 100, "policy_kink_pre": 0.5, "policy_kink_post": 0.5},
            "change",
        ),
        (make_dik_synthetic, {"n": 101}, "even"),
    ],
)
def test_kink_synthetic_validates_design_parameters(maker, kwargs, message):
    with pytest.raises(ValueError, match=message):
        maker(rng=np.random.default_rng(0), **kwargs)
