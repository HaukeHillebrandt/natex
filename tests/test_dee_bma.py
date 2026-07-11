"""Tests for dee/bma.py: model weights (MLL/LOO/buffered stacking) + mixture posterior.

Plan task 5 contract: softmax correctness against hand-set scores, audit-8
mixture covariance formula, the label-per-draw aggregate-variance regression
(>= 1.5x over independent per-point labels), MC agreement with the analytic
mixture, buffered-fold geometry/determinism, stacking model selection on a
constructed scenario, and NaN-not-0.0 degenerate behavior.
"""

import numpy as np
import pytest

from natex.dee.bma import (
    MixturePosterior,
    ModelWeights,
    buffered_folds,
    buffered_stacking_weights,
    loo_weights,
    mixture_posterior,
    mll_weights,
)
from natex.dee.gp import GPPosterior, HeteroskedasticGP


def _tiny_gp(seed=0):
    rng = np.random.default_rng(seed)
    X = np.linspace(0.0, 1.0, 5)[:, None]
    y = rng.normal(size=5)
    return HeteroskedasticGP(
        lengthscale=1.0, outputscale=1.0, mean_const=0.0, X=X, y=y, noise_var=np.full(5, 0.1)
    )


# ------------------------------------------------------------ softmax baselines


def test_mll_weights_softmax_formula(monkeypatch):
    gp_bias, gp_direct = _tiny_gp(0), _tiny_gp(1)
    m1, m2 = -10.3, -11.7
    monkeypatch.setattr(gp_bias, "log_marginal_likelihood", lambda: m1)
    monkeypatch.setattr(gp_direct, "log_marginal_likelihood", lambda: m2)
    w = mll_weights(gp_bias, gp_direct)
    assert isinstance(w, ModelWeights)
    assert w.strategy == "mll"
    expected = np.exp(m1) / (np.exp(m1) + np.exp(m2))
    np.testing.assert_allclose(w.w_debias, expected, rtol=1e-12)


def test_mll_weights_equal_scores_half(monkeypatch):
    gp_bias, gp_direct = _tiny_gp(0), _tiny_gp(1)
    monkeypatch.setattr(gp_bias, "log_marginal_likelihood", lambda: -4.2)
    monkeypatch.setattr(gp_direct, "log_marginal_likelihood", lambda: -4.2)
    assert mll_weights(gp_bias, gp_direct).w_debias == 0.5


def test_loo_weights_softmax_formula(monkeypatch):
    gp_bias, gp_direct = _tiny_gp(0), _tiny_gp(1)
    s1, s2 = -7.25, -3.5
    monkeypatch.setattr(gp_bias, "loo_log_predictive", lambda: s1)
    monkeypatch.setattr(gp_direct, "loo_log_predictive", lambda: s2)
    w = loo_weights(gp_bias, gp_direct)
    assert w.strategy == "loo"
    expected = np.exp(s1) / (np.exp(s1) + np.exp(s2))
    np.testing.assert_allclose(w.w_debias, expected, rtol=1e-12)


def test_softmax_extreme_scores_stable(monkeypatch):
    # naive exp() would overflow/underflow; the weight must still be finite
    gp_bias, gp_direct = _tiny_gp(0), _tiny_gp(1)
    monkeypatch.setattr(gp_bias, "log_marginal_likelihood", lambda: -1e4)
    monkeypatch.setattr(gp_direct, "log_marginal_likelihood", lambda: -2e4)
    w = mll_weights(gp_bias, gp_direct)
    assert np.isfinite(w.w_debias)
    assert w.w_debias > 0.999


def test_softmax_nan_score_gives_nan_never_zero(monkeypatch):
    gp_bias, gp_direct = _tiny_gp(0), _tiny_gp(1)
    monkeypatch.setattr(gp_bias, "log_marginal_likelihood", lambda: float("nan"))
    monkeypatch.setattr(gp_direct, "log_marginal_likelihood", lambda: -3.0)
    assert np.isnan(mll_weights(gp_bias, gp_direct).w_debias)


# ------------------------------------------------------------ mixture posterior


def _two_point_mixture(w=0.3):
    post_a = GPPosterior(
        mean=np.array([1.0, -2.0]), cov=np.array([[0.5, 0.1], [0.1, 0.4]])
    )
    post_b = GPPosterior(
        mean=np.array([3.0, 1.0]), cov=np.array([[0.2, -0.05], [-0.05, 0.3]])
    )
    return post_a, post_b, mixture_posterior(post_a, post_b, w)


def test_mixture_covariance_audit8_formula():
    w = 0.3
    post_a, post_b, mix = _two_point_mixture(w)
    np.testing.assert_allclose(
        mix.mean, w * post_a.mean + (1 - w) * post_b.mean, rtol=1e-12
    )
    d = post_a.mean - post_b.mean
    expected_cov = w * post_a.cov + (1 - w) * post_b.cov + w * (1 - w) * np.outer(d, d)
    np.testing.assert_allclose(mix.cov, expected_cov, rtol=1e-12)
    assert mix.w == w


def test_mixture_posterior_rejects_invalid_w():
    post_a, post_b, _ = _two_point_mixture()
    with pytest.raises(ValueError):
        mixture_posterior(post_a, post_b, 1.5)
    with pytest.raises(ValueError):
        mixture_posterior(post_a, post_b, -0.1)


def test_mixture_posterior_nan_w_gives_nan_never_zero():
    post_a, post_b, _ = _two_point_mixture()
    mix = mixture_posterior(post_a, post_b, float("nan"))
    assert np.all(np.isnan(mix.mean))
    assert np.all(np.isnan(mix.cov))
    draws = mix.sample(np.random.default_rng(0), size=3)
    assert draws.shape == (3, 2)
    assert np.all(np.isnan(draws))


def _wide_mixture(m=4, w=0.5):
    mu_a = np.zeros(m)
    mu_b = np.full(m, 2.0)
    small = 0.05**2 * np.eye(m)
    post_a = GPPosterior(mean=mu_a, cov=small.copy())
    post_b = GPPosterior(mean=mu_b, cov=small.copy())
    return mixture_posterior(post_a, post_b, w)


def _independent_label_draws(mix, rng, size):
    """The audit-8 DEFECT, reimplemented inline: fresh label per query point."""
    m = mix.mean.shape[0]
    labels = rng.random((size, m)) < mix.w
    a = mix.post_a.sample(rng, size=size)
    b = mix.post_b.sample(rng, size=size)
    return np.where(labels, a, b)


def test_label_per_draw_aggregate_variance_regression_audit8():
    mix = _wide_mixture()
    n = 40_000
    shared = mix.sample(np.random.default_rng(101), size=n)
    indep = _independent_label_draws(mix, np.random.default_rng(202), size=n)
    assert isinstance(mix, MixturePosterior)
    var_shared = shared.mean(axis=1).var()
    var_indep = indep.mean(axis=1).var()
    # independent per-point labels average the bimodality away across query
    # points -> too-narrow aggregate intervals (the defect audit 8 flags)
    assert var_shared >= 1.5 * var_indep


def test_mixture_mc_agreement_with_analytic():
    rng = np.random.default_rng(7)
    mu_a = np.array([0.0, 0.5, -0.5, 1.0])
    mu_b = mu_a + np.array([2.0, 1.5, 2.5, 2.0])
    cov_a = 0.1 * np.eye(4)
    cov_b = 0.2 * np.eye(4)
    mix = mixture_posterior(GPPosterior(mu_a, cov_a), GPPosterior(mu_b, cov_b), 0.4)
    draws = mix.sample(rng, size=40_000)
    assert draws.shape == (40_000, 4)
    np.testing.assert_allclose(draws.mean(axis=0), mix.mean, rtol=0.1, atol=0.02)
    np.testing.assert_allclose(np.cov(draws.T), mix.cov, rtol=0.1, atol=0.02)


def test_mixture_sample_deterministic_and_requires_rng():
    mix = _wide_mixture()
    d1 = mix.sample(np.random.default_rng(5), size=11)
    d2 = mix.sample(np.random.default_rng(5), size=11)
    np.testing.assert_array_equal(d1, d2)
    with pytest.raises((ValueError, TypeError)):
        mix.sample(None)
    with pytest.raises((ValueError, TypeError)):
        mix.sample(42)


# --------------------------------------------------------------- buffered folds


def test_buffered_folds_geometry_and_coverage():
    rng = np.random.default_rng(3)
    X = rng.uniform(-2.0, 2.0, size=(24, 2))
    buffer = 0.5
    folds = buffered_folds(X, n_folds=4, buffer=buffer, rng=np.random.default_rng(0))
    assert len(folds) == 4
    held_out = np.sort(np.concatenate([te for _, te in folds]))
    np.testing.assert_array_equal(held_out, np.arange(24))  # each held out once
    for train, test in folds:
        assert np.intersect1d(train, test).size == 0
        if train.size and test.size:
            d = np.sqrt(
                ((X[train][:, None, :] - X[test][None, :, :]) ** 2).sum(axis=2)
            )
            assert d.min() >= buffer  # no train center within buffer of any test center


def test_buffered_folds_deterministic_given_seed():
    X = np.random.default_rng(1).normal(size=(15, 2))
    f1 = buffered_folds(X, n_folds=3, buffer=0.3, rng=np.random.default_rng(9))
    f2 = buffered_folds(X, n_folds=3, buffer=0.3, rng=np.random.default_rng(9))
    for (tr1, te1), (tr2, te2) in zip(f1, f2, strict=True):
        np.testing.assert_array_equal(tr1, tr2)
        np.testing.assert_array_equal(te1, te2)


def test_buffered_folds_validation():
    X = np.random.default_rng(0).normal(size=(6, 2))
    with pytest.raises((ValueError, TypeError)):
        buffered_folds(X, n_folds=3, buffer=0.1, rng=None)
    with pytest.raises(ValueError):
        buffered_folds(X, n_folds=1, buffer=0.1, rng=np.random.default_rng(0))
    with pytest.raises(ValueError):
        buffered_folds(X, n_folds=7, buffer=0.1, rng=np.random.default_rng(0))
    with pytest.raises(ValueError):
        buffered_folds(X, n_folds=3, buffer=-0.5, rng=np.random.default_rng(0))
    Xbad = X.copy()
    Xbad[2, 0] = np.nan
    with pytest.raises(ValueError):
        buffered_folds(Xbad, n_folds=3, buffer=0.1, rng=np.random.default_rng(0))


# ----------------------------------------------------------- buffered stacking


def _stacking_scenario(seed, mirror=False, u=24):
    """tau_hat = obs - smooth bias + eps (model A correct) or the mirror case.

    The rough component is iid noise a GP cannot extrapolate; whichever surface
    (bias for A, tau for B) carries only the smooth part is learnable.
    """
    rng = np.random.default_rng(seed)
    X = np.sort(rng.uniform(-2.0, 2.0, size=u))[:, None]
    smooth = 1.5 * np.sin(1.5 * X[:, 0])
    rough = rng.normal(0.0, 2.0, size=u)
    eps = rng.normal(0.0, 0.05, size=u)
    noise_var = np.full(u, 0.05**2)
    if not mirror:
        obs = rough
        tau = obs - smooth + eps  # bias surface obs - tau = smooth - eps: learnable
    else:
        tau = smooth + eps  # direct surface learnable
        obs = tau + rough  # bias surface = rough: unlearnable
    return X, tau, obs, noise_var


def test_stacking_prefers_debias_model_when_bias_is_smooth():
    X, tau, obs, noise_var = _stacking_scenario(seed=0, mirror=False)
    w = buffered_stacking_weights(X, tau, obs, noise_var, rng=np.random.default_rng(0))
    assert w.strategy == "stacking"
    assert w.w_debias > 0.8
    assert "buffer" in w.detail and w.detail["buffer"] > 0.0


def test_stacking_prefers_direct_model_in_mirror_case():
    X, tau, obs, noise_var = _stacking_scenario(seed=0, mirror=True)
    w = buffered_stacking_weights(X, tau, obs, noise_var, rng=np.random.default_rng(0))
    assert w.w_debias < 0.2


def test_stacking_deterministic_given_seed():
    X, tau, obs, noise_var = _stacking_scenario(seed=2)
    w1 = buffered_stacking_weights(X, tau, obs, noise_var, rng=np.random.default_rng(4))
    w2 = buffered_stacking_weights(X, tau, obs, noise_var, rng=np.random.default_rng(4))
    assert w1.w_debias == w2.w_debias


def test_stacking_reduces_folds_when_few_centers():
    X, tau, obs, noise_var = _stacking_scenario(seed=1, u=4)
    w = buffered_stacking_weights(
        X, tau, obs, noise_var, rng=np.random.default_rng(0), n_folds=5, buffer=0.0
    )
    assert w.detail["n_folds"] == 4  # reduced from 5 with a diagnostic


def test_stacking_degenerate_u2_gives_nan_no_exception():
    X, tau, obs, noise_var = _stacking_scenario(seed=1, u=2)
    w = buffered_stacking_weights(X, tau, obs, noise_var, rng=np.random.default_rng(0))
    assert isinstance(w, ModelWeights)
    assert np.isnan(w.w_debias)
    assert w.strategy == "stacking"


def test_stacking_requires_rng():
    X, tau, obs, noise_var = _stacking_scenario(seed=0)
    with pytest.raises((ValueError, TypeError)):
        buffered_stacking_weights(X, tau, obs, noise_var, rng=None)
