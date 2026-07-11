"""Tests for dee/gp.py: exact heteroskedastic RBF GP on core deps.

Plan task 3 contract: analytic 2-point kriging, heteroskedastic pull,
closed-form LOO == brute force (rtol 1e-6), prior lengthscale/variance
properties, rng determinism, NaN-never-0.0 policy.
"""

import numpy as np
import pytest

from natex.dee.gp import GPPosterior, HeteroskedasticGP, rbf_kernel, sample_gp_prior


def _easy_problem(seed=7, n=30):
    rng = np.random.default_rng(seed)
    X = np.linspace(0.0, 5.0, n)[:, None]
    noise_var = np.full(n, 0.01)
    y = np.sin(X[:, 0]) + rng.normal(0.0, np.sqrt(noise_var))
    return X, y, noise_var


# ---------------------------------------------------------------- rbf kernel


def test_rbf_kernel_values_and_shape():
    A = np.array([[0.0], [1.0]])
    B = np.array([[0.0], [1.0], [2.0]])
    K = rbf_kernel(A, B, lengthscale=1.0, outputscale=2.0)
    assert K.shape == (2, 3)
    assert np.isclose(K[0, 0], 2.0)
    assert np.isclose(K[0, 1], 2.0 * np.exp(-0.5))
    assert np.isclose(K[0, 2], 2.0 * np.exp(-2.0))
    # symmetry on identical inputs
    K2 = rbf_kernel(B, B, lengthscale=0.7, outputscale=1.3)
    assert np.allclose(K2, K2.T)
    assert np.allclose(np.diag(K2), 1.3)


# ------------------------------------------------------- analytic 2-point GP


def test_two_point_kriging_analytic():
    X = np.array([[0.0], [1.0]])
    y = np.array([0.0, 1.0])
    noise = np.full(2, 1e-12)
    gp = HeteroskedasticGP(
        lengthscale=1.0, outputscale=1.0, mean_const=0.0, X=X, y=y, noise_var=noise
    )
    post = gp.posterior(X)
    assert isinstance(post, GPPosterior)
    np.testing.assert_allclose(post.mean, y, atol=1e-6)
    assert np.all(np.diag(post.cov) < 1e-8)
    # hand-computed kriging mean at x = 0.5
    xq = np.array([[0.5]])
    k = np.array([np.exp(-0.125), np.exp(-0.125)])
    K = np.array([[1.0, np.exp(-0.5)], [np.exp(-0.5), 1.0]]) + 1e-12 * np.eye(2)
    expected = k @ np.linalg.solve(K, y)
    post_q = gp.posterior(xq)
    np.testing.assert_allclose(post_q.mean[0], expected, atol=1e-8)


def test_heteroskedastic_pull():
    # duplicate x: the low-noise observation (y=0) dominates the posterior mean
    X = np.array([[2.0], [2.0]])
    y = np.array([0.0, 10.0])
    noise = np.array([1e-4, 1e4])
    gp = HeteroskedasticGP(
        lengthscale=1.0, outputscale=1.0, mean_const=0.0, X=X, y=y, noise_var=noise
    )
    post = gp.posterior(np.array([[2.0]]))
    assert abs(post.mean[0]) < 0.05


# ------------------------------------------------------------------ MLL fit


def test_fit_improves_mll_over_every_start():
    X, y, noise_var = _easy_problem()
    gp = HeteroskedasticGP.fit(X, y, noise_var, rng=np.random.default_rng(0))
    assert np.isfinite(gp.log_marginal_likelihood())
    report = gp.fit_report
    assert report is not None and len(report["starts"]) >= 1
    best = report["best_mll"]
    assert np.isclose(best, gp.log_marginal_likelihood(), rtol=1e-8)
    for start in report["starts"]:
        assert best >= start["init_mll"] - 1e-9


def test_fit_requires_generator():
    X, y, noise_var = _easy_problem()
    with pytest.raises((ValueError, TypeError)):
        HeteroskedasticGP.fit(X, y, noise_var, rng=None)
    with pytest.raises((ValueError, TypeError)):
        HeteroskedasticGP.fit(X, y, noise_var, rng=42)


# ---------------------------------------------------------------------- LOO


def test_loo_closed_form_equals_brute_force():
    rng = np.random.default_rng(3)
    n = 12
    X = rng.uniform(-2.0, 2.0, size=(n, 2))
    y = np.sin(X[:, 0]) + 0.5 * X[:, 1] + rng.normal(0.0, 0.1, size=n)
    noise = rng.uniform(0.005, 0.05, size=n)
    gp = HeteroskedasticGP(
        lengthscale=1.2, outputscale=0.8, mean_const=0.3, X=X, y=y, noise_var=noise
    )
    closed = gp.loo_log_predictive()

    brute = 0.0
    for i in range(n):
        keep = np.arange(n) != i
        gp_i = HeteroskedasticGP(
            lengthscale=1.2,
            outputscale=0.8,
            mean_const=0.3,
            X=X[keep],
            y=y[keep],
            noise_var=noise[keep],
        )
        post = gp_i.posterior(X[i : i + 1])
        mu = post.mean[0]
        var = post.cov[0, 0] + noise[i]  # predictive of the NOISY y_i
        brute += -0.5 * np.log(2.0 * np.pi * var) - (y[i] - mu) ** 2 / (2.0 * var)
    np.testing.assert_allclose(closed, brute, rtol=1e-6)


# ------------------------------------------------------------- prior draws


def test_prior_draw_variance_and_lengthscale():
    X = np.linspace(0.0, 10.0, 400)[:, None]
    rng = np.random.default_rng(11)
    outputscale = 2.0
    rough = sample_gp_prior(X, lengthscale=0.1, outputscale=outputscale, rng=rng, size=200)
    smooth = sample_gp_prior(X, lengthscale=1.0, outputscale=outputscale, rng=rng, size=200)
    assert rough.shape == (200, 400)
    emp_var = rough.var()
    assert 0.75 * outputscale < emp_var < 1.25 * outputscale
    mssd_rough = np.mean(np.diff(rough, axis=1) ** 2)
    mssd_smooth = np.mean(np.diff(smooth, axis=1) ** 2)
    assert mssd_rough > mssd_smooth


def test_prior_requires_generator():
    X = np.linspace(0.0, 1.0, 5)[:, None]
    with pytest.raises((ValueError, TypeError)):
        sample_gp_prior(X, lengthscale=1.0, outputscale=1.0, rng=None)


def test_posterior_sample_shape_and_determinism():
    X, y, noise_var = _easy_problem()
    gp = HeteroskedasticGP(
        lengthscale=1.0, outputscale=1.0, mean_const=0.0, X=X, y=y, noise_var=noise_var
    )
    Xq = np.linspace(0.0, 5.0, 7)[:, None]
    post = gp.posterior(Xq)
    s1 = post.sample(np.random.default_rng(5), size=9)
    s2 = post.sample(np.random.default_rng(5), size=9)
    assert s1.shape == (9, 7)
    np.testing.assert_array_equal(s1, s2)


# ------------------------------------------------------------- determinism


def test_fit_determinism_same_seed_bitwise():
    X, y, noise_var = _easy_problem()
    a = HeteroskedasticGP.fit(X, y, noise_var, rng=np.random.default_rng(42))
    b = HeteroskedasticGP.fit(X, y, noise_var, rng=np.random.default_rng(42))
    assert a.lengthscale == b.lengthscale
    assert a.outputscale == b.outputscale
    assert a.mean_const == b.mean_const
    np.testing.assert_array_equal(
        a.posterior(X[:3]).mean, b.posterior(X[:3]).mean
    )


def test_fit_seed_robustness_easy_problem():
    X, y, noise_var = _easy_problem()
    a = HeteroskedasticGP.fit(X, y, noise_var, rng=np.random.default_rng(0))
    b = HeteroskedasticGP.fit(X, y, noise_var, rng=np.random.default_rng(1))
    np.testing.assert_allclose(a.lengthscale, b.lengthscale, rtol=1e-3)
    np.testing.assert_allclose(a.outputscale, b.outputscale, rtol=1e-3)
    np.testing.assert_allclose(a.mean_const, b.mean_const, rtol=1e-3, atol=1e-6)


def test_prior_determinism():
    X = np.linspace(0.0, 1.0, 20)[:, None]
    d1 = sample_gp_prior(X, 0.5, 1.0, rng=np.random.default_rng(9), size=3)
    d2 = sample_gp_prior(X, 0.5, 1.0, rng=np.random.default_rng(9), size=3)
    np.testing.assert_array_equal(d1, d2)


# -------------------------------------------------------------- NaN policy


def test_fit_drops_nan_rows():
    X, y, noise_var = _easy_problem()
    y = y.copy()
    y[4] = np.nan
    gp = HeteroskedasticGP.fit(X, y, noise_var, rng=np.random.default_rng(0))
    assert gp.X.shape[0] == X.shape[0] - 1
    assert gp.fit_report["n_dropped"] == 1
    assert np.isfinite(gp.log_marginal_likelihood())


def test_all_nan_gives_nan_posterior_never_zero():
    X, y, noise_var = _easy_problem(n=6)
    y = np.full_like(y, np.nan)
    gp = HeteroskedasticGP.fit(X, y, noise_var, rng=np.random.default_rng(0))
    assert np.isnan(gp.log_marginal_likelihood())
    post = gp.posterior(np.array([[1.0], [2.0]]))
    assert post.mean.shape == (2,)
    assert post.cov.shape == (2, 2)
    assert np.all(np.isnan(post.mean))
    assert np.all(np.isnan(post.cov))
    assert np.isnan(gp.loo_log_predictive())
    draws = post.sample(np.random.default_rng(0), size=4)
    assert draws.shape == (4, 2)
    assert np.all(np.isnan(draws))


def test_nan_noise_rows_dropped_too():
    X, y, noise_var = _easy_problem()
    noise_var = noise_var.copy()
    noise_var[[2, 5]] = np.nan
    gp = HeteroskedasticGP.fit(X, y, noise_var, rng=np.random.default_rng(0))
    assert gp.X.shape[0] == X.shape[0] - 2
    assert gp.fit_report["n_dropped"] == 2
