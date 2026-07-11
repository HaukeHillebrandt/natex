"""Tests for dee/noise.py: hierarchical stage-1 chi-square noise smoothing.

Plan task 4 contract: recovery beats raw SE^2 (log-MSE halved), outlier
shrinkage, exact scipy digamma/polygamma constants, df=None fallback,
data-scaled floor, NaN propagation (never 0.0), rng discipline.
"""

import numpy as np
import pytest
from scipy.special import digamma, polygamma

from natex.dee.noise import log_se2_bias, log_se2_measurement_var, smooth_noise


# ------------------------------------------------------------------ recovery


def test_recovery_beats_raw_se2():
    # Constant true sigma^2 = 4 over 40 centers, se2 ~ (4/df) * chi2(df=10).
    # Calibrated across data seeds 0-9 (all pass with margin); seed 3 pinned.
    u, df_val, sigma2 = 40, 10.0, 4.0
    data_rng = np.random.default_rng(3)
    X = np.linspace(0.0, 8.0, u)[:, None]
    se2 = (sigma2 / df_val) * data_rng.chisquare(df_val, size=u)
    df = np.full(u, df_val)
    smoothed = smooth_noise(X, se2, df, rng=np.random.default_rng(0))
    assert smoothed.shape == (u,)
    assert np.all(np.isfinite(smoothed)) and np.all(smoothed > 0.0)
    log_truth = np.log(sigma2)
    mse_raw = float(np.mean((np.log(se2) - log_truth) ** 2))
    mse_smooth = float(np.mean((np.log(smoothed) - log_truth) ** 2))
    assert mse_smooth < 0.5 * mse_raw


# ---------------------------------------------------------- outlier shrinkage


def test_outlier_shrinks_toward_local_median():
    # One center at 50x the ambient se2 among 30 smooth neighbors.
    u, k = 31, 15
    X = np.linspace(0.0, 6.0, u)[:, None]
    se2 = 1.0 + 0.1 * np.sin(X[:, 0])
    se2[k] = 50.0 * float(np.median(np.delete(se2, k)))
    df = np.full(u, 10.0)
    smoothed = smooth_noise(X, se2, df, rng=np.random.default_rng(0))
    local_median = float(np.median(np.delete(smoothed, k)))
    assert np.isfinite(smoothed[k])
    assert smoothed[k] < 10.0 * local_median


# --------------------------------------------------- chi-square debias moments


def test_chi_square_constants_match_scipy():
    # For df=10: bias = psi(5) - log(5), measurement variance = psi_1(5).
    df = np.array([10.0])
    np.testing.assert_allclose(log_se2_bias(df), digamma(5.0) - np.log(5.0), rtol=1e-12)
    np.testing.assert_allclose(log_se2_measurement_var(df), polygamma(1, 5.0), rtol=1e-12)
    # vectorized over heterogeneous df
    dfs = np.array([2.0, 10.0, 33.0])
    np.testing.assert_allclose(log_se2_bias(dfs), digamma(dfs / 2.0) - np.log(dfs / 2.0))
    np.testing.assert_allclose(log_se2_measurement_var(dfs), polygamma(1, dfs / 2.0))


# ------------------------------------------------------------ df=None fallback


def test_df_none_fallback_finite_positive():
    data_rng = np.random.default_rng(2)
    X = data_rng.normal(size=(25, 2))
    se2 = np.exp(data_rng.normal(0.0, 0.3, size=25))
    out = smooth_noise(X, se2, None, rng=np.random.default_rng(5))
    assert out.shape == (25,)
    assert np.all(np.isfinite(out)) and np.all(out > 0.0)


# ------------------------------------------------------------------ floor, NaN


def test_zero_se2_floored_and_nan_propagates():
    u = 20
    X = np.linspace(0.0, 4.0, u)[:, None]
    se2 = np.full(u, 2.0)
    se2[3] = 0.0
    se2[7] = np.nan
    df = np.full(u, 10.0)
    out = smooth_noise(X, se2, df, rng=np.random.default_rng(0), floor_frac=1e-3)
    floor = 1e-3 * float(np.median(se2[np.isfinite(se2)]))
    assert floor > 0.0
    # NaN se2 in => NaN out at that index only
    assert np.isnan(out[7])
    keep = np.arange(u) != 7
    assert np.all(np.isfinite(out[keep]))
    # zero se2 in => output at (data-scaled) floor or above, never 0.0
    assert out[3] >= floor
    assert np.all(out[keep] >= floor)


# --------------------------------------------------------------- rng discipline


def test_requires_explicit_generator():
    X = np.zeros((3, 1))
    se2 = np.ones(3)
    with pytest.raises(ValueError, match="Generator"):
        smooth_noise(X, se2, None, rng=None)


def test_deterministic_given_seed():
    data_rng = np.random.default_rng(11)
    X = data_rng.normal(size=(20, 1))
    se2 = np.exp(data_rng.normal(0.0, 0.5, size=20))
    df = np.full(20, 8.0)
    out1 = smooth_noise(X, se2, df, rng=np.random.default_rng(42))
    out2 = smooth_noise(X, se2, df, rng=np.random.default_rng(42))
    np.testing.assert_array_equal(out1, out2)


# ------------------------------------------------------------------ df clipping


def test_df_below_one_clipped_with_diagnostic():
    u = 12
    X = np.linspace(0.0, 2.0, u)[:, None]
    se2 = np.full(u, 1.5)
    df = np.full(u, 6.0)
    df[0] = 0.25
    with pytest.warns(UserWarning, match="clipped"):
        out = smooth_noise(X, se2, df, rng=np.random.default_rng(0))
    assert np.all(np.isfinite(out)) and np.all(out > 0.0)
