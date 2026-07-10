"""Task 6: KDD Eqs 17-19 fidelity options + label-noise protocol P(T_rho = T) = rho."""

import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr

from natex.data.synthetic import draw_confounder, inject_label_noise, make_synthetic


def test_defaults_unchanged():
    # New kwargs at their defaults must reproduce phase-1 output frame-for-frame
    # (guards accidental rng-consumption reordering).
    a, Da = make_synthetic(n=500, rng=np.random.default_rng(0))
    b, Db = make_synthetic(
        n=500,
        rng=np.random.default_rng(0),
        boundary=0.5,
        min_region_frac=0.05,
        heteroskedastic=False,
        confounder="normal",
    )
    pd.testing.assert_frame_equal(a.df, b.df)
    np.testing.assert_array_equal(Da, Db)


def test_random_boundary_mass_bounds():
    masses = []
    for seed in range(20):
        _, D = make_synthetic(n=4000, boundary="random", rng=np.random.default_rng(seed))
        mass = D.mean()
        assert 0.05 <= mass <= 0.95
        masses.append(mass)
    # boundaries are redrawn per seed, so region masses must not all coincide
    assert len(np.unique(np.round(masses, 6))) > 1


def test_unknown_boundary_string_raises():
    with pytest.raises(ValueError):
        make_synthetic(n=100, boundary="diagonal", rng=np.random.default_rng(0))


def _squared_resid_vs_meanx(heteroskedastic: bool, seed: int) -> float:
    ds, _ = make_synthetic(
        n=20000,
        zeta=0.0,
        tau=0.0,
        kind="real",
        heteroskedastic=heteroskedastic,
        rng=np.random.default_rng(seed),
    )
    x = ds.X
    T = ds.T
    A = np.column_stack([np.ones(len(T)), x])
    beta, *_ = np.linalg.lstsq(A, T, rcond=None)
    resid = T - A @ beta
    return spearmanr(resid**2, x.mean(axis=1)).statistic


def test_heteroskedastic_variance_tracks_x():
    assert _squared_resid_vs_meanx(heteroskedastic=True, seed=3) > 0.1
    assert _squared_resid_vs_meanx(heteroskedastic=False, seed=3) < 0.05


def test_uniform_confounder_range():
    u = draw_confounder(10000, "uniform", np.random.default_rng(0))
    assert u.min() >= 0.0
    assert u.max() <= 1.0
    assert 0.45 < u.mean() < 0.55
    v = draw_confounder(10000, "normal", np.random.default_rng(1))
    assert (v < 0).any()
    with pytest.raises(ValueError):
        draw_confounder(10, "cauchy", np.random.default_rng(2))


def test_inject_label_noise_exact_rate():
    T = np.random.default_rng(0).binomial(1, 0.5, size=20000).astype(float)

    same = inject_label_noise(T, 1.0, np.random.default_rng(1))
    np.testing.assert_array_equal(same, T)
    assert same is not T  # never mutates / never aliases the input

    noisy = inject_label_noise(T, 0.8, np.random.default_rng(2))
    assert 0.78 < (noisy == T).mean() < 0.82

    coin = inject_label_noise(T, 0.5, np.random.default_rng(3))
    assert 0.48 < (coin == T).mean() < 0.52

    a = inject_label_noise(T, 0.7, np.random.default_rng(9))
    b = inject_label_noise(T, 0.7, np.random.default_rng(9))
    np.testing.assert_array_equal(a, b)

    with pytest.raises(ValueError):
        inject_label_noise(T, 0.4, np.random.default_rng(4))
    with pytest.raises(ValueError):
        inject_label_noise(np.array([0.1, 0.9, 0.5]), 0.8, np.random.default_rng(5))
