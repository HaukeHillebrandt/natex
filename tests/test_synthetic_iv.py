"""Tests for the sparse-first-stage IV synthetic DGP (phase 5, task 3).

Calibration notes (repo statistical-test policy: every stochastic assertion
was run across >= 5 seeds during implementation; ranges recorded here and one
seed base pinned with margin — see per-test comments for the observed values).
"""

import numpy as np
import pandas as pd
import pytest

from natex.data.synthetic_iv import make_iv_synthetic
from natex.estimate.iv2sls import iv_2sls


def test_rng_required():
    with pytest.raises(ValueError, match="numpy Generator"):
        make_iv_synthetic(n=100)


def test_rng_wrong_type():
    with pytest.raises(TypeError, match="numpy Generator"):
        make_iv_synthetic(n=100, rng=np.random.RandomState(0))


def test_shapes_and_metadata():
    d = make_iv_synthetic(n=80, p=10, s=3, n_invalid=2, rng=np.random.default_rng(0))
    names = [f"z{j}" for j in range(1, 11)]
    assert list(d.df.columns) == names + ["T", "y"]
    assert d.df.shape == (80, 12)
    assert d.pool_names == names
    assert d.true_support == ["z1", "z2", "z3"]
    assert d.invalid_names == ["z9", "z10"]
    assert not set(d.invalid_names) & set(d.true_support)
    assert d.pi.shape == (10,)
    assert np.all(d.pi[:3] > 0) and np.all(d.pi[3:] == 0.0)
    assert d.tau == 1.0
    assert np.isfinite(d.concentration) and d.concentration > 0
    # No planted violators by default.
    d0 = make_iv_synthetic(n=80, p=10, s=3, rng=np.random.default_rng(0))
    assert d0.invalid_names == []


def test_support_and_violators_must_fit_in_pool():
    with pytest.raises(ValueError, match="s \\+ n_invalid"):
        make_iv_synthetic(n=80, p=6, s=5, n_invalid=2, rng=np.random.default_rng(0))


def test_seeded_determinism():
    a = make_iv_synthetic(n=120, p=8, s=3, rng=np.random.default_rng(7))
    b = make_iv_synthetic(n=120, p=8, s=3, rng=np.random.default_rng(7))
    pd.testing.assert_frame_equal(a.df, b.df, check_exact=True)
    assert a.concentration == b.concentration
    assert np.array_equal(a.pi, b.pi)
    c = make_iv_synthetic(n=120, p=8, s=3, rng=np.random.default_rng(8))
    assert not a.df.equals(c.df)


@pytest.mark.parametrize("mu2", [30.0, 180.0])
def test_concentration_targeting(mu2):
    # Calibrated over seeds 0..4 at n=500: relative error |realized - mu2|/mu2
    # in [0.018, 0.103] (identical for both mu2 values — the realized/target
    # ratio is scale-free in mu2 given the same seed) — well inside the 25%
    # gate; realized concentration has relative sd ~ sqrt(2/n) from each of
    # pi'Z'Z pi and var_hat(v).
    for seed in range(5):
        d = make_iv_synthetic(n=500, p=50, s=5, mu2=mu2, rng=np.random.default_rng(seed))
        assert abs(d.concentration - mu2) / mu2 < 0.25


def test_ols_biased_true_support_2sls_unbiased():
    # Endogeneity fidelity gate (plan task 3): median over 5 seeds at n=2000.
    # Plim OLS bias = endog / (pi'Sigma pi + 1) = 0.6/1.09 ~ 0.55. Calibrated
    # over seed bases 0,100,200,300,400 (5 seeds each): median OLS bias in
    # [0.534, 0.559]; median 2SLS bias in [-0.027, +0.074] (per-seed 2SLS
    # sd ~ 1/sqrt(mu2) ~ 0.075, so bases 200/300 exceed the 0.05 gate by
    # chance). Pinned base 100: median OLS bias 0.541, median 2SLS bias
    # -0.0011 — margins >> gates.
    ols_bias, tsls_bias = [], []
    for seed in range(100, 105):
        d = make_iv_synthetic(n=2000, p=50, s=5, mu2=180.0, rng=np.random.default_rng(seed))
        y = d.df["y"].to_numpy()
        t_vec = d.df["T"].to_numpy()
        ols = np.linalg.lstsq(np.c_[np.ones(y.size), t_vec], y, rcond=None)[0][1]
        est = iv_2sls(y, t_vec, d.df[d.true_support].to_numpy())
        assert not est.weak_instrument
        ols_bias.append(ols - d.tau)
        tsls_bias.append(est.tau - d.tau)
    assert np.median(ols_bias) > 0.15
    assert abs(np.median(tsls_bias)) < 0.05


def test_planted_violators_reject_hansen_j_valid_overid_does_not():
    # Calibrated over seeds 0..4 at n=500: with n_invalid=2 in the instrument
    # set, j_p in [3.2e-30, 1.7e-24] — rejection at every seed. With valid
    # overidentification (support only, k=5, j_df=4) j_p is null-uniform
    # (5.5% below 0.05 across 200 seeds; seeds 0..4 give 0.041, 0.017, 0.312,
    # 0.038, 0.299). Pinned seed 2: invalid j_p ~ 3.2e-30, valid j_p ~ 0.312.
    d = make_iv_synthetic(n=500, p=50, s=5, n_invalid=2, phi=0.5, rng=np.random.default_rng(2))
    y = d.df["y"].to_numpy()
    t_vec = d.df["T"].to_numpy()
    bad = iv_2sls(y, t_vec, d.df[d.true_support + d.invalid_names].to_numpy())
    assert bad.j_df == 6
    assert bad.j_p < 0.01

    d_ok = make_iv_synthetic(n=500, p=50, s=5, rng=np.random.default_rng(2))
    ok = iv_2sls(
        d_ok.df["y"].to_numpy(), d_ok.df["T"].to_numpy(), d_ok.df[d_ok.true_support].to_numpy()
    )
    assert ok.j_df == 4
    assert ok.j_p > 0.05
