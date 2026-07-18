"""Tests for dee/vknn.py effect + balance additions (phase-4 task 2).

experiment_effects delegates to the phase-1 frozen-side estimators via the
Discovery duck-typing contract; balance_filter reuses the phase-1 placebo
battery and never reads y; NaN (never 0.0) on missing outcomes.
"""

import numpy as np
import pandas as pd
import pytest

from natex.data.spec import Dataset, DatasetSpec
from natex.dee.vknn import (
    QuasiExperiment,
    VKNNResult,
    balance_filter,
    experiment_effects,
    voronoi_knn_repair,
)
from natex.rdd.lord3 import lord3_scan


def sharp_dataset(n=1200, seed=0):
    """Sharp 1-D design: T = 1{x >= 0}, y = 1 + 2*T + 0.2*eps."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(-2.0, 2.0, n)
    T = (x >= 0.0).astype(float)
    y = 1.0 + 2.0 * T + 0.2 * rng.standard_normal(n)
    df = pd.DataFrame({"x": x, "T": T, "y": y})
    spec = DatasetSpec(treatment="T", outcome="y", forcing=["x"], covariates=["x"])
    return Dataset(df, spec)


def repaired_sharp(seed=0):
    ds = sharp_dataset(seed=seed)
    scan = lord3_scan(ds, k=60, rng=np.random.default_rng(seed + 1))
    res = voronoi_knn_repair(ds, scan.discoveries, m_prime=1, k_prime=200, t_side=30)
    return ds, res


def hand_experiment(ds, center, lo, hi):
    """QuasiExperiment over rows [lo, hi) with the 1-D hyperplane at ``center``."""
    members = np.arange(lo, hi)
    Zs = ds.Z_std
    nhat = np.array([1.0])
    s = (Zs[members] - Zs[center]) @ nhat
    centroid = Zs[members].mean(axis=0)
    projected = centroid - ((centroid - Zs[center]) @ nhat) * nhat
    return QuasiExperiment(
        center_index=int(center),
        members=members,
        group1=s >= 0.0,
        normal=nhat,
        llr=1.0,
        centroid=centroid,
        projected_center=projected,
    )


def two_experiment_setup(extra_cov=None, seed=2):
    """T jumps 0->1 at x=25 and 1->0 at x=75; hand-built experiments at both."""
    n = 100
    x = np.arange(float(n))
    T = ((x >= 25) & (x < 75)).astype(float)
    rng = np.random.default_rng(seed)
    y = 1.0 + 2.0 * T + 0.2 * rng.standard_normal(n)
    df = pd.DataFrame({"x": x, "T": T, "y": y})
    covariates = ["x"]
    if extra_cov is not None:
        df["c"] = extra_cov
        covariates.append("c")
    spec = DatasetSpec(treatment="T", outcome="y", forcing=["x"], covariates=covariates)
    ds = Dataset(df, spec)
    exps = [hand_experiment(ds, 25, 10, 41), hand_experiment(ds, 75, 60, 91)]
    res = VKNNResult(
        experiments=exps,
        accepted=np.array([0, 1]),
        rejected=np.array([], dtype=int),
        k_prime=31,
        t_side=5,
    )
    return ds, res


def test_effects_recover_tau_on_sharp_synthetic():
    ds, res = repaired_sharp()
    assert len(res.experiments) == 1
    e = res.experiments[0]
    assert e.group1.any() and not e.group1.all()  # straddles the cutoff
    ests = experiment_effects(ds, res)
    assert len(ests) == 1
    est = ests[0]
    assert est.method == "2sls"
    assert 1.6 <= est.tau <= 2.4
    assert est.weak_instrument is False
    assert est.n_used > 0


def test_effects_method_dispatch_and_validation():
    ds, res = repaired_sharp(seed=3)
    wald = experiment_effects(ds, res, method="wald")
    assert all(w.method == "wald" for w in wald)
    assert 1.6 <= wald[0].tau <= 2.4
    with pytest.raises(ValueError, match="method"):
        experiment_effects(ds, res, method="ols")


def test_nan_outcome_gives_nan_never_zero_and_leaves_others_alone():
    ds, res = two_experiment_setup()
    clean = experiment_effects(ds, res)
    assert all(1.0 <= abs(c.tau) <= 3.0 for c in clean)
    ds.df.loc[res.experiments[0].members, "y"] = np.nan
    poisoned, other = experiment_effects(ds, res)
    assert np.isnan(poisoned.tau) and np.isnan(poisoned.se)
    assert np.isnan(poisoned.ci[0]) and np.isnan(poisoned.ci[1])
    assert not (poisoned.tau == 0.0)
    assert poisoned.n_used == 0
    assert other.tau == clean[1].tau and other.se == clean[1].se
    assert other.n_used == clean[1].n_used


def test_balance_filter_drops_jumping_covariate_keeps_clean():
    rng = np.random.default_rng(7)
    x = np.arange(100.0)
    # covariate jumps by 5 exactly at experiment A's boundary (x = 25)
    c = 5.0 * (x >= 25.0) + 1.5 * rng.standard_normal(100)
    ds, res = two_experiment_setup(extra_cov=c)
    filtered = balance_filter(ds, res, alpha=0.05)
    assert [e.center_index for e in filtered.experiments] == [75]
    assert np.array_equal(filtered.accepted, [1])
    # new object; input untouched
    assert filtered is not res
    assert [e.center_index for e in res.experiments] == [25, 75]
    assert np.array_equal(res.accepted, [0, 1])
    # tiny alpha keeps both
    lax = balance_filter(ds, res, alpha=1e-9)
    assert [e.center_index for e in lax.experiments] == [25, 75]


def test_issue_34_balance_filter_keeps_vacuous_battery():
    """Issue #34: ``placebo_tests`` reports ``passed=None`` when the only
    covariate is the forcing column (vacuous battery). The balance filter must
    keep those experiments — only an actual False drops one; a bare
    ``dtype=bool`` coercion of None would silently drop every experiment
    whose battery had nothing to test."""
    ds, res = two_experiment_setup()  # covariates = forcing only
    filtered = balance_filter(ds, res, alpha=0.05)
    assert [e.center_index for e in filtered.experiments] == [25, 75]
    assert np.array_equal(filtered.accepted, [0, 1])


def test_balance_filter_never_reads_y():
    rng = np.random.default_rng(7)
    x = np.arange(100.0)
    c = 5.0 * (x >= 25.0) + 1.5 * rng.standard_normal(100)
    ds, res = two_experiment_setup(extra_cov=c)
    ds.df["y"] = np.nan  # poison the outcome; filtering must not care
    filtered = balance_filter(ds, res, alpha=0.05)
    assert [e.center_index for e in filtered.experiments] == [75]
