"""Estimators must tolerate NaN outcomes: mask to finite y, report n_used,
and return all-NaN estimates (never 0.0, never an exception) when
underdetermined. LSO outcomes are 13-56% missing, so this is load-bearing."""

import math

import numpy as np
import pytest

from natex.data.spec import Dataset
from natex.data.synthetic import make_synthetic
from natex.estimate.local2sls import local_2sls, wald_estimate
from natex.rdd.lord3 import lord3_scan

ESTIMATORS = [local_2sls, wald_estimate]


@pytest.fixture(scope="module")
def discovered():
    ds, _ = make_synthetic(n=1500, zeta=4.0, kind="real", rng=np.random.default_rng(0))
    res = lord3_scan(ds, k=40, rng=np.random.default_rng(1))
    return ds, res.discoveries[0]


def _with_nan_y(ds, rows):
    df = ds.df.copy()
    df.loc[np.asarray(rows), ds.spec.outcome] = np.nan
    return Dataset(df, ds.spec.model_copy())


def test_clean_data_reports_full_n_used(discovered):
    ds, d = discovered
    for estimator in ESTIMATORS:
        est = estimator(ds, d)
        assert est.n_used == d.members.size


@pytest.mark.parametrize("estimator", ESTIMATORS)
def test_partial_nan_outcome_close_to_clean(discovered, estimator):
    ds, d = discovered
    tau_clean = estimator(ds, d).tau
    rng = np.random.default_rng(7)
    n_drop = int(round(0.3 * d.members.size))
    dropped = rng.choice(d.members, size=n_drop, replace=False)
    est = estimator(_with_nan_y(ds, dropped), d)
    assert abs(est.tau - tau_clean) < 1.0
    assert est.n_used == d.members.size - n_drop


@pytest.mark.parametrize("estimator", ESTIMATORS)
def test_all_nan_outcome_returns_nan_not_zero(discovered, estimator):
    ds, d = discovered
    est = estimator(_with_nan_y(ds, d.members), d)
    assert math.isnan(est.tau)
    assert math.isnan(est.se)
    assert math.isnan(est.ci[0]) and math.isnan(est.ci[1])
    assert est.weak_instrument is True
    assert est.n_used == 0
    assert est.tau != 0.0  # NaN compares unequal; the point is it is not a silent zero


@pytest.mark.parametrize("estimator", ESTIMATORS)
def test_one_sided_after_masking_returns_nan(discovered, estimator):
    ds, d = discovered
    g1 = d.group1.astype(bool)
    assert g1.sum() >= 8, "fixture must leave a full one-sided group after masking"
    est = estimator(_with_nan_y(ds, d.members[~g1]), d)
    assert math.isnan(est.tau)
    assert math.isnan(est.se)
    assert est.weak_instrument is True
    assert est.n_used == int(g1.sum())
