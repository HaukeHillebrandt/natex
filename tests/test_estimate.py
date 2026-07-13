import numpy as np
import pandas as pd
import pytest

from natex.data.spec import Dataset, DatasetSpec
from natex.data.synthetic import make_synthetic
from natex.estimate.local2sls import local_2sls, wald_estimate
from natex.rdd.lord3 import Discovery, lord3_scan


def _discovered(zeta=4.0, tau=2.0, n=2000, kind="binary", seed=0):
    ds, _ = make_synthetic(n=n, zeta=zeta, tau=tau, kind=kind, rng=np.random.default_rng(seed))
    res = lord3_scan(ds, k=60, rng=np.random.default_rng(seed + 1))
    return ds, res.discoveries[0]


def _sharp_discovery(seed=0, n=200, treatment=None):
    """Hand-built sharp design: T = 1{z >= 0} exactly (deterministic first stage)."""
    rng = np.random.default_rng(seed)
    z = rng.uniform(-1.0, 1.0, size=n)
    T = (z >= 0).astype(float) if treatment is None else np.full(n, float(treatment))
    y = 2.0 * T + 0.3 * z + rng.normal(0.0, 0.1, size=n)
    df = pd.DataFrame({"z": z, "T": T, "y": y})
    spec = DatasetSpec(treatment="T", outcome="y", forcing=["z"], covariates=["z"])
    ds = Dataset(df, spec)
    zs = ds.df["z"].to_numpy()
    d = Discovery(
        center_index=int(np.argmin(np.abs(zs))),
        k=n,
        llr=0.0,
        normal=np.array([1.0]),
        members=np.arange(n),
        group1=zs >= 0,
    )
    return ds, d


def test_2sls_recovers_tau_binary():
    ds, d = _discovered()
    est = local_2sls(ds, d)
    assert est.method == "2sls"
    assert est.ci[0] < 2.0 < est.ci[1]
    assert est.first_stage_t > 2.0 and not est.weak_instrument


def test_wald_close_to_2sls():
    ds, d = _discovered(seed=3)
    a, b = local_2sls(ds, d), wald_estimate(ds, d)
    assert abs(a.tau - b.tau) < 1.5


def test_no_outcome_raises():
    ds, d = _discovered(seed=5)
    ds.spec.outcome = None
    with pytest.raises(ValueError, match="outcome"):
        local_2sls(ds, d)


def test_issue_4_sharp_design_inf_first_stage_t_never_weak_both_estimators():
    # Deterministic first stage (sharp T = 1{z >= 0}): side variances are
    # exactly 0 and the 2SLS-side HC1 se is float noise (~1e-16). The t must
    # be +inf on BOTH estimators — never a meaningless ~1e15 float-noise value
    # (2sls) nor NaN (wald) — and the weak-IV flags must agree: not weak.
    ds, d = _sharp_discovery()
    a = local_2sls(ds, d)
    b = wald_estimate(ds, d)
    assert a.first_stage_jump == pytest.approx(1.0, abs=1e-8)
    assert b.first_stage_jump == pytest.approx(1.0, abs=1e-12)
    assert np.isinf(a.first_stage_t) and a.first_stage_t > 0
    assert np.isinf(b.first_stage_t) and b.first_stage_t > 0
    assert a.weak_instrument is False
    assert b.weak_instrument is False
    assert a.ci[0] < 2.0 < a.ci[1]
    assert np.isfinite(b.tau)  # wald tau is trend-biased here by design; only finiteness matters


def test_issue_4_constant_treatment_keeps_nan_t_and_weak_flag():
    # No first stage at all (constant T): the t stays NaN — a zero se must
    # not be promoted to inf when there is no jump — and both flags are weak.
    ds, d = _sharp_discovery(treatment=1.0)
    a = local_2sls(ds, d)
    b = wald_estimate(ds, d)
    assert np.isnan(a.first_stage_t)
    assert np.isnan(b.first_stage_t)
    assert a.weak_instrument is True
    assert b.weak_instrument is True
    assert np.isnan(b.tau)  # wald dt == 0 path: NaN, never 0.0
