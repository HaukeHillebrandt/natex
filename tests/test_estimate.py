import numpy as np
import pytest

from natex.data.synthetic import make_synthetic
from natex.estimate.local2sls import local_2sls, wald_estimate
from natex.rdd.lord3 import lord3_scan


def _discovered(zeta=4.0, tau=2.0, n=2000, kind="binary", seed=0):
    ds, _ = make_synthetic(n=n, zeta=zeta, tau=tau, kind=kind, rng=np.random.default_rng(seed))
    res = lord3_scan(ds, k=60, rng=np.random.default_rng(seed + 1))
    return ds, res.discoveries[0]


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
