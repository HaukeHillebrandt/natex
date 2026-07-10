import numpy as np
import pytest

from natex.estimate.local2sls import local_2sls
from natex.rdd.lord3 import lord3_scan
from natex.validate.randomization import randomization_test

pytestmark = pytest.mark.backtest


@pytest.fixture()
def ds(load_or_skip):
    return load_or_skip("test_score_2012")


def test_registry_spec_matches_phase1_handbuilt(ds):
    # The registry must reproduce the phase-1 hand-built Dataset exactly.
    assert ds.spec.treatment == "treat"
    assert ds.spec.outcome == "posttest"
    assert ds.spec.forcing == ["age", "pretest"]
    assert ds.spec.covariates == [
        "gender", "sped", "frlunch", "esol", "black", "white",
        "hispanic", "asian", "age", "pretest",
    ]
    # 2766 raw data rows (checked by verify()); Dataset drops the 160 rows with
    # NaN in scan columns, exactly as the phase-1 hand-built Dataset did.
    assert ds.n == 2606


def test_recovers_cutoff_215_and_forcing_variable(ds):
    res = lord3_scan(ds, k=50, rng=np.random.default_rng(0))
    top = res.top(10)
    hits = [d for d in top if abs(ds.Z[d.center_index][1] - 215) < 5]
    assert hits, f"no top-10 center near pretest 215; top centers: {[ds.Z[d.center_index].tolist() for d in top]}"
    # pretest must dominate the forcing influence for the best hit
    best = hits[0]
    assert abs(best.normal[1]) > abs(best.normal[0])


def test_scan_significant_and_effect_bracketed(ds):
    res = lord3_scan(ds, k=50, rng=np.random.default_rng(1))
    rep = randomization_test(ds, res, Q=19, rng=np.random.default_rng(2), scan_kwargs={"k": 50})
    assert rep.p_value <= 0.05
    hits = [d for d in res.top(10) if abs(ds.Z[d.center_index][1] - 215) < 5]
    est = local_2sls(ds, hits[0])
    # treatment goes to LOW scorers: expect tau approx +10 on posttest, generous bracket
    assert 4.0 < abs(est.tau) < 16.0
    assert not est.weak_instrument
