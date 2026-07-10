import numpy as np

from natex.data.synthetic import make_synthetic
from natex.rdd.lord3 import lord3_scan
from natex.rdd.metrics import normalized_information_gain


def test_scan_finds_planted_boundary_real_T():
    rng = np.random.default_rng(0)
    ds, D = make_synthetic(n=1500, zeta=4.0, kind="real", rng=rng)
    res = lord3_scan(ds, k=40, rng=np.random.default_rng(1))
    top = res.top(10)
    # at least one of the top-10 centers sits near the true boundary
    boundary_hit = False
    for d in top:
        raw = ds.Z[d.center_index]
        if np.any(np.abs(raw - 0.5) < 0.15):
            boundary_hit = True
    assert boundary_hit
    # and its split aligns with the truth reasonably well
    nig = normalized_information_gain(D, top[0].members, top[0].group1)
    assert nig > 0.2


def test_scan_binary_model_autoselects():
    rng = np.random.default_rng(2)
    ds, _ = make_synthetic(n=1200, zeta=3.0, kind="binary", rng=rng)
    res = lord3_scan(ds, k=40, rng=np.random.default_rng(3))
    assert res.model == "bernoulli"
    assert res.discoveries[0].llr > 0


def test_outcome_never_read():
    rng = np.random.default_rng(4)
    ds, _ = make_synthetic(n=400, rng=rng)
    ds.df["y"] = np.nan  # poison the outcome; scan must not care
    res = lord3_scan(ds, k=20, rng=np.random.default_rng(5))
    assert np.isfinite(res.discoveries[0].llr)


def test_determinism():
    ds, _ = make_synthetic(n=400, rng=np.random.default_rng(6))
    a = lord3_scan(ds, k=20, rng=np.random.default_rng(7))
    b = lord3_scan(ds, k=20, rng=np.random.default_rng(7))
    assert a.discoveries[0].llr == b.discoveries[0].llr
    assert a.discoveries[0].center_index == b.discoveries[0].center_index
