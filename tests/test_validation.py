import numpy as np

from natex.data.synthetic import make_synthetic
from natex.rdd.lord3 import lord3_scan
from natex.validate.density import density_test
from natex.validate.honest import honest_split
from natex.validate.placebo import hc1_ols, placebo_tests, signed_distance


def test_honest_split_disjoint_and_deterministic():
    a1, b1 = honest_split(100, rng=np.random.default_rng(0))
    a2, b2 = honest_split(100, rng=np.random.default_rng(0))
    assert set(a1) & set(b1) == set()
    assert len(a1) + len(b1) == 100
    np.testing.assert_array_equal(a1, a2)


def test_hc1_recovers_slope():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(500, 2))
    y = 1.0 + 2.0 * x[:, 0] - 3.0 * x[:, 1] + rng.normal(size=500)
    X = np.c_[np.ones(500), x]
    beta, se = hc1_ols(X, y)
    assert abs(beta[1] - 2.0) < 0.2 and abs(beta[2] + 3.0) < 0.2
    assert np.all(se > 0)


def test_placebo_passes_on_clean_synthetic():
    rng = np.random.default_rng(2)
    ds, _ = make_synthetic(n=1500, zeta=4.0, kind="real", px=3, pz=2, rng=rng)
    res = lord3_scan(ds, k=40, rng=np.random.default_rng(3))
    rep = placebo_tests(ds, res.discoveries[0])
    # x2 (non-forcing covariate) is smooth through the boundary -> should pass
    assert rep.passed


def test_density_smoke():
    rng = np.random.default_rng(4)
    ds, _ = make_synthetic(n=1500, zeta=4.0, kind="real", rng=rng)
    res = lord3_scan(ds, k=40, rng=np.random.default_rng(5))
    rep = density_test(ds, res.discoveries[0])
    assert 0.0 <= rep.p_value <= 1.0
    s = signed_distance(ds, res.discoveries[0])
    assert s.shape == (res.discoveries[0].members.size,)
    assert np.isfinite(s).all()
