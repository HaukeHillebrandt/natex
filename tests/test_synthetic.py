import numpy as np

from natex.data.synthetic import make_synthetic


def test_real_T_jump_matches_zeta():
    rng = np.random.default_rng(0)
    ds, D = make_synthetic(n=20000, zeta=3.0, kind="real", rng=rng)
    T = ds.T
    # conditional means differ by ~zeta after netting the smooth part only weakly matters at this n
    assert 2.0 < T[D].mean() - T[~D].mean() < 4.5


def test_binary_T_logodds_shift():
    rng = np.random.default_rng(1)
    ds, D = make_synthetic(n=60000, zeta=2.0, kind="binary", rng=rng)
    p_in, p_out = ds.T[D].mean(), ds.T[~D].mean()
    # log-odds gap should be near zeta (up to confounder smoothing), NOT p-scale gap of zeta
    gap = np.log(p_in / (1 - p_in)) - np.log(p_out / (1 - p_out))
    assert 1.2 < gap < 2.8


def test_determinism():
    a, Da = make_synthetic(n=500, rng=np.random.default_rng(7))
    b, Db = make_synthetic(n=500, rng=np.random.default_rng(7))
    np.testing.assert_array_equal(a.T, b.T)
    np.testing.assert_array_equal(Da, Db)
