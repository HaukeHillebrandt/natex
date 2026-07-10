import numpy as np

from natex.scan.statistics import normal_glr_common_variance, normal_llr_all_splits


def test_hand_computed_example():
    # r = [1,1,-1,-1], unit weights, split = first two vs last two.
    # Group means fit perfectly => LLR = sum(r^2)/2 = 2.0
    r = np.array([1.0, 1.0, -1.0, -1.0])
    w = np.ones(4)
    G = np.array([[True], [True], [False], [False]])
    np.testing.assert_allclose(normal_llr_all_splits(r, w, G), [2.0])


def test_nonnegative_and_empty_side_zero():
    rng = np.random.default_rng(0)
    for _ in range(200):
        k = rng.integers(3, 30)
        r = rng.normal(size=k)
        w = 1.0 / rng.uniform(0.1, 2.0, size=k)
        G = rng.random((k, 8)) < 0.5
        llr = normal_llr_all_splits(r, w, G)
        assert np.all(llr >= -1e-12)
        empty = (G.sum(axis=0) == 0) | (G.sum(axis=0) == k)
        assert np.all(llr[empty] == 0.0)


def test_scale_invariance():
    # r -> a*r with w -> w/a^2 leaves the LLR unchanged (precision-weighted form)
    rng = np.random.default_rng(1)
    r = rng.normal(size=12)
    w = 1.0 / rng.uniform(0.5, 2.0, size=12)
    G = rng.random((12, 5)) < 0.5
    a = 3.7
    np.testing.assert_allclose(
        normal_llr_all_splits(r, w, G), normal_llr_all_splits(a * r, w / a**2, G), rtol=1e-10
    )


def test_glr_common_variance_monotone_in_separation():
    r_weak = np.array([0.1, 0.1, -0.1, -0.1])
    r_strong = np.array([1.0, 1.0, -1.0, -1.0]) + 1e-6  # avoid RSS1 == 0
    G = np.array([[True], [True], [False], [False]])
    assert normal_glr_common_variance(r_strong, G)[0] > normal_glr_common_variance(r_weak, G)[0]
