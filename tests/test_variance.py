import numpy as np

from natex.scan.neighborhoods import knn_indices, local_residual_variance


def test_own_neighborhood_not_reverse_neighbors():
    # Asymmetric layout: point 3's own 3-NN = {3,2,1}, but the set of points
    # having 3 in THEIR neighborhood differs -> variances must differ.
    z = np.array([[0.0], [0.1], [0.2], [10.0]])
    r = np.array([0.0, 1.0, -1.0, 5.0])
    idx = knn_indices(z, k=3)
    v = local_residual_variance(r, idx)
    own = np.var(r[idx[3]], ddof=1)
    assert np.isclose(v[3], max(own, 1e-3 * np.var(r, ddof=1)))


def test_floor_is_data_scaled():
    z = np.linspace(0, 1, 50)[:, None]
    r = np.zeros(50)
    r[0] = 100.0  # global variance >> 0; local variances of constant stretches = 0
    idx = knn_indices(z, k=5)
    v = local_residual_variance(r, idx)
    assert np.all(v >= 1e-3 * np.var(r, ddof=1) - 1e-12)
