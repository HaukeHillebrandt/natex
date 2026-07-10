"""Task 4: scan geometry cache — Kmax-NN prefix reuse, replica reuse, center subsets."""

import numpy as np
import pytest

from natex.data.spec import Dataset
from natex.data.synthetic import make_synthetic
from natex.rdd.lord3 import fit_treatment_model, lord3_scan
from natex.scan.geometry import ScanGeometry, build_geometry
from natex.scan.neighborhoods import knn_indices
from natex.validate.randomization import _draw_null_treatment, randomization_test


def _dataset(n=400, kind="real", seed=0):
    ds, _ = make_synthetic(n=n, zeta=3.0, kind=kind, rng=np.random.default_rng(seed))
    return ds


def test_idx_matches_knn_indices():
    ds = _dataset(n=400)
    Z = ds.Z_std
    geom = build_geometry(Z, 25)
    assert isinstance(geom, ScanGeometry)
    assert geom.k == 25
    np.testing.assert_array_equal(geom.idx, knn_indices(Z, 25))


def test_shrink_prefix_exact():
    ds = _dataset(n=400)
    Z = ds.Z_std
    small = build_geometry(Z, 40).shrink(15)
    assert small.k == 15
    np.testing.assert_array_equal(small.idx, knn_indices(Z, 15))


def test_shrink_rejects_growth():
    ds = _dataset(n=100)
    geom = build_geometry(ds.Z_std, 10)
    with pytest.raises(ValueError):
        geom.shrink(11)


def test_shrink_cache_is_fresh():
    ds = _dataset(n=100)
    Z = ds.Z_std
    geom = build_geometry(Z, 10)
    geom.partitions_for(0, Z)
    small = geom.shrink(5)
    assert small._partitions == {}
    # parent cache untouched, and lazy cache returns the same object on re-access
    G1, keep1 = geom.partitions_for(0, Z)
    G2, keep2 = geom.partitions_for(0, Z)
    assert G1 is G2 and keep1 is keep2


def test_scan_with_geometry_identical():
    ds = _dataset(n=400)
    res_plain = lord3_scan(ds, k=25)
    res_geom = lord3_scan(ds, k=25, geometry=build_geometry(ds.Z_std, 25))
    assert len(res_plain.discoveries) == len(res_geom.discoveries)
    for a, b in zip(res_plain.top(10), res_geom.top(10), strict=True):
        assert a.center_index == b.center_index
        assert a.llr == b.llr
        np.testing.assert_array_equal(a.group1, b.group1)


def test_centers_subset():
    ds = _dataset(n=350)
    n = ds.n
    centers = np.arange(0, n, 7)
    res_full = lord3_scan(ds, k=20)
    res_sub = lord3_scan(ds, k=20, centers=centers)
    assert res_sub.centers is not None
    np.testing.assert_array_equal(res_sub.centers, centers)
    allowed = set(centers.tolist())
    full_llr = {d.center_index: d.llr for d in res_full.discoveries}
    assert res_sub.discoveries  # subset scan finds something on this signal
    for d in res_sub.discoveries:
        assert d.center_index in allowed
        assert d.llr == full_llr[d.center_index]


def _per_replica_geometry_null(dataset, scan_result, Q, rng, scan_kwargs):
    """Phase-1 equivalent path: fresh geometry built for every replica scan."""
    scan_kwargs = dict(scan_kwargs or {})
    scan_kwargs.setdefault("k", scan_result.k)
    kind = scan_result.model
    X, T, Z = dataset.X, dataset.T, dataset.Z_std
    predict, _ = fit_treatment_model(X, T, kind, scan_kwargs.get("degree", 1))
    fitted = predict(X)
    sigma2 = None
    if kind == "normal":
        from natex.scan.neighborhoods import local_residual_variance

        sigma2 = local_residual_variance(T - fitted, knn_indices(Z, scan_kwargs["k"]))
    else:
        fitted = np.clip(fitted, 1e-6, 1 - 1e-6)
    null_max = np.empty(Q)
    for q_i in range(Q):
        t_star = _draw_null_treatment(kind, fitted, sigma2, rng)
        df_star = dataset.df.copy()
        df_star[dataset.spec.treatment] = t_star
        ds_star = Dataset(df_star, dataset.spec)
        geom = build_geometry(ds_star.Z_std, scan_kwargs["k"])  # rebuilt per replica
        res_star = lord3_scan(ds_star, model=kind, geometry=geom, **scan_kwargs)
        null_max[q_i] = res_star.discoveries[0].llr if res_star.discoveries else 0.0
    observed = scan_result.discoveries[0].llr
    p = (1.0 + float(np.sum(null_max >= observed))) / (Q + 1.0)
    return p, null_max


def test_randomization_bitwise_parity_and_single_knn_build(monkeypatch):
    import natex.scan.neighborhoods as nb

    ds = _dataset(n=300, kind="binary", seed=2)
    res = lord3_scan(ds, k=20)

    # phase-1 path: geometry rebuilt for every replica
    p_legacy, null_legacy = _per_replica_geometry_null(
        ds, res, Q=5, rng=np.random.default_rng(3), scan_kwargs={"k": 20}
    )

    calls = {"n": 0}
    real = nb.knn_indices

    def counting(z_std, k):
        calls["n"] += 1
        return real(z_std, k)

    monkeypatch.setattr(nb, "knn_indices", counting)
    rep = randomization_test(ds, res, Q=5, rng=np.random.default_rng(3), scan_kwargs={"k": 20})
    assert calls["n"] == 1  # one geometry build shared by all replicas
    assert rep.p_value == p_legacy
    np.testing.assert_array_equal(rep.null_max_llrs, null_legacy)
