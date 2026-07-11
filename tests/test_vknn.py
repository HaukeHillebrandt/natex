"""Tests for dee/vknn.py: corrected DEE Algorithm 1 (Voronoi-KNN repair) + M' selection.

Regression targets from docs/math_audit_final.md: repo risk 1 (explicit inputs),
risk 2 (first candidate thresholded like every other), risk 3 (deterministic
direct projection, no RNG), audit-23 tie convention, audit-7 radius diagnostic.
"""

import inspect

import numpy as np
import pandas as pd

from natex.data.spec import Dataset, DatasetSpec
from natex.dee.vknn import (
    QuasiExperiment,
    VKNNResult,
    experiment_radius,
    select_m_prime,
    voronoi_knn_repair,
)
from natex.rdd.lord3 import Discovery, LoRD3Result


def make_dataset(Z, outcome="y"):
    Z = np.asarray(Z, dtype=float)
    cols = [f"z{j}" for j in range(Z.shape[1])]
    df = pd.DataFrame(Z, columns=cols)
    df["T"] = (Z[:, 0] >= np.median(Z[:, 0])).astype(float)
    if outcome is not None:
        df[outcome] = np.linspace(-1.0, 1.0, Z.shape[0])
    spec = DatasetSpec(treatment="T", outcome=outcome, forcing=cols, covariates=cols)
    return Dataset(df, spec)


def make_discovery(center, normal, llr):
    normal = np.asarray(normal, dtype=float)
    normal = normal / np.linalg.norm(normal)
    return Discovery(
        center_index=int(center),
        k=5,
        llr=float(llr),
        normal=normal,
        members=np.array([int(center)]),
        group1=np.array([True]),
    )


def grid_2d():
    """60 points on a 10x6 integer grid; row index = 10*y + x."""
    xs, ys = np.meshgrid(np.arange(10.0), np.arange(6.0))
    return np.c_[xs.ravel(), ys.ravel()]


def test_disjointness_and_ownership():
    ds = make_dataset(grid_2d())
    d_a = make_discovery(22, [1.0, 0.0], 10.0)  # (x=2, y=2)
    d_b = make_discovery(37, [1.0, 0.0], 9.0)  # (x=7, y=3)
    res = voronoi_knn_repair(ds, [d_a, d_b], m_prime=2, k_prime=30, t_side=5)
    assert isinstance(res, VKNNResult)
    assert len(res.experiments) == 2
    e1, e2 = res.experiments
    assert e1.center_index == 22 and e2.center_index == 37  # acceptance = LLR rank order
    assert set(e1.members.tolist()) & set(e2.members.tolist()) == set()
    # every member's nearest accepted center is its owner
    Zs = ds.Z_std
    d2 = np.stack([((Zs - Zs[22]) ** 2).sum(axis=1), ((Zs - Zs[37]) ** 2).sum(axis=1)])
    owner = np.argmin(d2, axis=0)
    assert np.all(owner[e1.members] == 0)
    assert np.all(owner[e2.members] == 1)
    for e in res.experiments:
        # per-side support and the audit-23 sign convention (>= 0 => group 1)
        assert int(e.group1.sum()) >= 5 and int((~e.group1).sum()) >= 5
        s = (Zs[e.members] - Zs[e.center_index]) @ e.normal
        assert np.array_equal(e.group1, s >= 0)
        assert e.center_index in e.members.tolist()
        assert e.members.size <= 30
    assert np.array_equal(res.accepted, [0, 1])
    assert res.rejected.size == 0


def test_first_center_thresholded():
    """Repo risk 2 regression: the FIRST candidate faces the support test too."""
    ds = make_dataset(np.arange(20.0)[:, None])
    d = make_discovery(2, [1.0], 5.0)  # only 2 points strictly below x=2
    res = voronoi_knn_repair(ds, [d], m_prime=1, k_prime=20, t_side=5)
    assert res.experiments == []
    assert np.array_equal(res.rejected, [0])
    assert res.accepted.size == 0


def test_rejection_restores_state():
    ds = make_dataset(np.arange(30.0)[:, None])
    d_a = make_discovery(10, [1.0], 10.0)
    d_b = make_discovery(5, [1.0], 9.0)  # would leave A only {8, 9} below its plane
    baseline = voronoi_knn_repair(ds, [d_a], m_prime=1, k_prime=30, t_side=5)
    assert len(baseline.experiments) == 1
    res = voronoi_knn_repair(ds, [d_a, d_b], m_prime=2, k_prime=30, t_side=5)
    assert len(res.experiments) == 1
    assert res.experiments[0].center_index == 10
    assert np.array_equal(res.experiments[0].members, baseline.experiments[0].members)
    assert np.array_equal(res.experiments[0].group1, baseline.experiments[0].group1)
    assert np.array_equal(res.accepted, [0])
    assert np.array_equal(res.rejected, [1])


def test_projection_on_plane_hand_formula_idempotent():
    ds = make_dataset(grid_2d())
    d = make_discovery(22, [1.0, 1.0], 5.0)
    res = voronoi_knn_repair(ds, [d], m_prime=1, k_prime=30, t_side=5)
    assert len(res.experiments) == 1
    e = res.experiments[0]
    Zs = ds.Z_std
    nhat = e.normal
    p = e.projected_center
    # on the frozen hyperplane
    assert abs(float((p - Zs[22]) @ nhat)) < 1e-12
    # centroid and the hand projection formula, bitwise
    centroid = Zs[e.members].mean(axis=0)
    assert np.array_equal(e.centroid, centroid)
    hand = centroid - ((centroid - Zs[22]) @ nhat) * nhat
    assert np.array_equal(p, hand)
    # idempotent: projecting again is a no-op
    again = p - ((p - Zs[22]) @ nhat) * nhat
    assert np.allclose(again, p, atol=1e-12)


def test_rng_free_determinism():
    assert "rng" not in inspect.signature(voronoi_knn_repair).parameters
    ds = make_dataset(grid_2d())
    cands = [make_discovery(22, [1.0, 0.0], 10.0), make_discovery(37, [1.0, 1.0], 9.0)]
    r1 = voronoi_knn_repair(ds, cands, m_prime=2, k_prime=30, t_side=5)
    r2 = voronoi_knn_repair(ds, cands, m_prime=2, k_prime=30, t_side=5)
    assert len(r1.experiments) == len(r2.experiments)
    for a, b in zip(r1.experiments, r2.experiments, strict=True):
        assert np.array_equal(a.members, b.members)
        assert np.array_equal(a.group1, b.group1)
        assert np.array_equal(a.projected_center, b.projected_center)


def test_tie_break_earlier_accepted_center():
    # symmetric 1-D configuration: x = -10..10, centers at -4 and +4, x=0 is
    # exactly equidistant (bitwise, by symmetry of Z_std) => earlier-accepted owns it
    ds = make_dataset(np.arange(-10.0, 11.0)[:, None])
    i0, ia, ib = 10, 6, 14  # rows of x = 0, -4, +4
    d_a = make_discovery(ia, [1.0], 8.0)
    d_b = make_discovery(ib, [1.0], 7.0)
    res = voronoi_knn_repair(ds, [d_a, d_b], m_prime=2, k_prime=50, t_side=3)
    assert len(res.experiments) == 2
    e_a, e_b = res.experiments
    assert i0 in e_a.members.tolist()
    assert i0 not in e_b.members.tolist()


def test_unsorted_candidates_are_sorted_defensively():
    ds = make_dataset(grid_2d())
    d_a = make_discovery(22, [1.0, 0.0], 10.0)
    d_b = make_discovery(37, [1.0, 0.0], 9.0)
    r1 = voronoi_knn_repair(ds, [d_b, d_a], m_prime=2, k_prime=30, t_side=5)
    r2 = voronoi_knn_repair(ds, [d_a, d_b], m_prime=2, k_prime=30, t_side=5)
    assert [e.center_index for e in r1.experiments] == [
        e.center_index for e in r2.experiments
    ]
    assert r1.experiments[0].llr >= r1.experiments[1].llr


def test_select_m_prime():
    discs = [make_discovery(i, [1.0], llr) for i, llr in enumerate([9.0, 7.0, 5.0, 3.0, 1.0])]
    scan = LoRD3Result(discoveries=discs, model="normal", k=5)
    null = np.repeat(np.arange(5.0), 20)  # 95th percentile = 4.0
    assert select_m_prime(scan, null, level=0.95) == 3  # strictly greater: 9, 7, 5
    assert select_m_prime(scan, np.full(20, 100.0)) == 0  # all null above => none


def test_m_prime_zero_and_k_prime_clamp():
    ds = make_dataset(np.arange(12.0)[:, None])
    d = make_discovery(6, [1.0], 5.0)
    res0 = voronoi_knn_repair(ds, [d], m_prime=0)
    assert res0.experiments == []
    assert res0.accepted.size == 0 and res0.rejected.size == 0
    res = voronoi_knn_repair(ds, [d], m_prime=1, k_prime=10_000, t_side=3)
    assert len(res.experiments) == 1
    assert res.experiments[0].members.size == 12  # clamped to n, not an error


def test_y_blindness():
    ds_y = make_dataset(grid_2d(), outcome="y")
    ds_y.df["y"] = np.nan  # poison the outcome; repair must not care
    ds_none = make_dataset(grid_2d(), outcome=None)
    cands = [make_discovery(22, [1.0, 0.0], 10.0), make_discovery(37, [1.0, 0.0], 9.0)]
    r1 = voronoi_knn_repair(ds_y, cands, m_prime=2, k_prime=30, t_side=5)
    r2 = voronoi_knn_repair(ds_none, cands, m_prime=2, k_prime=30, t_side=5)
    assert len(r1.experiments) == len(r2.experiments) == 2
    for a, b in zip(r1.experiments, r2.experiments, strict=True):
        assert np.array_equal(a.members, b.members)
        assert np.array_equal(a.group1, b.group1)
        assert np.array_equal(a.projected_center, b.projected_center)


def test_experiment_radius_hand_computed():
    Z = np.array([[0.0, 0.0], [3.0, 0.0], [0.0, 4.0], [1.0, 1.0], [2.0, 2.0], [5.0, 5.0]])
    ds = make_dataset(Z)
    Zs = ds.Z_std
    members = np.array([0, 1, 2])
    e = QuasiExperiment(
        center_index=0,
        members=members,
        group1=np.array([True, True, False]),
        normal=np.array([1.0, 0.0]),
        llr=1.0,
        centroid=Zs[members].mean(axis=0),
        projected_center=Zs[0],
    )
    expected = float(np.max(np.linalg.norm(Zs[members] - Zs[0], axis=1)))
    assert experiment_radius(ds, e) == expected
