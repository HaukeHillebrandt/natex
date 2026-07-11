"""Unit tests for greedy discovery clustering (multi-cutoff assertions)."""

from __future__ import annotations

import numpy as np

from natex.rdd.lord3 import Discovery, LoRD3Result
from natex.rdd.metrics import cluster_discoveries


def _discovery(center_index: int, llr: float) -> Discovery:
    return Discovery(
        center_index=center_index,
        k=3,
        llr=llr,
        normal=np.array([1.0]),
        members=np.arange(3),
        group1=np.array([True, False, False]),
    )


def _result(llrs: list[float]) -> LoRD3Result:
    discoveries = [_discovery(i, llr) for i, llr in enumerate(llrs)]
    return LoRD3Result(discoveries=discoveries, model="normal", k=3)


def test_merges_within_tol_and_orders():
    Z = np.array([[0.0], [0.5], [10.0], [10.4], [30.0]])
    res = _result([9.0, 8.0, 7.0, 6.0, 5.0])
    clusters = cluster_discoveries(res, Z, tol=1.0)
    assert len(clusters) == 3
    assert [c.size for c in clusters] == [2, 2, 1]
    assert [c.center_z[0] for c in clusters] == [0.0, 10.0, 30.0]
    assert [c.max_llr for c in clusters] == [9.0, 7.0, 5.0]
    assert [c.representative.center_index for c in clusters] == [0, 2, 4]


def test_per_dimension_tol():
    # dim 0 differences all within 5.0; dim 1 splits rows 0/2 from row 1.
    Z = np.array([[0.0, 0.0], [1.0, 0.5], [2.0, 0.05]])
    res = _result([9.0, 8.0, 7.0])
    clusters = cluster_discoveries(res, Z, tol=np.array([5.0, 0.1]))
    assert len(clusters) == 2
    assert [c.size for c in clusters] == [2, 1]
    assert clusters[0].representative.center_index == 0
    assert clusters[1].representative.center_index == 1


def test_top_limits_input():
    Z = np.array([[0.0], [0.5], [10.0], [10.4], [30.0]])
    res = _result([9.0, 8.0, 7.0, 6.0, 5.0])
    clusters = cluster_discoveries(res, Z, tol=1.0, top=2)
    assert len(clusters) == 1
    assert clusters[0].size == 2
    assert clusters[0].max_llr == 9.0


def test_empty_result():
    res = LoRD3Result(discoveries=[], model="normal", k=3)
    assert cluster_discoveries(res, np.zeros((0, 1)), tol=1.0) == []


def test_center_z_is_a_copy():
    Z = np.array([[0.0], [30.0]])
    res = _result([9.0, 8.0])
    clusters = cluster_discoveries(res, Z, tol=1.0)
    Z[0, 0] = 99.0
    assert clusters[0].center_z[0] == 0.0
