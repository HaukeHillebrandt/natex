"""Task 5: coarse-to-fine scan — seeded subsample, localization, full-res rescan,
and the spec 6b never-silently-truncate coverage contract."""

import numpy as np
import pytest

from natex.data.synthetic import make_synthetic
from natex.rdd.lord3 import lord3_scan
from natex.rdd.metrics import normalized_information_gain
from natex.scan.coarse import CoarseToFineResult, coarse_to_fine_scan


def _dataset(n, seed=0, zeta=3.0, kind="real"):
    ds, D = make_synthetic(n=n, zeta=zeta, kind=kind, rng=np.random.default_rng(seed))
    return ds, D


def test_finds_planted_boundary_cheaply():
    ds, D = _dataset(n=6000, zeta=4.0)
    res = coarse_to_fine_scan(
        ds, k=40, n_coarse=600, top_m=10, rng=np.random.default_rng(1)
    )
    assert isinstance(res, CoarseToFineResult)
    top = res.result.discoveries[0]
    nig = normalized_information_gain(D, top.members, top.group1)
    assert nig > 0.4
    assert res.frac_centers_scanned < 0.3


def test_deterministic():
    ds, _ = _dataset(n=800)
    a = coarse_to_fine_scan(ds, k=25, n_coarse=200, top_m=5, rng=np.random.default_rng(1))
    b = coarse_to_fine_scan(ds, k=25, n_coarse=200, top_m=5, rng=np.random.default_rng(1))
    ta, tb = a.result.discoveries[0], b.result.discoveries[0]
    assert (ta.center_index, ta.llr) == (tb.center_index, tb.llr)
    np.testing.assert_array_equal(a.fine_centers, b.fine_centers)


def test_reports_coverage():
    ds, _ = _dataset(n=800)
    res = coarse_to_fine_scan(
        ds, k=25, n_coarse=200, top_m=5, rng=np.random.default_rng(2)
    )
    assert 0 < res.frac_centers_scanned <= 1
    assert {"n_coarse", "top_m", "radius_mult", "k", "model", "degree"} <= set(res.params)
    fc = res.fine_centers
    assert fc.ndim == 1
    np.testing.assert_array_equal(fc, np.unique(fc))  # sorted unique


def test_rng_required():
    ds, _ = _dataset(n=300)
    with pytest.raises(ValueError):
        coarse_to_fine_scan(ds, k=20, rng=None)


def test_small_n_degenerates_to_full():
    ds, _ = _dataset(n=300)
    res = coarse_to_fine_scan(ds, k=20, rng=np.random.default_rng(3))
    # every point is a coarse center
    np.testing.assert_array_equal(
        np.sort(np.asarray(res.coarse_result.centers)), np.arange(ds.n)
    )
    assert res.frac_centers_scanned == 1.0
    full = lord3_scan(ds, k=20)
    got = [(d.center_index, d.llr) for d in res.result.top(5)]
    want = [(d.center_index, d.llr) for d in full.top(5)]
    assert got == want
