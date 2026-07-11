"""Tests for the scaled DEE simulation-1 DGP (phase 4, task 7).

Statistical-test calibration (phase-4 policy: >=5 seeds, pin one, record ranges):

- exact bias identity (n=200k, constant_surfaces=(2, 3)): max abs deviation of the
  per-region contrast from tau0+beta0 across seeds 0-9 was 0.0213; tolerance 0.05
  (>2x margin). Seed 0 pinned.
- binned GP-surface identity (n=120k, l=0.5): mean cell error across seeds 0-4 was
  0.014-0.028 with all 36 cells usable; tolerance 0.15 (>5x margin). Seed 1 pinned.
- discoverability (n=3000, k=50): seeds 0-4 all produced multiple top-10 discoveries
  within 0.1 raw units of a corner boundary with max |normal component| > 0.7
  (typical dist 0.005-0.05, typical alignment 0.85-1.0). Seed 4 pinned.
"""

import numpy as np
import pytest

from natex.data.spec import Dataset
from natex.data.synthetic_dee import DEETruth, make_dee_synthetic
from natex.rdd.lord3 import lord3_scan


def _xmat(ds: Dataset) -> np.ndarray:
    return ds.df[["x0", "x1"]].to_numpy(dtype=float)


def _regions(ds: Dataset, thresholds: tuple[float, float]) -> np.ndarray:
    """0/1/2 region labels from the nested corner instruments (01 impossible)."""
    x = _xmat(ds)
    b1, b2 = thresholds
    z1 = (x[:, 0] >= b1) & (x[:, 1] >= b1)
    z2 = (x[:, 0] >= b2) & (x[:, 1] >= b2)
    return z1.astype(int) + z2.astype(int)


def _corner_boundary_dist(p: np.ndarray, b: float) -> float:
    """Distance from raw-unit point p to the L-shaped corner boundary at threshold b."""
    dx, dy = float(p[0] - b), float(p[1] - b)
    d_vert = abs(dx) if dy >= 0.0 else float(np.hypot(dx, dy))  # {x0=b, x1>=b}
    d_horiz = abs(dy) if dx >= 0.0 else float(np.hypot(dx, dy))  # {x1=b, x0>=b}
    return min(d_vert, d_horiz)


def test_exact_bias_identity_per_region():
    """Load-bearing: mean(y|D=1) - mean(y|D=0) - tau0 == beta0 in every region."""
    ds, truth = make_dee_synthetic(
        n=200_000, constant_surfaces=(2.0, 3.0), rng=np.random.default_rng(0)
    )
    region = _regions(ds, truth.thresholds)
    D, y = ds.T, ds.df["y"].to_numpy(dtype=float)
    for reg in (0, 1, 2):
        m = region == reg
        contrast = y[m & (D == 1.0)].mean() - y[m & (D == 0.0)].mean()
        assert abs(contrast - 2.0 - 3.0) < 0.05, f"region {reg}: contrast {contrast}"


def test_binned_gp_surface_identity():
    """Weaker binned identity with GP-sampled surfaces: cell-mean contrast tracks tau+beta."""
    ds, truth = make_dee_synthetic(
        n=120_000,
        cate_lengthscale=0.5,
        bias_lengthscale=0.5,
        rng=np.random.default_rng(1),
    )
    assert np.std(truth.cate_train) > 0.01  # GP path actually exercised
    assert np.std(truth.bias_train) > 0.01
    x = _xmat(ds)
    D, y = ds.T, ds.df["y"].to_numpy(dtype=float)
    target = truth.cate_train + truth.bias_train
    edges = np.linspace(0.0, 1.0, 7)  # 6x6 cells; thresholds (1/3, 2/3) sit on cell edges
    errs = []
    for i in range(6):
        for j in range(6):
            cell = (
                (x[:, 0] >= edges[i])
                & (x[:, 0] < edges[i + 1])
                & (x[:, 1] >= edges[j])
                & (x[:, 1] < edges[j + 1])
            )
            n1 = int((cell & (D == 1.0)).sum())
            n0 = int((cell & (D == 0.0)).sum())
            if n1 < 200 or n0 < 200:
                continue
            contrast = y[cell & (D == 1.0)].mean() - y[cell & (D == 0.0)].mean()
            errs.append(abs(contrast - target[cell].mean()))
    assert len(errs) >= 20  # nearly all cells usable at this n
    assert float(np.mean(errs)) < 0.15


def test_overlap_every_region_has_both_arms():
    ds, truth = make_dee_synthetic(
        n=5000, constant_surfaces=(2.0, 3.0), rng=np.random.default_rng(2)
    )
    region = _regions(ds, truth.thresholds)
    D = ds.T
    for reg in (0, 1, 2):
        m = region == reg
        assert (D[m] == 1.0).any(), f"region {reg} has no treated rows"
        assert (D[m] == 0.0).any(), f"region {reg} has no control rows"


def test_treatment_matches_complier_types_exactly():
    """D = 1[G=3] + 1[G=1] Z1 + 1[G=2] Z2, bitwise."""
    ds, truth = make_dee_synthetic(
        n=4000, constant_surfaces=(1.0, 1.0), rng=np.random.default_rng(3)
    )
    x = _xmat(ds)
    b1, b2 = truth.thresholds
    z1 = (x[:, 0] >= b1) & (x[:, 1] >= b1)
    z2 = (x[:, 0] >= b2) & (x[:, 1] >= b2)
    g = truth.complier_type
    expected = (g == 3) | ((g == 1) & z1) | ((g == 2) & z2)
    np.testing.assert_array_equal(ds.T, expected.astype(float))
    assert set(np.unique(g).tolist()) <= {0, 1, 2, 3}


def test_lord3_discovers_corner_boundaries():
    """The DGP feeds the phase-1 scan: a top-10 discovery hugs a corner boundary."""
    ds, truth = make_dee_synthetic(
        n=3000, constant_surfaces=(2.0, 3.0), rng=np.random.default_rng(4)
    )
    res = lord3_scan(ds, k=50, model="bernoulli", rng=np.random.default_rng(104))
    x = _xmat(ds)
    hit = False
    for d in res.top(10):
        p = x[d.center_index]
        dist = min(_corner_boundary_dist(p, b) for b in truth.thresholds)
        axis_aligned = float(np.max(np.abs(d.normal))) > 0.7
        if dist < 0.1 and axis_aligned:
            hit = True
            break
    assert hit, "no top-10 discovery near a corner boundary with an axis-aligned normal"


def test_determinism_and_shapes():
    kw = dict(n=2000, grid=10, cate_lengthscale=0.4, bias_lengthscale=0.6)
    ds_a, tr_a = make_dee_synthetic(rng=np.random.default_rng(5), **kw)
    ds_b, tr_b = make_dee_synthetic(rng=np.random.default_rng(5), **kw)
    assert ds_a.df.equals(ds_b.df)
    assert isinstance(tr_a, DEETruth)
    for name in ("cate_train", "bias_train", "cate_query", "bias_query", "query"):
        np.testing.assert_array_equal(getattr(tr_a, name), getattr(tr_b, name))
    np.testing.assert_array_equal(tr_a.complier_type, tr_b.complier_type)
    assert tr_a.thresholds == tr_b.thresholds
    assert tr_a.cate_train.shape == (2000,)
    assert tr_a.bias_train.shape == (2000,)
    assert tr_a.cate_query.shape == (100,)
    assert tr_a.bias_query.shape == (100,)
    assert tr_a.query.shape == (100, 2)
    assert tr_a.complier_type.shape == (2000,)
    assert ds_a.spec.treatment == "D"
    assert ds_a.spec.outcome == "y"
    assert ds_a.spec.forcing == ["x0", "x1"]


def test_type_frequencies_match_probs():
    probs = (0.4, 0.2, 0.1, 0.3)
    n = 20_000
    _, truth = make_dee_synthetic(
        n=n, type_probs=probs, constant_surfaces=(0.0, 0.0), rng=np.random.default_rng(6)
    )
    for t, p in enumerate(probs):
        freq = float((truth.complier_type == t).mean())
        assert abs(freq - p) < 3.0 * np.sqrt(p * (1.0 - p) / n), f"type {t}: freq {freq}"


def test_constant_surfaces_fill_truth_exactly():
    _, truth = make_dee_synthetic(
        n=500, grid=5, constant_surfaces=(2.5, -1.5), rng=np.random.default_rng(7)
    )
    assert np.all(truth.cate_train == 2.5)
    assert np.all(truth.bias_train == -1.5)
    assert np.all(truth.cate_query == 2.5)
    assert np.all(truth.bias_query == -1.5)


def test_rng_required():
    with pytest.raises(ValueError):
        make_dee_synthetic(n=100)


@pytest.mark.parametrize(
    "thresholds", [(0.5, 0.5), (0.7, 0.3), (0.0, 0.5), (0.5, 1.0), (-0.1, 0.5)]
)
def test_invalid_thresholds_raise(thresholds):
    with pytest.raises(ValueError):
        make_dee_synthetic(n=100, thresholds=thresholds, rng=np.random.default_rng(8))


def test_invalid_type_probs_raise():
    with pytest.raises(ValueError):
        make_dee_synthetic(
            n=100, type_probs=(0.5, 0.5, 0.0, 0.0), rng=np.random.default_rng(9)
        )
    with pytest.raises(ValueError):
        make_dee_synthetic(
            n=100, type_probs=(0.5, 0.3, 0.3, 0.3), rng=np.random.default_rng(9)
        )
