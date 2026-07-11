"""Tests for the ch.6 synthetic DiD DGP (Eqs 6.22-6.28, audit-repaired) — phase 3, task 6."""

import numpy as np
import pytest

from natex.data.synthetic_did import DiDTruth, make_did_synthetic

BASE = dict(n=800, d=4, V=8, periods=10, zeta=10.0, tau=10.0, s_dims=2, s_values=2)


def test_truth_mask_matches_pandas_recomputation():
    ds, truth = make_did_synthetic(rng=np.random.default_rng(0), **BASE)
    assert isinstance(truth, DiDTruth)
    assert len(truth.included) == BASE["d"]
    recomputed = np.ones(ds.n, dtype=bool)
    for j, inc in enumerate(truth.included):
        values = np.arange(1, BASE["V"] + 1)[np.asarray(inc, dtype=bool)]
        recomputed &= ds.df[f"x{j}"].isin(values).to_numpy()
    np.testing.assert_array_equal(recomputed, truth.record_mask)
    # exactly s_dims dimensions are constrained (s_values < V)
    constrained = [j for j, inc in enumerate(truth.included) if not np.all(inc)]
    assert len(constrained) == BASE["s_dims"]
    for j in constrained:
        assert int(np.sum(truth.included[j])) == BASE["s_values"]
    assert truth.record_mask.any() and not truth.record_mask.all()


def test_theta_jump_magnitude():
    # Time-trend-free DGP: mean theta of treated records post - pre ~= zeta.
    # Calibration across seeds 0-9 (zeta=10, n=4000): |diff - 10| <= 0.48;
    # the +-1.5 tolerance from the plan carries a wide margin. Seed 3 pinned.
    ds, truth = make_did_synthetic(
        n=4000, d=4, V=8, periods=10, zeta=10.0, tau=10.0, rng=np.random.default_rng(3)
    )
    theta = ds.T
    t = ds.df["t"].to_numpy(dtype=float)
    post = t >= truth.t0
    jump = theta[truth.record_mask & post].mean() - theta[truth.record_mask & ~post].mean()
    assert abs(jump - 10.0) < 1.5


def test_determinism_same_seed():
    ds_a, tr_a = make_did_synthetic(rng=np.random.default_rng(7), **BASE)
    ds_b, tr_b = make_did_synthetic(rng=np.random.default_rng(7), **BASE)
    assert ds_a.df.equals(ds_b.df)
    np.testing.assert_array_equal(tr_a.record_mask, tr_b.record_mask)
    for a, b in zip(tr_a.included, tr_b.included, strict=True):
        np.testing.assert_array_equal(a, b)
    assert (tr_a.t0, tr_a.zeta, tr_a.tau) == (tr_b.t0, tr_b.zeta, tr_b.tau)


def test_rng_required():
    with pytest.raises(ValueError):
        make_did_synthetic(**BASE)


def test_binary_theta_variant():
    ds, truth = make_did_synthetic(theta_kind="binary", rng=np.random.default_rng(1), **BASE)
    theta = ds.T
    assert set(np.unique(theta)) <= {0.0, 1.0}
    t = ds.df["t"].to_numpy(dtype=float)
    post = t >= truth.t0
    # the thresholded latent still jumps: treated-post P(theta=1) exceeds treated-pre
    assert theta[truth.record_mask & post].mean() > theta[truth.record_mask & ~post].mean()


def test_hetero_group_only_changes_y():
    ds_h, tr_h = make_did_synthetic(hetero_group=True, rng=np.random.default_rng(5), **BASE)
    ds_p, tr_p = make_did_synthetic(hetero_group=False, rng=np.random.default_rng(5), **BASE)
    # identical draws up to the y-noise term: x, t, theta and the truth coincide
    cols = [f"x{j}" for j in range(BASE["d"])] + ["t", "theta"]
    assert ds_h.df[cols].equals(ds_p.df[cols])
    np.testing.assert_array_equal(tr_h.record_mask, tr_p.record_mask)
    assert tr_h.t0 == tr_p.t0
    assert not np.allclose(ds_h.df["y"].to_numpy(), ds_p.df["y"].to_numpy())
    # s_g = s_I union a nonempty random untreated subset
    assert tr_p.hetero_mask is None
    sg = tr_h.hetero_mask
    assert sg is not None
    assert np.all(sg[tr_h.record_mask])  # s_I subset of s_g
    assert np.any(sg & ~tr_h.record_mask)  # some untreated records included


def test_hetero_shock_variant():
    # Prose-intent correlated variant: shared per-period shocks on s_c only.
    ds_s, tr_s = make_did_synthetic(
        hetero_group=True, hetero_kind="shock", rng=np.random.default_rng(5), **BASE
    )
    ds_p, tr_p = make_did_synthetic(hetero_group=False, rng=np.random.default_rng(5), **BASE)
    cols = [f"x{j}" for j in range(BASE["d"])] + ["t", "theta"]
    assert ds_s.df[cols].equals(ds_p.df[cols])
    sc = tr_s.hetero_mask
    assert sc is not None and sc.any()
    assert not np.any(sc & tr_s.record_mask)  # shocks never touch s_I
    assert tr_s.hetero_shocks is not None
    assert tr_s.hetero_shocks.shape == (BASE["periods"],)
    assert tr_p.hetero_shocks is None
    # y differs from the plain draw exactly on s_c, by the shared shock * t.
    dy = ds_s.df["y"].to_numpy() - ds_p.df["y"].to_numpy()
    assert np.all(dy[~sc] == 0.0)
    t = ds_s.df["t"].to_numpy()
    expected = tr_s.hetero_shocks[t.astype(np.int64) - 1] * t
    np.testing.assert_allclose(dy[sc], expected[sc], rtol=1e-12)


def test_hetero_validation_errors():
    with pytest.raises(ValueError, match="hetero_kind"):
        make_did_synthetic(n=100, hetero_kind="bogus", rng=np.random.default_rng(0))
    with pytest.raises(ValueError, match="hetero_scale"):
        make_did_synthetic(n=100, hetero_scale=0.0, rng=np.random.default_rng(0))


def test_dataset_wiring():
    ds, _ = make_did_synthetic(rng=np.random.default_rng(2), **BASE)
    spec = ds.spec
    assert spec.time == "t"
    assert spec.unit is None
    assert spec.forcing == []
    assert spec.covariates == [f"x{j}" for j in range(BASE["d"])]
    assert spec.treatment == "theta"
    assert spec.outcome == "y"
    assert ds.Z.shape == (ds.n, 0)
    assert ds.n == BASE["n"]
    # covariate values live on the thesis grid 1..V
    x0 = ds.df["x0"].to_numpy()
    assert x0.min() >= 1 and x0.max() <= BASE["V"]


def test_t0_in_middle_half_with_two_sided_data():
    for seed in range(10):
        ds, truth = make_did_synthetic(rng=np.random.default_rng(seed), **BASE)
        assert 3.0 <= truth.t0 <= 8.0  # middle half of times 1..10
        t = ds.df["t"].to_numpy(dtype=float)
        assert np.any(t < truth.t0) and np.any(t >= truth.t0)


def test_validation_errors():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        make_did_synthetic(n=100, d=2, s_dims=3, rng=rng)
    with pytest.raises(ValueError):
        make_did_synthetic(n=100, V=4, s_values=5, rng=rng)
    with pytest.raises(ValueError):
        make_did_synthetic(n=100, periods=2, rng=rng)
    with pytest.raises(ValueError):
        make_did_synthetic(n=100, theta_kind="bogus", rng=rng)
    with pytest.raises(ValueError):
        make_did_synthetic(n=100, V=1, rng=rng)
    with pytest.raises(ValueError):
        make_did_synthetic(n=0, rng=rng)
