"""Tests for SC donor scoring/selection/ATT (phase 5, task 7).

Calibration notes (repo statistical-test policy: stochastic assertions run
across >= 5 seeds during implementation; observed ranges in per-test
comments, one seed pinned with margin).

Pre-only honesty: donor scoring, ranking, and simplex weights read ONLY
pre-t0 outcome columns — mutation tests multiply every post-t0 outcome by
10 and require scores/donors/weights bitwise unchanged.
"""

import numpy as np
import pandas as pd
import pytest

from natex.data.spec import Dataset, DatasetSpec
from natex.data.synthetic_sc import make_sc_synthetic
from natex.iv.donors import (
    DonorScore,
    DonorSelectionResult,
    select_donors,
    select_donors_from_dataset,
    unit_time_matrix,
)


def _matrix(df):
    return unit_time_matrix(df, "unit", "time", "y")


# ---------------------------------------------------------------- unit_time_matrix


def test_unit_time_matrix_mean_aggregates_duplicates_and_nans_missing_cells():
    df = pd.DataFrame(
        {
            "unit": ["a", "a", "a", "b"],
            "time": [1, 1, 2, 1],
            "y": [1.0, 3.0, 5.0, 7.0],
        }
    )
    Y, units, times = unit_time_matrix(df, "unit", "time", "y")
    assert list(units) == ["a", "b"]
    assert times.tolist() == [1.0, 2.0]
    assert Y.shape == (2, 2)
    assert Y[0, 0] == 2.0  # duplicate (unit, time) rows mean-aggregate
    assert Y[0, 1] == 5.0
    assert Y[1, 0] == 7.0
    assert np.isnan(Y[1, 1])  # missing cell is NaN, never 0


def test_unit_time_matrix_sorts_units_and_times():
    df = pd.DataFrame(
        {
            "unit": ["b", "a", "b", "a"],
            "time": [2, 2, 1, 1],
            "y": [4.0, 3.0, 2.0, 1.0],
        }
    )
    Y, units, times = unit_time_matrix(df, "unit", "time", "y")
    assert list(units) == ["a", "b"]
    assert times.tolist() == [1.0, 2.0]
    assert Y.tolist() == [[1.0, 3.0], [2.0, 4.0]]


def test_unit_time_matrix_missing_column_raises():
    df = pd.DataFrame({"unit": ["a"], "time": [1], "y": [1.0]})
    with pytest.raises(ValueError, match="nope"):
        unit_time_matrix(df, "unit", "time", "nope")


# ---------------------------------------------------------------- recovery / ATT


def test_donor_recovery_pinned_seed():
    # Observed over seeds 0-9 (defaults, n_donors=8): true donors all
    # selected in 10/10 seeds; sum of simplex weight on true donors in
    # [0.80, 1.00]. Seed 0 pinned (weight sum 0.92); plan gate 0.7.
    d = make_sc_synthetic(rng=np.random.default_rng(0))
    Y, units, times = _matrix(d.df)
    res = select_donors(Y, units, times, d.treated_unit, d.t0, n_donors=8)
    assert isinstance(res, DonorSelectionResult)
    assert len(res.donors) == 8
    assert set(d.true_donors) <= set(res.donors)
    w_true = sum(
        w for u, w in zip(res.donors, res.weights, strict=True) if u in d.true_donors
    )
    assert w_true >= 0.7


def test_att_accuracy_pinned_seed():
    # Observed |att_post - 10| over seeds 0-9 (n_donors=8): [0.01, 0.92].
    # Seed 0 pinned (0.92); plan gate 1.5 (effect=10, noise=0.5).
    d = make_sc_synthetic(rng=np.random.default_rng(0))
    Y, units, times = _matrix(d.df)
    res = select_donors(Y, units, times, d.treated_unit, d.t0, n_donors=8)
    assert np.isfinite(res.att_post)
    assert abs(res.att_post - d.effect) <= 1.5
    assert res.pre_rmspe < res.post_rmspe  # effect shows up post only
    # effect_by_time is the gap; post-period mean of defined gaps == att_post
    post = res.times >= d.t0
    gap_post = res.effect_by_time[post]
    assert np.isclose(np.nanmean(gap_post), res.att_post)


# ---------------------------------------------------------------- pre-only honesty


@pytest.mark.parametrize("scoring", ["rmse", "corr"])
def test_post_outcome_mutation_never_moves_scores_or_weights(scoring):
    d = make_sc_synthetic(rng=np.random.default_rng(3))
    mutated = d.df.copy()
    post = mutated["time"] >= d.t0
    mutated.loc[post, "y"] *= 10.0  # donors AND treated

    Ya, ua, ta = _matrix(d.df)
    Yb, ub, tb = _matrix(mutated)
    a = select_donors(Ya, ua, ta, d.treated_unit, d.t0, n_donors=8, scoring=scoring)
    b = select_donors(Yb, ub, tb, d.treated_unit, d.t0, n_donors=8, scoring=scoring)

    # Selection artifacts bitwise unchanged.
    assert list(a.donors) == list(b.donors)
    assert np.array_equal(a.weights, b.weights)
    assert len(a.scores) == len(b.scores)
    for sa, sb in zip(a.scores, b.scores, strict=True):
        assert isinstance(sa, DonorScore)
        assert sa.unit == sb.unit
        assert sa.rank == sb.rank
        assert (sa.pre_rmse == sb.pre_rmse) or (
            np.isnan(sa.pre_rmse) and np.isnan(sb.pre_rmse)
        )
        assert (sa.pre_corr == sb.pre_corr) or (
            np.isnan(sa.pre_corr) and np.isnan(sb.pre_corr)
        )
    # Pre-period counterfactual and pre-RMSPE bitwise unchanged.
    pre = a.times < d.t0
    assert np.array_equal(a.y0_hat[pre], b.y0_hat[pre], equal_nan=True)
    assert a.pre_rmspe == b.pre_rmspe
    # Only post-period estimation moves.
    assert a.att_post != b.att_post
    assert a.post_rmspe != b.post_rmspe


# ---------------------------------------------------------------- pool sizing


def test_n_donors_none_uses_all_complete_candidates():
    d = make_sc_synthetic(n_units=12, rng=np.random.default_rng(1))
    Y, units, times = _matrix(d.df)
    res = select_donors(Y, units, times, d.treated_unit, d.t0, n_donors=None)
    assert len(res.donors) == 11  # all complete candidates
    assert len(res.scores) == 11
    assert res.extras["n_candidates"] == 11
    assert res.extras["n_dropped_incomplete"] == 0
    assert res.weights.shape == (11,)


def test_n_donors_k_returns_exactly_topk():
    d = make_sc_synthetic(n_units=12, rng=np.random.default_rng(1))
    Y, units, times = _matrix(d.df)
    res_all = select_donors(Y, units, times, d.treated_unit, d.t0, n_donors=None)
    res3 = select_donors(Y, units, times, d.treated_unit, d.t0, n_donors=3)
    assert len(res3.donors) == 3
    assert list(res3.donors) == [s.unit for s in res_all.scores[:3]]
    # scores still cover ALL complete candidates, ranked
    assert len(res3.scores) == 11
    assert [s.rank for s in res3.scores] == list(range(1, 12))


def test_n_donors_must_be_positive():
    d = make_sc_synthetic(n_units=6, rng=np.random.default_rng(1))
    Y, units, times = _matrix(d.df)
    with pytest.raises(ValueError, match="n_donors"):
        select_donors(Y, units, times, d.treated_unit, d.t0, n_donors=0)


# ---------------------------------------------------------------- scoring methods


def _tiny_panel():
    """Deterministic panel: donor 'level' close in level but weakly correlated;
    donor 'shape' offset by +10 but perfectly correlated with treated."""
    times = [0.0, 1.0, 2.0, 3.0, 4.0]
    rows = []
    treated = [1.0, 2.0, 3.0, 4.0, 5.0]
    shape = [11.0, 12.0, 13.0, 14.0, 15.0]  # corr 1.0, rmse 10
    level = [1.5, 1.4, 3.6, 3.4, 5.6]  # small rmse, corr < 1
    for name, ys in (("treated", treated), ("shape", shape), ("level", level)):
        rows += [{"unit": name, "time": t, "y": y} for t, y in zip(times, ys, strict=True)]
    return pd.DataFrame(rows)


def test_scoring_rmse_vs_corr_rank_differently():
    Y, units, times = _matrix(_tiny_panel())
    by_rmse = select_donors(Y, units, times, "treated", 4.0, scoring="rmse")
    by_corr = select_donors(Y, units, times, "treated", 4.0, scoring="corr")
    assert by_rmse.scores[0].unit == "level"
    assert by_corr.scores[0].unit == "shape"
    assert by_corr.scores[0].pre_corr > by_corr.scores[1].pre_corr


def test_pre_corr_nan_below_three_points():
    Y, units, times = _matrix(_tiny_panel())
    res = select_donors(Y, units, times, "treated", 2.0)  # 2 pre times
    assert all(np.isnan(s.pre_corr) for s in res.scores)
    assert all(np.isfinite(s.pre_rmse) for s in res.scores)


def test_unknown_scoring_raises():
    Y, units, times = _matrix(_tiny_panel())
    with pytest.raises(ValueError, match="scoring"):
        select_donors(Y, units, times, "treated", 4.0, scoring="mse")


# ---------------------------------------------------------------- failure paths


def test_single_unit_panel_nan_att_inf_rmspe():
    df = pd.DataFrame(
        {"unit": ["treated"] * 5, "time": range(5), "y": [1.0, 2.0, 3.0, 4.0, 5.0]}
    )
    Y, units, times = _matrix(df)
    res = select_donors(Y, units, times, "treated", 3.0)
    assert np.isnan(res.att_post)
    assert res.pre_rmspe == float("inf")  # never 0 on failure
    assert res.post_rmspe == float("inf")
    assert res.donors == []
    assert res.scores == []
    assert res.weights.size == 0
    assert np.all(np.isnan(res.y0_hat))
    assert np.all(np.isnan(res.effect_by_time))
    assert "failure" in res.extras


def test_treated_unobserved_pre_t0_inf_rmspe():
    rows = []
    for name in ("treated", "d1", "d2"):
        for t in range(4):
            y = np.nan if (name == "treated" and t < 2) else float(t + 1)
            rows.append({"unit": name, "time": t, "y": y})
    df = pd.DataFrame(rows)
    Y, units, times = _matrix(df)
    res = select_donors(Y, units, times, "treated", 2.0)
    assert res.pre_rmspe == float("inf")
    assert res.pre_rmspe != 0.0
    assert np.isnan(res.att_post)
    assert "failure" in res.extras


def test_no_complete_candidate_nan_everywhere():
    rows = []
    for name in ("treated", "d1"):
        for t in range(4):
            y = np.nan if (name == "d1" and t == 1) else float(t + 1)
            rows.append({"unit": name, "time": t, "y": y})
    df = pd.DataFrame(rows)
    Y, units, times = _matrix(df)
    res = select_donors(Y, units, times, "treated", 2.0)
    assert res.donors == []
    assert np.isnan(res.att_post)
    assert res.pre_rmspe == float("inf")
    assert res.extras["n_candidates"] == 1
    assert res.extras["n_dropped_incomplete"] == 1
    assert "failure" in res.extras


def test_no_post_period_nan_att():
    Y, units, times = _matrix(_tiny_panel())
    res = select_donors(Y, units, times, "treated", 10.0)  # t0 beyond max time
    assert np.isnan(res.att_post)
    assert res.post_rmspe == float("inf")
    assert np.isfinite(res.pre_rmspe)
    assert len(res.donors) == 2  # selection itself still works


def test_unknown_treated_unit_raises():
    Y, units, times = _matrix(_tiny_panel())
    with pytest.raises(ValueError, match="treated_unit"):
        select_donors(Y, units, times, "nope", 4.0)


# ---------------------------------------------------------------- Dataset adapter


def _as_dataset(df, t0):
    df = df.copy()
    df["T"] = ((df["unit"] == "treated") & (df["time"] >= t0)).astype(float)
    spec = DatasetSpec(
        treatment="T", outcome="y", forcing=[], covariates=[], time="time", unit="unit"
    )
    return Dataset(df, spec)


def test_dataset_adapter_matches_direct_call():
    d = make_sc_synthetic(n_units=10, rng=np.random.default_rng(2))
    ds = _as_dataset(d.df, d.t0)
    via_ds = select_donors_from_dataset(ds, d.treated_unit, d.t0, n_donors=5)
    Y, units, times = _matrix(d.df)
    direct = select_donors(Y, units, times, d.treated_unit, d.t0, n_donors=5)
    assert list(via_ds.donors) == list(direct.donors)
    assert np.array_equal(via_ds.weights, direct.weights)
    assert np.array_equal(via_ds.y0_hat, direct.y0_hat, equal_nan=True)
    assert via_ds.att_post == direct.att_post


def test_dataset_adapter_requires_unit_time_outcome():
    d = make_sc_synthetic(n_units=6, rng=np.random.default_rng(2))
    df = d.df.copy()
    df["T"] = 0.0
    spec = DatasetSpec(
        treatment="T", outcome="y", forcing=[], covariates=[], time="time", unit=None
    )
    ds = Dataset(df, spec)
    with pytest.raises(ValueError, match="spec.unit"):
        select_donors_from_dataset(ds, "treated", d.t0)
