"""Tests for the honest instrument-discovery pipeline (phase 5, task 5).

Statistical-test policy: every stochastic assertion below was calibrated
across >= 5 seeds during implementation; the pinned seed and the observed
ranges are recorded in a comment next to each threshold.
"""

import numpy as np
import pytest

from natex.data.synthetic_iv import make_iv_synthetic
from natex.iv.pipeline import InstrumentDiscovery, discover_instruments
from natex.iv.search import select_instruments
from natex.validate.honest import honest_split


def _boosted_iv_frame(seed, n=2000, p=20, s=4, mu2=180.0, n_invalid=0, phi=0.5, boost=0.4):
    """BCCH draw whose LAST TWO pool columns get real first-stage strength.

    make_iv_synthetic plants violators with pi = 0 (exclusion-only), so Lasso
    never selects them. Adding ``bump = boost * (z_{p-1} + z_p)`` to T and
    ``tau * bump`` to y keeps the structural equation ``y = tau T + e (+ phi
    violators)`` exact while making the last two columns relevant — selected
    violators when n_invalid = 2, valid extra instruments when n_invalid = 0.
    """
    data = make_iv_synthetic(
        n=n, p=p, s=s, mu2=mu2, n_invalid=n_invalid, phi=phi,
        rng=np.random.default_rng(seed),
    )
    df = data.df.copy()
    last_two = data.pool_names[-2:]
    bump = boost * df[last_two].sum(axis=1)
    df["T"] = df["T"] + bump
    df["y"] = df["y"] + data.tau * bump
    return data, df, last_two


def test_end_to_end_tau_recovery_honest():
    # Calibrated across seeds 0..7: |tau - 1| in [0.003, 0.179] (> 0.15 at
    # seeds 1, 4, 5 — the estimation half has only n=500 rows, 2SLS sd ~ 0.1);
    # ar_kind == "interval" and weak False at 8/8 seeds; the AR interval
    # covers 1.0 at 7/8 seeds (misses at seed 4). Pinned seed 2
    # (|tau-1| = 0.013, F_est = 29.3, ar_ci = (0.633, 1.278)).
    data = make_iv_synthetic(n=1000, p=40, s=4, mu2=180.0, rng=np.random.default_rng(2))
    res = discover_instruments(
        data.df, "T", data.pool_names, outcome="y", rng=np.random.default_rng(2)
    )
    assert isinstance(res, InstrumentDiscovery)
    assert res.honest is True
    assert res.estimate is not None
    assert abs(res.estimate.tau - 1.0) < 0.15
    assert res.estimate.ar_kind == "interval"
    lo, hi = res.estimate.ar_ci
    assert lo < 1.0 < hi
    assert res.estimate.weak_instrument is False
    assert res.search.weak is False


def test_honest_split_mechanics_selection_bitwise_and_rows_disjoint():
    data = make_iv_synthetic(n=600, p=20, s=3, mu2=150.0, rng=np.random.default_rng(1))
    res = discover_instruments(
        data.df, "T", data.pool_names, outcome="y", rng=np.random.default_rng(42)
    )
    idx_d = res.extras["idx_discovery"]
    idx_e = res.extras["idx_estimation"]
    # estimation rows and discovery rows are disjoint and exhaustive
    assert len(set(idx_d) & set(idx_e)) == 0
    assert sorted([*idx_d, *idx_e]) == list(range(len(data.df)))
    assert res.n_discovery == len(idx_d) == 300
    assert res.n_estimation == len(idx_e) == 300
    # selection is bitwise-identical to running select_instruments on the
    # discovery-half rows directly (rng consumed by the split alone)
    idx_d2, idx_e2 = honest_split(len(data.df), 0.5, rng=np.random.default_rng(42))
    assert np.array_equal(idx_d, idx_d2)
    assert np.array_equal(idx_e, idx_e2)
    pool = data.df[data.pool_names].to_numpy()
    manual = select_instruments(
        data.df["T"].to_numpy()[idx_d2], pool[idx_d2], pool_names=data.pool_names
    )
    assert res.search.selected == manual.selected
    assert np.array_equal(res.search.pi_lasso, manual.pi_lasso)
    assert np.array_equal(res.search.pi_post, manual.pi_post)
    assert res.search.lam == manual.lam
    assert res.search.first_stage_F == manual.first_stage_F
    # the estimate used exactly the estimation half (no NaNs in this DGP)
    assert res.estimate.n_used == res.n_estimation


def test_planted_violators_flagged_by_estimation_half_j():
    # Calibrated across seeds 0..4: violator run selects both violators 5/5
    # and j_p in [5.2e-25, 1.7e-15]; all-valid run j_p in [0.034, 0.87]
    # (null-uniform, seed 2 lands below 0.05 by chance). Pinned seed 0
    # (violator j_p = 1.6e-19, valid j_p = 0.41).
    _, df_bad, last_two = _boosted_iv_frame(seed=0, n_invalid=2)
    pool_names = [c for c in df_bad.columns if c.startswith("z")]
    bad = discover_instruments(
        df_bad, "T", pool_names, outcome="y", rng=np.random.default_rng(0)
    )
    assert set(last_two) <= set(bad.search.selected)  # violators were selected
    assert bad.estimate.j_p is not None
    assert bad.estimate.j_p < 0.05

    _, df_ok, _ = _boosted_iv_frame(seed=0, n_invalid=0)
    ok = discover_instruments(
        df_ok, "T", pool_names, outcome="y", rng=np.random.default_rng(0)
    )
    assert ok.estimate.j_p is not None
    assert ok.estimate.j_p > 0.05


def test_weak_regime_reports_unbounded_or_disjoint_ar_never_fake_interval():
    # mu2 = 2: plugin lam would empty the selection, so an explicit small
    # lambda forces a nonempty weak selection. Calibrated across seeds 0..4:
    # selection nonempty 5/5 (k in [3, 6]), estimation-half F in [0.73, 2.06],
    # weak True 5/5, ar_kind "unbounded" at seeds 0, 2, 4 and "disjoint" at
    # seeds 1, 3 — never "interval". Pinned seed 0 (unbounded, F = 0.73).
    data = make_iv_synthetic(n=800, p=10, s=3, mu2=2.0, rng=np.random.default_rng(0))
    res = discover_instruments(
        data.df, "T", data.pool_names, outcome="y", lam=25.0,
        rng=np.random.default_rng(0),
    )
    assert len(res.search.selected) >= 1
    assert res.estimate.weak_instrument is True
    assert res.estimate.ar_kind in {"unbounded", "disjoint"}
    assert res.estimate.ar_ci is None  # never a fake tight interval


def test_outcome_none_still_selects():
    data = make_iv_synthetic(n=600, p=20, s=3, mu2=150.0, rng=np.random.default_rng(2))
    frame = data.df.drop(columns=["y"])  # discovery reads no y — it can't
    res = discover_instruments(
        frame, "T", data.pool_names, outcome=None, rng=np.random.default_rng(0)
    )
    assert res.estimate is None
    assert len(res.search.selected) >= 1
    assert res.n_discovery + res.n_estimation == len(frame)


def test_empty_selection_gives_nan_estimate_with_reason():
    # mu2 = 0 pure-noise pool: plugin selection is empty (calibrated in
    # test_iv_search: 5/5 empty at seeds 0..4).
    data = make_iv_synthetic(n=500, p=50, s=5, mu2=0.0, rng=np.random.default_rng(0))
    res = discover_instruments(
        data.df, "T", data.pool_names, outcome="y", rng=np.random.default_rng(0)
    )
    assert res.search.selected == []
    assert res.estimate is not None
    assert np.isnan(res.estimate.tau)
    assert not res.estimate.tau == 0.0
    assert res.estimate.weak_instrument is True
    assert res.estimate.extras["reason"] == "empty selection"


def test_honest_true_without_rng_raises():
    data = make_iv_synthetic(n=100, p=5, s=2, mu2=50.0, rng=np.random.default_rng(3))
    with pytest.raises(ValueError, match="Generator"):
        discover_instruments(data.df, "T", data.pool_names, outcome="y")


def test_same_seed_identical_split_and_results():
    data = make_iv_synthetic(n=600, p=20, s=3, mu2=150.0, rng=np.random.default_rng(4))
    a = discover_instruments(
        data.df, "T", data.pool_names, outcome="y", rng=np.random.default_rng(7)
    )
    b = discover_instruments(
        data.df, "T", data.pool_names, outcome="y", rng=np.random.default_rng(7)
    )
    assert np.array_equal(a.extras["idx_discovery"], b.extras["idx_discovery"])
    assert np.array_equal(a.extras["idx_estimation"], b.extras["idx_estimation"])
    assert a.search.selected == b.search.selected
    assert a.estimate.tau == b.estimate.tau
    assert a.estimate.se == b.estimate.se
    assert a.estimate.j_p == b.estimate.j_p


def test_dishonest_full_sample_carries_caveat():
    data = make_iv_synthetic(n=600, p=20, s=3, mu2=150.0, rng=np.random.default_rng(5))
    res = discover_instruments(data.df, "T", data.pool_names, outcome="y", honest=False)
    assert res.honest is False
    assert "caveat" in res.extras
    assert "not corrected" in res.extras["caveat"]
    assert res.n_discovery == res.n_estimation == len(data.df)
    # full-sample selection matches a direct full-sample call
    pool = data.df[data.pool_names].to_numpy()
    manual = select_instruments(data.df["T"].to_numpy(), pool, pool_names=data.pool_names)
    assert res.search.selected == manual.selected


def test_invalid_arguments_raise():
    data = make_iv_synthetic(n=100, p=5, s=2, mu2=50.0, rng=np.random.default_rng(6))
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="column"):
        discover_instruments(data.df, "nope", data.pool_names, outcome="y", rng=rng)
    with pytest.raises(ValueError, match="column"):
        discover_instruments(data.df, "T", ["z1", "ghost"], outcome="y", rng=rng)
    with pytest.raises(ValueError, match="column"):
        discover_instruments(data.df, "T", data.pool_names, outcome="ghost", rng=rng)
    with pytest.raises(ValueError, match="column"):
        discover_instruments(
            data.df, "T", data.pool_names, outcome="y", controls=["ghost"], rng=rng
        )
    with pytest.raises(ValueError, match="frac_discovery"):
        discover_instruments(
            data.df, "T", data.pool_names, outcome="y", frac_discovery=1.0, rng=rng
        )
    with pytest.raises(TypeError, match="Generator"):
        discover_instruments(data.df, "T", data.pool_names, outcome="y", rng=123)
