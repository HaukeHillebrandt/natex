"""Tests for Abadie in-space RMSPE-ratio placebo inference (phase 5, task 8).

Protocol under test (audit item 1/5 lineage): every complete donor of the
treated run is refit as pseudo-treated under the IDENTICAL selection rule,
with the treated unit removed from every placebo panel; ratio = post/pre
RMSPE (sign-agnostic, so two-sided by construction); p is the +1-rank form
(1 + #{placebo ratio >= treated}) / (n_used + 1); fewer than 5 usable
placebos -> p = NaN (phase-3 _MIN_USABLE policy, never a fake 1.0).
Deterministic throughout: no rng inside sc_placebo_test.

Calibration notes (repo statistical-test policy: stochastic assertions run
across >= 5 seeds during implementation; observed values in per-test
comments, one seed pinned with margin).
"""

import numpy as np
import pandas as pd
import pytest

from natex.data.synthetic_sc import make_sc_synthetic
from natex.iv.donors import (
    SCPlaceboReport,
    _rmspe_ratio,
    sc_placebo_test,
    select_donors,
    unit_time_matrix,
)


def _matrix(df):
    return unit_time_matrix(df, "unit", "time", "y")


# ---------------------------------------------------------------- signal / null


def test_signal_exact_plus_one_rank_p():
    # Observed over seeds 0-9 (effect=10, noise=0.5, n_units=20): treated
    # ratio in [15.7, 29.7], max placebo ratio in [1.79, 6.57] -> treated is
    # the strict max in 10/10 seeds, p == 1/20 every time. Seed 0 pinned
    # (treated 27.1 vs max placebo 2.81).
    d = make_sc_synthetic(effect=10.0, noise=0.5, n_units=20, rng=np.random.default_rng(0))
    Y, units, times = _matrix(d.df)
    rep = sc_placebo_test(Y, units, times, d.treated_unit, d.t0)
    assert isinstance(rep, SCPlaceboReport)
    assert rep.ratios.size == 19  # every donor usable
    assert rep.n_skipped == 0
    assert rep.ratio_treated > rep.ratios.max()  # treated has the largest ratio
    assert rep.p_value == 1.0 / 20.0  # exact +1-rank arithmetic with 19 placebos
    # ratios sorted descending, aligned with placebo_units, treated excluded
    assert np.all(np.diff(rep.ratios) <= 0)
    assert len(rep.placebo_units) == 19
    assert d.treated_unit not in rep.placebo_units


def test_null_calibration_across_seeds():
    # effect=0, seeds 0-9. Plan gate: p >= 0.05 in >= 8/10 (with 19 usable
    # placebos min possible p is exactly 1/20 = 0.05, so this also guards
    # that no placebo is skipped). Observed p over seeds 0-9: [0.35, 1.00,
    # 0.70, 0.60, 0.75, 0.90, 0.90, 1.00, 0.30, 0.75] -> 10/10 at >= 0.05,
    # mean 0.725 (null E[p] = 0.525 with 19 placebos); calibrated extra
    # gate mean >= 0.3.
    ps = []
    for seed in range(10):
        d = make_sc_synthetic(
            effect=0.0, noise=0.5, n_units=20, rng=np.random.default_rng(seed)
        )
        Y, units, times = _matrix(d.df)
        rep = sc_placebo_test(Y, units, times, d.treated_unit, d.t0)
        ps.append(rep.p_value)
    ps = np.asarray(ps)
    assert np.isfinite(ps).all()
    assert int(np.sum(ps >= 0.05)) >= 8
    assert float(ps.mean()) >= 0.3


# ---------------------------------------------------------------- +1-rank arithmetic


def test_plus_one_rank_matches_manual_recompute():
    # 5 placebos (n_units=6) is the smallest usable count; every ratio is
    # recomputed manually through the public select_donors API on the
    # treated-row-deleted panel and must match bitwise (same rule, same
    # deterministic fitter).
    d = make_sc_synthetic(n_units=6, rng=np.random.default_rng(7))
    Y, units, times = _matrix(d.df)
    rep = sc_placebo_test(Y, units, times, d.treated_unit, d.t0)

    res_t = select_donors(Y, units, times, d.treated_unit, d.t0)
    assert rep.ratio_treated == res_t.post_rmspe / res_t.pre_rmspe

    i_t = int(np.flatnonzero(units == d.treated_unit)[0])
    Yp = np.delete(Y, i_t, axis=0)
    up = np.delete(units, i_t)
    manual = {}
    for u in [s.unit for s in res_t.scores]:  # every complete donor
        r = select_donors(Yp, up, times, u, d.t0)
        manual[u] = r.post_rmspe / r.pre_rmspe

    got = dict(zip(rep.placebo_units, rep.ratios, strict=True))
    assert got == manual
    expected_p = (1 + sum(v >= rep.ratio_treated for v in manual.values())) / (len(manual) + 1)
    assert rep.p_value == expected_p
    assert rep.ratios.size == 5


def test_four_placebos_p_nan():
    # n_used = 4 < 5 -> p = NaN (never a fake 1.0); the ratios themselves
    # are still reported and the treated ratio stays finite.
    d = make_sc_synthetic(n_units=5, rng=np.random.default_rng(7))
    Y, units, times = _matrix(d.df)
    rep = sc_placebo_test(Y, units, times, d.treated_unit, d.t0)
    assert rep.ratios.size == 4
    assert rep.n_skipped == 0
    assert np.isnan(rep.p_value)
    assert np.isfinite(rep.ratio_treated)


def test_no_usable_placebos_nan_p():
    # Two-unit panel: the single placebo panel has no donors left once the
    # treated row is removed -> failed placebo fit, counted in n_skipped.
    rows = [
        {"unit": name, "time": t, "y": float(t + off)}
        for name, off in (("treated", 0.0), ("d1", 1.0))
        for t in range(6)
    ]
    Y, units, times = _matrix(pd.DataFrame(rows))
    rep = sc_placebo_test(Y, units, times, "treated", 4.0)
    assert np.isnan(rep.p_value)
    assert rep.ratios.size == 0
    assert rep.n_skipped == 1
    assert rep.extras["n_failed"] == 1


# ---------------------------------------------------------------- poor-fit exclusion


def test_exclude_poor_fit_drops_noisy_placebo():
    # One donor's PRE outcomes get a deterministic +/-8 sawtooth: as a
    # pseudo-treated unit its donors cannot match, so its pre-RMSPE blows
    # past 2x the treated unit's and mult=2 drops it, counted in n_skipped;
    # p is recomputed over the survivors (whose ratios are unchanged).
    # Observed over seeds 0-4 (n_units=12): unit03 dropped in 5/5 (its
    # placebo pre-RMSPE 7.5-8.6 vs treated 0.49-0.70); other extreme donors
    # may also exceed mult=2 (2-5 total drops), n_used stays >= 5 in 5/5.
    # Seed 1 pinned (5 dropped, n_used 6, p 0.083 -> 0.143).
    d = make_sc_synthetic(n_units=12, rng=np.random.default_rng(1))
    df = d.df.copy()
    noisy = "unit03"
    pre = (df["unit"] == noisy) & (df["time"] < d.t0)
    saw = 8.0 * np.where(np.arange(int(pre.sum())) % 2 == 0, 1.0, -1.0)
    df.loc[pre, "y"] = df.loc[pre, "y"].to_numpy() + saw
    Y, units, times = _matrix(df)

    base = sc_placebo_test(Y, units, times, d.treated_unit, d.t0)
    excl = sc_placebo_test(Y, units, times, d.treated_unit, d.t0, exclude_poor_fit=2.0)

    assert noisy in base.placebo_units
    assert noisy not in excl.placebo_units
    assert noisy in excl.extras["poor_fit_units"]
    assert excl.extras["placebo_pre_rmspe"][noisy] > 2.0 * excl.extras["treated_pre_rmspe"]

    n_dropped = len(excl.extras["poor_fit_units"])
    assert n_dropped >= 1
    assert excl.n_skipped == base.n_skipped + n_dropped
    assert excl.ratios.size == base.ratios.size - n_dropped
    assert excl.n_skipped == (
        excl.extras["n_failed"] + excl.extras["n_zero_pre_rmspe"] + excl.extras["n_poor_fit"]
    )
    # survivors keep their ratios; p recomputed over the survivor count
    base_ratios = dict(zip(base.placebo_units, base.ratios, strict=True))
    excl_ratios = dict(zip(excl.placebo_units, excl.ratios, strict=True))
    assert set(excl_ratios) == set(base_ratios) - set(excl.extras["poor_fit_units"])
    assert all(excl_ratios[u] == base_ratios[u] for u in excl_ratios)
    expected_p = (1 + int(np.sum(excl.ratios >= excl.ratio_treated))) / (excl.ratios.size + 1)
    assert excl.p_value == expected_p


def test_exclude_poor_fit_must_be_positive():
    d = make_sc_synthetic(n_units=6, rng=np.random.default_rng(0))
    Y, units, times = _matrix(d.df)
    with pytest.raises(ValueError, match="exclude_poor_fit"):
        sc_placebo_test(Y, units, times, d.treated_unit, d.t0, exclude_poor_fit=0.0)


# ---------------------------------------------------------------- treated exclusion


def test_treated_never_in_placebo_pools():
    d = make_sc_synthetic(n_units=6, rng=np.random.default_rng(3))
    Y, units, times = _matrix(d.df)
    rep = sc_placebo_test(Y, units, times, d.treated_unit, d.t0)
    pools = rep.extras["placebo_pools"]
    assert set(pools) == set(rep.placebo_units)  # nothing skipped on this panel
    for u, pool in pools.items():
        assert len(pool) >= 1
        assert d.treated_unit not in pool  # treated excluded from every placebo pool
        assert u not in pool  # a pseudo-treated unit is not its own donor


def test_placebos_use_identical_selection_rule():
    # n_donors/scoring propagate: with n_donors=3 every placebo pool has
    # exactly 3 donors (4 candidates are available once treated is removed).
    d = make_sc_synthetic(n_units=6, rng=np.random.default_rng(3))
    Y, units, times = _matrix(d.df)
    rep = sc_placebo_test(Y, units, times, d.treated_unit, d.t0, n_donors=3, scoring="corr")
    assert rep.extras["n_donors"] == 3
    assert rep.extras["scoring"] == "corr"
    for pool in rep.extras["placebo_pools"].values():
        assert len(pool) == 3


# ---------------------------------------------------------------- determinism / ratio policy


def test_deterministic_rng_free():
    d = make_sc_synthetic(n_units=8, rng=np.random.default_rng(5))
    Y, units, times = _matrix(d.df)
    a = sc_placebo_test(Y, units, times, d.treated_unit, d.t0)
    b = sc_placebo_test(Y, units, times, d.treated_unit, d.t0)
    assert a.p_value == b.p_value
    assert a.ratio_treated == b.ratio_treated
    assert np.array_equal(a.ratios, b.ratios)
    assert a.placebo_units == b.placebo_units
    assert a.n_skipped == b.n_skipped


def test_rmspe_ratio_never_fake_zero_on_failure():
    inf = float("inf")
    assert np.isnan(_rmspe_ratio(inf, inf))  # failed fit (select_donors failure path)
    assert np.isnan(_rmspe_ratio(inf, 3.0))  # post/inf must not fabricate 0.0
    assert np.isnan(_rmspe_ratio(2.0, inf))  # no defined post period
    assert _rmspe_ratio(0.0, 2.0) == inf  # genuinely perfect pre fit
    assert np.isnan(_rmspe_ratio(0.0, 0.0))  # 0/0 undefined
    assert _rmspe_ratio(2.0, 1.0) == 0.5
