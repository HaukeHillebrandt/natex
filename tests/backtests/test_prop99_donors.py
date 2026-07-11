"""Backtest: Prop 99 synthetic-control donor selection (phase 5, spec section 8 row 6 reuse).

ADH (2010) target on the same 39-state x 31-year smoking panel as
``test_prop99.py``: treated_unit = California, t0 = 1989, outcome = cigsale.
Protocol (fixed before calibration, phase-5 plan task 11): ``unit_time_matrix``
on (state, year, cigsale), scoring="rmse", ``n_donors=None`` (full complete
pool) and ``n_donors=8`` variants, placebo with ``exclude_poor_fit=None``.
Everything below is deterministic — no rng anywhere in the donor path.

Calibration run of record (2026-07-11, real CSV, macOS arm64):

* n_donors=8 pre-trend ranks: Montana(1), Idaho(2), West Virginia(3), Iowa(4),
  Colorado(5), Nebraska(6), Connecticut(7), Wisconsin(8) — exactly 3 of the
  ADH five {Colorado, Connecticut, Montana, Nevada, Utah} in the top 8.
  Nevada and Utah are NOT top-8 by raw pre-RMSE: level distance to
  California's trajectory penalizes Nevada (high-smoking, ~190 packs) and
  Utah (low-smoking, ~65) even though both match California's SHAPE — the
  simplex fit, which can mix levels, recovers them (reconciliation note for
  docs/method_cards/iv_sc.md; same donor set as the phase-3 SuDDDS synthetic
  control: Utah/Montana/Nevada/Connecticut).
* Full pool nonzero weights: Utah .394, Montana .232, Nevada .205,
  Connecticut .109, New Hampshire .045, Colorado .015 — summed weight on the
  ADH five = 0.955; att_post = -19.514 (matches the canonical ADH ~-19
  average through 2000 and the phase-3 finding -19.5; the thesis Table 6.1
  -8.96 reflects a shorter effective post window — same reconciliation as
  test_prop99.py). pre_rmspe 1.656 (8-donor variant: att -22.648, pre 3.671).
* Placebo (exclude_poor_fit=None, all 38 placebos usable, 0 skipped):
  treated ratio 12.440 (full pool) / 6.570 (n_donors=8); California ranks
  3/39 in BOTH variants -> p = 3/39 = 0.077. Gap vs ADH's p = 1/39 (CA most
  extreme): Missouri (23.9) and Virginia (19.8) out-rank CA here because our
  variant fits outcomes only (no covariate V-weights) and applies NO
  poor-pre-fit exclusion — ADH's figure discards placebos with pre-MSPE far
  above California's before ranking. Pinned as a regression below.
"""

import numpy as np
import pytest

from natex.iv.donors import sc_placebo_test, select_donors, unit_time_matrix
from natex.iv.search import select_instruments

pytestmark = pytest.mark.backtest

TREATED = "California"
T0 = 1989
ADH_FIVE = frozenset({"Colorado", "Connecticut", "Montana", "Nevada", "Utah"})


@pytest.fixture(scope="module")
def matrix(load_or_skip):
    ds = load_or_skip("prop99")
    Y, units, times = unit_time_matrix(ds.df, "state", "year", "cigsale")
    assert Y.shape == (39, 31) and not np.isnan(Y).any()  # balanced panel, no empty cells
    return Y, units, times


@pytest.fixture(scope="module")
def top8(matrix):
    Y, units, times = matrix
    return select_donors(Y, units, times, TREATED, T0, n_donors=8, scoring="rmse")


@pytest.fixture(scope="module")
def full(matrix):
    Y, units, times = matrix
    return select_donors(Y, units, times, TREATED, T0, n_donors=None, scoring="rmse")


# ---------------------------------------------------------------------------
# 1: ADH donor set — pre-trend ranks (top-8) and full-pool simplex weight
# ---------------------------------------------------------------------------


def test_top8_pretrend_ranks_recover_adh_donors(top8):
    assert len(top8.donors) == 8
    assert top8.extras["n_candidates"] == 38 and top8.extras["n_dropped_incomplete"] == 0
    # Plan contract: >= 3 of the ADH five in the top-8 pre-trend ranks.
    hits = ADH_FIVE & set(top8.donors)
    assert len(hits) >= 3
    # Pinned run of record: exactly these three (Nevada/Utah excluded by
    # level-RMSE — see module docstring), Montana the single best pre fit.
    assert hits == {"Colorado", "Connecticut", "Montana"}
    assert top8.scores[0].unit == "Montana" and top8.scores[0].rank == 1


def test_full_pool_weight_concentrates_on_adh_five(full):
    assert len(full.donors) == 38  # full complete pool
    assert full.extras["converged"]
    w = dict(zip(full.donors, full.weights, strict=True))
    adh_weight = float(sum(w[d] for d in ADH_FIVE))
    # Plan contract >= 0.5; pinned band around the observed 0.955.
    assert adh_weight >= 0.5
    assert 0.90 <= adh_weight <= 1.0 + 1e-9
    # The four heavyweight donors are ADH's own (Utah/Montana/Nevada/Connecticut).
    top4 = sorted(w, key=w.get, reverse=True)[:4]
    assert set(top4) == {"Utah", "Montana", "Nevada", "Connecticut"}


# ---------------------------------------------------------------------------
# 2: effect — negative ATT within the reconciled full-post-period band
# ---------------------------------------------------------------------------


def test_att_post_negative_within_band(full, top8):
    # Plan contract [-35, -5]; pinned bands around the observed run of record
    # (full pool -19.514, top-8 -22.648; see module docstring reconciliation).
    for res, lo, hi in ((full, -25.0, -14.0), (top8, -30.0, -15.0)):
        assert res.att_post < 0
        assert -35.0 <= res.att_post <= -5.0
        assert lo <= res.att_post <= hi
    # More donors -> weakly better pre fit (observed 1.656 vs 3.671).
    assert full.pre_rmspe < top8.pre_rmspe
    assert full.pre_rmspe < 3.0 and full.post_rmspe > 10.0  # ADH-style pre/post contrast


# ---------------------------------------------------------------------------
# 3: in-space placebo — RMSPE-ratio rank and +1-rank p-value
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_donors", [None, 8], ids=["full", "top8"])
def test_placebo_rank_and_p_value(matrix, n_donors):
    Y, units, times = matrix
    rep = sc_placebo_test(
        Y, units, times, TREATED, T0, n_donors=n_donors, scoring="rmse", exclude_poor_fit=None
    )
    assert rep.extras["n_used"] == 38 and rep.n_skipped == 0  # every placebo usable
    assert np.isfinite(rep.ratio_treated) and rep.ratio_treated > 1.0
    rank = 1 + int(np.sum(rep.ratios >= rep.ratio_treated))
    # Plan contract: top-5 rank, p <= 0.2. Observed rank 3/39 in both variants
    # (p = 0.077); +1 rank of slack because every placebo ratio comes from an
    # SLSQP refit (same convention as test_prop99.py's synthetic-control pin).
    assert rank <= 5
    assert rep.p_value <= 0.2
    assert rep.p_value <= 4 / 39
    # Gap vs ADH's 1/39 is the two poor-pre-fit placebos (module docstring).
    outrankers = {u for u, r in zip(rep.placebo_units, rep.ratios, strict=False)
                  if r >= rep.ratio_treated}
    assert outrankers <= {"Missouri", "Virginia", "Georgia"}


# ---------------------------------------------------------------------------
# 4: determinism — two runs bitwise-identical (no rng in the donor path)
# ---------------------------------------------------------------------------


def test_deterministic_bitwise(matrix, top8, full):
    Y, units, times = matrix
    for first, n_donors in ((top8, 8), (full, None)):
        again = select_donors(Y, units, times, TREATED, T0, n_donors=n_donors, scoring="rmse")
        assert again.donors == first.donors
        np.testing.assert_array_equal(again.weights, first.weights)  # bitwise, not approx
        np.testing.assert_array_equal(again.y0_hat, first.y0_hat)
        assert again.att_post == first.att_post
        assert again.pre_rmspe == first.pre_rmspe and again.post_rmspe == first.post_rmspe


# ---------------------------------------------------------------------------
# 5: OPTIONAL Egger IV stretch (non-blocking; spec section 10 stress case)
#
# Statutory 1[wpop >= c] dummies vs decoy dummies as instruments for council
# size, controls = smooth log-population polynomial, same wpop < 20000 protocol
# restriction as test_egger_koethenbuerger.py. Selection reads only
# (T, pool, controls) — never an outcome (phase-5 honesty policy).
#
# CALIBRATED FINDING (2026-07-11, for docs/method_cards/iv_sc.md, task 12):
# in-sample, rcsize is EXACTLY the statutory step function of wpop
# (8 + 4*[>=1001] + 2*[>=2001] + 2*[>=3001] + 4*[>=5001] + 4*[>=10001], zero
# residual), so the BCCH plug-in loadings psi_j = sqrt(mean(z_j^2 eps^2))
# collapse to 0 on iteration, the penalty vanishes, and the final Lasso
# SUPPORT balloons to all 10 dummies — the strict "decoy-free support" claim
# FAILS (xfail below). What DOES hold, and is pinned as a passing regression:
# the plug-in Lasso coefficients put ~4-orders-of-magnitude more mass on
# statutory dummies (|pi| ~ 2-4) than decoys (|pi| < 5e-4), and post-Lasso
# OLS recovers the statutory jumps (4, 2, 2, 4, 4) with decoy coefficients
# ~0 — statutory dummies are selected "ahead of" decoys in coefficient mass,
# just not by support exclusion in this zero-noise degenerate first stage.
# ---------------------------------------------------------------------------

STATUTORY = (1001, 2001, 3001, 5001, 10001)
DECOYS = (1501, 2501, 4001, 7001, 15001)  # non-statutory population points
STAT_NAMES = [f"stat_{c}" for c in STATUTORY]
DECOY_NAMES = [f"decoy_{c}" for c in DECOYS]


@pytest.fixture(scope="module")
def egger_search(load_or_skip):
    ds = load_or_skip("egger_koethenbuerger")
    df = ds.df[ds.df["wpop"] < 20000].reset_index(drop=True)  # phase-2 protocol restriction
    pop = df["wpop"].to_numpy(dtype=float)
    T = df["rcsize"].to_numpy(dtype=float)
    pool = np.column_stack([(pop >= c).astype(float) for c in (*STATUTORY, *DECOYS)])
    lp = np.log(pop)
    lp = (lp - lp.mean()) / lp.std()
    controls = np.column_stack([lp, lp**2, lp**3])  # smooth population polynomial
    return select_instruments(T, pool, controls=controls, pool_names=STAT_NAMES + DECOY_NAMES)


@pytest.mark.xfail(
    strict=False,
    reason="deterministic first stage: zero residuals collapse the plug-in penalty, so the "
    "Lasso support keeps the decoys too (documented finding, method card task 12)",
)
def test_egger_stretch_strict_decoy_free_support(egger_search):
    assert set(egger_search.selected) == set(STAT_NAMES)


def test_egger_statutory_dummies_dominate_decoys(egger_search):
    # The non-xfail slice of the stretch: pinned coefficient-mass ordering.
    assert set(STAT_NAMES) <= set(egger_search.selected)  # all 5 statutory selected
    assert np.isfinite(egger_search.first_stage_F) and not egger_search.weak
    assert egger_search.partial_r2 == pytest.approx(1.0)  # exact statutory first stage
    pi_lasso = dict(zip(STAT_NAMES + DECOY_NAMES, egger_search.pi_lasso, strict=True))
    pi_post = dict(zip(STAT_NAMES + DECOY_NAMES, egger_search.pi_post, strict=True))
    assert max(abs(pi_lasso[d]) for d in DECOY_NAMES) < 0.01 * min(
        abs(pi_lasso[s]) for s in STAT_NAMES
    )
    for name, jump in zip(STAT_NAMES, (4.0, 2.0, 2.0, 4.0, 4.0), strict=True):
        assert pi_post[name] == pytest.approx(jump, abs=1e-3)  # statutory council-size jumps
    assert max(abs(pi_post[d]) for d in DECOY_NAMES) < 1e-6
