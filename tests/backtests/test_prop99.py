"""Backtest: California Proposition 99 (thesis ch.6 section 6.4.3, spec section 8 row 6).

SuDDDS on the Abadie-Diamond-Hainmueller smoking panel (39 states x 31 years,
1970-2000; binary treatment = California from 1989). Thesis parity target: all
three LLR methods perfectly rediscover (California, 1989); effect signs match
Table 6.1 (DD -10.94, Synthetic Control -8.96, GESS -6.67).

Protocol (fixed before calibration): windows (5, 8, 10), bins=4, restarts=8,
seed 0, and ``degree=0`` for the treatment background. degree=0 is REQUIRED,
not tuning: theta is a pure policy dummy whose ONLY time variation is the
candidate discontinuity, so any global time polynomial can only fit the jump
itself — with degree=1 the fitted slope leaks the step into all 38 control
states' residuals and manufactures a spurious whole-panel optimum at t0=1980
(llr 8.53) that traps greedy/wcc; with degree=0 (unit effects only) every
method recovers California exactly (llr 13.82-13.86). The thesis never reports
its background for this experiment (spec section 10 unreported-hyperparameter
risk); see docs/method_cards/suddds.md.

Investigated deviations from the thesis's printed Table 6.1 values (full
details + reconciliation in the method card; every number below is the
deterministic run of record, calibrated 2026-07-11):

* Effect magnitudes: our estimators use the FULL post period 1989-2000
  (Eq 6.18 as printed, count-corrected). The gap accumulates (~-5 packs in
  1989 to ~-30+ by 2000), so full-period means are ~2-2.5x the thesis values,
  which match a ~5-year effective post window (e.g. our dd mean gap restricted
  to 1989-1993 is -18.8; symmetric W=5 2x2 DD is -12.2 vs printed -10.94).
  Sign and ordering agree; the synthetic estimate -19.5 matches the canonical
  ADH result (~-19 average through 2000, donors Utah/Montana/Nevada/
  Connecticut).
* Significance: the audit-item-5 CORRECTED test (two-sided studentized
  |tau/se|, +1-rank, 38 enumerated placebo states) does NOT reproduce the
  thesis's "all significant at 5%": p <= 0.05 would require California to be
  the single most extreme state (1/39 = 0.026; 2/39 = 0.051), and genuinely
  non-parallel placebo states (Missouri, West Virginia: flat post-shifted
  gaps with tiny se) out-t California's trending gap. Observed ranks are
  pinned below as regressions. The significance claim that DOES survive the
  corrected pipeline is the DISCOVERY calibration: the Bernoulli-model scan
  against direct Bernoulli(p_hat) nulls (audit items 2 + 19) gives
  p = 0.01 at Q = 99.
"""

import numpy as np
import pytest

from natex.did.effects import (
    did_effect,
    placebo_dimension_tests,
    tau_randomization_test,
)
from natex.did.metrics import subset_precision_recall
from natex.did.panel import build_panel
from natex.did.suddds import suddds_scan
from natex.validate.panel import (
    anticipation_test,
    composition_test,
    panel_randomization_test,
)

pytestmark = pytest.mark.backtest

WINDOWS = (5.0, 8.0, 10.0)
BINS = 4
RESTARTS = 8
DEGREE = 0  # unit effects only — see module docstring (a policy dummy has no trend)
METHODS = ("greedy", "wcc", "single_delta")


def _scan(ds, panel, method, model="normal"):
    return suddds_scan(
        ds,
        windows=WINDOWS,
        restarts=RESTARTS,
        model=model,
        method=method,
        bins=BINS,
        degree=DEGREE,
        rng=np.random.default_rng(0),
        panel=panel,
    )


@pytest.fixture(scope="module")
def ds(load_or_skip):
    return load_or_skip("prop99")


@pytest.fixture(scope="module")
def panel(ds):
    return build_panel(ds, bins=BINS)


@pytest.fixture(scope="module")
def california(ds):
    mask = ds.df["state"].to_numpy() == "California"
    assert int(mask.sum()) == 31  # one record per year, 1970-2000
    return mask


@pytest.fixture(scope="module")
def scans(ds, panel):
    """One normal-model scan per method (thesis-parity path, audit 19 forceable)."""
    return {method: _scan(ds, panel, method) for method in METHODS}


@pytest.fixture(scope="module")
def bernoulli_scan(ds, panel):
    return _scan(ds, panel, "wcc", model="bernoulli")


@pytest.fixture(scope="module")
def discovery(scans):
    """The single_delta top discovery conditions the effect/validation tests."""
    return scans["single_delta"].discoveries[0]


# ---------------------------------------------------------------------------
# 1-2: discovery (thesis section 6.4.3: perfect recovery by all methods)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", METHODS)
def test_scan_recovers_california_1989(scans, california, method):
    top = scans[method].discoveries[0]
    assert top.t0 == 1989.0
    assert np.array_equal(top.mask, california)  # exactly the 31 CA records
    precision, recall, _ = subset_precision_recall(top.mask, california)
    assert precision == 1.0 and recall == 1.0


def test_bernoulli_model_recovers(bernoulli_scan, california):
    # Audit 19: the corrected default model for a binary theta. Observed:
    # exact recovery (precision = recall = 1.0, llr 13.86); the assertion
    # floor (CA subset of mask, precision >= 0.9) is the plan's contract.
    top = bernoulli_scan.discoveries[0]
    assert top.t0 == 1989.0
    assert not np.any(california & ~top.mask)  # California subset of mask
    precision, _, _ = subset_precision_recall(top.mask, california)
    assert precision >= 0.9


# ---------------------------------------------------------------------------
# 3: LLR significance (fitted-null Monte Carlo, +1-rank, audit items 1/2/18)
# ---------------------------------------------------------------------------


def test_llr_significant(ds, bernoulli_scan):
    # Audit-19-consistent calibration: Bernoulli scan against direct
    # Bernoulli(p_hat) null draws (audit 2). Calibration run of record
    # (2026-07-11, seed 1): p = 0.0100 — observed max LLR 13.86 vs null
    # max 7.95, q90 4.67.
    rep = panel_randomization_test(
        ds,
        bernoulli_scan,
        Q=99,
        rng=np.random.default_rng(1),
        null="bernoulli",
        scan_kwargs={"degree": DEGREE, "bins": BINS},
    )
    assert rep.null_kind == "bernoulli"
    assert rep.p_value <= 0.05


def test_ar1_null_documented_conservative(ds, scans):
    # Structural finding (investigated, method card "validation battery"):
    # the dependence-preserving ar1_unit null CANNOT calibrate a deterministic
    # policy dummy. 38 of 39 units have identically-zero residuals, so the
    # pooled AR(1) fit absorbs California's step as autocorrelated noise
    # (phi ~ 0.94) and every replica hands all 39 units that noise — null max
    # LLRs (~54) dwarf the observed 13.82 and p pins at 1.0. Documented as a
    # regression so a future null-model change is visible.
    rep = panel_randomization_test(
        ds,
        scans["single_delta"],
        Q=19,
        rng=np.random.default_rng(2),
        null="ar1_unit",
        scan_kwargs={"degree": DEGREE, "bins": BINS},
    )
    assert rep.null_kind == "ar1_unit"
    assert rep.p_value >= 0.5  # observed: 1.0


# ---------------------------------------------------------------------------
# 4: effects conditioned on the recovered discovery (Table 6.1 comparison)
# ---------------------------------------------------------------------------


def test_effects_in_line_with_table_6_1(panel, discovery):
    # Bands are the INVESTIGATED full-post-period values (module docstring);
    # deterministic run of record: dd -27.349, synthetic -19.514 (SLSQP),
    # gess -26.653 (control = Montana, pre_mse 18.35).
    dd = did_effect(panel, discovery, control="dd")
    synth = did_effect(panel, discovery, control="synthetic")
    gess = did_effect(panel, discovery, control="gess")

    assert dd.tau < 0 and -30.0 <= dd.tau <= -20.0
    assert synth.tau < 0 and -25.0 <= synth.tau <= -14.0
    assert gess.tau < 0 and -32.0 <= gess.tau <= -20.0
    # Synthetic control exists to fix dd's non-parallel pre fit (ADH's point):
    assert synth.pre_mse < dd.pre_mse  # observed: 2.74 vs 51.23
    # Binary theta: dose normalization auto-skipped (audit 19).
    assert dd.dose is None and synth.dose is None and gess.dose is None

    # Corrected two-sided studentized placebo test (audit 5), enumerate mode:
    # all 38 single-profile placebos = Abadie's placebo-in-space. The pinned
    # ranks are the investigated truth (module docstring): the thesis's 5%
    # claims do NOT survive the corrected statistic on this panel. dd/gess
    # placebo effects are optimizer-free, hence exact rank pins; synthetic
    # placebos each refit SLSQP weights, so that rank gets one rank of slack.
    rep_dd = tau_randomization_test(panel, discovery, control="dd")
    rep_sy = tau_randomization_test(panel, discovery, control="synthetic")
    rep_ge = tau_randomization_test(panel, discovery, control="gess")
    for rep in (rep_dd, rep_sy, rep_ge):
        assert rep.mode == "enumerate"
        assert rep.q == 38  # every placebo state usable (gess needed the
        # matched-dim seeding repair; before it, 38/38 failed)
        assert rep.observed < 0  # two-sided statistic, negative effect visible
    assert rep_dd.p_value == pytest.approx(13 / 39)  # observed rank 13/39
    assert rep_sy.p_value <= 8 / 39  # observed rank 7/39 (+1 slack for SLSQP)
    assert rep_ge.p_value == pytest.approx(9 / 39)  # observed rank 9/39


# ---------------------------------------------------------------------------
# 5: validation battery (audit 18 replacements for McCrary-in-time)
# ---------------------------------------------------------------------------


def test_validation_battery(ds, panel, discovery):
    # Composition: since issue #16 the counts are restricted to the
    # discovery's own subset mask, and prop99's discovery masks a single
    # unit (California) — one usable row, 10 pre / 10 post records. A
    # one-row table admits no independence test, so the report is
    # degenerate by design: NaN p, passed=False, never a silent pass.
    comp = composition_test(panel, discovery)
    assert not comp.passed
    assert np.isnan(comp.p_value)
    np.testing.assert_array_equal(comp.table, [[10, 10]])

    # Anticipation: the policy dummy is identically 0 before T0=1988, so the
    # issue-#12 pre-period refit is EXACT — every residual 0, audit-24
    # data-scaled variance floor 0, no noise scale for the placebo z. That is
    # the normal-model analog of the one-class Bernoulli guard: degenerate
    # all-NaN report, passed=False (all-degenerate-fails rule, never a
    # silent pass). Substantively there was no pre-1988 treatment movement,
    # but the test honestly reports "cannot test" rather than fabricating
    # p=1 from a 0/0 statistic.
    ant = anticipation_test(
        panel, discovery, shifts=(1, 2, 3), model="normal", degree=DEGREE
    )
    assert not ant.passed
    assert np.all(np.isnan(ant.estimates))
    assert np.all(np.isnan(ant.p_holm))

    # Per-dimension composition placebos, Holm alpha = 0.05: prop99's derived
    # covariates are state-level time-invariant summaries, so composition
    # cannot move at the cutoff — every free dimension's gaps are exactly 0
    # (zero-movement studentization repair) and Holm p = 1.0 for all four
    # free dims (mean_retprice, mean_age15to24, mean_beer, cigsale_1980).
    pdim = placebo_dimension_tests(
        panel, discovery, control="dd", rng=np.random.default_rng(0)
    )
    assert pdim.passed
    assert len(pdim.p_holm) == 4
    usable = [p for p in pdim.p_holm.values() if not np.isnan(p)]
    assert usable and all(p > 0.05 for p in usable)


# ---------------------------------------------------------------------------
# 6: y-blindness — discovery must be bitwise independent of the outcome
# ---------------------------------------------------------------------------


def test_scan_never_reads_outcome(ds, load_or_skip):
    ds_noy = load_or_skip("prop99", outcome=None)
    assert ds_noy.y is None
    res_with = _scan(ds, None, "single_delta")
    res_without = _scan(ds_noy, None, "single_delta")
    assert len(res_with.discoveries) == len(res_without.discoveries)
    for a, b in zip(res_with.discoveries, res_without.discoveries, strict=True):
        assert a.llr == b.llr  # bitwise, not approx
        assert a.t0 == b.t0
        np.testing.assert_array_equal(a.mask, b.mask)
