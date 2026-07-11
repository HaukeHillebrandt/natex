"""CI-small slices of the IV selection / SC recovery benchmark (phase-5 task 10).

Seeded 3-seed slices of ``benchmarks/run_iv_selection.py`` at the full grid
config (n=500, p=50, s=5); each test < 10 s. Pinned seeds follow the phase-4
statistical-test policy: calibrate over a wider seed range first, pin typical
passing seeds, record the observed tables here.

Calibration (2026-07-11, seeds 0..19, harness defaults n=500 p=50 s=5
endog=0.6 tau=1.0; the harness spawns (data, selection, split) streams from
``default_rng(seed)``, so these tables are NOT comparable seed-by-seed with
the napkin's raw ``default_rng(seed)`` select_instruments table):

Strong regime (mu2=180, plugin lam): recall_top3 == 1.0 in 19/20 seeds
(miss: seed 1 — rho_z=0.5 Toeplitz shrinkage absorbs z3, the known
select_instruments failure mode); precision == 1.0 and weak == 0 in 20/20
(selection is a subset of true_support, F 29.5-76.8); bias_2sls < bias_ols
in 20/20 with a wide margin (bias_2sls 0.001-0.163 vs bias_ols 0.380-0.512;
plim OLS bias endog/(pi' Sigma pi + 1) ~ 0.44). Pinned (0, 2, 4): typical
passing seeds, margins bias_ols - bias_2sls = 0.27-0.43.

Weak regime (mu2=8): the plug-in penalty (lam ~ 177) honestly REFUSES to
select in 17/20 seeds, and the three nonempty seeds (1, 10, 11) have
F 12.2-17.9 with bounded AR sets — no estimation-stage gap is visible at
the plug-in lambda. The slice therefore holds selection open with explicit
lam=60 (napkin convention, mu2=2 + lam=25 there): n_selected 5-13,
first-stage F 2.8-5.6, weak == 1 in 20/20. Full-sample AR is still
"interval" in 19/20 (boundedness <=> homoskedastic F > F_crit ~ 2.2 at
n=500) — the gap lives on the HONEST estimation half, where F halves:
honest_ar_kind unbounded/disjoint in 17/20 seeds (interval: 10, 11;
empty: 15) while the Wald CI is finite in 20/20 full-sample AND 20/20
estimation-half runs — Wald never admits weakness. Pinned (0, 1, 2):
honest_ar_kind (unbounded, unbounded, unbounded), all Wald CIs finite.

SC slice (noise=0.5, defaults n_units=20 n_pre=15 n_post=10 k_true=3
effect=10, n_donors=8): donor_recovery == 1.0 in 10/10 seeds 0..9;
abs_att_error 0.010-0.921 (mean 0.33), weight_on_true 0.80-1.00. The 1.5
mean-error gate has > 1.6x margin on the WORST single seed. Pinned (0, 1, 2):
errors (0.921, 0.010, 0.322), mean 0.42.
"""

import importlib.util
from math import isnan
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "benchmarks" / "run_iv_selection.py"


def _load_harness():
    spec = importlib.util.spec_from_file_location("run_iv_selection", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


harness = _load_harness()

STRONG_SEEDS = (0, 2, 4)  # pinned after calibration (docstring table)
WEAK_SEEDS = (0, 1, 2)
SC_SEEDS = (0, 1, 2)
MU2_STRONG = 180.0
MU2_WEAK = 8.0
WEAK_LAM = 60.0  # plugin refuses selection at mu2=8 (docstring); explicit weak penalty
SC_NOISE = 0.5
TAU = 1.0  # harness default; biases in the rows are vs this truth


@pytest.fixture(scope="module")
def strong_rows():
    return [harness.run_iv_replication(s, MU2_STRONG) for s in STRONG_SEEDS]


@pytest.fixture(scope="module")
def weak_rows():
    return [harness.run_iv_replication(s, MU2_WEAK, lam=WEAK_LAM) for s in WEAK_SEEDS]


# ------------------------------------------------------------- strong regime


def test_strong_regime_perfect_top3_recall(strong_rows):
    """mu2=180: every pinned seed recovers all of {z1, z2, z3} (mean == 1.0),
    with clean precision and no weak flag."""
    recalls = [row["recall_top3"] for row in strong_rows]
    assert float(np.mean(recalls)) == 1.0, f"recall_top3 by seed: {recalls}"
    for seed, row in zip(STRONG_SEEDS, strong_rows, strict=True):
        assert row["precision"] == 1.0, f"seed {seed}: precision {row['precision']}"
        assert row["weak"] == 0.0, f"seed {seed}: flagged weak, F {row['first_stage_F']}"


def test_strong_regime_2sls_beats_ols_bias(strong_rows):
    """mu2=180: post-Lasso 2SLS |tau_hat - tau| < OLS |tau_hat - tau|, 3/3 seeds."""
    for seed, row in zip(STRONG_SEEDS, strong_rows, strict=True):
        assert row["bias_2sls"] < row["bias_ols"], (
            f"seed {seed}: 2SLS bias {row['bias_2sls']} >= OLS bias {row['bias_ols']}"
        )


# --------------------------------------------------------------- weak regime


def test_weak_regime_ar_admits_weakness_wald_never_does(weak_rows):
    """mu2=8 honesty gap: the estimation-half AR set goes unbounded/disjoint
    in >= 1/3 seeds while every Wald CI (full sample and estimation half)
    stays finite — Wald never admits weakness."""
    for seed, row in zip(WEAK_SEEDS, weak_rows, strict=True):
        assert row["wald_finite"] == 1.0, f"seed {seed}: full-sample Wald CI not finite"
        # a defined coverage value means the estimation-half Wald CI was finite
        assert not isnan(row["honest_wald_covers"]), f"seed {seed}: honest Wald CI not finite"
    kinds = [row["honest_ar_kind"] for row in weak_rows]
    assert sum(k in ("unbounded", "disjoint") for k in kinds) >= 1, f"honest ar_kind: {kinds}"


def test_weak_regime_flagged_weak(weak_rows):
    """The first-stage weak flag (audit item 10) fires in every pinned weak seed."""
    for seed, row in zip(WEAK_SEEDS, weak_rows, strict=True):
        assert row["weak"] == 1.0, f"seed {seed}: first-stage F {row['first_stage_F']}"


# ------------------------------------------------------------------ SC slice


def test_sc_donor_recovery_and_att():
    """noise=0.5: all k_true donors inside the top-8 pool in 3/3 seeds and
    mean |ATT - effect| < 1.5."""
    rows = [harness.run_sc_replication(s, SC_NOISE) for s in SC_SEEDS]
    for seed, row in zip(SC_SEEDS, rows, strict=True):
        assert row["donor_recovery"] == 1.0, (
            f"seed {seed}: recovered {row['n_true_recovered']}/{row['k_true']}"
        )
    errs = [row["abs_att_error"] for row in rows]
    assert float(np.mean(errs)) < 1.5, f"abs ATT errors: {errs}"


# ------------------------------------------- harness contract (no file I/O)


def test_iv_grid_returns_documented_columns():
    """2-cell mini-grid: exact IV_RESULT_COLUMNS, one row per (seed, mu2) cell."""
    df = harness.run_iv_grid((0, 1), (MU2_STRONG,))
    assert list(df.columns) == list(harness.IV_RESULT_COLUMNS)
    assert len(df) == 2
    assert sorted(df["seed"]) == [0, 1]
    assert set(df["mu2"]) == {MU2_STRONG}


def test_sc_grid_returns_documented_columns():
    df = harness.run_sc_grid((0,), (SC_NOISE,))
    assert list(df.columns) == list(harness.SC_RESULT_COLUMNS)
    assert len(df) == 1
    assert df.iloc[0]["seed"] == 0 and df.iloc[0]["noise"] == SC_NOISE


def test_iv_replication_determinism(weak_rows):
    """Identical (seed, mu2, lam) => identical row, bitwise on floats
    (one rng root; plugin/explicit-lambda selection is itself rng-free)."""
    again = harness.run_iv_replication(WEAK_SEEDS[0], MU2_WEAK, lam=WEAK_LAM)
    first = weak_rows[0]
    assert set(again) == set(first)
    for key, value in first.items():
        other = again[key]
        same = value == other or (
            isinstance(value, float) and isnan(value) and isnan(other)
        )
        assert same, f"{key}: {value!r} != {other!r}"
