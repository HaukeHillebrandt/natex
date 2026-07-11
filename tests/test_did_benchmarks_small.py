"""CI-small slices of the ch.6 synthetic discovery benchmarks (phase-3 task 6).

Config per the plan: d=3, V=4, periods=10, n=1500, s = 2 values x 2 dims,
averaged over 3 pinned seeds. All calls are seeded (deterministic), so the
thresholds are hard assertions, not flaky statistics.

Calibration (seeds 0-7, zeta=8, restarts=8, windows=(3.0,), scan rng
1000+seed): wcc and greedy recover F = 1.0 on every seed; single_delta
recovers F = 1.0 on 7/8 seeds and fails to F = 0.40 on seed 2 (t0
misidentification — the documented Alg 6 multimodality). zeta=0 top LLRs lie
in [5.5, 8.2] vs zeta=8 top LLRs >= 18.1. Complexity s_dims=3 at zeta=10:
single_delta F = 1.0 on 6/8 seeds, 0.0 on seeds 3 and 6 (mean 0.75). The
pinned seeds (0, 1, 2) include the single_delta seed-2 failure, so the mean-F
margin is honest, not cherry-picked.
"""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from natex.data.synthetic_did import make_did_synthetic
from natex.did.metrics import subset_precision_recall
from natex.did.suddds import suddds_scan

SCRIPT = Path(__file__).resolve().parents[1] / "benchmarks" / "run_did_curves.py"

SEEDS = (0, 1, 2)
CFG = dict(n=1500, d=3, V=4, periods=10, tau=10.0, s_dims=2, s_values=2)
SCAN = dict(windows=(3.0,), restarts=8, bins=4)


def _scan(ds, method, seed, **kw):
    return suddds_scan(ds, method=method, rng=np.random.default_rng(1000 + seed), **SCAN, **kw)


def _metrics(result, truth):
    """(P, R, F, llr) of the top discovery; NaN metrics when nothing was found."""
    if not result.discoveries:
        return float("nan"), float("nan"), float("nan"), float("-inf")
    top = result.discoveries[0]
    p, r, f = subset_precision_recall(top.mask, truth.record_mask)
    return p, r, f, top.llr


# ---------------------------------------------------------------------------
# subset_precision_recall unit tests
# ---------------------------------------------------------------------------


def test_metrics_perfect_and_disjoint():
    a = np.array([True, True, False, False])
    assert subset_precision_recall(a, a) == (1.0, 1.0, 1.0)
    b = np.array([False, False, True, True])
    assert subset_precision_recall(a, b) == (0.0, 0.0, 0.0)


def test_metrics_partial_overlap():
    pred = np.array([True, True, True, False])
    true = np.array([True, False, False, True])
    p, r, f = subset_precision_recall(pred, true)
    assert p == pytest.approx(1 / 3)
    assert r == pytest.approx(1 / 2)
    assert f == pytest.approx(2 * (1 / 3) * (1 / 2) / (1 / 3 + 1 / 2))


def test_metrics_empty_masks_are_nan_never_zero():
    empty = np.zeros(4, dtype=bool)
    some = np.array([True, False, True, False])
    p, r, f = subset_precision_recall(empty, some)
    assert np.isnan(p) and r == 0.0 and np.isnan(f)
    p, r, f = subset_precision_recall(some, empty)
    assert p == 0.0 and np.isnan(r) and np.isnan(f)


def test_metrics_shape_mismatch_raises():
    with pytest.raises(ValueError):
        subset_precision_recall(np.zeros(3, dtype=bool), np.zeros(4, dtype=bool))


# ---------------------------------------------------------------------------
# discovery benchmark: one shared run cache (module-scoped for runtime)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def zeta8_runs():
    """(seed, method) -> (P, R, F, llr) at zeta=8; wcc/greedy on the forced
    heuristic priority branch (exhaustive_max_values=0), single_delta default."""
    out = {}
    for seed in SEEDS:
        ds, truth = make_did_synthetic(zeta=8.0, rng=np.random.default_rng(seed), **CFG)
        for method, kw in [
            ("single_delta", {}),
            ("wcc", {"exhaustive_max_values": 0}),
            ("greedy", {"exhaustive_max_values": 0}),
        ]:
            out[(seed, method)] = _metrics(_scan(ds, method, seed, **kw), truth)
    return out


def test_mean_f_at_zeta8(zeta8_runs):
    # Calibrated means over seeds (0, 1, 2): single_delta 0.80, wcc 1.00.
    for method in ("single_delta", "wcc"):
        mean_f = np.mean([zeta8_runs[(seed, method)][2] for seed in SEEDS])
        assert mean_f >= 0.6, f"{method}: mean F {mean_f:.3f} < 0.6"


def test_greedy_recall_at_most_wcc_recall(zeta8_runs):
    # Heuristic priority branch forced for both (exhaustive_max_values=0).
    greedy = np.mean([zeta8_runs[(seed, "greedy")][1] for seed in SEEDS])
    wcc = np.mean([zeta8_runs[(seed, "wcc")][1] for seed in SEEDS])
    assert greedy <= wcc + 0.05


def test_exact_branch_greedy_equals_wcc():
    # At V=4 the default exhaustive_max_values=12 activates the exact
    # per-dimension enumeration (audit 16): both double-beta methods score the
    # same candidate set and the priority ordering is irrelevant, so the top
    # LLRs coincide exactly (the exact branch draws nothing from rng either).
    ds, _ = make_did_synthetic(zeta=8.0, rng=np.random.default_rng(0), **CFG)
    greedy = _scan(ds, "greedy", 0)
    wcc = _scan(ds, "wcc", 0)
    assert greedy.discoveries and wcc.discoveries
    assert greedy.discoveries[0].llr == wcc.discoveries[0].llr


def test_zeta0_llr_below_every_zeta8_llr(zeta8_runs):
    # Same pipeline (single_delta) on zeta=0 data: pure-noise top LLRs
    # (calibrated 5.5-8.2) sit strictly below every zeta=8 top LLR
    # (calibrated minimum 18.1, on the seed-2 single_delta failure).
    zeta8_llrs = [zeta8_runs[(seed, "single_delta")][3] for seed in SEEDS]
    for seed in SEEDS:
        ds0, _ = make_did_synthetic(zeta=0.0, rng=np.random.default_rng(seed), **CFG)
        res0 = _scan(ds0, "single_delta", seed)
        llr0 = res0.discoveries[0].llr if res0.discoveries else float("-inf")
        assert llr0 < min(zeta8_llrs)


def test_complexity_three_dims_still_recovers():
    # Fig 6.2 spot-check: s_dims=3 at zeta=10, single_delta. Calibrated F over
    # seeds (0, 1, 2) is 1.0 each (seeds 3 and 6 fail to 0.0; not pinned —
    # mean over seeds 0-7 is 0.75, also >= 0.5).
    fs = []
    for seed in SEEDS:
        ds, truth = make_did_synthetic(
            zeta=10.0, rng=np.random.default_rng(seed), **{**CFG, "s_dims": 3}
        )
        fs.append(_metrics(_scan(ds, "single_delta", seed), truth)[2])
    assert np.mean(fs) >= 0.5


# ---------------------------------------------------------------------------
# full-curve script smoke test
# ---------------------------------------------------------------------------


def test_script_writes_csvs(tmp_path):
    """The curves script writes both CSVs; plots skip gracefully when
    matplotlib (optional extra) is missing."""
    cmd = [
        sys.executable,
        str(SCRIPT),
        "--zetas", "8",
        "--complexities", "1",
        "--methods", "single_delta",
        "--n-experiments", "1",
        "--n", "400",
        "--restarts", "2",
        "--out", str(tmp_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, proc.stderr
    magnitude = pd.read_csv(tmp_path / "did_curve_magnitude.csv")
    complexity = pd.read_csv(tmp_path / "did_curve_complexity.csv")
    assert len(magnitude) == 1 and len(complexity) == 1
    for df, key in ((magnitude, "zeta"), (complexity, "s_dims")):
        for col in (key, "method", "precision_mean", "recall_mean", "f_mean",
                    "t0_hit_rate", "llr_mean", "n_experiments"):
            assert col in df.columns, f"missing column {col}"
    assert (magnitude["n_experiments"] == 1).all()
