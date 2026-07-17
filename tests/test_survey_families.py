"""kink, iv, sc, bunching and dee family runners (phase survey task 6).

Contract under test: the five remaining families run REUSING the existing
modules (regression_kink/sensitivity_grid, discover_instruments,
unit_time_matrix/select_donors/sc_placebo_test, binned_poisson_jump,
lord3_scan + dee_debias); verdicts follow the plan's gates (Holm over
declared cutoffs/thresholds at ALPHA, SC placebo at SC_ALPHA, weak-IV
demotion per audit 10, dee's runtime gate on a credible rdd); ambiguous sc
treated units and missing outcomes surface as ``needs_input`` — never a
crash; guidance config hints flow into the declared inputs and the recorded
override.

Stochastic calibration (>= 5 seeds each during implementation; one pinned):

- panel DGP (12 units x 16 years, effect 10, noise 0.5, DGP seed 0; survey
  seeds 0-4 through the full survey): sc credible with p = 1/12 on 5/5
  (rng-free given the DGP; DGP seeds 0-4 gave p in {1/12, 2/12, 4/12} with
  n_used = 11 always); did null with finite p in [0.3, 0.8] on 5/5 at q=9;
  rdd lands on the honest "no rdd configuration could be scanned" null.
  Survey seed 0 pinned.
- kink DGP (n=800, slopes 0.5 -> 2.5 at z=2, noise 0.3, DGP seeds 0-4,
  rng-free): tau in [1.88, 2.19] (gate [1.0, 3.0]), p <= 1e-43; no-kink p
  in [0.078, 0.975] (all above ALPHA). Seed 0 pinned (tau 2.19).
- iv DGP (make_iv_synthetic n=96, p=8, s=3, mu2=180, binarized T, DGP seed
  base 100 per napkin): through the survey, DGP seeds 100-104 at survey
  seed 0 were all credible with estimation-half F in [14.5, 39.9]; survey
  seeds 0-4 at DGP seed 100 gave F in [8.0, 29.9] (seed 1 an honest weak-F
  null — the n=96 honest split halves to 48 rows). Pinned pair (DGP 100,
  survey 0): F = 24.0. The mu2=0 pure-noise pool refused selection on 3/3
  survey seeds.
- bunching reuses the task-4 calibration (tests/test_validation.py): seed-0
  tripled-mass p = 0.0, untouched-uniform p = 0.979 (rng-free given the
  draw).
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pandas as pd
import pytest

from natex.data.synthetic import make_synthetic
from natex.data.synthetic_iv import make_iv_synthetic
from natex.llm import MockBackend
from natex.survey import survey
from natex.survey.registry import FAMILY_ORDER
from natex.survey.runner import ALPHA, FamilyResult

_BUDGET = {"q": 9, "k": 25}  # small explicit test budget (plan task 5 convention)


# ---------------------------------------------------------------- DGP helpers


def _panel_df(seed=0, n_units=12, years=range(1990, 2006), t0=2000,
              effect=10.0, noise=0.5, treated=("u03",)):
    """12 string units x 16 int years; binary T switches on for `treated`
    at t0 with an additive effect on y (tests/test_donors.py DGP idea)."""
    rng = np.random.default_rng(seed)
    units = [f"u{i:02d}" for i in range(n_units)]
    fe = rng.normal(0.0, 2.0, n_units)
    rows = []
    for i, u in enumerate(units):
        for t in years:
            tr = 1 if (u in treated and t >= t0) else 0
            y = fe[i] + 0.3 * (t - 1990) + effect * tr + rng.normal(0.0, noise)
            rows.append((u, t, tr, y))
    return pd.DataFrame(rows, columns=["state", "year", "T", "y"])


def _kink_df(seed=0, n=800, kink=True):
    """Piecewise-linear DGP: slope 0.5 below z=2, 2.5 above (kink 2.0)."""
    rng = np.random.default_rng(seed)
    z = rng.uniform(0.0, 4.0, n)
    if kink:
        y = 1.0 + 0.5 * np.minimum(z, 2.0) + 2.5 * np.maximum(z - 2.0, 0.0)
    else:
        y = 1.0 + 0.5 * z
    return pd.DataFrame({"z": z, "y": y + rng.normal(0.0, 0.3, n)})


def _iv_df(seed=100, mu2=180.0):
    """make_iv_synthetic frame with T binarized at its median so the profiler
    finds a treatment candidate; n=96 keeps every other family below its row
    floor (rdd needs 100, dee 200), isolating the iv runner."""
    d = make_iv_synthetic(n=96, p=8, s=3, mu2=mu2, rng=np.random.default_rng(seed))
    df = d.df.copy()
    df["T"] = (df["T"] > df["T"].median()).astype(int)
    return df, d.pool_names


# ---------------------------------------------------------------- panel e2e


def test_panel_shape_end_to_end(tmp_path):
    df = _panel_df(seed=0)
    res = survey(df, rng=np.random.default_rng(0), out_dir=tmp_path / "out",
                 budget=_BUDGET)

    assert list(res.families) == list(FAMILY_ORDER)  # all seven present

    did = res.families["did"]
    assert did.status in {"credible", "null"}, did.reason
    assert np.isfinite(did.key_numbers["p_value"])

    sc = res.families["sc"]
    assert sc.status in {"credible", "null"}, sc.reason
    p = sc.key_numbers["p_value"]
    n_placebos = sc.key_numbers["n_placebos"]
    assert np.isfinite(p)
    # +1-rank granularity: p is an integer multiple of 1/(n_used+1)
    assert abs(p * (n_placebos + 1) - round(p * (n_placebos + 1))) < 1e-9
    assert sc.diagnostics["treated_unit"] == "u03"  # auto-derived from T
    assert sc.diagnostics["t0"] == 2000.0
    assert np.isfinite(sc.key_numbers["att_post"])

    # 192 rows < 200: dee under its row floor regardless of extras
    assert res.families["dee"].status in {"skipped", "needs_input"}


# ---------------------------------------------------------------- kink


def test_kink_declared_cutoff(tmp_path):
    res = survey(_kink_df(seed=0), rng=np.random.default_rng(0),
                 out_dir=tmp_path / "out", cutoffs={"z": 2.0})
    kink = res.families["kink"]
    assert kink.status == "credible", kink.reason
    assert 1.0 <= kink.key_numbers["tau"] <= 3.0  # true slope change 2.0
    assert kink.key_numbers["min_holm_p"] <= ALPHA
    assert np.isfinite(kink.key_numbers["bandwidth"])
    assert kink.key_numbers["n_used"] > 0
    rows = kink.diagnostics["sensitivity"]["z"]
    assert len(rows) == 3  # bandwidths {0.5, 1, 2} x bw
    assert all(set(r) == {"bandwidth", "tau", "se"} for r in rows)

    res0 = survey(_kink_df(seed=0, kink=False), rng=np.random.default_rng(0),
                  out_dir=tmp_path / "out0", cutoffs={"z": 2.0})
    assert res0.families["kink"].status == "null"


# ---------------------------------------------------------------- iv


def test_iv_declared_instruments(tmp_path):
    df, pool = _iv_df(seed=100, mu2=180.0)
    res = survey(df, rng=np.random.default_rng(0), out_dir=tmp_path / "out",
                 instruments=pool)
    iv = res.families["iv"]
    assert iv.status == "credible", iv.reason
    assert iv.key_numbers["n_selected"] >= 1
    assert iv.key_numbers["first_stage_F"] > 10
    assert np.isfinite(iv.key_numbers["tau"])
    assert iv.key_numbers["n_discovery"] + iv.key_numbers["n_estimation"] == len(df)

    df0, pool0 = _iv_df(seed=100, mu2=0.0)  # pi = 0: pure-noise pool
    res0 = survey(df0, rng=np.random.default_rng(0), out_dir=tmp_path / "out0",
                  instruments=pool0)
    iv0 = res0.families["iv"]
    assert iv0.status == "null"
    assert "no instrument selected" in iv0.reason


# ---------------------------------------------------------------- bunching


def test_bunching_declared_threshold(tmp_path):
    rng = np.random.default_rng(0)  # task-4 calibrated draw (seed 0)
    s = rng.uniform(-1.0, 1.0, 2000)
    bump = s[(s >= 0.0) & (s < 0.1)]
    gap = np.concatenate([s, bump, bump])  # mass on [0, 0.1] tripled

    res = survey(pd.DataFrame({"v": gap}), rng=np.random.default_rng(0),
                 out_dir=tmp_path / "gap", thresholds={"v": 0.0})
    b = res.families["bunching"]
    assert b.status == "credible", b.reason
    assert b.key_numbers["min_holm_p"] <= ALPHA
    assert b.key_numbers["n_finite"] == len(gap)
    assert np.isfinite(b.key_numbers["theta"])

    res0 = survey(pd.DataFrame({"v": s}), rng=np.random.default_rng(0),
                  out_dir=tmp_path / "null", thresholds={"v": 0.0})
    assert res0.families["bunching"].status == "null"

    # audit 18: a threshold on a calendar-time column carries the caveat
    years = pd.DataFrame({"year": np.tile(np.arange(1990, 2010), 15)})
    resy = survey(years, rng=np.random.default_rng(0),
                  out_dir=tmp_path / "year", thresholds={"year": 2000.0})
    by = resy.families["bunching"]
    assert by.status in {"credible", "null"}
    assert "information-free" in " ".join(by.diagnostics["caveats"])


# ---------------------------------------------------------------- sc


def test_sc_needs_input_when_ambiguous(tmp_path):
    df = _panel_df(seed=0, treated=("u03", "u07"))  # TWO ever-treated units
    res = survey(df, rng=np.random.default_rng(0), out_dir=tmp_path / "out",
                 budget=_BUDGET)
    sc = res.families["sc"]
    assert sc.status == "needs_input"
    assert "found 2" in sc.reason
    assert "treated_unit" in sc.reason  # tells the user what to provide


# ---------------------------------------------------------------- dee


def test_dee_gate_no_gp_extra(monkeypatch, tmp_path):
    """find_spec -> None for torch/gpytorch: dee never runs and the reason
    names the extra. With find_spec real and torch actually absent (core-only
    CI) the same skip holds through the identical predicate."""
    real = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util, "find_spec",
        lambda name, *a: None if name in ("torch", "gpytorch") else real(name, *a),
    )
    df = pd.DataFrame(np.random.default_rng(0).normal(size=(250, 4)),
                      columns=[f"x{i}" for i in range(4)])
    res = survey(df, rng=np.random.default_rng(1), out_dir=tmp_path / "out")
    dee = res.families["dee"]
    assert dee.status == "skipped"
    assert 'natex-discovery[gp]' in dee.reason
    assert dee.applicability["heuristic"]["status"] == "inapplicable"


def test_dee_skipped_without_credible_rdd(monkeypatch, tmp_path):
    """Runtime gate on top of applicability: rdd forced to 'null' => dee is
    skipped with the exact no-validated-discovery reason (cheap unit — no
    scan, no GP; runs in every environment via the find_spec stub)."""
    from natex.survey import runner as runner_mod

    real = importlib.util.find_spec
    monkeypatch.setattr(  # make dee applicable even without the gp extra
        importlib.util, "find_spec",
        lambda name, *a: object() if name in ("torch", "gpytorch") else real(name, *a),
    )
    monkeypatch.setattr(
        runner_mod, "_run_rdd",
        lambda *a, **k: FamilyResult(family="rdd", status="null",
                                     reason="forced null (test)"),
    )
    ds, _ = make_synthetic(n=300, px=2, pz=2, zeta=6.0, kind="binary",
                           rng=np.random.default_rng(0))
    res = survey(ds.df, rng=np.random.default_rng(0), out_dir=tmp_path / "out",
                 budget=_BUDGET)
    dee = res.families["dee"]
    assert dee.status == "skipped"
    assert dee.reason == "no validated rdd discovery to debias"


@pytest.mark.skipif(importlib.util.find_spec("torch") is None
                    or importlib.util.find_spec("gpytorch") is None,
                    reason="gp extra not installed")
def test_dee_runs_after_credible_rdd(tmp_path):
    """When rdd is credible on the synthetic, dee actually executes and lands
    on credible|null (a degenerate surface is an honest null, never a crash).

    q must be >= 19: the +1-rank scan p has floor 1/(q+1), so q=9 can never
    clear ALPHA=0.05. Calibrated across survey seeds 0-4 on
    make_synthetic(n=400, px=2, pz=2, zeta=8.0, DGP seed 0) with budget
    q=19/k=30: rdd credible (p=0.05) on 5/5 seeds; dee reached an honest
    null (2 usable experiments < 3) on 5/5. Survey seed 0 pinned.
    """
    ds, _ = make_synthetic(n=400, px=2, pz=2, zeta=8.0, kind="binary",
                           rng=np.random.default_rng(0))
    res = survey(ds.df, rng=np.random.default_rng(0), out_dir=tmp_path / "out",
                 budget={"q": 19, "k": 30})
    assert res.families["rdd"].status == "credible", res.families["rdd"].reason
    dee = res.families["dee"]
    assert dee.status in {"credible", "null"}, dee.reason
    assert "w_debias" in dee.key_numbers
    assert dee.key_numbers["n_experiments"] >= 0


# ---------------------------------------------------------------- guidance


def test_guidance_hint_flows_to_config(tmp_path):
    """A MockBackend kink-cutoff config hint reaches the runner (kink executes
    at the proposed cutoff) and the applicability block records the override.

    MockBackend pops in study()'s fixed task order understand -> prepare ->
    search_plan before method_applicability; three empty dicts fail schema
    validation and fall back to NullBackend heuristics (recorded), leaving
    the fourth reply for method_applicability.
    """
    mock = MockBackend([
        {}, {}, {},  # understand / prepare / search_plan -> heuristic fallback
        {
            "families": [
                {
                    "family": "kink",
                    "run": True,
                    "reason": "policy kink at z=1.5 per context",
                    "config_hints": {"cutoffs": [{"column": "z", "value": 1.5}]},
                }
            ]
        },
    ])
    df = _kink_df(seed=0)  # no declared cutoff: kink is needs_input heuristically
    res = survey(df, rng=np.random.default_rng(0), out_dir=tmp_path / "out",
                 guidance=mock)
    kink = res.families["kink"]
    assert kink.status in {"credible", "null"}  # it RAN at the proposed cutoff
    assert kink.diagnostics["cutoffs"] == {"z": 1.5}
    assert kink.applicability["override"] == {
        "heuristic_said": False,
        "analyst_said": True,
        "reason": "policy kink at z=1.5 per context",
    }
    assert res.guidance_log_path == "intake/guidance_log.jsonl"
