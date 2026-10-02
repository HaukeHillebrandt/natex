"""Survey runner skeleton (phase survey task 5): SurveyResult, rdd/did, isolation.

Contract under test: ``survey()`` requires an explicit rng; always returns ALL
SEVEN families in FAMILY_ORDER with statuses from the fixed 5-value vocabulary
(credible|null|skipped|needs_input|failed); rdd runs end-to-end via
``natex.discover`` on the synthetic-shape CSV; a raising family runner is
isolated (status "failed", verbatim error, survey completes); per-family rng
sub-streams come from ONE upfront ``rng.spawn(7)`` so declaring an extra
family never shifts another family's stream; ``survey.json`` round-trips
through ``SurveyResult.load`` with an identical families dict.

Stochastic assertions: the rdd verdict on the synthetic CSV is only pinned to
{"credible", "null"} (both acceptable at q=9), and spawn stability is bitwise
equality under a fixed seed — checked across seeds 0/7 during implementation;
no margin needed.
"""

import json

import numpy as np
import pandas as pd
import pytest

from natex.data.synthetic import make_synthetic
from natex.llm import MockBackend
from natex.survey import SurveyResult, survey
from natex.survey.registry import FAMILY_ORDER
from natex.survey.runner import ALPHA

_BUDGET = {"q": 9, "k": 25}  # small explicit test budget (plan task 5)

_STATUSES = {"credible", "null", "inconclusive", "skipped", "needs_input", "failed"}


def _write_synthetic_csv(root):
    """make_synthetic(n=300) binary-treatment CSV with a decoy binary column
    'holiday' inserted BEFORE T (recipe from tests/test_cli_study.py)."""
    ds, _ = make_synthetic(
        n=300, px=3, pz=2, zeta=6.0, kind="binary", rng=np.random.default_rng(0)
    )
    df = ds.df.copy()
    df.insert(df.columns.get_loc("T"), "holiday",
              np.random.default_rng(1).integers(0, 2, len(df)))
    path = root / "synthetic.csv"
    df.to_csv(path, index=False)
    return path


def _plain_cross_section(seed=0, n=200):
    """Pure rng normals x0..x3: no binary column, no panel, nothing to run."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame(rng.normal(size=(n, 4)), columns=[f"x{i}" for i in range(4)])


def test_survey_requires_rng(tmp_path):
    df = _plain_cross_section()
    with pytest.raises(ValueError):
        survey(df, out_dir=tmp_path / "out")


def test_rdd_shape_end_to_end(tmp_path):
    csv = _write_synthetic_csv(tmp_path)
    out = tmp_path / "out"
    res = survey(str(csv), rng=np.random.default_rng(0), out_dir=out, budget=_BUDGET)

    # ALL SEVEN families, fixed order.
    assert list(res.families) == list(FAMILY_ORDER)

    rdd = res.families["rdd"]
    assert rdd.status in {"credible", "null"}, rdd.reason
    assert np.isfinite(rdd.key_numbers["p_value"])
    # key_numbers is FLAT name->number: no nested dicts/lists (report contract)
    assert all(not isinstance(v, (dict, list)) for v in rdd.key_numbers.values())
    assert rdd.details_path == "families/rdd.json"
    assert (out / "families" / "rdd.json").exists()
    assert res.coverage["rdd"] is not None  # discover's searched block surfaced

    did = res.families["did"]
    assert did.status in {"skipped", "needs_input"}
    assert did.reason

    kink = res.families["kink"]
    assert kink.status == "needs_input"
    assert kink.reason == "no pre-declared cutoff (kink is candidate evaluation, not discovery)"

    path = out / "survey.json"
    assert path.exists()
    loaded = SurveyResult.load(path)
    assert loaded.families == res.families


def test_plain_cross_section_lists_all_seven(tmp_path):
    out = tmp_path / "out"
    res = survey(_plain_cross_section(), rng=np.random.default_rng(1), out_dir=out)
    assert list(res.families) == list(FAMILY_ORDER)
    for fam in res.families.values():
        assert fam.status in {"skipped", "needs_input"}  # none "failed"
        assert fam.reason
    assert len(res.coverage["not_run"]) == 7
    assert res.coverage["ran"] == []
    assert (out / "survey.json").exists()


def test_failure_isolation(monkeypatch, tmp_path):
    # NOTE: ``import natex.survey.runner`` would fail here — natex/__init__
    # binds the name ``survey`` to the function (discover precedent), so the
    # attribute path is shadowed; go through the subpackage module instead.
    from natex.survey import runner as runner_mod

    def boom(*args, **kwargs):
        raise RuntimeError("boom-xyzzy")

    monkeypatch.setattr(runner_mod, "_run_rdd", boom)
    csv = _write_synthetic_csv(tmp_path)
    out = tmp_path / "out"
    res = survey(str(csv), rng=np.random.default_rng(0), out_dir=out, budget=_BUDGET)

    rdd = res.families["rdd"]
    assert rdd.status == "failed"
    assert rdd.error == "boom-xyzzy"  # verbatim str(exc)
    assert "boom-xyzzy" in rdd.diagnostics["traceback"]
    # survey completed: all seven present, other families untouched
    assert list(res.families) == list(FAMILY_ORDER)
    assert res.families["did"].status in {"skipped", "needs_input"}
    assert res.families["kink"].status == "needs_input"
    assert (out / "survey.json").exists()


def test_spawn_stability(tmp_path):
    """Declaring a bunching threshold (an extra family) never shifts rdd's stream."""
    csv = _write_synthetic_csv(tmp_path)
    res_a = survey(str(csv), rng=np.random.default_rng(7), out_dir=tmp_path / "a",
                   budget=_BUDGET)
    res_b = survey(str(csv), rng=np.random.default_rng(7), out_dir=tmp_path / "b",
                   budget=_BUDGET, thresholds={"x0": 0.5})
    assert res_a.families["rdd"].key_numbers == res_b.families["rdd"].key_numbers
    assert res_a.families["rdd"].status == res_b.families["rdd"].status


def test_status_vocabulary(tmp_path):
    res = survey(_plain_cross_section(seed=3), rng=np.random.default_rng(4),
                 out_dir=tmp_path / "out")
    assert {f.status for f in res.families.values()} <= _STATUSES
    # survey.json statuses match too
    saved = json.loads((tmp_path / "out" / "survey.json").read_text())
    assert {f["status"] for f in saved["families"].values()} <= _STATUSES


def _mixed_time_panel(root):
    """Issue #49 shape: numeric decimal-year ``t`` plus a string ``date`` column.

    ``t`` is neither name- nor value-time-like, so the profiled panel
    candidate (and the null backend's did candidate) uses the string ``date``
    column — the declared ``--time t`` must still bind the did family.
    """
    rng = np.random.default_rng(3)
    units = [f"u{i}" for i in range(6)]
    periods = np.arange(10)
    rows = []
    for u_index, u in enumerate(units):
        adopt = 4 + (u_index % 3)
        for p in periods:
            treated = int(p >= adopt)
            rows.append({
                "u": u,
                "date": f"2023-{p + 1:02d}-01",
                "t": 2023.0 + p / 12.0 + 0.001,
                "treat": treated,
                "y": 0.3 * p + 0.8 * treated + 0.05 * rng.standard_normal(),
            })
    df = pd.DataFrame(rows)
    path = root / "panel.csv"
    df.to_csv(path, index=False)
    return path


def test_declared_time_binds_the_did_family_over_candidate_guesses(tmp_path):
    """Issue #49: --time must bind even when a ranked did candidate disagrees."""
    csv = _mixed_time_panel(tmp_path)
    out = tmp_path / "out"

    res = survey(
        csv, rng=np.random.default_rng(0), out_dir=out,
        budget=_BUDGET, time="t", unit="u",
    )

    did = res.families["did"]
    assert "time column must be numeric" not in (did.error or "")
    assert "time column must be numeric" not in did.reason
    assert did.status in ("credible", "null"), (did.status, did.reason, did.error)
    assert did.diagnostics.get("searched") is not None


def test_kink_outcome_guess_skips_declared_design_columns_and_counters(tmp_path):
    """Issue #52: declared unit/time and monotone counters are never auto outcomes."""
    rng = np.random.default_rng(9)
    n = 120
    df = pd.DataFrame({
        "run": np.arange(n),                                   # monotone int counter
        "sector": np.repeat(np.arange(11, 31), 6).astype(float),  # declared unit codes
        "x0": np.linspace(0.0, 1.0, n) + 0.001 * rng.standard_normal(n),
        "y": rng.normal(10.0, 1.0, n),
    })
    csv = tmp_path / "roles.csv"
    df.to_csv(csv, index=False)

    res = survey(
        csv, rng=np.random.default_rng(0), out_dir=tmp_path / "out",
        budget=_BUDGET, unit="sector", cutoffs={"x0": 0.5},
    )

    per_cutoff = res.families["kink"].diagnostics.get("per_cutoff", {})
    assert per_cutoff, res.families["kink"].reason
    assert per_cutoff["x0"]["outcome"] == "y"


def test_rdd_family_flags_mechanical_time_step_rediscovery(tmp_path):
    """Issue #52: treatment = deterministic step in a time column is never credible."""
    rng = np.random.default_rng(2)
    years = np.repeat(np.arange(2000, 2060), 4).astype(float)
    df = pd.DataFrame({
        "year": years,
        "post": (years >= 2030.0).astype(int),
        "activity": rng.normal(5.0, 1.0, years.size),
        "output": rng.normal(0.0, 1.0, years.size),
    })
    csv = tmp_path / "mechanical.csv"
    df.to_csv(csv, index=False)

    res = survey(
        csv, rng=np.random.default_rng(0), out_dir=tmp_path / "out",
        budget={"q": 39, "k": 25},
    )

    rdd = res.families["rdd"]
    # a vacuous design is "could not test", never "tested and found nothing"
    assert rdd.status == "inconclusive", (rdd.status, rdd.reason)
    mechanical = rdd.diagnostics.get("mechanical_step")
    assert mechanical, (rdd.status, rdd.reason, rdd.diagnostics.get("caveats"))
    assert mechanical["treatment"] == "post"
    assert mechanical["column"] == "year"
    assert any("deterministic step" in c for c in rdd.diagnostics["caveats"])


def test_kink_family_says_when_a_group_contrast_is_not_expressible(tmp_path):
    """Issue #52: a pooled kink null must say the group (DiK) contrast is out of scope."""
    rng = np.random.default_rng(11)
    t = np.tile(np.arange(40, dtype=float) + 0.5, 2)
    group = np.r_[np.zeros(40), np.ones(40)]
    df = pd.DataFrame({
        "t": t,
        "g": group.astype(int),
        "y": 0.1 * t + 0.4 * np.maximum(t - 20.0, 0.0) * group
        + 0.3 * rng.standard_normal(t.size),
    })
    csv = tmp_path / "pooled.csv"
    df.to_csv(csv, index=False)

    res = survey(
        csv, rng=np.random.default_rng(0), out_dir=tmp_path / "out",
        budget=_BUDGET, cutoffs={"t": 20.0},
    )

    kink = res.families["kink"]
    caveats = " ".join(kink.diagnostics["caveats"])
    assert "cannot express" in caveats and "--design dik" in caveats
    if kink.status == "null":
        assert "group" in kink.reason


def test_sc_family_aggregates_a_multi_treated_indicator(tmp_path):
    """Issue #50: >1 ever-treated unit aggregates into one treated series
    instead of a refusal."""
    rng = np.random.default_rng(7)
    units = [f"u{i}" for i in range(6)]
    periods = np.arange(12)
    rows = []
    for u_index, u in enumerate(units):
        base = 1.0 + 0.1 * u_index
        for p in periods:
            treated = int(u_index < 2 and p >= 6)
            rows.append({
                "ticker": u,
                "period": float(p),
                "treated_flag": treated,
                "value": base + 0.2 * p + 0.5 * treated
                + 0.02 * rng.standard_normal(),
            })
    csv = tmp_path / "multi.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)

    res = survey(
        csv, rng=np.random.default_rng(0), out_dir=tmp_path / "out",
        budget=_BUDGET, unit="ticker", time="period",
    )

    sc = res.families["sc"]
    # four donors remain after aggregation: the placebo test refuses (<5),
    # which is an inconclusive verdict, never a null one
    assert sc.status in ("credible", "null", "inconclusive"), (sc.status, sc.reason, sc.error)
    aggregated = sc.diagnostics.get("treated_units_aggregated")
    assert aggregated == ["u0", "u1"]
    assert "aggregate" in str(sc.diagnostics.get("treated_unit"))
    assert any("aggregat" in c for c in sc.diagnostics["caveats"])


def test_sc_family_with_numeric_units_and_string_declared_treated_unit(tmp_path):
    """Declared string treated_unit='1' matches numeric unit column in survey panel."""
    rng = np.random.default_rng(42)
    units = np.arange(1, 8)
    periods = np.arange(10)
    rows = []
    for u in units:
        base = 1.0 + 0.1 * float(u)
        for p in periods:
            rows.append({
                "unit_id": int(u),
                "year": float(p),
                "val": base + 0.2 * float(p) + 0.02 * rng.standard_normal(),
            })
    csv = tmp_path / "numeric_panel.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)

    mock = MockBackend([
        {}, {}, {},  # understand / prepare / search_plan -> heuristic fallback
        {
            "families": [
                {
                    "family": "sc",
                    "run": True,
                    "reason": "evaluate unit 1",
                    "config_hints": {"treated_unit": "1", "t0": 5.0},
                }
            ]
        },
    ])

    res = survey(
        csv,
        guidance=mock,
        rng=np.random.default_rng(0),
        out_dir=tmp_path / "out",
        budget=_BUDGET,
        unit="unit_id",
        time="year",
    )
    sc = res.families["sc"]
    assert sc.status in ("credible", "null"), (sc.status, sc.reason, sc.error)
    assert sc.diagnostics["treated_unit"] == 1




def test_rdd_family_scans_the_full_role_space_and_reports_every_outcome(tmp_path):
    """The survey's rdd family must not stop at the plan's one or two
    candidates: every binary column x every single non-time forcing column
    is scanned (coverage recorded), and the best boundary is estimated against
    every candidate outcome so the report says which outcome jumps there."""
    csv = _write_synthetic_csv(tmp_path)  # binary T + decoy 'holiday'; x0..x2, y
    res = survey(str(csv), rng=np.random.default_rng(0), out_dir=tmp_path / "out",
                 budget=_BUDGET)
    rdd = res.families["rdd"]
    cov = res.coverage["rdd"]
    assert cov["role_space"]["treatments"] == ["holiday", "T"]
    assert cov["exhaustive_candidates"] >= 6  # 2 treatments x (3 singles + joint), deduped
    assert cov["n_scanned"] >= 6
    by_outcome = rdd.diagnostics["effects_by_outcome"]
    assert "y" in by_outcome and "2sls" in by_outcome["y"]
    assert rdd.diagnostics["best_candidate"]["treatment"] == "T"


def _three_unit_panel(seed=0, years=range(1990, 2006), t0=2000):
    """Three string units x 16 years, one treated from t0: too few donors for
    the in-space RMSPE-ratio placebo (needs >= 5 usable placebos)."""
    rng = np.random.default_rng(seed)
    rows = []
    for i, u in enumerate(("u00", "u01", "u02")):
        for t in years:
            tr = int(u == "u01" and t >= t0)
            rows.append((u, t, tr, 2.0 * i + 0.3 * (t - 1990) + 10.0 * tr + rng.normal(0, 0.5)))
    return pd.DataFrame(rows, columns=["state", "year", "T", "y"])


def test_sc_family_with_too_few_placebos_is_inconclusive_not_null(tmp_path):
    """A refused placebo test (p = NaN) means the design could not be tested;
    reporting it as a null result would fabricate a negative finding."""
    res = survey(_three_unit_panel(), rng=np.random.default_rng(0),
                 out_dir=tmp_path / "out", budget=_BUDGET)
    sc = res.families["sc"]
    assert sc.status == "inconclusive", (sc.status, sc.reason)
    assert "placebo" in sc.reason
    assert sc.key_numbers["p_value"] is None


def test_scanless_rdd_family_is_inconclusive_not_null(tmp_path):
    """Every enumerated rdd configuration invalid: nothing was tested, so the
    verdict is inconclusive with the coverage count in the reason."""
    from natex.survey import runner as runner_mod

    class _Rep:
        searched = {"n_invalid": 2, "n_total": 2, "n_scanned": 0}
        configs = []

        @staticmethod
        def best():
            return None

    res = runner_mod._scanless_result("rdd", _Rep(), {"caveats": []})
    assert res.status == "inconclusive"
    assert "2 invalid of 2 enumerated" in res.reason


def _calendar_series(n=240, kink=0.0, curve=0.0, seed=0):
    """Monthly calendar-time series ('year' is time-like by name): optional
    convex curvature (an exponential with no event anywhere) and/or a real
    slope change of ``kink`` per year at 2015."""
    rng = np.random.default_rng(seed)
    t = 2005.0 + np.arange(n) / 12.0
    y = 0.1 * (t - 2005.0) + kink * np.maximum(t - 2015.0, 0.0)
    if curve:
        y = y + np.exp(curve * (t - 2005.0))
    return pd.DataFrame({"year": t, "y": y + 0.1 * rng.standard_normal(n)})


def test_calendar_time_kink_is_gated_by_placebo_calibration_not_hc1(tmp_path):
    """Issue #51: on a smooth convex series HC1 finds a 'kink' at almost any
    date. The gate for a calendar-time running variable is the placebo-
    calibrated p over shifted cutoffs; the HC1 p is reported as nominal only."""
    res = survey(_calendar_series(curve=0.25), rng=np.random.default_rng(0),
                 out_dir=tmp_path / "out", cutoffs={"year": 2015.0})
    kink = res.families["kink"]
    pc = kink.diagnostics["per_cutoff"]["year"]
    assert pc["inference"] == "placebo_calibrated"
    assert pc["n_placebos"] >= 19
    assert pc["nominal_p"] < 0.05  # the oversized HC1 p
    assert pc["placebo_calibrated_p"] > 0.05  # the honest one
    assert kink.status == "null", (kink.status, kink.reason)
    assert "placebo" in kink.reason
    assert kink.key_numbers["placebo_calibrated_p"] == pc["placebo_calibrated_p"]


def test_calendar_time_kink_with_a_real_bend_stays_credible(tmp_path):
    res = survey(_calendar_series(kink=2.0), rng=np.random.default_rng(0),
                 out_dir=tmp_path / "out", cutoffs={"year": 2015.0})
    kink = res.families["kink"]
    pc = kink.diagnostics["per_cutoff"]["year"]
    assert pc["placebo_calibrated_p"] <= 0.05
    assert kink.status == "credible", (kink.status, kink.reason)


def test_calendar_time_kink_with_too_few_placebo_positions_is_inconclusive(tmp_path):
    """16 quarterly values (4 rows each) cannot host 19 placebo cutoffs: the
    floor 1/(n+1) sits above alpha, so the cutoff is not gateable."""
    rng = np.random.default_rng(1)
    q = np.repeat(2020.0 + np.arange(16) / 4.0, 4)
    df = pd.DataFrame({"year": q, "y": 0.5 * (q - 2020.0) + 0.05 * rng.standard_normal(q.size)})
    res = survey(df, rng=np.random.default_rng(0), out_dir=tmp_path / "out",
                 cutoffs={"year": 2022.0})
    kink = res.families["kink"]
    pc = kink.diagnostics["per_cutoff"]["year"]
    assert pc["n_placebos"] < 19
    assert pc["min_attainable_p"] > ALPHA
    assert kink.status == "inconclusive", (kink.status, kink.reason)
    assert "placebo" in kink.reason
