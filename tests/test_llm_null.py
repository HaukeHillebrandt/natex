"""NullBackend deterministic profile-only heuristics (phase llm-analyst task 3).

Contract under test: every Null ``understand``/``prepare``/``search_plan`` content
validates through ``Understanding``/``PrepPlan``/``SearchPlan`` — this is the fallback
``study()`` relies on when no LLM backend is configured. All heuristics derive from
the request payload alone (no rng, no data access), so responses are bitwise
deterministic.
"""

import json

import numpy as np
import pandas as pd
import pytest

from natex.intake.plans import SearchPlan, Understanding
from natex.intake.prep import PrepPlan
from natex.intake.profiler import profile
from natex.llm import GuidanceBackend, GuidanceRequest, NullBackend


def _panel_df():
    """Same construction as tests/test_profiler.py::test_profile_panel_dataset."""
    rng = np.random.default_rng(0)
    units, years = 40, 10
    return pd.DataFrame(
        {
            "state": np.repeat([f"s{i}" for i in range(units)], years),
            "year": np.tile(np.arange(1990, 1990 + years), units),
            "smoking": rng.normal(50, 5, units * years),
            "treated": np.zeros(units * years, dtype=int),
        }
    )


def _fake_test_score_df(n=60):
    """Fake of the MDRC test-score CSV (columns as tests/test_cli.py, continuous posttest)."""
    rng = np.random.default_rng(0)
    cols = [
        "ID", "gender", "sped", "frlunch", "esol", "black", "white", "hispanic",
        "asian", "age", "pretest", "cutoff", "treat", "posttest",
    ]
    df = pd.DataFrame({c: rng.integers(0, 2, n) for c in cols})
    df["pretest"] = rng.integers(170, 268, n)
    df["posttest"] = rng.normal(220.0, 20.0, n)
    df["treat"] = (df["pretest"] < 215).astype(int)
    return df


def _profile_dict(df):
    return json.loads(profile(df).to_json())


def _understand(df, context=None):
    be = NullBackend()
    req = GuidanceRequest(task="understand", payload={"profile": _profile_dict(df), "context": context})
    return be.complete(req).content


def test_null_backend_is_guidance_backend():
    assert isinstance(NullBackend(), GuidanceBackend)
    assert NullBackend().name == "null"


def test_determinism_same_payload_same_bytes():
    be = NullBackend()
    prof = _profile_dict(_panel_df())
    for task, payload in [
        ("understand", {"profile": prof, "context": None}),
        ("search_plan", {"profile": prof, "understanding": _understand(_panel_df()), "context": None}),
    ]:
        req = GuidanceRequest(task=task, payload=payload)
        r1 = be.complete(req)
        r2 = be.complete(req)
        assert r1.content == r2.content
        assert r1.raw_text == r2.raw_text  # bitwise
        assert r1.backend == "null"
        assert json.loads(r1.raw_text) == r1.content


def test_shape_panel():
    content = _understand(_panel_df())
    u = Understanding.model_validate(content)
    assert u.shape == "panel"
    assert u.unit_of_observation == "state"
    assert u.did_structures and u.did_structures[0].unit == "state"
    assert u.did_structures[0].time == "year"
    assert u.notes == "NullBackend heuristics (no LLM)"


def test_shape_time_series():
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"year": np.arange(1900, 2020), "gdp": rng.normal(100.0, 10.0, 120)})
    u = Understanding.model_validate(_understand(df))
    assert u.shape == "time-series"
    assert u.unit_of_observation == "row"


def test_shape_aggregated_cells():
    rng = np.random.default_rng(2)
    df = pd.DataFrame(
        {"cellmean": rng.normal(0.0, 1.0, 100), "count": rng.integers(1, 50, 100)}
    )
    u = Understanding.model_validate(_understand(df))
    assert u.shape == "aggregated-cells"


def test_shape_cross_section_fallback():
    rng = np.random.default_rng(3)
    df = pd.DataFrame(
        {
            "x": rng.normal(0.0, 1.0, 50),
            "T": rng.integers(0, 2, 50),
            "y": rng.normal(0.0, 1.0, 50),
        }
    )
    u = Understanding.model_validate(_understand(df))
    assert u.shape == "cross-section"
    assert [g.column for g in u.treatments] == ["T"]
    assert u.treatments[0].reason == "binary 0/1 column"
    outcome_cols = [g.column for g in u.outcomes]
    assert "x" in outcome_cols and "y" in outcome_cols and "T" not in outcome_cols
    assert [g.column for g in u.forcing] == ["x", "y"]


def test_understand_forcing_excludes_time_like():
    # 400 distinct time-like values >= 20 uniques would be a forcing candidate by profile;
    # the Null heuristic removes time-like columns from forcing guesses.
    df = _panel_df()
    u = Understanding.model_validate(_understand(df))
    assert "year" not in [g.column for g in u.forcing]
    assert "year" not in [g.column for g in u.outcomes]


def test_understand_quirks():
    rng = np.random.default_rng(4)
    n = 50
    df = pd.DataFrame({"x": rng.normal(0.0, 1.0, n), "const": np.ones(n)})
    df["holey"] = np.where(np.arange(n) < 35, 1.5, np.nan)  # 30% missing
    content = _understand(df)
    u = Understanding.model_validate(content)
    assert "const: constant column" in u.quirks
    assert "holey: 30% missing" in u.quirks


def test_prepare_small_profile_no_subsample_drops_bad_columns():
    rng = np.random.default_rng(5)
    n = 100
    df = pd.DataFrame({"x": rng.normal(0.0, 1.0, n), "const": np.zeros(n)})
    df["gone"] = np.where(np.arange(n) < 30, 1.0, np.nan)  # 70% missing
    be = NullBackend()
    payload = {"profile": _profile_dict(df), "understanding": {}, "seed": 7, "context": None}
    content = be.complete(GuidanceRequest(task="prepare", payload=payload)).content
    plan = PrepPlan.model_validate(content)  # contract: always a valid PrepPlan
    assert plan.subsample is None
    assert "const" in plan.drop_cols and "gone" in plan.drop_cols
    assert "x" not in plan.drop_cols
    assert plan.column_roles == {} and plan.filters == [] and plan.encodings == {}


def test_prepare_large_profile_subsamples_with_payload_seed():
    prof = _profile_dict(_fake_test_score_df())
    prof["n_rows"] = 60000  # fabricated large profile
    be = NullBackend()
    payload = {"profile": prof, "understanding": {}, "seed": 123, "context": None}
    content = be.complete(GuidanceRequest(task="prepare", payload=payload)).content
    plan = PrepPlan.model_validate(content)
    assert plan.subsample is not None
    assert plan.subsample.n == 20000
    assert plan.subsample.seed == 123


def test_search_plan_fake_test_score():
    df = _fake_test_score_df()
    be = NullBackend()
    prof = _profile_dict(df)
    und = _understand(df)
    payload = {"profile": prof, "understanding": und, "context": None}
    content = be.complete(GuidanceRequest(task="search_plan", payload=payload)).content
    plan = SearchPlan.model_validate(content)  # contract: always a valid SearchPlan
    assert plan.candidates, "expected rdd candidates from binary treatment guesses"
    treat = [c for c in plan.candidates if c.treatment == "treat" and c.design == "rdd"]
    assert treat and treat[0].forcing  # nonempty forcing, model-validated anyway
    assert [c.priority for c in plan.candidates] == list(range(len(plan.candidates)))
    assert plan.budget == {"k": 50, "q": 99, "coarse": False, "n_coarse": 2000}
    assert plan.ranked()[0].priority == 0


def test_search_plan_panel_yields_did_candidate():
    df = _panel_df()
    be = NullBackend()
    payload = {"profile": _profile_dict(df), "understanding": _understand(df), "context": None}
    content = be.complete(GuidanceRequest(task="search_plan", payload=payload)).content
    plan = SearchPlan.model_validate(content)
    dids = [c for c in plan.candidates if c.design == "did"]
    assert dids and dids[0].unit == "state" and dids[0].time == "year"
    assert dids[0].treatment == "treated"
    # the only forcing guess (smoking) is consumed as outcome -> the rdd candidate is skipped
    assert all(c.design == "did" for c in plan.candidates)
    assert [c.priority for c in plan.candidates] == list(range(len(plan.candidates)))


def test_interpret_discovery_names_design_dominant_forcing_and_location():
    be = NullBackend()
    payload = {
        "candidate": {"design": "rdd", "treatment": "T"},
        "summary": {
            "design": "rdd",
            "forcing_influence": {"x0": 0.12, "x1": -0.93},
            "center_z": [0.5, 1.0],
            "llr": 3.2,
        },
        "context": None,
    }
    content = be.complete(GuidanceRequest(task="interpret_discovery", payload=payload)).content
    assert content["matched_policies"] == []
    assert content["confounded_risk"] == "unknown"
    assert content["note"] == "NullBackend: no domain knowledge applied"
    assert "rdd" in content["summary"]
    assert "x1" in content["summary"]  # max |influence|
    assert "center_z" in content["summary"]


def test_audit_assumptions_never_vetoes():
    be = NullBackend()
    payload = {"candidate": {"design": "rdd"}, "validation": {"p_value": 0.02}}
    content = be.complete(GuidanceRequest(task="audit_assumptions", payload=payload)).content
    assert content["veto"] is False
    assert content["excludability"] == "unreviewed"
    assert content["monotonicity"] == "unreviewed"
    assert content["sutva"] == "unreviewed"
    assert content["caveats"]


def test_review_control_group_never_vetoes():
    be = NullBackend()
    payload = {"profile": {}, "expansions": [{"a": 1}, {"a": 2}, {"a": 3}], "n_control": 4}
    content = be.complete(GuidanceRequest(task="review_control_group", payload=payload)).content
    assert content["veto"] is False
    assert content["face_valid"] is None
    assert "n_expansions=3" in content["reason"]


def test_unknown_task_raises_value_error():
    be = NullBackend()
    bogus = GuidanceRequest.model_construct(task="bogus", payload={}, schema_hint={})
    with pytest.raises(ValueError, match="bogus"):
        be.complete(bogus)
