"""Blind-vs-informed guidance eval scaffold (phase llm-analyst task 10).

Contract under test: ``make_eval_cases`` is Generator-deterministic and plants
a known true design behind a decoy binary column inserted BEFORE the true
treatment; ``rank_of_truth`` returns the index of the first ranked candidate
matching the truth (rdd: design + treatment + forcing superset-or-equal;
did: design + treatment + time); ``run_guidance_eval`` measurably separates
the blind NullBackend arm from an informed backend arm. CI uses MockBackend
ONLY — API arms are manual via ``benchmarks/guidance_eval.py``, never CI.

Statistical-test policy calibration (seeds 0-4, n_rdd=2, include_did=False,
run 2026-07-12): every case at every seed gave rank_null == 1 and
rank_backend == 0 — the decoy binary column sits before ``T`` so NullBackend's
column-order heuristics rank the decoy treatment first, while the informed
mock pins the truth at priority 0. The pinned assertions (rank_backend == 0,
rank_null >= 1) hold with margin at every calibrated seed; all five seeds run
in CI because the pipeline is profile-only and cheap at n=400.
"""

import json

import numpy as np
import pandas as pd
import pytest

from natex.guidance_eval import (
    DECOY_NAMES,
    EVAL_COLUMNS,
    EvalCase,
    make_eval_cases,
    rank_of_truth,
    run_guidance_eval,
)
from natex.intake.plans import DesignCandidate, SearchPlan
from natex.intake.profiler import profile
from natex.llm import GuidanceRequest, MockBackend, NullBackend


def test_make_eval_cases_requires_rng():
    with pytest.raises(ValueError, match="Generator"):
        make_eval_cases()


def test_make_eval_cases_deterministic():
    a = make_eval_cases(n_rdd=2, include_did=True, rng=np.random.default_rng(0))
    b = make_eval_cases(n_rdd=2, include_did=True, rng=np.random.default_rng(0))
    assert [c.name for c in a] == [c.name for c in b]
    assert [c.context for c in a] == [c.context for c in b]
    for ca, cb in zip(a, b, strict=True):
        pd.testing.assert_frame_equal(ca.df, cb.df)


def test_rdd_cases_have_decoy_before_treatment():
    cases = make_eval_cases(n_rdd=3, include_did=False, rng=np.random.default_rng(0))
    assert len(cases) == 3
    for case in cases:
        assert case.design == "rdd"
        assert case.treatment == "T"
        assert case.forcing == ("x0", "x1")
        assert case.time is None
        cols = list(case.df.columns)
        decoys = [c for c in cols if c in DECOY_NAMES]
        assert len(decoys) == 1
        # the decoy binary column sits BEFORE the true treatment T
        assert cols.index(decoys[0]) < cols.index("T")
        assert set(case.df[decoys[0]].unique()) <= {0, 1}
        # one pure-noise numeric decoy appended at the end
        assert cols[-1] == "noise"
        assert case.df["noise"].nunique() > 20


def test_did_case_appended_when_requested():
    cases = make_eval_cases(n_rdd=1, include_did=True, rng=np.random.default_rng(0))
    assert [c.design for c in cases] == ["rdd", "did"]
    did = cases[-1]
    assert did.treatment == "theta"
    assert did.forcing == ()
    assert did.time == "year"
    assert {"unit", "year", "theta", "y"} <= set(did.df.columns)
    # the profiler must see the (unit, year) panel or Null never proposes did
    assert profile(did.df).panel_candidates == [("unit", "year")]


def _rdd_case():
    return EvalCase(
        name="hand", df=pd.DataFrame(), context="", design="rdd",
        treatment="T", forcing=("x0", "x1"), time=None,
    )


def test_rank_of_truth_first_even_when_listed_last():
    # priority 0 wins regardless of list position; forcing may be a superset
    plan = SearchPlan(candidates=[
        DesignCandidate(design="rdd", treatment="holiday", forcing=["x0", "x1"], priority=1),
        DesignCandidate(design="rdd", treatment="T", forcing=["x1", "x0", "x2"], priority=0),
    ])
    assert rank_of_truth(plan, _rdd_case()) == 0


def test_rank_of_truth_skips_non_matching_candidates():
    plan = SearchPlan(candidates=[
        DesignCandidate(design="rdd", treatment="holiday", forcing=["x0", "x1"], priority=0),
        DesignCandidate(design="rdd", treatment="T", forcing=["x0"], priority=1),  # no superset
        DesignCandidate(design="rdd", treatment="T", forcing=["x0", "x1"], priority=2),
    ])
    assert rank_of_truth(plan, _rdd_case()) == 2


def test_rank_of_truth_none_when_absent():
    plan = SearchPlan(candidates=[
        DesignCandidate(design="rdd", treatment="T", forcing=["x2"]),
    ])
    assert rank_of_truth(plan, _rdd_case()) is None


def test_rank_of_truth_did_matches_on_time():
    case = EvalCase(
        name="hand-did", df=pd.DataFrame(), context="", design="did",
        treatment="theta", forcing=(), time="year",
    )
    plan = SearchPlan(candidates=[
        DesignCandidate(design="rdd", treatment="theta", forcing=["y"], priority=0),
        DesignCandidate(design="did", treatment="theta", unit="unit", time="t", priority=1),
        DesignCandidate(design="did", treatment="theta", unit="unit", time="year", priority=2),
    ])
    assert rank_of_truth(plan, case) == 2


def _informed_mock(case: EvalCase) -> MockBackend:
    """Fresh informed mock per case: Null-style understand/prepare (built by
    calling NullBackend on the case's profile, so the mock stays aligned with
    the payload contract) + a search plan with the TRUE config at priority 0."""
    null = NullBackend()
    prof = json.loads(profile(case.df).to_json())
    und = null.complete(GuidanceRequest(
        task="understand", payload={"profile": prof, "context": case.context},
    )).content
    prep = null.complete(GuidanceRequest(
        task="prepare",
        payload={"profile": prof, "understanding": und, "seed": 0, "context": case.context},
    )).content
    if case.design == "rdd":
        truth = {
            "design": "rdd", "treatment": case.treatment, "outcome": "y",
            "forcing": list(case.forcing), "priority": 0, "rationale": "context hint",
        }
    else:
        truth = {
            "design": "did", "treatment": case.treatment, "outcome": "y",
            "unit": "unit", "time": case.time, "priority": 0, "rationale": "context hint",
        }
    return MockBackend([und, prep, {"candidates": [truth], "budget": {}}])


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_informed_mock_beats_blind_null(seed):
    # runtime guard: n_rdd=2, include_did=False keeps the CI slice fast
    frame = run_guidance_eval(_informed_mock, n_rdd=2, include_did=False, seed=seed)
    assert list(frame.columns) == EVAL_COLUMNS
    assert len(frame) == 2
    assert frame["rank_backend"].notna().all()
    assert (frame["rank_backend"] == 0).all()
    rdd = frame[frame["design"] == "rdd"]
    assert rdd["rank_null"].notna().all()
    assert (rdd["rank_null"] >= 1).all()


def test_informed_mock_hits_did_truth():
    frame = run_guidance_eval(_informed_mock, n_rdd=1, include_did=True, seed=0)
    assert list(frame["design"]) == ["rdd", "did"]
    assert frame["rank_backend"].notna().all()
    assert (frame["rank_backend"] == 0).all()


def test_null_smoke_mode_arms_identical():
    # make_backend -> None is the blind arm twice (the runner's --backend null)
    frame = run_guidance_eval(lambda case: None, n_rdd=2, include_did=False, seed=0)
    assert list(frame.columns) == EVAL_COLUMNS
    assert frame["rank_null"].dtype == "Int64"
    assert frame["rank_backend"].dtype == "Int64"
    pd.testing.assert_series_equal(
        frame["rank_backend"], frame["rank_null"], check_names=False,
    )
    assert (frame["n_candidates_backend"] == frame["n_candidates_null"]).all()
