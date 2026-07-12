"""Understanding / DesignCandidate / SearchPlan model contracts (phase llm-analyst task 3)."""

import pytest
from pydantic import ValidationError

from natex.intake.plans import (
    ColumnGuess,
    DesignCandidate,
    DiDStructure,
    SearchPlan,
    Understanding,
)


def test_understanding_defaults_and_shape_literal():
    u = Understanding(shape="cross-section")
    assert u.unit_of_observation == "row"
    assert u.treatments == [] and u.outcomes == [] and u.forcing == []
    assert u.did_structures == [] and u.quirks == [] and u.notes == ""
    with pytest.raises(ValidationError):
        Understanding(shape="hexagonal")


def test_understanding_nested_guesses():
    u = Understanding(
        shape="panel",
        unit_of_observation="state",
        treatments=[ColumnGuess(column="treated", reason="binary 0/1 column")],
        did_structures=[DiDStructure(unit="state", time="year")],
    )
    assert u.treatments[0].column == "treated"
    assert u.did_structures[0].time == "year"
    assert u.did_structures[0].reason == ""


def test_rdd_candidate_requires_nonempty_forcing():
    with pytest.raises(ValidationError, match="forcing"):
        DesignCandidate(design="rdd", treatment="T")
    ok = DesignCandidate(design="rdd", treatment="T", forcing=["x0"])
    assert ok.forcing == ["x0"]


def test_did_candidate_requires_time():
    with pytest.raises(ValidationError, match="time"):
        DesignCandidate(design="did", treatment="T", unit="state")
    ok = DesignCandidate(design="did", treatment="T", unit="state", time="year")
    assert ok.time == "year"


def test_ranked_stable_sort_by_priority():
    a = DesignCandidate(design="rdd", treatment="a", forcing=["x"], priority=1)
    b = DesignCandidate(design="rdd", treatment="b", forcing=["x"], priority=0)
    c = DesignCandidate(design="rdd", treatment="c", forcing=["y"], priority=1)
    plan = SearchPlan(candidates=[a, b, c])
    assert [x.treatment for x in plan.ranked()] == ["b", "a", "c"]
    # original list order untouched
    assert [x.treatment for x in plan.candidates] == ["a", "b", "c"]


def test_key_dedup_forcing_order_insensitive():
    a = DesignCandidate(design="rdd", treatment="T", forcing=["x1", "x0"])
    b = DesignCandidate(design="rdd", treatment="T", forcing=["x0", "x1"], priority=5)
    assert a.key() == b.key() == ("rdd", "T", ("x0", "x1"))
    other = DesignCandidate(design="rdd", treatment="T", forcing=["x0"])
    assert other.key() != a.key()


def test_key_did_semantics():
    d1 = DesignCandidate(design="did", treatment="T", unit="state", time="year")
    d2 = DesignCandidate(
        design="did", treatment="T", unit="state", time="year", priority=3, rationale="why not"
    )
    assert d1.key() == d2.key() == ("did", "T", "state", "year")
    rdd = DesignCandidate(design="rdd", treatment="T", forcing=["year"])
    assert d1.key() != rdd.key()


def test_search_plan_json_round_trip():
    plan = SearchPlan(
        candidates=[
            DesignCandidate(
                design="rdd", treatment="T", outcome="y", forcing=["x0", "x1"], priority=1
            ),
            DesignCandidate(design="did", treatment="T", unit="state", time="year", priority=0),
        ],
        budget={"k": 50, "q": 99, "coarse": False, "n_coarse": 2000},
    )
    back = SearchPlan.model_validate_json(plan.model_dump_json())
    assert back == plan
    assert [c.design for c in back.ranked()] == ["did", "rdd"]
