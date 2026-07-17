"""``method_applicability`` guidance task: recorded overrides, hint hygiene, fallback.

Deterministic (MockBackend/NullBackend only — no rng in the code under test, no
network): no statistical calibration needed. Contract under test (plan task 3):
``resolve_applicability`` fires at most ONE guidance request, the analyst may
override heuristics BOTH ways with every override recorded
(``heuristic_said``/``analyst_said``/``reason``), invalid config hints are
dropped and recorded (never a crash), explicit user declarations always win,
and any backend/parse failure falls back to heuristics per family with
``guidance_error`` recorded.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from natex.intake.profiler import profile
from natex.llm import GuidanceRequest, MockBackend, NullBackend
from natex.llm.api import _STRIP_KEYS, _strict_schema
from natex.llm.backends import TASK_INSTRUCTIONS, TASKS
from natex.survey.applicability import (
    ApplicabilityResponse,
    resolve_applicability,
)
from natex.survey.registry import FAMILY_ORDER, DeclaredInputs


def _cross_section(n: int = 300) -> pd.DataFrame:
    """Binary T, numeric z/y, non-numeric state; no panel structure."""
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "T": np.tile([0, 1], n // 2),
            "z": rng.normal(size=n),
            "y": rng.normal(size=n),
            "state": np.tile(["a", "b", "c", "d"], n // 4),
        }
    )


def test_task_registered():
    assert "method_applicability" in TASKS
    assert "method_applicability" in TASK_INSTRUCTIONS
    req = GuidanceRequest(task="method_applicability", payload={})
    assert req.task == "method_applicability"


def test_null_guidance_matches_no_guidance():
    prof = profile(_cross_section())
    declared = DeclaredInputs()

    plans_none, merged_none = resolve_applicability(prof, None, declared, None)
    plans_null, merged_null = resolve_applicability(prof, None, declared, NullBackend())

    assert tuple(plans_none) == FAMILY_ORDER
    assert tuple(plans_null) == FAMILY_ORDER
    assert {n: (p.run, p.reason) for n, p in plans_none.items()} == {
        n: (p.run, p.reason) for n, p in plans_null.items()
    }
    for plans in (plans_none, plans_null):
        for p in plans.values():
            assert p.override is None
            assert p.guidance_error is None
    assert merged_none == declared
    assert merged_null == declared


def test_mock_override_both_ways():
    prof = profile(_cross_section())
    declared = DeclaredInputs()
    mock = MockBackend(
        [
            {
                "families": [
                    {
                        "family": "rdd",
                        "run": False,
                        "reason": "context says treatment was assigned alphabetically",
                        "config_hints": {},
                    },
                    {
                        "family": "kink",
                        "run": True,
                        "reason": "policy kink at z=1.5 per context",
                        "config_hints": {"cutoffs": [{"column": "z", "value": 1.5}]},
                    },
                ]
            }
        ]
    )

    plans, merged = resolve_applicability(prof, None, declared, mock, context="ctx")

    # ONE request, correct task/payload/schema
    assert len(mock.requests) == 1
    req = mock.requests[0]
    assert req.task == "method_applicability"
    assert set(req.payload) == {"profile", "context", "declared", "families", "heuristics"}
    assert req.payload["context"] == "ctx"
    assert [f["name"] for f in req.payload["families"]] == list(FAMILY_ORDER)
    assert set(req.payload["heuristics"]) == set(FAMILY_ORDER)
    assert req.schema_hint == ApplicabilityResponse.model_json_schema()

    # rdd: heuristic applicable -> analyst run=False, override recorded
    assert plans["rdd"].heuristic.status == "applicable"
    assert plans["rdd"].run is False
    assert plans["rdd"].override == {
        "heuristic_said": True,
        "analyst_said": False,
        "reason": "context says treatment was assigned alphabetically",
    }

    # kink: heuristic needs_input -> analyst run=True, override recorded
    assert plans["kink"].heuristic.status == "needs_input"
    assert plans["kink"].run is True
    assert plans["kink"].override == {
        "heuristic_said": False,
        "analyst_said": True,
        "reason": "policy kink at z=1.5 per context",
    }
    assert [(h.column, h.value) for h in plans["kink"].config_hints.cutoffs] == [("z", 1.5)]

    # omitted families keep their heuristic decision, no override
    for name in FAMILY_ORDER:
        if name in ("rdd", "kink"):
            continue
        p = plans[name]
        assert p.run is (p.heuristic.status == "applicable"), name
        assert p.reason == p.heuristic.reason, name
        assert p.override is None, name

    # merged declared gains the analyst cutoff hint
    assert merged.cutoffs == {"z": 1.5}


def test_hint_hygiene():
    prof = profile(_cross_section())
    declared = DeclaredInputs(cutoffs={"z": 1.5})
    mock = MockBackend(
        [
            {
                "families": [
                    {
                        "family": "kink",
                        "run": True,
                        "reason": "run with hints",
                        "config_hints": {
                            "cutoffs": [
                                {"column": "nope", "value": 1.0},  # nonexistent
                                {"column": "state", "value": 2.0},  # non-numeric
                                {"column": "z", "value": 9.0},  # declared wins
                                {"column": "y", "value": 0.5},  # valid, new
                            ]
                        },
                    }
                ]
            }
        ]
    )

    plans, merged = resolve_applicability(prof, None, declared, mock)

    err = plans["kink"].guidance_error
    assert err is not None
    assert "nope" in err
    assert "state" in err
    # invalid hints dropped from the cleaned plan hints
    assert [(h.column, h.value) for h in plans["kink"].config_hints.cutoffs] == [
        ("z", 9.0),
        ("y", 0.5),
    ]
    # explicit declaration is NOT overwritten; the valid new hint merges
    assert merged.cutoffs == {"z": 1.5, "y": 0.5}


def test_backend_failure_falls_back():
    prof = profile(_cross_section())
    declared = DeclaredInputs()
    mock = MockBackend([])  # exhausted mock raises RuntimeError on complete()

    plans, merged = resolve_applicability(prof, None, declared, mock)

    assert tuple(plans) == FAMILY_ORDER
    for name, p in plans.items():
        assert p.run is (p.heuristic.status == "applicable"), name
        assert p.reason == p.heuristic.reason, name
        assert p.override is None, name
        assert p.guidance_error is not None, name
        assert "method_applicability" in p.guidance_error, name
    assert merged == declared


def test_unknown_family_dropped_and_recorded():
    prof = profile(_cross_section())
    mock = MockBackend(
        [{"families": [{"family": "quantile_bandit", "run": True, "reason": "x"}]}]
    )
    plans, merged = resolve_applicability(prof, None, DeclaredInputs(), mock)
    assert "quantile_bandit" not in plans
    assert tuple(plans) == FAMILY_ORDER
    for p in plans.values():  # heuristic decisions kept, drop recorded
        assert p.run is (p.heuristic.status == "applicable")
        assert p.guidance_error is not None and "quantile_bandit" in p.guidance_error


def _walk(node):
    yield node
    if isinstance(node, dict):
        for v in node.values():
            yield from _walk(v)
    elif isinstance(node, list):
        for v in node:
            yield from _walk(v)


def test_schema_hint_is_strict_safe():
    """Regression against the dict-field trap: strict-mode conversion must keep
    the families item schema intact (lists of keyed submodels, no dict fields)."""
    strict = _strict_schema(ApplicabilityResponse.model_json_schema())
    for node in _walk(strict):
        if isinstance(node, dict):
            for key in _STRIP_KEYS:
                assert key not in node
            if node.get("type") == "object":
                assert node["additionalProperties"] is False
    # families item schema survives with its keyed submodels
    decision = strict["$defs"]["FamilyDecision"]
    assert set(decision["properties"]) == {"family", "run", "reason", "config_hints"}
    hints = strict["$defs"]["ConfigHints"]
    assert set(hints["properties"]) == {
        "cutoffs",
        "instruments",
        "thresholds",
        "treated_unit",
        "t0",
    }
    value_hint = strict["$defs"]["ConfigValueHint"]
    assert set(value_hint["properties"]) == {"column", "value"}
