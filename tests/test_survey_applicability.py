"""Heuristic per-family applicability: content-blind verdicts from registry predicates."""

from __future__ import annotations

import importlib.util

import numpy as np
import pandas as pd
import pytest

from natex.intake.profiler import profile
from natex.survey.applicability import FamilyVerdict, heuristic_applicability
from natex.survey.registry import FAMILY_ORDER, DeclaredInputs

_STATUSES = {"applicable", "inapplicable", "needs_input"}


def _cross_section(n: int = 300) -> pd.DataFrame:
    """Binary T, numeric z, numeric y; no time-like or id column."""
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "T": np.tile([0, 1], n // 2),
            "z": rng.normal(size=n),
            "y": rng.normal(size=n),
        }
    )


def _panel() -> pd.DataFrame:
    """8 states x 16 years grid: string unit, int year, binary T, numeric y."""
    rng = np.random.default_rng(1)
    states = [f"s{i}" for i in range(8)]
    years = list(range(1990, 2006))
    rows = [(s, t) for s in states for t in years]
    return pd.DataFrame(
        {
            "state": [s for s, _ in rows],
            "year": [t for _, t in rows],
            "T": [int(s in {"s0", "s1"} and t >= 1998) for s, t in rows],
            "y": rng.normal(size=len(rows)),
        }
    )


def test_cross_section_skips_did_and_kink_in_time():
    verdicts = heuristic_applicability(profile(_cross_section()), None, DeclaredInputs())

    assert verdicts["rdd"].status == "applicable"
    assert verdicts["rdd"].reason == "all requirements met"
    assert verdicts["rdd"].unmet == []

    assert verdicts["did"].status == "inapplicable"
    assert "panel" in verdicts["did"].reason
    assert "unit and time" in verdicts["did"].reason
    assert verdicts["did"].unmet == ["needs_panel"]

    assert verdicts["kink"].status == "needs_input"
    assert (
        "no pre-declared cutoff (kink is candidate evaluation, not discovery)"
        in verdicts["kink"].reason
    )
    assert verdicts["kink"].unmet == ["needs_declared_cutoff"]

    assert verdicts["sc"].status == "inapplicable"
    assert verdicts["sc"].unmet == ["needs_panel"]

    assert verdicts["bunching"].status == "needs_input"
    assert verdicts["bunching"].unmet == ["needs_declared_threshold"]

    assert verdicts["iv"].status == "needs_input"
    assert verdicts["iv"].unmet == ["needs_candidate_instruments"]


def test_panel_enables_did_and_sc():
    verdicts = heuristic_applicability(profile(_panel()), None, DeclaredInputs())
    assert verdicts["did"].status == "applicable"
    assert verdicts["did"].reason == "all requirements met"
    assert verdicts["sc"].status == "applicable"
    assert verdicts["sc"].reason == "all requirements met"


def test_declared_cutoff_enables_kink():
    declared = DeclaredInputs(cutoffs={"z": 1.5})
    verdicts = heuristic_applicability(profile(_cross_section()), None, declared)
    assert verdicts["kink"].status == "applicable"
    assert verdicts["kink"].reason == "all requirements met"
    assert verdicts["kink"].unmet == []


def test_declared_instruments_enable_iv():
    declared = DeclaredInputs(instruments=["z"])
    verdicts = heuristic_applicability(profile(_cross_section()), None, declared)
    assert verdicts["iv"].status == "applicable"
    assert verdicts["iv"].unmet == []


def test_thresholds_enable_bunching():
    declared = DeclaredInputs(thresholds={"z": 0.0})
    verdicts = heuristic_applicability(profile(_cross_section()), None, declared)
    assert verdicts["bunching"].status == "applicable"
    assert verdicts["bunching"].unmet == []


def test_min_rows_gate():
    floors = {"rdd": 100, "did": 60, "kink": 60, "iv": 80, "sc": 40, "bunching": 60, "dee": 200}
    verdicts = heuristic_applicability(profile(_cross_section(n=20)), None, DeclaredInputs())
    for family, floor in floors.items():
        assert verdicts[family].status == "inapplicable", family
        assert f"at least {floor} rows" in verdicts[family].reason, family
        assert "min_rows" in verdicts[family].unmet, family


def test_dee_needs_gp_and_outcome(monkeypatch):
    # Missing gp extra -> inapplicable with the install hint in the reason.
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    verdicts = heuristic_applicability(profile(_cross_section()), None, DeclaredInputs())
    assert verdicts["dee"].status == "inapplicable"
    assert 'natex-discovery[gp]' in verdicts["dee"].reason
    assert verdicts["dee"].unmet == ["needs_gp_extra"]

    # No continuous non-time outcome (gp present) -> inapplicable naming the outcome.
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: object())
    rng = np.random.default_rng(2)
    df = pd.DataFrame(
        {
            "T": np.tile([0, 1], 150),
            "time_score": rng.normal(size=300),  # time-like by name, so not an outcome
        }
    )
    verdicts = heuristic_applicability(profile(df), None, DeclaredInputs())
    assert verdicts["dee"].status == "inapplicable"
    assert "outcome" in verdicts["dee"].reason
    assert verdicts["dee"].unmet == ["needs_outcome"]


def test_all_seven_always_present():
    inputs = [
        (profile(_cross_section()), DeclaredInputs()),
        (profile(_panel()), DeclaredInputs(cutoffs={"y": 0.0}, instruments=["y"])),
        (profile(_cross_section(n=20)), DeclaredInputs()),
    ]
    for prof, declared in inputs:
        verdicts = heuristic_applicability(prof, None, declared)
        assert list(verdicts) == list(FAMILY_ORDER)
        for name, verdict in verdicts.items():
            assert isinstance(verdict, FamilyVerdict)
            assert verdict.family == name
            assert verdict.status in _STATUSES
            assert verdict.reason.strip()


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-q"])
