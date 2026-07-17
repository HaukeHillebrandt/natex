"""Method-family registry: fixed order, declarative requirements, content-blind predicates."""

from __future__ import annotations

import importlib.util
import re

import pandas as pd
import pytest

from natex.intake.profiler import ColumnProfile, profile
from natex.survey.registry import FAMILIES, FAMILY_ORDER, DeclaredInputs

_ALLOWED_PROFILE_ATTRS = {
    "n_rows",
    "columns",
    "panel_candidates",
    "forcing_candidates",
    "treatment_candidates",
}


def _dummy_profile():
    """3-column dummy profile built through the real profiler."""
    df = pd.DataFrame(
        {
            "score": [float(i) for i in range(30)],
            "treated": [0, 1] * 15,
            "y": [float(i) * 0.5 for i in range(30)],
        }
    )
    return profile(df)


class _RecordingProfile:
    """Stub that records every attribute a predicate reads, returning realistic values."""

    _VALUES = {
        "n_rows": 500,
        "columns": [
            ColumnProfile(
                name="score", dtype="float64", n_unique=500, missing_frac=0.0,
                is_numeric=True, is_binary=False, is_time_like=False,
            ),
            ColumnProfile(
                name="treated", dtype="int64", n_unique=2, missing_frac=0.0,
                is_numeric=True, is_binary=True, is_time_like=False,
            ),
            ColumnProfile(
                name="y", dtype="float64", n_unique=480, missing_frac=0.0,
                is_numeric=True, is_binary=False, is_time_like=False,
            ),
        ],
        "panel_candidates": [],
        "forcing_candidates": [],
        "treatment_candidates": [],
        "boundary_values": {},
    }

    def __init__(self) -> None:
        self.accessed: set[str] = set()

    def __getattr__(self, name: str):
        self.accessed.add(name)
        try:
            return self._VALUES[name]
        except KeyError:
            raise AttributeError(name) from None


def test_family_order_and_keys():
    assert list(FAMILIES) == list(FAMILY_ORDER)
    assert FAMILY_ORDER == ("rdd", "did", "kink", "iv", "sc", "bunching", "dee")
    assert len(FAMILIES) == 7
    for name, fam in FAMILIES.items():
        assert fam.name == name
        assert fam.title.strip()
        assert fam.caveat.strip()
        assert len(fam.description) >= 200, f"{name} description is not a real paragraph"
        assert "TODO" not in fam.description


def test_requirement_keys_are_declarative():
    prof = _dummy_profile()
    pattern = re.compile(r"^needs_[a-z0-9_]+$|^min_rows$")
    for fam in FAMILIES.values():
        assert fam.requirements, f"{fam.name} has no requirements"
        for req in fam.requirements:
            assert pattern.match(req.key), f"{fam.name}:{req.key} is not declarative"
            assert req.description.strip()
            assert isinstance(req.user_suppliable, bool)
            result = req.check(prof, None, DeclaredInputs())
            assert isinstance(result, bool)


def test_predicates_profile_only():
    """First half of the generalizability guard: predicates read profile metadata only."""
    accessed: set[str] = set()
    for fam in FAMILIES.values():
        for req in fam.requirements:
            stub = _RecordingProfile()
            req.check(stub, None, DeclaredInputs())
            accessed |= stub.accessed
    assert accessed <= _ALLOWED_PROFILE_ATTRS, f"predicates read {accessed - _ALLOWED_PROFILE_ATTRS}"


def test_gp_predicate_env_only(monkeypatch):
    req = next(r for r in FAMILIES["dee"].requirements if r.key == "needs_gp_extra")
    prof = _dummy_profile()
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    assert req.check(prof, None, DeclaredInputs()) is False
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: object())
    assert req.check(prof, None, DeclaredInputs()) is True


def test_no_bunching_hyphenation_traps():
    """Registry prose stays clean of 'None'/'nan' word tokens (report string hygiene)."""
    parts: list[str] = []
    for fam in FAMILIES.values():
        parts += [fam.title, fam.description, fam.caveat]
        parts += [req.description for req in fam.requirements]
    text = " ".join(parts)
    assert not re.search(r"\bnan\b", text, re.IGNORECASE)
    assert not re.search(r"\bNone\b", text)


def test_kink_cutoff_requirement_is_user_suppliable():
    req = next(r for r in FAMILIES["kink"].requirements if r.key == "needs_declared_cutoff")
    assert req.user_suppliable is True
    assert req.description == "no pre-declared cutoff (kink is candidate evaluation, not discovery)"
    assert req.check(_dummy_profile(), None, DeclaredInputs(cutoffs={"score": 10.0})) is True


def test_gp_requirement_names_install_hint():
    req = next(r for r in FAMILIES["dee"].requirements if r.key == "needs_gp_extra")
    assert req.user_suppliable is False
    assert 'pip install "natex-discovery[gp]"' in req.description


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-q"])
