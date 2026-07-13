import json

import numpy as np
import pandas as pd
import pytest

from natex.intake.profiler import profile


def test_profile_panel_dataset():
    rng = np.random.default_rng(0)
    units, years = 40, 10
    df = pd.DataFrame(
        {
            "state": np.repeat([f"s{i}" for i in range(units)], years),
            "year": np.tile(np.arange(1990, 1990 + years), units),
            "smoking": rng.normal(50, 5, units * years),
            "treated": np.zeros(units * years, dtype=int),
        }
    )
    prof = profile(df)
    assert prof.n_rows == 400
    assert ("state", "year") in prof.panel_candidates
    assert "smoking" in prof.forcing_candidates
    assert "treated" in prof.treatment_candidates
    parsed = json.loads(prof.to_json())
    assert parsed["n_rows"] == 400


def test_issue_6_structural_missingness_evidence():
    """Issue #6: the profile carries enough evidence for a data-blind backend
    to distinguish structurally prefix-missing columns (sensor activated at
    some boundary row) from randomly holey ones, plus the boundary value of
    every monotone column so the boundary is expressible as a row filter."""
    n, b = 10, 6
    df = pd.DataFrame(
        {
            "days": np.arange(n, dtype=float),  # monotone, fully observed
            "m": [np.nan] * b + [1.0, np.nan, 3.0, 4.0],  # prefix-missing metric
        }
    )
    prof = profile(df)
    by = {c.name: c for c in prof.columns}
    assert by["days"].is_monotone
    assert by["days"].first_valid_index == 0
    # no missing cells => prefix fraction undefined: None, never 0.0-coerced
    assert by["days"].prefix_missing_frac is None
    assert by["m"].first_valid_index == b
    assert by["m"].prefix_missing_frac == pytest.approx(b / (b + 1))
    assert not by["m"].is_monotone  # has missing cells
    # the monotone column's value at the candidate boundary row
    assert prof.boundary_values[b]["days"] == float(b)
    # JSON round trip (str keys) stays parseable
    parsed = json.loads(prof.to_json())
    assert parsed["boundary_values"][str(b)]["days"] == float(b)


def test_issue_6_all_missing_and_nonmonotone_columns():
    df = pd.DataFrame(
        {
            "gone": [np.nan, np.nan, np.nan],
            "wiggle": [2.0, 1.0, 3.0],
        }
    )
    prof = profile(df)
    by = {c.name: c for c in prof.columns}
    assert by["gone"].first_valid_index is None
    assert by["gone"].prefix_missing_frac is None
    assert not by["wiggle"].is_monotone
    assert prof.boundary_values == {}


def test_time_like_detection():
    df = pd.DataFrame({"year": [2001, 2002], "score": [1.5, 2.5], "when": ["2020-01-01", "2020-02-01"]})
    prof = profile(df)
    by_name = {c.name: c for c in prof.columns}
    assert by_name["year"].is_time_like
    assert by_name["when"].is_time_like
    assert not by_name["score"].is_time_like
