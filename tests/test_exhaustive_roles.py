"""Helpers behind exhaustive role enumeration: ``Dataset.with_outcome`` (a
row-aligned outcome swap for per-outcome effect estimation at one discovered
boundary) and ``mechanical_step_column`` (a treatment that is a global
deterministic step in a time column is the design's own adoption boundary,
not a scannable natural experiment — issue #52)."""

import numpy as np
import pandas as pd
import pytest

from natex.data.spec import Dataset, DatasetSpec
from natex.intake.profiler import mechanical_step_column


def test_with_outcome_swaps_the_outcome_column_without_redoing_row_deletion():
    df = pd.DataFrame({
        "T": [0, 1, 0, 1, 0, 1],
        "z": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
        "y": [1.0, 2.0, np.nan, 4.0, 5.0, 6.0],
        "w": [9.0, 8.0, 7.0, 6.0, 5.0, 4.0],
    })
    ds = Dataset(df, DatasetSpec(treatment="T", outcome="y", forcing=["z"], covariates=["z"]))
    other = ds.with_outcome("w")
    assert other.spec.outcome == "w"
    assert other.n == ds.n
    assert other.df.index.equals(ds.df.index)  # same rows, same order: members stay aligned
    np.testing.assert_array_equal(other.y, df["w"].to_numpy())
    assert ds.spec.outcome == "y"  # the original is untouched
    with pytest.raises(ValueError, match="not in dataframe"):
        ds.with_outcome("nope")


def test_mechanical_step_column_names_the_time_column_of_a_global_step():
    years = np.repeat(np.arange(2000, 2010), 3).astype(float)
    df = pd.DataFrame({
        "year": years,
        "post": (years >= 2005).astype(int),
        "mixed": (np.arange(30) % 2).astype(int),
    })
    assert mechanical_step_column(df, "post", ["year"]) == "year"
    assert mechanical_step_column(df, "mixed", ["year"]) is None  # not a step in time
    assert mechanical_step_column(df, "post", []) is None  # no time column offered
