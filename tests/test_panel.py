"""Tests for the DiD panel data layer (phase 3, task 1)."""

import numpy as np
import pandas as pd
import pytest

from natex.data.spec import Dataset, DatasetSpec
from natex.did.panel import build_panel, quantile_bins

# ---------------------------------------------------------------------------
# quantile_bins
# ---------------------------------------------------------------------------


def test_quantile_bins_known_codes():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    codes, edges = quantile_bins(x, bins=4)
    np.testing.assert_array_equal(codes, [0, 0, 1, 1, 2, 2, 3, 3])
    assert codes.dtype == np.int64
    assert edges.shape == (5,)
    np.testing.assert_allclose(edges, [1.0, 2.75, 4.5, 6.25, 8.0])


def test_quantile_bins_constant_column_single_bin():
    x = np.full(10, 3.14)
    codes, edges = quantile_bins(x, bins=4)
    np.testing.assert_array_equal(codes, np.zeros(10, dtype=np.int64))
    assert np.unique(codes).size == 1


def test_quantile_bins_monotone_in_x():
    rng = np.random.default_rng(0)
    x = np.sort(rng.normal(size=200))
    codes, _ = quantile_bins(x, bins=5)
    assert np.all(np.diff(codes) >= 0)


def test_quantile_bins_rejects_nonfinite():
    with pytest.raises(ValueError, match="finite"):
        quantile_bins(np.array([1.0, np.nan, 2.0]), bins=2)


# ---------------------------------------------------------------------------
# build_panel on a toy panel DataFrame
# ---------------------------------------------------------------------------


def panel_df() -> pd.DataFrame:
    # 12 rows; 2 categorical dims, 1 continuous dim (12 distinct values),
    # 1 binary 0/1 numeric dim, a unit column, and a time column.
    return pd.DataFrame(
        {
            "group": ["a", "b", "a", "b", "a", "b", "a", "b", "a", "b", "a", "a"],
            "region": ["n", "s", "e", "n", "s", "e", "n", "s", "e", "n", "s", "n"],
            "income": [10.0, 25.0, 3.0, 47.0, 8.0, 31.0, 19.0, 55.0, 2.0, 40.0, 15.0, 22.0],
            "flag": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0],
            "state": ["CA", "NY", "CA", "TX", "NY", "TX", "CA", "NY", "TX", "CA", "NY", "CA"],
            "year": [1980, 1981, 1982, 1983, 1984, 1985, 1986, 1987, 1988, 1989, 1990, 1991],
            "T": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1],
            "y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
        }
    )


def panel_dataset() -> Dataset:
    spec = DatasetSpec(
        treatment="T",
        outcome="y",
        forcing=[],
        covariates=["group", "region", "income", "flag"],
        time="year",
        unit="state",
    )
    return Dataset(panel_df(), spec)


def test_build_panel_codes_and_dim_sizes():
    panel = build_panel(panel_dataset(), bins=4)
    assert panel.n == 12
    assert panel.m == 4
    assert panel.codes.shape == (12, 4)
    assert panel.codes.dtype == np.int64
    assert panel.dim_names == ["group", "region", "income", "flag"]
    # group: {a, b}; region: {e, n, s}; income: 12 distinct -> 4 quantile bins;
    # flag: binary {0, 1} coded by its values, NOT quantile-split.
    assert panel.dim_sizes == (2, 3, 4, 2)


def test_build_panel_binary_column_not_quantile_split():
    panel = build_panel(panel_dataset(), bins=4)
    j = panel.dim_names.index("flag")
    np.testing.assert_allclose(np.asarray(panel.dim_values[j], dtype=float), [0.0, 1.0])
    np.testing.assert_array_equal(panel.codes[:, j], panel_df()["flag"].to_numpy())


def test_build_panel_arrays_and_unit():
    panel = build_panel(panel_dataset(), bins=4)
    np.testing.assert_allclose(panel.t, panel_df()["year"].to_numpy(dtype=float))
    np.testing.assert_allclose(panel.theta, panel_df()["T"].to_numpy(dtype=float))
    assert panel.y is not None
    np.testing.assert_allclose(panel.y, panel_df()["y"].to_numpy(dtype=float))
    # unit codes decode back to the original state labels
    decoded = np.asarray(panel.unit_values)[panel.unit]
    np.testing.assert_array_equal(decoded, panel_df()["state"].to_numpy())
    assert panel.unit.dtype == np.int64
    assert panel.unit.min() >= 0
    assert panel.unit.max() == len(panel.unit_values) - 1


def test_profile_id_equal_iff_codes_equal():
    df = panel_df()
    # duplicate row 0's dim values into row 11 so at least one profile repeats
    for c in ["group", "region", "income", "flag"]:
        df.loc[11, c] = df.loc[0, c]
    spec = DatasetSpec(
        treatment="T",
        outcome="y",
        forcing=[],
        covariates=["group", "region", "income", "flag"],
        time="year",
        unit="state",
    )
    panel = build_panel(Dataset(df, spec), bins=4)
    pid = panel.profile_id
    assert pid.shape == (12,)
    assert pid[0] == pid[11]
    for i in range(panel.n):
        for k in range(panel.n):
            same_codes = bool(np.all(panel.codes[i] == panel.codes[k]))
            assert (pid[i] == pid[k]) == same_codes


def test_subset_mask_matches_pandas_reimplementation():
    df = panel_df()
    panel = build_panel(panel_dataset(), bins=4)

    def masks_for(allowed: dict[str, list]) -> list[np.ndarray]:
        out = []
        for j, name in enumerate(panel.dim_names):
            if name in allowed:
                out.append(np.isin(panel.dim_values[j], allowed[name]))
            else:
                out.append(np.ones(len(panel.dim_values[j]), dtype=bool))
        return out

    # subset 1: group in {a}
    got = panel.subset_mask(masks_for({"group": ["a"]}))
    np.testing.assert_array_equal(got, df["group"].isin(["a"]).to_numpy())
    # subset 2: group in {b} AND region in {n, s}
    got = panel.subset_mask(masks_for({"group": ["b"], "region": ["n", "s"]}))
    expected = (df["group"].isin(["b"]) & df["region"].isin(["n", "s"])).to_numpy()
    np.testing.assert_array_equal(got, expected)
    # subset 3: flag in {1} AND region in {e}
    got = panel.subset_mask(masks_for({"flag": [1], "region": ["e"]}))
    expected = (df["flag"].isin([1]) & df["region"].isin(["e"])).to_numpy()
    np.testing.assert_array_equal(got, expected)


def test_subset_mask_all_true_is_full_dataset():
    panel = build_panel(panel_dataset(), bins=4)
    full = [np.ones(k, dtype=bool) for k in panel.dim_sizes]
    assert panel.subset_mask(full).all()


def test_build_panel_unit_defaults_to_profile_id():
    spec = DatasetSpec(
        treatment="T",
        outcome="y",
        forcing=[],
        covariates=["group", "region", "income", "flag"],
        time="year",
    )
    panel = build_panel(Dataset(panel_df(), spec), bins=4)
    # each profile is one unit: unit codes partition exactly like profile_id
    pid = panel.profile_id
    for i in range(panel.n):
        for k in range(panel.n):
            assert (panel.unit[i] == panel.unit[k]) == (pid[i] == pid[k])
    assert panel.unit.min() >= 0
    assert panel.unit.max() == len(panel.unit_values) - 1


def test_build_panel_requires_time():
    spec = DatasetSpec(
        treatment="T",
        outcome="y",
        forcing=[],
        covariates=["group", "region", "income", "flag"],
    )
    ds = Dataset(panel_df(), spec)
    with pytest.raises(ValueError, match="time"):
        build_panel(ds)


# ---------------------------------------------------------------------------
# DatasetSpec unit/time validation, empty forcing
# ---------------------------------------------------------------------------


def test_missing_unit_column_raises():
    spec = DatasetSpec(
        treatment="T", outcome="y", forcing=[], covariates=["group"], time="year", unit="nope"
    )
    with pytest.raises(ValueError, match="unit"):
        Dataset(panel_df(), spec)


def test_missing_time_column_raises():
    spec = DatasetSpec(
        treatment="T", outcome="y", forcing=[], covariates=["group"], time="nope", unit="state"
    )
    with pytest.raises(ValueError, match="time"):
        Dataset(panel_df(), spec)


def test_nonnumeric_time_column_raises():
    spec = DatasetSpec(
        treatment="T", outcome="y", forcing=[], covariates=["group"], time="state"
    )
    with pytest.raises(ValueError, match="time"):
        Dataset(panel_df(), spec)


def test_nan_unit_and_time_rows_dropped_never_outcome():
    df = panel_df()
    df.loc[2, "state"] = np.nan
    df.loc[5, "year"] = np.nan
    df.loc[7, "y"] = np.nan  # outcome NaN must NOT drop the row
    spec = DatasetSpec(
        treatment="T",
        outcome="y",
        forcing=[],
        covariates=["group", "region", "income", "flag"],
        time="year",
        unit="state",
    )
    ds = Dataset(df, spec)
    assert ds.n == 10  # rows 2 and 5 dropped; row 7 kept
    assert np.isnan(ds.y).sum() == 1


def test_empty_forcing_end_to_end():
    ds = panel_dataset()
    assert ds.spec.forcing == []
    assert ds.Z.shape == (ds.n, 0)
    assert ds.n == 12


def test_from_csv_unit_passthrough(tmp_path):
    p = tmp_path / "panel.csv"
    panel_df().to_csv(p, index=False)
    ds = Dataset.from_csv(
        p,
        treatment="T",
        outcome="y",
        forcing=[],
        covariates=["group", "region", "income", "flag"],
        time="year",
        unit="state",
    )
    assert ds.spec.unit == "state"
    assert ds.spec.time == "year"
    panel = build_panel(ds)
    assert panel.n == 12
