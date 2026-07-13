import numpy as np
import pandas as pd
import pytest

from natex.data.spec import Dataset, DatasetSpec


def toy_df():
    return pd.DataFrame(
        {
            "age": [1.0, 2.0, 3.0, 4.0],
            "score": [10.0, 20.0, 30.0, 40.0],
            "group": ["a", "b", "a", "b"],
            "T": [0, 0, 1, 1],
            "y": [0.1, 0.2, 0.3, 0.4],
        }
    )


def test_explicit_spec_shapes():
    spec = DatasetSpec(treatment="T", outcome="y", forcing=["age"], covariates=["age", "score", "group"])
    ds = Dataset(toy_df(), spec)
    assert ds.n == 4
    assert ds.T.shape == (4,) and ds.T.dtype == float
    assert ds.Z.shape == (4, 1)
    assert ds.X.shape[0] == 4 and ds.X.shape[1] >= 4  # age, score, group one-hot(2)
    assert ds.treatment_is_binary is True
    np.testing.assert_allclose(ds.Z_std.mean(axis=0), 0.0, atol=1e-12)
    np.testing.assert_allclose(ds.Z_std.std(axis=0), 1.0, atol=1e-12)


def test_from_csv_defaults(tmp_path):
    p = tmp_path / "d.csv"
    toy_df().to_csv(p, index=False)
    ds = Dataset.from_csv(p, treatment="T", outcome="y")
    assert set(ds.spec.forcing) == {"age", "score"}  # numeric, non-T/y
    assert "group" in ds.spec.covariates


def test_nonnumeric_forcing_rejected():
    spec = DatasetSpec(treatment="T", outcome="y", forcing=["group"], covariates=["group"])
    with pytest.raises(ValueError, match="forcing"):
        Dataset(toy_df(), spec)


def test_standardize_bitwise_consistent_with_Z_std():
    spec = DatasetSpec(
        treatment="T", outcome="y", forcing=["age", "score"], covariates=["age", "score", "group"]
    )
    ds = Dataset(toy_df(), spec)
    assert np.array_equal(ds.standardize(ds.Z), ds.Z_std)  # bitwise


def test_standardize_zero_variance_column_passes_through():
    df = toy_df()
    df["const"] = 3.0
    spec = DatasetSpec(
        treatment="T", outcome="y", forcing=["age", "const"], covariates=["age", "const"]
    )
    ds = Dataset(df, spec)
    out = ds.standardize(np.array([[2.0, 10.0]]))
    assert out[0, 1] == 10.0 - 3.0  # centered, unscaled (sd 0 -> 1)
    assert np.array_equal(ds.standardize(ds.Z), ds.Z_std)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"treatment": "nope"},
        {"outcome": "nope"},
        {"forcing": ["age", "nope"], "covariates": ["age", "nope"]},
        {"covariates": ["age", "score", "nope"]},
        {"time": "nope"},
        {"unit": "nope"},
    ],
    ids=["treatment", "outcome", "forcing", "covariate", "time", "unit"],
)
def test_issue_26_missing_role_column_raises_eagerly(kwargs):
    """Issue #26: EVERY declared role — outcome included — must fail at
    construction with a ValueError naming the column, not a raw KeyError from
    ``ds.y`` after an expensive discovery scan."""
    base = dict(treatment="T", outcome="y", forcing=["age"], covariates=["age", "score"])
    base.update(kwargs)
    spec = DatasetSpec(**base)
    with pytest.raises(ValueError, match="nope"):
        Dataset(toy_df(), spec)


def test_issue_26_from_csv_missing_outcome_raises(tmp_path):
    p = tmp_path / "d.csv"
    toy_df().to_csv(p, index=False)
    with pytest.raises(ValueError, match="yy"):
        Dataset.from_csv(p, treatment="T", outcome="yy")


def test_issue_26_nan_outcome_values_still_tolerated():
    """Guard: only the outcome COLUMN's existence is validated — NaN outcome
    VALUES must never listwise-delete scan rows (load-bearing LSO policy)."""
    df = toy_df()
    df.loc[0, "y"] = np.nan
    spec = DatasetSpec(treatment="T", outcome="y", forcing=["age"], covariates=["age", "score"])
    ds = Dataset(df, spec)
    assert ds.n == 4


def test_standardize_shape_errors():
    spec = DatasetSpec(
        treatment="T", outcome="y", forcing=["age", "score"], covariates=["age", "score"]
    )
    ds = Dataset(toy_df(), spec)
    with pytest.raises(ValueError, match="shape"):
        ds.standardize(np.zeros((3, 5)))
    with pytest.raises(ValueError, match="shape"):
        ds.standardize(np.zeros(4))
