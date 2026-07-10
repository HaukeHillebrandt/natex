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
