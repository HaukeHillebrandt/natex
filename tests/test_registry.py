"""Registry/loader tests against a tmp fake data root — no real data needed in CI."""

import numpy as np
import pandas as pd
import pytest

from natex.data.registry import REGISTRY, data_root, load_dataset, locate, verify

EXPECTED_NAMES = {
    "test_score_2012",
    "academic_probation",
    "ed_visits",
    "inpatient_visits",
    "egger_koethenbuerger",
    "prop99",
}

TEST_SCORE_HEADER = [
    "ID", "gender", "sped", "frlunch", "esol", "black", "white", "hispanic",
    "asian", "age", "pretest", "cutoff", "treat", "posttest",
]

EGGER_HEADER = [
    "id", "year", "pop", "tratea", "trateb", "tratep", "exptot", "exppers",
    "expsach", "expsachinv", "debt", "wpop1983", "wpop1989", "wpop1995",
    "wpop2001", "rlcsize", "wpop", "rcsize", "phase",
]


def _write_test_score(root, n=30):
    rng = np.random.default_rng(0)
    df = pd.DataFrame({c: rng.integers(0, 2, n) for c in TEST_SCORE_HEADER})
    df["ID"] = np.arange(n)
    df["age"] = rng.integers(10, 14, n)
    df["pretest"] = rng.integers(170, 268, n)
    df["cutoff"] = 215
    df["treat"] = (df["pretest"] < 215).astype(int)
    df["posttest"] = df["pretest"] + rng.normal(0, 5, n)
    path = root / "test_score_2012" / "RDD_Guide_Dataset_0.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def _write_egger(root, wpop, filename="EggerKoethenbuerger_AEJ_Data (1).csv"):
    n = len(wpop)
    rng = np.random.default_rng(1)
    df = pd.DataFrame({c: rng.normal(size=n) for c in EGGER_HEADER})
    df["id"] = np.arange(n)
    df["year"] = 1990
    df["wpop"] = wpop
    df["rcsize"] = rng.integers(8, 20, n)
    path = root / filename
    df.to_csv(path, index=False)
    return path


def _write_academic_probation(root, n=25):
    rng = np.random.default_rng(2)
    df = pd.DataFrame(
        {
            "probation_year1": rng.integers(0, 2, n),
            "GPA_year2": rng.normal(2.5, 0.5, n),
            "left_school": rng.integers(0, 2, n),
            "dist_from_cut": rng.normal(0, 0.8, n),
            "hsgrade_pct": rng.uniform(1, 100, n),
            "totcredits_year1": rng.integers(3, 6, n),
            "age_at_entry": rng.integers(17, 22, n),
            "sex": rng.choice(["M", "F"], n),
            "bpl_north_america": rng.integers(0, 2, n),
            "mtongue": rng.choice(["English", "French", "Other"], n),
            "loc_campus1": rng.integers(0, 2, n),
            "loc_campus2": rng.integers(0, 2, n),
        }
    )
    path = root / "AcademicProbation_LSO_2010" / "data_orig.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def test_registry_names():
    assert set(REGISTRY) == EXPECTED_NAMES
    for name, info in REGISTRY.items():
        assert info.name == name
        assert info.source.strip()


def test_data_root_requires_env(monkeypatch):
    monkeypatch.delenv("NATEX_DATA", raising=False)
    with pytest.raises(RuntimeError, match="NATEX_DATA"):
        data_root(None)


def test_load_dataset_maps_columns(tmp_path):
    _write_test_score(tmp_path, n=30)
    ds = load_dataset("test_score_2012", root=tmp_path)
    assert ds.spec.treatment == "treat"
    assert ds.spec.outcome == "posttest"
    assert ds.spec.forcing == ["age", "pretest"]
    assert ds.n == 30


def test_locate_glob_fallback(tmp_path):
    wpop = np.array([500.0, 1500.0, 2500.0, 6000.0, 12000.0])
    written = _write_egger(tmp_path, wpop)
    assert locate("egger_koethenbuerger", root=tmp_path) == written
    ds = load_dataset("egger_koethenbuerger", root=tmp_path)
    assert ds.spec.forcing == ["log_pop"]
    assert "log_pop" in ds.df.columns
    np.testing.assert_allclose(
        ds.df["log_pop"].to_numpy(), np.log(ds.df["wpop"].to_numpy())
    )


def test_egger_drops_missing_and_nonpositive_wpop(tmp_path):
    wpop = np.array([500.0, 1500.0, 2500.0, 6000.0, 12000.0, 0.0, np.nan])
    _write_egger(tmp_path, wpop)
    ds = load_dataset("egger_koethenbuerger", root=tmp_path)
    assert ds.n == 5
    assert np.all(np.isfinite(ds.df["log_pop"].to_numpy()))


def test_verify_missing_and_row_mismatch(tmp_path):
    st = verify("ed_visits", root=tmp_path)
    assert not st.found
    assert not st.ok
    assert st.path is None
    assert REGISTRY["ed_visits"].source in st.message

    path = tmp_path / "ED_visits" / "P03_ED_Analysis_File.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {"months_23": range(10), "all": range(10), "priv_all": range(10)}
    ).to_csv(path, index=False)
    st = verify("ed_visits", root=tmp_path)
    assert st.found
    assert not st.ok
    assert st.n_rows == 10
    assert "161" in st.message


def test_locate_missing_names_source(tmp_path):
    with pytest.raises(FileNotFoundError, match="openICPSR"):
        locate("academic_probation", root=tmp_path)


def test_outcome_override_and_none(tmp_path):
    _write_academic_probation(tmp_path, n=25)
    ds_none = load_dataset("academic_probation", root=tmp_path, outcome=None)
    assert ds_none.spec.outcome is None
    assert ds_none.y is None
    ds_over = load_dataset("academic_probation", root=tmp_path, outcome="left_school")
    assert ds_over.spec.outcome == "left_school"
    assert ds_over.y is not None
    ds_default = load_dataset("academic_probation", root=tmp_path)
    assert ds_default.spec.outcome == "GPA_year2"
