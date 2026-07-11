"""Prop 99 registry entry + `natex fetch-data` — synthetic fixtures, no network."""

from __future__ import annotations

import io
import urllib.request

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from natex.cli import app
from natex.data.registry import REGISTRY, load_dataset

RAW_COLUMNS = ["state", "year", "cigsale", "lnincome", "beer", "age15to24", "retprice"]

PROP99_COVARIATES = [
    "mean_lnincome",
    "mean_retprice",
    "mean_age15to24",
    "mean_beer",
    "cigsale_1975",
    "cigsale_1980",
    "cigsale_1988",
]


def _prop99_frame(states=("Alabama", "California", "Colorado"), seed=0) -> pd.DataFrame:
    """Synthetic raw Prop-99-shaped panel: len(states) x 6 years, 7 raw columns.

    Puts NaN into lnincome/beer for the earliest year to exercise the
    NaN-skipping per-state means (mirrors the real file, where lnincome starts
    1972 and beer covers 1984-1997 only).
    """
    rng = np.random.default_rng(seed)
    years = [1975, 1980, 1988, 1989, 1990, 1991]
    rows = []
    for s in states:
        for yr in years:
            rows.append(
                {
                    "state": s,
                    "year": float(yr),
                    "cigsale": float(rng.uniform(40, 300)),
                    "lnincome": np.nan if yr == 1975 else float(rng.uniform(9, 11)),
                    "beer": np.nan if yr < 1988 else float(rng.uniform(15, 40)),
                    "age15to24": float(rng.uniform(0.1, 0.2)),
                    "retprice": float(rng.uniform(30, 250)),
                }
            )
    return pd.DataFrame(rows, columns=RAW_COLUMNS)


def _write_prop99(root, df: pd.DataFrame | None = None):
    df = _prop99_frame() if df is None else df
    path = root / "prop99" / "smoking_data.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


# ---------------------------------------------------------------- registry


def test_prop99_registered_with_fetch_url():
    info = REGISTRY["prop99"]
    assert info.relpath == "prop99/smoking_data.csv"
    assert info.time == "year"
    assert info.unit == "state"
    assert info.forcing == ()
    assert info.n_rows == 1209
    assert info.fetch_url is not None and info.fetch_url.startswith("https://")
    assert "fetch-data" in info.source


def test_new_datasetinfo_fields_default_none_for_existing_entries():
    for name, info in REGISTRY.items():
        if name == "prop99":
            continue
        assert info.time is None
        assert info.unit is None
        assert info.fetch_url is None


def test_load_prop99_prepares_did_columns(tmp_path):
    _write_prop99(tmp_path)
    ds = load_dataset("prop99", root=tmp_path)
    assert ds.spec.time == "year"
    assert ds.spec.unit == "state"
    assert ds.spec.forcing == []
    assert ds.spec.treatment == "treated"
    assert ds.spec.outcome == "cigsale"
    assert ds.spec.covariates == PROP99_COVARIATES
    # 3 states x 6 years, nothing dropped: derived spec columns have no NaN.
    assert ds.n == 18
    spec_cols = ["treated", "cigsale", "year", *PROP99_COVARIATES]
    assert not ds.df[spec_cols].isna().any().any()
    # treated is exactly 1.0 for California from 1989 on, 0.0 elsewhere.
    expect = (
        (ds.df["state"] == "California") & (ds.df["year"] >= 1989)
    ).astype(float)
    assert (ds.df["treated"] == expect).all()
    assert ds.df["treated"].sum() == 3.0  # CA 1989, 1990, 1991
    # Derived covariates are state-level (constant within each state).
    for col in PROP99_COVARIATES:
        assert (ds.df.groupby("state")[col].nunique() == 1).all()
    # Year pulls match the raw cigsale values.
    raw = _prop99_frame()
    ca_1980 = raw.loc[(raw.state == "California") & (raw.year == 1980), "cigsale"].item()
    assert (
        ds.df.loc[ds.df.state == "California", "cigsale_1980"] == ca_1980
    ).all()
    # NaN-skipping means: Alabama's mean_lnincome ignores the NaN 1975 entry.
    al_mean = raw.loc[raw.state == "Alabama", "lnincome"].mean()  # pandas skips NaN
    got = ds.df.loc[ds.df.state == "Alabama", "mean_lnincome"].iloc[0]
    assert got == pytest.approx(al_mean)


def test_prepare_prop99_rejects_missing_raw_columns(tmp_path):
    df = _prop99_frame().drop(columns=["retprice"])
    _write_prop99(tmp_path, df)
    with pytest.raises(ValueError, match="retprice"):
        load_dataset("prop99", root=tmp_path)


# ---------------------------------------------------------------- datasets CLI


def test_datasets_lists_prop99_missing_with_fetch_instructions(tmp_path):
    runner = CliRunner()
    result = runner.invoke(app, ["datasets", "--root", str(tmp_path)])
    assert result.exit_code == 0, result.output
    line = next(ln for ln in result.output.splitlines() if ln.startswith("prop99"))
    assert "missing" in line
    assert "fetch-data" in line  # source carries the fetch instructions


# ---------------------------------------------------------------- fetch-data CLI


def _fixture_bytes(n_states=39) -> bytes:
    """1209-row fixture (39 states x 31 years) so verify() passes post-download."""
    states = [f"State{i:02d}" for i in range(n_states - 1)] + ["California"]
    rng = np.random.default_rng(1)
    rows = []
    for s in states:
        for yr in range(1970, 2001):
            rows.append(
                {
                    "state": s,
                    "year": float(yr),
                    "cigsale": float(rng.uniform(40, 300)),
                    "lnincome": float(rng.uniform(9, 11)),
                    "beer": float(rng.uniform(15, 40)),
                    "age15to24": float(rng.uniform(0.1, 0.2)),
                    "retprice": float(rng.uniform(30, 250)),
                }
            )
    buf = io.StringIO()
    pd.DataFrame(rows, columns=RAW_COLUMNS).to_csv(buf, index=False)
    return buf.getvalue().encode()


def _patch_urlopen(monkeypatch, payload: bytes, calls: list | None = None):
    def fake_urlopen(url, *args, **kwargs):
        if calls is not None:
            calls.append(url)
        return io.BytesIO(payload)

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)


def test_fetch_data_downloads_and_verifies(tmp_path, monkeypatch):
    payload = _fixture_bytes()
    calls: list = []
    _patch_urlopen(monkeypatch, payload, calls)
    runner = CliRunner()
    result = runner.invoke(app, ["fetch-data", "prop99", "--root", str(tmp_path)])
    assert result.exit_code == 0, result.output
    dest = tmp_path / "prop99" / "smoking_data.csv"
    assert dest.read_bytes() == payload
    assert calls == [REGISTRY["prop99"].fetch_url]
    assert "1209" in result.output
    # No leftover temp files from the atomic write.
    assert list(dest.parent.iterdir()) == [dest]


def test_fetch_data_refuses_overwrite_without_force(tmp_path, monkeypatch):
    payload = _fixture_bytes()
    _patch_urlopen(monkeypatch, payload)
    runner = CliRunner()
    assert runner.invoke(app, ["fetch-data", "prop99", "--root", str(tmp_path)]).exit_code == 0
    dest = tmp_path / "prop99" / "smoking_data.csv"
    marker = b"do-not-clobber"
    dest.write_bytes(marker)

    result = runner.invoke(app, ["fetch-data", "prop99", "--root", str(tmp_path)])
    assert result.exit_code != 0
    assert "--force" in result.output
    assert dest.read_bytes() == marker  # untouched

    result = runner.invoke(app, ["fetch-data", "prop99", "--root", str(tmp_path), "--force"])
    assert result.exit_code == 0, result.output
    assert dest.read_bytes() == payload


def test_fetch_data_login_gated_prints_source_and_exits_1(tmp_path, monkeypatch):
    def boom(*args, **kwargs):  # network must never be touched
        raise AssertionError("urlopen called for a login-gated dataset")

    monkeypatch.setattr(urllib.request, "urlopen", boom)
    runner = CliRunner()
    result = runner.invoke(app, ["fetch-data", "academic_probation", "--root", str(tmp_path)])
    assert result.exit_code == 1
    assert "openICPSR" in result.output  # info.source printed
    assert not (tmp_path / "AcademicProbation_LSO_2010").exists()


def test_fetch_data_unknown_dataset_errors(tmp_path):
    runner = CliRunner()
    result = runner.invoke(app, ["fetch-data", "nope", "--root", str(tmp_path)])
    assert result.exit_code != 0
    assert "nope" in result.output
