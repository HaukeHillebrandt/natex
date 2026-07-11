import json

import numpy as np
import pandas as pd
from typer.testing import CliRunner

from natex.cli import app
from natex.data.registry import REGISTRY
from natex.data.synthetic import make_synthetic


def test_discover_end_to_end(tmp_path):
    ds, _ = make_synthetic(n=500, zeta=4.0, kind="real", rng=np.random.default_rng(0))
    csv = tmp_path / "d.csv"
    ds.df.to_csv(csv, index=False)
    runner = CliRunner()
    result = runner.invoke(
        app,
        ["discover", str(csv), "--treatment", "T", "--outcome", "y",
         "--k", "25", "--q", "9", "--seed", "0", "--out", str(tmp_path / "out")],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads((tmp_path / "out" / "results.json").read_text())
    assert payload["scan"]["model"] == "normal"
    assert 0.0 < payload["scan"]["p_value"] <= 1.0
    assert len(payload["discoveries"]) > 0
    assert payload["effects"]["2sls"]["tau"] is not None


def _write_fake_test_score(root, n=30):
    """Minimal fake of the MDRC test-score CSV (columns only; row count differs)."""
    rng = np.random.default_rng(0)
    cols = [
        "ID", "gender", "sped", "frlunch", "esol", "black", "white", "hispanic",
        "asian", "age", "pretest", "cutoff", "treat", "posttest",
    ]
    df = pd.DataFrame({c: rng.integers(0, 2, n) for c in cols})
    df["pretest"] = rng.integers(170, 268, n)
    df["treat"] = (df["pretest"] < 215).astype(int)
    path = root / "test_score_2012" / "RDD_Guide_Dataset_0.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def test_datasets_lists_all_with_fake_root(tmp_path):
    """`natex datasets` on a root holding only the fake test-score CSV: 5 lines,
    one found, four missing with their fetch-instruction (source) text; exit 0."""
    _write_fake_test_score(tmp_path)
    runner = CliRunner()
    result = runner.invoke(app, ["datasets", "--root", str(tmp_path)])
    assert result.exit_code == 0, result.output
    lines = [ln for ln in result.output.splitlines() if ln.strip()]
    assert len(lines) == 5
    assert sum("found" in ln for ln in lines) == 1
    assert sum("missing" in ln for ln in lines) == 4
    found_line = next(ln for ln in lines if "found" in ln)
    assert "test_score_2012" in found_line
    for name in ("academic_probation", "ed_visits", "inpatient_visits", "egger_koethenbuerger"):
        line = next(ln for ln in lines if ln.startswith(name))
        assert "missing" in line
        assert REGISTRY[name].source in line


def test_discover_coarse_smoke(tmp_path):
    """--coarse writes the section-6b coverage block into results.json."""
    ds, _ = make_synthetic(n=800, zeta=4.0, kind="real", rng=np.random.default_rng(0))
    csv = tmp_path / "d.csv"
    ds.df.to_csv(csv, index=False)
    runner = CliRunner()
    result = runner.invoke(
        app,
        ["discover", str(csv), "--treatment", "T", "--outcome", "y",
         "--k", "25", "--q", "9", "--seed", "0", "--coarse", "--n-coarse", "200",
         "--out", str(tmp_path / "out")],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads((tmp_path / "out" / "results.json").read_text())
    coarse = payload["coarse"]
    assert 0.0 < coarse["frac_centers_scanned"] <= 1.0
    assert coarse["n_coarse"] == 200
    assert coarse["k"] == 25
    p = payload["scan"]["p_value"]
    assert p is not None and 0.0 < p <= 1.0
    assert payload["params"]["coarse"] is True


def test_discover_degree_passthrough(tmp_path):
    """--degree 2 runs end to end and is recorded in results.json params."""
    ds, _ = make_synthetic(n=400, zeta=4.0, kind="real", rng=np.random.default_rng(1))
    csv = tmp_path / "d.csv"
    ds.df.to_csv(csv, index=False)
    runner = CliRunner()
    result = runner.invoke(
        app,
        ["discover", str(csv), "--treatment", "T", "--outcome", "y",
         "--k", "25", "--q", "9", "--seed", "0", "--degree", "2",
         "--out", str(tmp_path / "out")],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads((tmp_path / "out" / "results.json").read_text())
    assert payload["params"]["degree"] == 2
