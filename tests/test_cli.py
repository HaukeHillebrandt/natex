import json
import subprocess
import sys

import numpy as np
import pandas as pd
from typer.testing import CliRunner

from natex.cli import app
from natex.data.registry import REGISTRY
from natex.data.synthetic import make_synthetic
from natex.data.synthetic_did import make_did_synthetic


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
    """`natex datasets` on a root holding only the fake test-score CSV: 6 lines,
    one found, five missing with their fetch-instruction (source) text; exit 0."""
    _write_fake_test_score(tmp_path)
    runner = CliRunner()
    result = runner.invoke(app, ["datasets", "--root", str(tmp_path)])
    assert result.exit_code == 0, result.output
    lines = [ln for ln in result.output.splitlines() if ln.strip()]
    assert len(lines) == 6
    assert sum("found" in ln for ln in lines) == 1
    assert sum("missing" in ln for ln in lines) == 5
    found_line = next(ln for ln in lines if "found" in ln)
    assert "test_score_2012" in found_line
    for name in ("academic_probation", "ed_visits", "inpatient_visits",
                 "egger_koethenbuerger", "prop99"):
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


def test_discover_did_smoke(tmp_path):
    """--design did: full scan + validation + three-control effects payload.

    DGP seed 1 is a config where the seeded scan recovers the planted subset
    exactly (nonempty subset_values, non-NaN effects), so the payload checks
    exercise the non-degenerate path.
    """
    ds, _ = make_did_synthetic(n=400, d=2, V=3, zeta=8.0, rng=np.random.default_rng(1))
    csv = tmp_path / "did.csv"
    ds.df.to_csv(csv, index=False)
    runner = CliRunner()
    result = runner.invoke(
        app,
        ["discover", str(csv), "--design", "did", "--treatment", "theta",
         "--outcome", "y", "--time", "t", "--q", "9", "--restarts", "2",
         "--windows", "4", "--seed", "0", "--out", str(tmp_path / "out")],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads((tmp_path / "out" / "results.json").read_text())
    did = payload["did"]
    top = did["discoveries"][0]
    assert isinstance(top["t0"], float)
    assert isinstance(top["llr"], float)
    assert isinstance(top["subset_values"], dict)
    p = did["scan"]["p_value"]
    assert 0.0 < p <= 1.0
    assert set(did["effects"]) == {"dd", "synthetic", "gess"}
    for block in did["effects"].values():
        assert {"tau", "se", "p", "pre_mse", "dose"} <= set(block)
        assert isinstance(block["tau"], float)  # non-null: recovered subset
        assert 0.0 < block["p"] <= 1.0
    # spec 6b obligation: the bundle always reports what was searched.
    searched = did["searched"]
    assert searched["windows"] == [4.0]
    assert searched["restarts"] == 2
    assert searched["method"] == "single_delta"
    assert searched["model"] == "normal"
    assert searched["dims"] == ["x0", "x1"]
    assert searched["bin_counts"] == {"x0": 3, "x1": 3}
    assert "validation" in did


def test_discover_did_requires_time(tmp_path):
    """--design did without --time: nonzero exit, message names --time."""
    ds, _ = make_did_synthetic(n=50, d=2, V=3, zeta=8.0, rng=np.random.default_rng(0))
    csv = tmp_path / "did.csv"
    ds.df.to_csv(csv, index=False)
    runner = CliRunner()
    result = runner.invoke(
        app,
        ["discover", str(csv), "--design", "did", "--treatment", "theta",
         "--out", str(tmp_path / "out")],
    )
    assert result.exit_code != 0
    assert "--time" in result.output


def test_import_surface_matplotlib_free():
    """New DiD exports importable from natex; module import stays matplotlib-free."""
    code = (
        "import sys, natex\n"
        "from natex import (suddds_scan, SuDDDSResult, DiDDiscovery, build_panel,\n"
        "                   make_did_synthetic, did_effect)\n"
        "assert 'matplotlib' not in sys.modules\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


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
