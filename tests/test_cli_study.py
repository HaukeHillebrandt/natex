"""CLI `natex study` + plan-driven `natex discover --plan` (phase llm-analyst task 9).

Contract under test: `_make_backend` factory (null -> None; unknown backend and
missing [llm] extras exit 2 cleanly, no traceback); `natex study` writes
intake_report.json / prep_plan.json / guidance_log.jsonl and echoes shape,
candidates and warnings; `natex discover --plan` loads the IntakeReport,
prepares the Dataset, merges budgets and reports full search coverage
(spec 6b: budget cuts are listed as skipped_budget, never dropped); no plan
and no treatment exits 2 naming both options.
"""

import json
import sys

import numpy as np
import pandas as pd
from typer.testing import CliRunner

from natex.cli import app
from natex.data.synthetic import make_synthetic

runner = CliRunner()


def _write_fake_test_score(root, n=60):
    """Fake of the MDRC test-score CSV (columns as tests/test_cli.py; continuous
    posttest as tests/test_study.py so NullBackend yields rdd candidates)."""
    rng = np.random.default_rng(0)
    cols = [
        "ID", "gender", "sped", "frlunch", "esol", "black", "white", "hispanic",
        "asian", "age", "pretest", "cutoff", "treat", "posttest",
    ]
    df = pd.DataFrame({c: rng.integers(0, 2, n) for c in cols})
    df["pretest"] = rng.integers(170, 268, n)
    df["posttest"] = rng.normal(220.0, 20.0, n)
    df["treat"] = (df["pretest"] < 215).astype(int)
    path = root / "RDD_Guide_Dataset_0.csv"
    df.to_csv(path, index=False)
    return path


def _write_synthetic_csv(root):
    """make_synthetic(n=300) binary-treatment CSV with a decoy binary column
    'holiday' inserted BEFORE T, so the Null search plan has >= 2 candidates
    and ranks the decoy first (as tests/test_study.py)."""
    ds, _ = make_synthetic(
        n=300, px=3, pz=2, zeta=6.0, kind="binary", rng=np.random.default_rng(0)
    )
    df = ds.df.copy()
    df.insert(df.columns.get_loc("T"), "holiday",
              np.random.default_rng(1).integers(0, 2, len(df)))
    path = root / "synthetic.csv"
    df.to_csv(path, index=False)
    return path


def test_study_null_end_to_end(tmp_path):
    csv = _write_fake_test_score(tmp_path)
    out = tmp_path / "out"
    result = runner.invoke(
        app,
        ["study", str(csv), "--backend", "null", "--context", "MDRC test score demo",
         "--seed", "0", "--out", str(out)],
    )
    assert result.exit_code == 0, result.output
    assert (out / "intake_report.json").exists()
    assert (out / "prep_plan.json").exists()
    assert (out / "guidance_log.jsonl").exists()
    # echoed: shape + unit of observation, candidate count + top line, paths
    assert "cross-section" in result.output
    assert "candidates:" in result.output
    assert "treat" in result.output
    for name in ("intake_report.json", "prep_plan.json", "guidance_log.jsonl"):
        assert name in result.output
    report = json.loads((out / "intake_report.json").read_text())
    assert any(c["treatment"] == "treat" for c in report["search_plan"]["candidates"])
    # NullBackend run: no fallbacks, so no "warning: " lines
    assert "warning:" not in result.output


def test_study_unknown_backend_exit_2(tmp_path):
    csv = _write_fake_test_score(tmp_path)
    result = runner.invoke(
        app, ["study", str(csv), "--backend", "bogus", "--out", str(tmp_path / "out")]
    )
    assert result.exit_code == 2
    assert "bogus" in result.output
    assert not (tmp_path / "out" / "intake_report.json").exists()


def test_study_missing_llm_extra_exit_2(tmp_path, monkeypatch):
    """A missing SDK exits 2 with the install message, no traceback."""
    csv = _write_fake_test_score(tmp_path)
    monkeypatch.setitem(sys.modules, "anthropic", None)
    result = runner.invoke(
        app, ["study", str(csv), "--backend", "anthropic", "--out", str(tmp_path / "out")]
    )
    assert result.exit_code == 2
    assert "natex-discovery[llm]" in result.output
    assert result.exception is None or isinstance(result.exception, SystemExit)


def test_study_then_discover_plan_round_trip(tmp_path):
    csv = _write_synthetic_csv(tmp_path)
    out1, out2 = tmp_path / "out1", tmp_path / "out2"
    res1 = runner.invoke(app, ["study", str(csv), "--seed", "0", "--out", str(out1)])
    assert res1.exit_code == 0, res1.output

    res2 = runner.invoke(
        app,
        ["discover", "--plan", str(out1 / "intake_report.json"), str(csv),
         "--q", "9", "--k", "25", "--seed", "0", "--out", str(out2)],
    )
    assert res2.exit_code == 0, res2.output
    assert "scanned" in res2.output
    payload = json.loads((out2 / "discover_report.json").read_text())
    assert payload["searched"]["n_total"] >= 1
    assert payload["searched"]["n_scanned"] >= 1
    valid_statuses = {"scanned", "skipped_budget", "failed", "invalid"}
    assert all(rec["status"] in valid_statuses for rec in payload["configs"])
    # CLI budget overrides merged over the plan's hints (explicit wins)
    assert payload["searched"]["budget"]["k"] == 25
    assert payload["searched"]["budget"]["q"] == 9


def test_issue_2_discover_plan_writes_results_bundle(tmp_path):
    """Issue #2: plan mode wrote ONLY discover_report.json, so `natex paper`
    rendered 'seed —' and 'No dataset metadata was recorded' despite the run
    having seed, dataset and intake in scope. Plan mode now ALSO saves a full
    ResultsBundle (results.json with the natex_bundle marker), which wins over
    the discover_report.json compat path on load; discover_report.json stays
    (documented output)."""
    import natex
    from natex.report.bundle import ResultsBundle

    csv = _write_synthetic_csv(tmp_path)
    out1, out2 = tmp_path / "out1", tmp_path / "out2"
    res1 = runner.invoke(app, ["study", str(csv), "--seed", "0", "--out", str(out1)])
    assert res1.exit_code == 0, res1.output
    res2 = runner.invoke(
        app,
        ["discover", "--plan", str(out1 / "intake_report.json"), str(csv),
         "--q", "9", "--k", "25", "--seed", "0", "--out", str(out2)],
    )
    assert res2.exit_code == 0, res2.output
    assert (out2 / "discover_report.json").exists()  # docs promise it
    payload = json.loads((out2 / "results.json").read_text())
    assert payload["natex_bundle"] == 1
    assert payload["seed"] == 0
    assert payload["natex_version"] == natex.__version__
    assert payload["params"]["k"] == 25 and payload["params"]["q"] == 9
    data = payload["data"]
    assert data["n_rows"] is not None and data["treatment"] is not None
    intake = payload["intake"]
    assert intake["source"] == str(csv)
    assert "understanding" in intake
    # load() resolves the bundle (path 1), not the discover_report compat path
    loaded = ResultsBundle.load(out2)
    assert loaded.results["seed"] == 0
    assert loaded.results["natex_version"] == natex.__version__


def test_discover_plan_max_configs_lists_skipped(tmp_path):
    """spec 6b through the CLI: --max-configs 1 on a >= 2 candidate plan lists
    the remainder as skipped_budget instead of silently dropping it."""
    csv = _write_synthetic_csv(tmp_path)
    out1, out2 = tmp_path / "out1", tmp_path / "out2"
    res1 = runner.invoke(app, ["study", str(csv), "--seed", "0", "--out", str(out1)])
    assert res1.exit_code == 0, res1.output
    plan = json.loads((out1 / "intake_report.json").read_text())
    assert len(plan["search_plan"]["candidates"]) >= 2

    res2 = runner.invoke(
        app,
        ["discover", "--plan", str(out1 / "intake_report.json"), str(csv),
         "--q", "9", "--k", "25", "--max-configs", "1", "--seed", "0",
         "--out", str(out2)],
    )
    assert res2.exit_code == 0, res2.output  # one config still scanned
    payload = json.loads((out2 / "discover_report.json").read_text())
    assert payload["searched"]["n_skipped_budget"] >= 1
    assert payload["searched"]["budget"]["max_configs"] == 1
    assert "skipped by budget" in res2.output

    # --max-configs 0: nothing scanned => exit 1, coverage still reported
    res3 = runner.invoke(
        app,
        ["discover", "--plan", str(out1 / "intake_report.json"), str(csv),
         "--q", "9", "--k", "25", "--max-configs", "0", "--seed", "0",
         "--out", str(tmp_path / "out3")],
    )
    assert res3.exit_code == 1
    assert "no configuration scanned successfully" in res3.output
    payload3 = json.loads((tmp_path / "out3" / "discover_report.json").read_text())
    assert payload3["searched"]["n_scanned"] == 0
    assert payload3["searched"]["n_skipped_budget"] == payload3["searched"]["n_total"]


def test_discover_no_plan_no_treatment_exit_2(tmp_path):
    csv = _write_fake_test_score(tmp_path)
    result = runner.invoke(app, ["discover", str(csv)])
    assert result.exit_code == 2
    assert "--treatment" in result.output
    assert "--plan" in result.output


def test_discover_bad_plan_path_exit_2(tmp_path):
    result = runner.invoke(
        app, ["discover", "--plan", str(tmp_path / "missing.json"),
              "--out", str(tmp_path / "out")]
    )
    assert result.exit_code == 2
    assert "missing.json" in result.output
