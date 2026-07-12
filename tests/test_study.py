"""natex.study() Stage-0 analyst pipeline + IntakeReport (phase llm-analyst task 6).

Contract under test: fixed task order (understand, prepare, search_plan) — the
MockBackend contract; uniform fallback-to-NullBackend policy recorded in
guidance_errors; PrepPlan validated against the real df and applied only by
natex code; subsample seed drawn from the pipeline Generator and stored IN the
plan; outcome-snooping warning is advisory only; explicit-Generator requirement.
"""

import json
import re

import numpy as np
import pandas as pd
import pytest

from natex import Dataset, IntakeReport, PrepPlan, study
from natex.data.synthetic import make_synthetic
from natex.llm import MockBackend


def _write_fake_test_score(root, n=60):
    """Fake of the MDRC test-score CSV (columns as tests/test_cli.py; continuous
    posttest as tests/test_llm_null.py so NullBackend yields rdd candidates)."""
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
    """make_synthetic binary-treatment CSV with a decoy binary column 'holiday'
    inserted BEFORE T, so NullBackend ranks a holiday candidate first."""
    ds, _ = make_synthetic(n=400, px=3, pz=2, zeta=6.0, kind="binary", rng=np.random.default_rng(0))
    df = ds.df.copy()
    df.insert(df.columns.get_loc("T"), "holiday", np.random.default_rng(1).integers(0, 2, len(df)))
    path = root / "synthetic.csv"
    df.to_csv(path, index=False)
    return path


def test_rng_required():
    with pytest.raises(ValueError, match="Generator"):
        study(pd.DataFrame({"a": [1.0, 2.0]}))


def test_null_end_to_end_fake_test_score(tmp_path):
    csv = _write_fake_test_score(tmp_path)
    out = tmp_path / "out"
    report = study(csv, context="MDRC test score demo", rng=np.random.default_rng(0), out=out)

    assert (out / "intake_report.json").exists()
    assert (out / "prep_plan.json").exists()
    log_path = out / "guidance_log.jsonl"
    assert log_path.exists()
    lines = [json.loads(ln) for ln in log_path.read_text().splitlines() if ln.strip()]
    assert [e["task"] for e in lines] == ["understand", "prepare", "search_plan"]
    assert all(e["backend"] == "null" for e in lines)

    assert report.understanding.shape == "cross-section"
    assert report.source == str(csv)
    assert report.guidance_log_path == str(log_path)
    assert report.guidance_errors == []
    treatments = [c.treatment for c in report.search_plan.candidates]
    assert "treat" in treatments

    loaded = IntakeReport.load(out / "intake_report.json")
    assert loaded.search_plan == report.search_plan
    assert loaded.prep_plan == report.prep_plan
    assert loaded.understanding == report.understanding

    ranked = report.search_plan.ranked()
    idx = next(i for i, c in enumerate(ranked) if c.treatment == "treat")
    ds = report.prepare(candidate=idx)
    assert isinstance(ds, Dataset)
    assert ds.spec.treatment == "treat"
    assert ds.n > 0
    # loaded report has _df=None but source is an existing csv path
    ds2 = IntakeReport.load(out / "intake_report.json").prepare(candidate=idx)
    assert ds2.spec.treatment == "treat"

    with pytest.raises(ValueError, match="candidate"):
        report.prepare(candidate=len(ranked))


def test_determinism_same_seed_same_json(tmp_path):
    csv = _write_fake_test_score(tmp_path)
    r1 = study(csv, context="ctx", rng=np.random.default_rng(0))
    r2 = study(csv, context="ctx", rng=np.random.default_rng(0))
    assert r1.to_json() == r2.to_json()


def test_mock_prepare_fallback_to_null(tmp_path):
    csv = _write_synthetic_csv(tmp_path)
    mock = MockBackend([
        {"shape": "cross-section"},
        {"version": 1, "drop_cols": ["no_such_column"]},
        {"candidates": [{"design": "rdd", "treatment": "T", "outcome": "y",
                         "forcing": ["x0"]}], "budget": {}},
    ])
    report = study(csv, guidance=mock, rng=np.random.default_rng(0))
    matching = [e for e in report.guidance_errors if re.search(r"prepare:.*unknown", e)]
    assert len(matching) == 1
    # the applied plan is the Null one (profile-only: nothing to drop here)
    assert report.prep_plan.drop_cols == []
    assert report.prep_plan.filters == []
    assert report.prep_plan.subsample is None
    # the run still completed with the mock search plan
    assert report.search_plan.candidates[0].treatment == "T"


def test_mock_prepare_failure_strict_raises(tmp_path):
    csv = _write_synthetic_csv(tmp_path)
    mock = MockBackend([
        {"shape": "cross-section"},
        {"version": 1, "drop_cols": ["no_such_column"]},
    ])
    with pytest.raises(ValueError, match="prepare"):
        study(csv, guidance=mock, rng=np.random.default_rng(0), strict=True)


def test_mock_search_plan_bogus_candidate_dropped(tmp_path):
    csv = _write_synthetic_csv(tmp_path)
    mock = MockBackend([
        {"shape": "cross-section"},
        {"version": 1},
        {"candidates": [
            {"design": "rdd", "treatment": "T", "outcome": "y", "forcing": ["x0", "x1"],
             "priority": 0},
            {"design": "rdd", "treatment": "T", "outcome": "y", "forcing": ["bogus_col"],
             "priority": 1},
        ], "budget": {}},
    ])
    report = study(csv, guidance=mock, rng=np.random.default_rng(0))
    assert len(report.search_plan.candidates) == 1
    assert report.search_plan.candidates[0].forcing == ["x0", "x1"]
    dropped = [e for e in report.guidance_errors if "bogus_col" in e]
    assert len(dropped) == 1 and "search_plan" in dropped[0]


def test_mock_ranks_truth_first_null_ranks_decoy(tmp_path):
    csv = _write_synthetic_csv(tmp_path)
    mock = MockBackend([
        {"shape": "cross-section", "treatments": [{"column": "T", "reason": "known"}]},
        {"version": 1},
        {"candidates": [
            {"design": "rdd", "treatment": "T", "outcome": "y", "forcing": ["x0", "x1"],
             "priority": 0},
            {"design": "rdd", "treatment": "holiday", "outcome": "y", "forcing": ["x0"],
             "priority": 1},
        ], "budget": {}},
    ])
    informed = study(csv, guidance=mock, rng=np.random.default_rng(0))
    assert informed.search_plan.ranked()[0].treatment == "T"
    # Null on the same csv ranks the decoy first (guards task-10 eval discrimination)
    blind = study(csv, rng=np.random.default_rng(0))
    assert blind.search_plan.ranked()[0].treatment == "holiday"


def test_snooping_warning_advisory_only(tmp_path):
    csv = _write_synthetic_csv(tmp_path)
    mock = MockBackend([
        {"shape": "cross-section"},
        {"version": 1, "filters": [{"col": "y", "op": ">", "value": 0.0}]},
        {"candidates": [{"design": "rdd", "treatment": "T", "outcome": "y",
                         "forcing": ["x0"]}], "budget": {}},
    ])
    report = study(csv, guidance=mock, rng=np.random.default_rng(0))
    warnings = [e for e in report.guidance_errors if "outcome" in e and "snooping" in e]
    assert warnings and "'y'" in warnings[0]
    # warning only: the plan was still applied
    assert report.prep_plan == PrepPlan.model_validate(
        {"version": 1, "filters": [{"col": "y", "op": ">", "value": 0.0}]}
    )
    assert any(ln.startswith("filter y > 0.0") for ln in report.prep_log)
