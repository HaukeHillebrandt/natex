"""CLI coverage for known-cutoff RKD and difference-in-kinks estimation."""

import json

import numpy as np
import pandas as pd
from typer.testing import CliRunner

from natex.cli import app
from natex.data.synthetic_kink import make_dik_synthetic, make_rkd_synthetic

runner = CliRunner()


def _strict_loads(text):
    def reject_nonfinite(value):
        raise AssertionError(f"non-finite JSON constant leaked: {value}")

    return json.loads(text, parse_constant=reject_nonfinite)


def test_sharp_rkd_cli_writes_nan_clean_result(tmp_path):
    data, truth = make_rkd_synthetic(
        n=5000, outcome_noise=0.25, rng=np.random.default_rng(1)
    )
    csv = tmp_path / "rkd.csv"
    data.df.to_csv(csv, index=False)
    out = tmp_path / "out"

    result = runner.invoke(
        app,
        [
            "kink",
            str(csv),
            "--design",
            "rkd",
            "--outcome",
            "y",
            "--running",
            "running",
            "--policy-kink",
            str(truth.policy_kink),
            "--bandwidth",
            "0.7",
            "--out",
            str(out),
        ],
    )

    assert result.exit_code == 0, result.output
    payload = _strict_loads((out / "kink.json").read_text())
    assert payload["estimate"]["method"] == "sharp_rkd"
    assert abs(payload["estimate"]["tau"] - truth.expected_rkd) < 0.2
    assert payload["params"]["contrast"] == "right_minus_left"
    assert payload["params"]["bandwidth"] == 0.7
    assert payload["estimate"]["weak_first_stage"] is False
    assert "results:" in result.output


def test_fuzzy_dik_cli_uses_time_threshold_and_records_identification_caveat(tmp_path):
    data, truth = make_dik_synthetic(
        n=10000, fuzzy=True, outcome_noise=0.3, rng=np.random.default_rng(2)
    )
    csv = tmp_path / "dik.csv"
    data.df.to_csv(csv, index=False)
    out = tmp_path / "out"

    result = runner.invoke(
        app,
        [
            "kink",
            str(csv),
            "--design",
            "dik",
            "--outcome",
            "y",
            "--running",
            "running",
            "--treatment",
            "policy",
            "--time",
            "post",
            "--t0",
            "1",
            "--bandwidth",
            "0.7",
            "--out",
            str(out),
        ],
    )

    assert result.exit_code == 0, result.output
    payload = _strict_loads((out / "kink.json").read_text())
    estimate = payload["estimate"]
    assert estimate["method"] == "fuzzy_dik"
    assert abs(estimate["tau"] - truth.expected_dik) < 0.35
    assert estimate["fieller_kind"] == "interval"
    assert estimate["first_stage_F"] > 10.0
    caveats = " ".join(payload["identification_caveats"]).lower()
    assert "composition" in caveats
    assert "same sign" in caveats
    assert payload["params"]["t0"] == 1.0


def test_sharp_dik_cli_accepts_known_policy_kink_change(tmp_path):
    data, truth = make_dik_synthetic(n=8000, rng=np.random.default_rng(3))
    csv = tmp_path / "dik.csv"
    data.df.to_csv(csv, index=False)
    out = tmp_path / "out"
    result = runner.invoke(
        app,
        [
            "kink",
            str(csv),
            "--design",
            "dik",
            "--outcome",
            "y",
            "--running",
            "running",
            "--policy-kink-change",
            str(truth.policy_kink_change),
            "--time",
            "post",
            "--t0",
            "1",
            "--bandwidth",
            "0.8",
            "--out",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.output
    payload = _strict_loads((out / "kink.json").read_text())
    assert payload["estimate"]["method"] == "sharp_dik"
    assert payload["estimate"]["first_stage"] == truth.policy_kink_change


def test_kink_cli_passes_covariates_and_cluster_column(tmp_path):
    data, truth = make_dik_synthetic(n=400, outcome_noise=0.1, rng=np.random.default_rng(4))
    df = data.df.copy()
    df["z"] = np.exp(df["running"])
    df["unit"] = np.arange(len(df)) // 2
    csv = tmp_path / "dik.csv"
    df.to_csv(csv, index=False)
    out = tmp_path / "out"
    result = runner.invoke(
        app,
        [
            "kink",
            str(csv),
            "--design",
            "dik",
            "--outcome",
            "y",
            "--running",
            "running",
            "--policy-kink-change",
            str(truth.policy_kink_change),
            "--time",
            "post",
            "--t0",
            "1",
            "--bandwidth",
            "1",
            "--covariates",
            "z",
            "--cluster",
            "unit",
            "--out",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.output
    payload = _strict_loads((out / "kink.json").read_text())
    assert payload["estimate"]["extras"]["inference"] == "CR1"
    assert payload["estimate"]["extras"]["n_clusters"] == 200
    assert payload["params"]["covariates"] == ["z"]


def test_weak_fuzzy_kink_serializes_nan_as_null(tmp_path):
    v = np.r_[np.linspace(-1.0, -0.02, 40), np.linspace(0.02, 1.0, 40)]
    df = pd.DataFrame({"v": v, "b": 0.4 * v, "y": np.maximum(v, 0.0)})
    csv = tmp_path / "weak.csv"
    df.to_csv(csv, index=False)
    out = tmp_path / "out"
    result = runner.invoke(
        app,
        [
            "kink",
            str(csv),
            "--outcome",
            "y",
            "--running",
            "v",
            "--treatment",
            "b",
            "--bandwidth",
            "1",
            "--kernel",
            "uniform",
            "--out",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.output
    payload = _strict_loads((out / "kink.json").read_text())
    assert payload["estimate"]["tau"] is None
    assert payload["estimate"]["first_stage_F"] is None
    assert payload["estimate"]["weak_first_stage"] is True


def test_kink_cli_requires_one_first_stage_source(tmp_path):
    data, truth = make_rkd_synthetic(n=100, rng=np.random.default_rng(5))
    csv = tmp_path / "rkd.csv"
    data.df.to_csv(csv, index=False)
    base = [
        "kink",
        str(csv),
        "--outcome",
        "y",
        "--running",
        "running",
        "--bandwidth",
        "1",
    ]
    missing = runner.invoke(app, base)
    both = runner.invoke(
        app,
        [*base, "--treatment", "policy", "--policy-kink", str(truth.policy_kink)],
    )
    assert missing.exit_code == 2
    assert "exactly one" in missing.output
    assert both.exit_code == 2
    assert "exactly one" in both.output


def test_kink_cli_dik_requires_time_and_t0(tmp_path):
    data, truth = make_dik_synthetic(n=100, rng=np.random.default_rng(6))
    csv = tmp_path / "dik.csv"
    data.df.to_csv(csv, index=False)
    result = runner.invoke(
        app,
        [
            "kink",
            str(csv),
            "--design",
            "dik",
            "--outcome",
            "y",
            "--running",
            "running",
            "--policy-kink-change",
            str(truth.policy_kink_change),
            "--bandwidth",
            "1",
        ],
    )
    assert result.exit_code == 2
    assert "--time" in result.output and "--t0" in result.output


def test_kink_cli_rejects_unknown_columns_without_traceback(tmp_path):
    data, _ = make_rkd_synthetic(n=100, rng=np.random.default_rng(7))
    csv = tmp_path / "rkd.csv"
    data.df.to_csv(csv, index=False)
    result = runner.invoke(
        app,
        [
            "kink",
            str(csv),
            "--outcome",
            "ghost",
            "--running",
            "running",
            "--policy-kink",
            "1",
            "--bandwidth",
            "1",
        ],
    )
    assert result.exit_code == 2
    assert "ghost" in result.output
    assert "Traceback" not in result.output
