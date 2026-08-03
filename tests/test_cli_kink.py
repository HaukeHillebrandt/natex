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
    assert result.exit_code == 1, result.output
    payload = _strict_loads((out / "kink.json").read_text())
    assert payload["estimate"]["tau"] is None
    assert payload["estimate"]["first_stage_F"] is None
    assert payload["estimate"]["weak_first_stage"] is True
    assert "results:" in result.output
    assert "estimation failed:" in result.output


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


def test_kink_cli_rejects_nonnumeric_estimation_columns_cleanly(tmp_path):
    data, _ = make_rkd_synthetic(n=100, rng=np.random.default_rng(8))
    df = data.df.copy()
    df["policy"] = "not-a-number"
    csv = tmp_path / "rkd.csv"
    df.to_csv(csv, index=False)
    result = runner.invoke(
        app,
        [
            "kink",
            str(csv),
            "--outcome",
            "y",
            "--running",
            "running",
            "--treatment",
            "policy",
            "--bandwidth",
            "1",
        ],
    )
    assert result.exit_code == 2
    assert "policy" in result.output
    assert "numeric" in result.output
    assert "Traceback" not in result.output


def test_kink_cli_drops_rows_with_missing_time_instead_of_calling_them_pre(tmp_path):
    data, truth = make_dik_synthetic(n=100, rng=np.random.default_rng(9))
    df = data.df.copy()
    df.loc[0, "post"] = np.nan
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
            "--kernel",
            "uniform",
            "--out",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.output
    payload = _strict_loads((out / "kink.json").read_text())
    assert payload["estimate"]["n_used"] == 99
    assert payload["estimate"]["extras"]["n_dropped_nonfinite"] == 1


def test_kink_cli_hac_lags_flag_switches_inference_and_echoes_warning(tmp_path):
    t = np.arange(-30, 30, dtype=float) + 0.5
    rho = 0.85
    rng = np.random.default_rng(2)
    noise = np.empty(t.size)
    noise[0] = rng.standard_normal()
    for i in range(1, t.size):
        noise[i] = rho * noise[i - 1] + 0.1 * rng.standard_normal()
    df = pd.DataFrame({"t": t, "y": 0.02 * t + noise})
    csv = tmp_path / "series.csv"
    df.to_csv(csv, index=False)

    base = [
        "kink",
        str(csv),
        "--design",
        "rkd",
        "--outcome",
        "y",
        "--running",
        "t",
        "--policy-kink",
        "1.0",
        "--bandwidth",
        "30",
        "--kernel",
        "uniform",
    ]

    plain = runner.invoke(app, [*base, "--out", str(tmp_path / "plain")])
    assert plain.exit_code == 0, plain.output
    plain_payload = _strict_loads((tmp_path / "plain" / "kink.json").read_text())
    assert plain_payload["estimate"]["extras"]["inference"] == "HC1"
    assert "autocorrelation_warning" in plain_payload["estimate"]["extras"]
    assert "warning:" in plain.output

    hac = runner.invoke(
        app, [*base, "--hac-lags", "4", "--out", str(tmp_path / "hac")]
    )
    assert hac.exit_code == 0, hac.output
    payload = _strict_loads((tmp_path / "hac" / "kink.json").read_text())
    assert payload["params"]["hac_lags"] == 4
    assert payload["estimate"]["extras"]["inference"] == "HAC"
    assert payload["estimate"]["extras"]["hac_lags"] == 4
    assert "warning:" not in hac.output
    assert (
        payload["estimate"]["reduced_form_se"]
        > plain_payload["estimate"]["reduced_form_se"]
    )


def test_kink_cli_rejects_hac_lags_combined_with_cluster(tmp_path):
    t = np.arange(-10, 10, dtype=float) + 0.5
    df = pd.DataFrame({"t": t, "y": 0.1 * t, "g": (np.arange(t.size) % 4)})
    csv = tmp_path / "series.csv"
    df.to_csv(csv, index=False)

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
            "t",
            "--policy-kink",
            "1.0",
            "--bandwidth",
            "10",
            "--cluster",
            "g",
            "--hac-lags",
            "2",
            "--out",
            str(tmp_path / "out"),
        ],
    )

    assert result.exit_code == 2
    assert "at most one" in result.output
    assert "Traceback" not in result.output


def test_kink_cli_weights_column_changes_inference_and_is_recorded(tmp_path):
    rng = np.random.default_rng(6)
    x = np.r_[np.linspace(-1.0, -0.05, 25), np.linspace(0.05, 1.0, 25)]
    sigma = 0.2 + rng.random(x.size)
    y = 0.4 * x + 0.9 * np.maximum(x, 0.0) + sigma * rng.standard_normal(x.size)
    df = pd.DataFrame({"x": x, "y": y, "w": 1.0 / sigma**2})
    csv = tmp_path / "weighted.csv"
    df.to_csv(csv, index=False)

    base = [
        "kink",
        str(csv),
        "--design",
        "rkd",
        "--outcome",
        "y",
        "--running",
        "x",
        "--policy-kink",
        "0.9",
        "--bandwidth",
        "1.0",
    ]
    plain = runner.invoke(app, [*base, "--out", str(tmp_path / "plain")])
    weighted = runner.invoke(
        app, [*base, "--weights", "w", "--out", str(tmp_path / "weighted")]
    )

    assert plain.exit_code == 0, plain.output
    assert weighted.exit_code == 0, weighted.output
    plain_payload = _strict_loads((tmp_path / "plain" / "kink.json").read_text())
    payload = _strict_loads((tmp_path / "weighted" / "kink.json").read_text())
    assert payload["params"]["weights"] == "w"
    assert payload["estimate"]["extras"]["user_weights"] is True
    assert plain_payload["estimate"]["extras"]["user_weights"] is False
    assert payload["estimate"]["tau"] != plain_payload["estimate"]["tau"]


def test_kink_cli_rejects_unknown_weights_column(tmp_path):
    df = pd.DataFrame({"x": np.linspace(-1, 1, 20), "y": np.zeros(20)})
    csv = tmp_path / "d.csv"
    df.to_csv(csv, index=False)

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
            "x",
            "--policy-kink",
            "1.0",
            "--bandwidth",
            "1.0",
            "--weights",
            "nope",
            "--out",
            str(tmp_path / "out"),
        ],
    )

    assert result.exit_code == 2
    assert "nope" in result.output
    assert "Traceback" not in result.output


def _group_dik_frame():
    t = np.tile(np.arange(-12, 12, dtype=float) + 0.5, 2)
    group = np.r_[np.zeros(24), np.ones(24)]
    rng = np.random.default_rng(41)
    y = (
        0.1 * t
        + 0.5 * np.maximum(t, 0.0)
        + 0.8 * np.maximum(t, 0.0) * group
        + 0.05 * rng.standard_normal(t.size)
    )
    return pd.DataFrame({"t": t, "g": group.astype(int), "y": y})


def test_kink_cli_group_dik_runs_with_neutral_labels_and_group_caveat(tmp_path):
    df = _group_dik_frame()
    csv = tmp_path / "group.csv"
    df.to_csv(csv, index=False)

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
            "t",
            "--group",
            "g",
            "--policy-kink-change",
            "1.0",
            "--bandwidth",
            "12",
            "--kernel",
            "uniform",
            "--out",
            str(tmp_path / "out"),
        ],
    )

    assert result.exit_code == 0, result.output
    payload = _strict_loads((tmp_path / "out" / "kink.json").read_text())
    assert payload["params"]["group"] == "g"
    assert payload["estimate"]["extras"]["dik_contrast"] == "group1_minus_group0"
    assert "group1_left" in payload["estimate"]["n_by_cell"]
    caveats = " ".join(payload["identification_caveats"])
    assert "across groups" in caveats
    assert "group-stable" in caveats
    assert "time-stable" not in caveats


def test_kink_cli_group_excludes_time_and_requires_dik(tmp_path):
    df = _group_dik_frame()
    csv = tmp_path / "group.csv"
    df.to_csv(csv, index=False)
    base = [
        "kink",
        str(csv),
        "--outcome",
        "y",
        "--running",
        "t",
        "--policy-kink-change",
        "1.0",
        "--bandwidth",
        "12",
        "--out",
        str(tmp_path / "out"),
    ]

    both = runner.invoke(
        app,
        [*base, "--design", "dik", "--group", "g", "--time", "t", "--t0", "0.5"],
    )
    assert both.exit_code == 2
    assert "not both" in both.output

    neither = runner.invoke(app, [*base, "--design", "dik"])
    assert neither.exit_code == 2
    assert "--group" in neither.output and "--time" in neither.output

    rkd = runner.invoke(
        app,
        [
            "kink",
            str(csv),
            "--design",
            "rkd",
            "--outcome",
            "y",
            "--running",
            "t",
            "--policy-kink",
            "1.0",
            "--bandwidth",
            "12",
            "--group",
            "g",
            "--out",
            str(tmp_path / "out"),
        ],
    )
    assert rkd.exit_code == 2
    assert "--design dik" in rkd.output


def test_kink_cli_group_column_must_be_binary(tmp_path):
    df = _group_dik_frame()
    df["g"] = np.arange(len(df))
    csv = tmp_path / "bad.csv"
    df.to_csv(csv, index=False)

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
            "t",
            "--group",
            "g",
            "--policy-kink-change",
            "1.0",
            "--bandwidth",
            "12",
            "--out",
            str(tmp_path / "out"),
        ],
    )

    assert result.exit_code == 2
    assert "0/1" in result.output
    assert "Traceback" not in result.output


def test_kink_cli_hc3_flag_widens_tiny_cell_se_and_echoes_warning(tmp_path):
    rng = np.random.default_rng(56)
    x = np.r_[np.linspace(-1.6, -0.1, 9), np.linspace(0.15, 0.55, 3)]
    y = 2.0 * x + 17.5 * np.maximum(x, 0.0) + 1.5 * rng.standard_normal(x.size)
    df = pd.DataFrame({"x": x, "y": y})
    csv = tmp_path / "tiny.csv"
    df.to_csv(csv, index=False)

    base = [
        "kink",
        str(csv),
        "--design",
        "rkd",
        "--outcome",
        "y",
        "--running",
        "x",
        "--policy-kink",
        "1.0",
        "--bandwidth",
        "1.6",
    ]
    hc1 = runner.invoke(app, [*base, "--out", str(tmp_path / "hc1")])
    hc3 = runner.invoke(app, [*base, "--hc", "hc3", "--out", str(tmp_path / "hc3")])

    assert hc1.exit_code == 0, hc1.output
    assert hc3.exit_code == 0, hc3.output
    assert "warning:" in hc1.output and "not calibrated" in hc1.output
    hc1_payload = _strict_loads((tmp_path / "hc1" / "kink.json").read_text())
    hc3_payload = _strict_loads((tmp_path / "hc3" / "kink.json").read_text())
    assert hc3_payload["params"]["hc"] == "hc3"
    assert hc3_payload["estimate"]["extras"]["inference"] == "HC3"
    assert hc3_payload["estimate"]["se"] > 1.3 * hc1_payload["estimate"]["se"]

    bad = runner.invoke(
        app, [*base, "--hc", "hc9", "--out", str(tmp_path / "bad")]
    )
    assert bad.exit_code == 2
    assert "hc9" in bad.output
