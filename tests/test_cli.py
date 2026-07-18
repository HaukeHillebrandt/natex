import ast
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from typer.testing import CliRunner

from natex.cli import app
from natex.data.registry import REGISTRY
from natex.data.synthetic import make_synthetic
from natex.data.synthetic_did import make_did_synthetic


def test_every_command_has_help_text():
    """Every row of the `natex --help` Commands table carries a non-empty
    description (convention pin — a command docstring's first line becomes
    the row text), and `discover --help` names the LoRD3 scan."""
    output = CliRunner().invoke(app, ["--help"]).output
    tail = output[output.index("Commands") :]
    rows: dict[str, str] = {}
    for line in tail.splitlines():
        m = re.match(r"^[│|]?\s{1,3}([a-z][a-z0-9-]+)\s*(.*)$", line)
        if m:
            rows[m.group(1)] = m.group(2).rstrip("│| ").strip()
    assert len(rows) >= 9, f"help parser lost commands: {sorted(rows)}"
    for name, desc in sorted(rows.items()):
        assert desc, f"command {name!r} has no help text in `natex --help`"
    assert "LoRD3" in CliRunner().invoke(app, ["discover", "--help"]).output


def test_every_option_has_help_text():
    """F-C2: every ``typer.Option(...)`` in the CLI carries non-empty help
    text, so no option row in ``<command> --help`` is a bare ``[required]`` or
    an undocumented default (positional arguments are out of scope)."""
    src = (Path(__file__).resolve().parents[1] / "src" / "natex" / "cli.py").read_text(
        encoding="utf-8"
    )
    missing = []
    for node in ast.walk(ast.parse(src)):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "Option"
            and getattr(node.func.value, "id", None) == "typer"
        ):
            helps = [k.value for k in node.keywords if k.arg == "help"]
            empty = helps and isinstance(helps[0], ast.Constant) and not helps[0].value
            if not helps or empty:
                missing.append(node.lineno)
    assert not missing, f"typer.Option without help= at src/natex/cli.py lines {missing}"


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


def test_issue_21_cli_coarse_uses_procedure_matched_replicas(tmp_path, monkeypatch):
    """Issue #21, CLI wiring: `discover --coarse` must calibrate the
    coarse-to-fine observed statistic with coarse-to-fine replicas."""
    import natex.cli as cli_mod

    captured = {}
    real = cli_mod.randomization_test

    def spy(ds, res, **kw):
        captured["search"] = kw.get("search")
        return real(ds, res, **kw)

    monkeypatch.setattr(cli_mod, "randomization_test", spy)
    ds, _ = make_synthetic(n=400, zeta=4.0, kind="real", rng=np.random.default_rng(0))
    csv = tmp_path / "d.csv"
    ds.df.to_csv(csv, index=False)
    result = CliRunner().invoke(
        app,
        ["discover", str(csv), "--treatment", "T", "--outcome", "y",
         "--k", "25", "--q", "9", "--seed", "0", "--coarse", "--n-coarse", "100",
         "--out", str(tmp_path / "out")],
    )
    assert result.exit_code == 0, result.output
    assert callable(captured["search"])


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


def test_issue_29_rdd_params_record_roles(tmp_path):
    """Issue #29: the plain rdd results.json params must record the run's
    treatment/outcome/forcing (the RESOLVED spec forcing, so the default
    all-numeric list is persisted) — otherwise `natex paper`/`brief` render
    'rdd: — ~ …' with the treatment name lost."""
    ds, _ = make_synthetic(n=300, zeta=4.0, kind="real", rng=np.random.default_rng(0))
    csv = tmp_path / "d.csv"
    ds.df.to_csv(csv, index=False)
    result = CliRunner().invoke(
        app,
        ["discover", str(csv), "--treatment", "T", "--outcome", "y",
         "--k", "25", "--q", "9", "--seed", "0", "--out", str(tmp_path / "out")],
    )
    assert result.exit_code == 0, result.output
    params = json.loads((tmp_path / "out" / "results.json").read_text())["params"]
    assert params["treatment"] == "T"
    assert params["outcome"] == "y"
    payload = json.loads((tmp_path / "out" / "results.json").read_text())
    assert params["forcing"] == list(payload["discoveries"][0]["forcing_influence"])


def test_issue_29_did_params_record_roles(tmp_path):
    """Issue #29, did shape: params must carry treatment/outcome/forcing
    alongside the already-persisted time/unit."""
    ds, _ = make_did_synthetic(n=400, d=2, V=3, zeta=8.0, rng=np.random.default_rng(1))
    csv = tmp_path / "did.csv"
    ds.df.to_csv(csv, index=False)
    result = CliRunner().invoke(
        app,
        ["discover", str(csv), "--design", "did", "--treatment", "theta",
         "--outcome", "y", "--time", "t", "--q", "9", "--restarts", "2",
         "--windows", "4", "--seed", "0", "--out", str(tmp_path / "out")],
    )
    assert result.exit_code == 0, result.output
    params = json.loads((tmp_path / "out" / "results.json").read_text())["params"]
    assert params["treatment"] == "theta"
    assert params["outcome"] == "y"
    assert params["forcing"] == []
    assert params["time"] == "t"


def test_issue_35_covariates_flag_restricts_rdd_scan_space(tmp_path):
    """Issue #35: dims default to ALL non-reserved columns, so an extra
    NaN-bearing metric listwise-deletes rows (here: every row) with no CLI way
    to exclude it. ``--covariates`` restricts the scan space; the resolved
    covariates are recorded in params (issue-29 pattern)."""
    ds, _ = make_synthetic(n=300, zeta=4.0, kind="real", rng=np.random.default_rng(0))
    df = ds.df.copy()
    df["label"] = [f"q{i % 4}" for i in range(len(df))]  # string decoy
    df["extra_metric"] = np.nan  # would listwise-delete EVERY row
    csv = tmp_path / "d.csv"
    df.to_csv(csv, index=False)
    args = ["discover", str(csv), "--treatment", "T", "--outcome", "y",
            "--k", "25", "--q", "9", "--seed", "0", "--out", str(tmp_path / "out")]
    # without the flag the NaN column silently deletes every scan row
    assert CliRunner().invoke(app, args).exit_code != 0
    result = CliRunner().invoke(app, args + ["--covariates", "x0,x1"])
    assert result.exit_code == 0, result.output
    payload = json.loads((tmp_path / "out" / "results.json").read_text())
    assert payload["params"]["covariates"] == ["x0", "x1"]
    assert payload["params"]["forcing"] == ["x0", "x1"]
    assert len(payload["discoveries"]) > 0
    assert set(payload["discoveries"][0]["forcing_influence"]) == {"x0", "x1"}


def test_issue_35_dims_flag_restricts_did_panel_dims(tmp_path):
    """Issue #35, did shape: a string quarter label silently enters the SuDDDS
    subset-search space; ``--dims`` (alias of ``--covariates``) restricts the
    panel dims."""
    ds, _ = make_did_synthetic(n=400, d=2, V=3, zeta=8.0, rng=np.random.default_rng(1))
    df = ds.df.copy()
    df["quarter_label"] = [f"Q{i % 4 + 1}" for i in range(len(df))]  # string decoy
    csv = tmp_path / "did.csv"
    df.to_csv(csv, index=False)
    args = ["discover", str(csv), "--design", "did", "--treatment", "theta",
            "--outcome", "y", "--time", "t", "--q", "9", "--restarts", "2",
            "--windows", "4", "--seed", "0", "--out", str(tmp_path / "out")]
    # without the flag the decoy becomes a scan dim — the reported silent entry
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 0, result.output
    searched = json.loads((tmp_path / "out" / "results.json").read_text())["did"]["searched"]
    assert "quarter_label" in searched["dims"]
    result = CliRunner().invoke(app, args + ["--dims", "x0,x1"])
    assert result.exit_code == 0, result.output
    payload = json.loads((tmp_path / "out" / "results.json").read_text())
    assert payload["params"]["covariates"] == ["x0", "x1"]
    assert payload["did"]["searched"]["dims"] == ["x0", "x1"]


def test_issue_35_covariates_with_plan_exits_2(tmp_path):
    """--covariates/--dims cannot combine with --plan (the plan's prep plan
    defines the scan space); rejected loudly BEFORE the plan file is read."""
    result = CliRunner().invoke(
        app, ["discover", "--plan", str(tmp_path / "never_written.json"),
              "--covariates", "x0"],
    )
    assert result.exit_code == 2
    assert "--plan" in result.output
    assert "prep plan" in result.output or "intake" in result.output


def test_issue_34_vacuous_placebo_records_null_not_true(tmp_path):
    """Issue #34: when the only covariate is the forcing column, the placebo
    battery is vacuous — results.json must record ``placebo_passed: null``
    plus an explicit ``placebo_note`` (never ``true`` with an empty Holm dict,
    field-level indistinguishable from a real pass), and the CLI echo must not
    claim the battery passed."""
    rng = np.random.default_rng(0)
    n = 300
    z = rng.normal(size=n)
    df = pd.DataFrame({
        "z": z,
        "T": (z >= 0).astype(float),
        "y": 1.0 + 0.5 * z + 2.0 * (z >= 0) + rng.normal(scale=0.5, size=n),
    })
    csv = tmp_path / "sharp.csv"
    df.to_csv(csv, index=False)
    result = CliRunner().invoke(
        app,
        ["discover", str(csv), "--treatment", "T", "--outcome", "y",
         "--forcing", "z", "--k", "25", "--q", "9", "--seed", "0",
         "--out", str(tmp_path / "out")],
    )
    assert result.exit_code == 0, result.output
    validation = json.loads((tmp_path / "out" / "results.json").read_text())["validation"]
    assert validation["placebo_passed"] is None
    assert validation["placebo_holm"] == {}
    assert "vacuous" in validation["placebo_note"]
    assert "placebo passed: True" not in result.output
    assert "no non-forcing covariate was testable" in result.output


def test_issue_28_discover_rdd_homogeneous_neighborhoods_exit_1(tmp_path):
    """Issue #28: two well-separated treatment-homogeneous clusters make the
    audit-item-21 fast path skip every center (discoveries=[]); the CLI must
    exit 1 with a diagnostic instead of an uncaught traceback out of
    randomization_test."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "z": np.concatenate(
                [rng.normal(0.0, 1.0, 50), rng.normal(100.0, 1.0, 50)]
            ),
            "t": np.repeat([0.0, 1.0], 50),
            "y": rng.normal(0.0, 1.0, 100),
        }
    )
    csv = tmp_path / "clusters.csv"
    df.to_csv(csv, index=False)
    result = CliRunner().invoke(
        app,
        ["discover", str(csv), "--treatment", "t", "--outcome", "y",
         "--forcing", "z", "--k", "20", "--q", "9", "--seed", "0",
         "--out", str(tmp_path / "out")],
    )
    assert result.exit_code == 1
    # clean typer.Exit, not an uncaught IndexError/ValueError traceback
    assert result.exception is None or isinstance(result.exception, SystemExit)
    assert "no scoreable neighborhood" in result.output


def test_issue_10_discover_did_binary_treatment_default_model(tmp_path):
    """Issue #10: `discover --design did` on a binary treatment with the
    default --method single_delta and --model auto must not crash — the CLI
    resolves auto -> normal exactly like the plan-mode runner (405a7ae);
    an explicit --model bernoulli still raises inside suddds_scan."""
    ds, _ = make_did_synthetic(n=400, d=2, V=3, zeta=8.0, theta_kind="binary",
                               rng=np.random.default_rng(1))
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
    assert payload["did"]["searched"]["model"] == "normal"
    assert payload["params"]["model"] == "normal"  # what actually ran


def test_issue_37_did_nan_randomization_p_prints_refusal(tmp_path):
    """Issue #37: on a few-profile panel the tau randomization test correctly
    refuses (p = NaN, < 5 usable placebos), but the CLI printed a bare
    'dd tau=... p=nan' with no hint. The echo must name the refusal and the
    manual placebo-in-space remedy, and results.json must record the reason.

    DGP: d=1, V=4 with a single planted profile leaves a pool of 3 placebo
    profiles < 5 minimum, so the refusal is structural (seed 0 pinned; the
    scan recovers the planted subset and effects are finite)."""
    ds, _ = make_did_synthetic(n=200, d=1, V=4, zeta=8.0, s_dims=1, s_values=1,
                               rng=np.random.default_rng(0))
    csv = tmp_path / "did.csv"
    ds.df.to_csv(csv, index=False)
    result = CliRunner().invoke(
        app,
        ["discover", str(csv), "--design", "did", "--treatment", "theta",
         "--outcome", "y", "--time", "t", "--q", "9", "--restarts", "2",
         "--windows", "4", "--seed", "0", "--out", str(tmp_path / "out")],
    )
    assert result.exit_code == 0, result.output
    assert "p=nan" not in result.output
    assert "randomization test refused" in result.output
    assert "usable placebos" in result.output
    assert "run a manual placebo-in-space battery" in result.output
    dd = json.loads((tmp_path / "out" / "results.json").read_text())["did"]["effects"]["dd"]
    assert dd["p"] is None  # NaN -> JSON null (house rule), never a fake number
    assert "only 3 usable placebos" in dd["p_refusal"]


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


def test_debias_smoke(tmp_path):
    """`natex debias` end to end on a tiny constant-surfaces DEE DGP.

    n=500 forces small pipeline knobs (the CLI defaults size k'=250 experiments,
    impossible at 500 rows): k'=60 / t_side=8 / m_prime=10 yield >= 3 usable
    experiments at seed 0. Asserts the plan's contract: exit 0, JSON payload
    with weights/experiments/grid/diagnostics, finite grid arrays (the _clean
    helper maps non-finite floats to None), and w_debias in [0, 1].
    """
    from natex.data.synthetic_dee import make_dee_synthetic

    ds, _ = make_dee_synthetic(
        n=500, constant_surfaces=(2.0, 3.0), type_probs=(0.1, 0.4, 0.4, 0.1),
        rng=np.random.default_rng(0),
    )
    csv = tmp_path / "dee.csv"
    ds.df.to_csv(csv, index=False)
    runner = CliRunner()
    result = runner.invoke(
        app,
        ["debias", str(csv), "--treatment", "D", "--outcome", "y",
         "--k", "25", "--m-prime", "10", "--k-prime", "60", "--t-side", "8",
         "--grid", "5", "--seed", "0", "--out", str(tmp_path / "out")],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads((tmp_path / "out" / "dee_result.json").read_text())
    assert {"params", "weights", "experiments", "grid", "diagnostics"} <= set(payload)
    w = payload["weights"]["w_debias"]
    assert w is not None and 0.0 <= w <= 1.0
    assert payload["weights"]["strategy"] == "stacking"
    assert payload["diagnostics"]["n_experiments_used"] >= 3
    exps = payload["experiments"]
    assert len(exps) >= 3
    for e in exps:
        assert {"center_z", "llr", "tau", "se", "first_stage_t",
                "weak_instrument", "n_members", "used"} <= set(e)
    grid = payload["grid"]
    assert len(grid["query"]) == 25  # 5x5 lattice over the 2 forcing dims
    assert all(v is not None for v in grid["cate_raw"])
    for key in ("cate_debiased", "cate_direct", "mixture"):
        for stat in ("mean", "sd"):
            vals = grid[key][stat]
            assert len(vals) == 25
            assert all(v is not None for v in vals), f"{key}.{stat} not finite"
    assert payload["params"]["m_prime_used"] == 10


def test_debias_requires_outcome(tmp_path):
    """debias without --outcome: nonzero exit, message names --outcome."""
    from natex.data.synthetic_dee import make_dee_synthetic

    ds, _ = make_dee_synthetic(
        n=100, constant_surfaces=(2.0, 3.0), rng=np.random.default_rng(0)
    )
    csv = tmp_path / "dee.csv"
    ds.df.to_csv(csv, index=False)
    runner = CliRunner()
    result = runner.invoke(app, ["debias", str(csv), "--treatment", "D"])
    assert result.exit_code != 0
    assert "--outcome" in result.output


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
