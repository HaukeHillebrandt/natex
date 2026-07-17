"""CLI ``natex survey`` + seeded determinism (phase survey task 9).

Contract under test: one command runs the full seven-family survey and prints
a fixed-width verdict table (FAMILY 9 / STATUS 12 / REASON truncated to 70)
followed by the report path and ``survey: <out>/survey.json``; exit 0 whenever
survey.json was written (failed families are a recorded outcome, not a CLI
failure). ``--cutoff/--threshold COL=VALUE`` and ``--instrument COL`` parse in
the CLI (malformed pair -> exit 2 naming the offending item) and reach the
applicability layer, asserted from survey.json. Same CSV/seed/flags => equal
survey.json payloads after dropping every ``created`` key and rewriting the
out-dir prefixes; a different seed changes at least one key_numbers value
(guards against accidentally seed-independent statistics). Unknown --backend
exits 2 via ``_make_backend``.

Stochastic assertions: determinism is bitwise under a fixed seed. The
different-seed check needs a statistic that actually moves with the seed: with
the plain synthetic CSV every executed number happens to be seed-pinned (the
scan llr is deterministic, rdd p sits at the replica ceiling, and the plugin
lasso refuses the weak x1 instrument, leaving only split-size constants), so
the determinism fixture adds a strong constructed instrument column iv1 —
then the iv family's F/tau/partial_r2 ride on the honest split drawn from the
family sub-generator (calibrated through the CLI across seeds 0-3: iv tau
0.221/0.222/0.272/0.240, first-stage F 579/696/607/610 — all distinct).
"""

import json

import numpy as np
from typer.testing import CliRunner

from natex.cli import app
from natex.data.synthetic import make_synthetic
from natex.survey.registry import FAMILY_ORDER

runner = CliRunner()

_FAST = ["--q", "9", "--k", "25"]  # small explicit test budget (plan task 5)


def _write_synthetic_csv(root):
    """make_synthetic(n=300) binary-treatment CSV with a decoy binary column
    'holiday' inserted BEFORE T (recipe from tests/test_cli_study.py)."""
    ds, _ = make_synthetic(
        n=300, px=3, pz=2, zeta=6.0, kind="binary", rng=np.random.default_rng(0)
    )
    df = ds.df.copy()
    df.insert(df.columns.get_loc("T"), "holiday",
              np.random.default_rng(1).integers(0, 2, len(df)))
    path = root / "synthetic.csv"
    df.to_csv(path, index=False)
    return path


def test_cli_survey_smoke(tmp_path):
    csv = _write_synthetic_csv(tmp_path)
    out = tmp_path / "out"
    result = runner.invoke(
        app, ["survey", str(csv), "--seed", "0", *_FAST, "--out", str(out)]
    )
    assert result.exit_code == 0, result.output

    # 7 fixed-width family lines: FAMILY padded to 9, STATUS padded to 12.
    lines = result.output.splitlines()
    for name in FAMILY_ORDER:
        fam_lines = [ln for ln in lines if ln.startswith(f"{name:<9}")]
        assert len(fam_lines) == 1, f"expected one {name} verdict line:\n{result.output}"
        assert len(fam_lines[0]) <= 9 + 12 + 70  # reason truncated terminal-safe
    assert "report: " in result.output
    assert f"survey: {out / 'survey.json'}" in result.output

    assert (out / "survey.json").exists()
    assert (out / "report.md").exists()
    assert (out / "families").is_dir()
    assert (out / "intake").is_dir()


def test_cli_survey_declared_flags(tmp_path):
    csv = _write_synthetic_csv(tmp_path)
    out = tmp_path / "out"
    result = runner.invoke(
        app,
        ["survey", str(csv), "--seed", "0", *_FAST, "--out", str(out),
         "--cutoff", "x0=0.5", "--threshold", "x0=0.5", "--instrument", "x1"],
    )
    assert result.exit_code == 0, result.output
    saved = json.loads((out / "survey.json").read_text())
    fams = saved["families"]

    # Declared inputs flipped the heuristics (needs_input -> applicable) ...
    for name in ("kink", "iv", "bunching"):
        assert fams[name]["applicability"]["heuristic"]["status"] == "applicable"
    # ... and the values themselves reached the family runners.
    assert fams["kink"]["diagnostics"]["cutoffs"] == {"x0": 0.5}
    assert fams["bunching"]["diagnostics"]["per_threshold"]["x0"]["threshold"] == 0.5
    assert fams["iv"]["diagnostics"]["pool"] == ["x1"]


def test_cli_survey_bad_cutoff(tmp_path):
    csv = _write_synthetic_csv(tmp_path)
    out = tmp_path / "out"
    result = runner.invoke(
        app, ["survey", str(csv), "--cutoff", "z:2", "--out", str(out)]
    )
    assert result.exit_code == 2
    assert "z:2" in result.output
    assert not (out / "survey.json").exists()


def _normalized_payload(path, out_dir):
    """survey.json payload with every ``created`` key dropped and the run's
    out-dir prefix rewritten to ``<out>`` in every string (plan task 9)."""

    def walk(obj):
        if isinstance(obj, dict):
            return {k: walk(v) for k, v in obj.items() if k != "created"}
        if isinstance(obj, list):
            return [walk(v) for v in obj]
        if isinstance(obj, str):
            return obj.replace(str(out_dir), "<out>")
        return obj

    return walk(json.loads(path.read_text()))


def _key_numbers(payload):
    return {n: f["key_numbers"] for n, f in payload["families"].items()}


def test_cli_survey_deterministic(tmp_path):
    # Strong constructed instrument for T: selected at every seed, and its
    # estimates ride on the rng-drawn honest split (module docstring
    # calibration). NO decoy column here — the intake ranks the decoy as the
    # treatment, which would starve iv1 of its first stage.
    ds, _ = make_synthetic(
        n=300, px=3, pz=2, zeta=6.0, kind="binary", rng=np.random.default_rng(0)
    )
    df = ds.df.copy()
    df["iv1"] = 2.0 * df["T"] + np.random.default_rng(2).normal(0, 0.5, len(df))
    csv = tmp_path / "synthetic_iv.csv"
    df.to_csv(csv, index=False)
    flags = [*_FAST, "--instrument", "iv1"]
    outs = {}
    for label, seed in (("a", 0), ("b", 0), ("c", 1)):
        out = tmp_path / label
        res = runner.invoke(
            app, ["survey", str(csv), "--seed", str(seed), *flags, "--out", str(out)]
        )
        assert res.exit_code == 0, res.output
        outs[label] = _normalized_payload(out / "survey.json", out)

    # Same CSV/seed/flags -> identical payloads across different out dirs.
    assert outs["a"] == outs["b"]
    # A different seed moves at least one key number.
    assert _key_numbers(outs["a"]) != _key_numbers(outs["c"])
    assert outs["c"]["seed"] == 1


def test_cli_survey_unknown_backend(tmp_path):
    csv = _write_synthetic_csv(tmp_path)
    result = runner.invoke(
        app, ["survey", str(csv), "--backend", "bogus", "--out", str(tmp_path / "out")]
    )
    assert result.exit_code == 2
    assert "bogus" in result.output
    assert not (tmp_path / "out" / "survey.json").exists()
