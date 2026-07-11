"""CLI tests for `natex instruments` and `natex donors` (phase-5 task 9).

Calibration (napkin + local runs): the instruments happy path uses the IV DGP
at (n=600, p=20, s=3, mu2=150, DGP seed 2) with CLI seed 0 — the honest split
selects ['z2'] (F ~ 45.3, weak False), estimation-half tau ~ 1.031 with
ar_kind "interval" and j_p None (just-identified). The mu2=0 pure-noise pool
(n=500, p=50, s=5, seed 0) yields an empty selection and an all-NaN estimate
— the NaN-clean JSON scenario. The donors path uses make_sc_synthetic
defaults (n_units=20, effect=10, noise=0.5, seed 0): n_donors=8 gives
att_post ~ 10.92 (task-7 tolerance |att - 10| <= 1.5), 19 ranked scores, and
the placebo test keeps all 19 placebos with treated ratio ~ 25.4 >> max
placebo ~ 3.13, so p == 1/20 exactly.
"""

import json

import numpy as np
from typer.testing import CliRunner

from natex.cli import app
from natex.data.synthetic_iv import make_iv_synthetic
from natex.data.synthetic_sc import make_sc_synthetic

runner = CliRunner()


def _strict_loads(text):
    """json.loads that rejects NaN/Infinity constants (_clean must map them to null)."""

    def _bad(s):
        raise AssertionError(f"non-finite constant leaked into JSON: {s}")

    return json.loads(text, parse_constant=_bad)


def _iv_csv(tmp_path, **kwargs):
    kwargs.setdefault("n", 600)
    kwargs.setdefault("p", 20)
    kwargs.setdefault("s", 3)
    kwargs.setdefault("mu2", 150.0)
    seed = kwargs.pop("seed", 2)
    data = make_iv_synthetic(rng=np.random.default_rng(seed), **kwargs)
    csv = tmp_path / "iv.csv"
    data.df.to_csv(csv, index=False)
    return csv, data


def _sc_csv(tmp_path, seed=0):
    d = make_sc_synthetic(rng=np.random.default_rng(seed))
    csv = tmp_path / "sc.csv"
    d.df.to_csv(csv, index=False)
    return csv, d


# ---------------------------------------------------------------------------
# natex instruments
# ---------------------------------------------------------------------------


def test_instruments_end_to_end(tmp_path):
    """Happy path: default pool (all numeric minus T/y), honest estimation block."""
    csv, data = _iv_csv(tmp_path)
    result = runner.invoke(
        app,
        ["instruments", str(csv), "--treatment", "T", "--outcome", "y",
         "--seed", "0", "--out", str(tmp_path / "out")],
    )
    assert result.exit_code == 0, result.output
    payload = _strict_loads((tmp_path / "out" / "instruments.json").read_text())
    sel = payload["selection"]
    assert sel["selected"], "selection must be non-empty on the strong DGP"
    assert set(sel["selected"]) <= set(data.pool_names)
    assert sel["weak"] is False
    assert sel["first_stage_F"] > 10.0
    assert sel["lam_source"] == "plugin"
    # default pool: every numeric column except treatment and outcome
    assert payload["params"]["pool"] == data.pool_names
    # honest split sizes
    assert payload["split"]["honest"] is True
    assert payload["split"]["n_discovery"] == 300
    assert payload["split"]["n_estimation"] == 300
    est = payload["estimate"]
    assert est["tau"] is not None and np.isfinite(est["tau"])
    assert abs(est["tau"] - data.tau) < 0.5
    assert est["ar_kind"] == "interval"
    assert est["ar_ci"][0] < est["tau"] < est["ar_ci"][1]
    assert est["j_p"] is None  # just-identified: never a fabricated value
    assert payload["caveat"] is None


def test_instruments_no_honest_sets_caveat(tmp_path):
    """--no-honest: full-sample selection + estimation, caveat string in payload."""
    csv, _ = _iv_csv(tmp_path)
    result = runner.invoke(
        app,
        ["instruments", str(csv), "--treatment", "T", "--outcome", "y",
         "--no-honest", "--out", str(tmp_path / "out")],
    )
    assert result.exit_code == 0, result.output
    payload = _strict_loads((tmp_path / "out" / "instruments.json").read_text())
    assert payload["split"]["honest"] is False
    assert payload["split"]["n_discovery"] == 600
    assert payload["split"]["n_estimation"] == 600
    assert isinstance(payload["caveat"], str)
    assert "post-selection" in payload["caveat"]
    assert "post-selection" in result.output  # echoed, not just buried in JSON


def test_instruments_selection_only_without_outcome(tmp_path):
    """No --outcome: selection runs (discovery never reads y), estimate is null."""
    csv, data = _iv_csv(tmp_path)
    result = runner.invoke(
        app,
        ["instruments", str(csv), "--treatment", "T",
         "--pool", ",".join(data.pool_names),
         "--seed", "0", "--out", str(tmp_path / "out")],
    )
    assert result.exit_code == 0, result.output
    payload = _strict_loads((tmp_path / "out" / "instruments.json").read_text())
    assert payload["estimate"] is None
    assert payload["selection"]["selected"]


def test_instruments_nan_clean_json_on_pure_noise_pool(tmp_path):
    """mu2=0 pool: empty selection, NaN diagnostics/estimate serialize as null."""
    csv, _ = _iv_csv(tmp_path, n=500, p=50, s=5, mu2=0.0, seed=0)
    result = runner.invoke(
        app,
        ["instruments", str(csv), "--treatment", "T", "--outcome", "y",
         "--seed", "0", "--out", str(tmp_path / "out")],
    )
    assert result.exit_code == 0, result.output
    text = (tmp_path / "out" / "instruments.json").read_text()
    payload = _strict_loads(text)  # raises if NaN/Infinity leaked
    sel = payload["selection"]
    assert sel["selected"] == []
    assert sel["first_stage_F"] is None  # NaN -> null, never 0.0
    assert sel["partial_r2"] is None
    assert sel["weak"] is True
    est = payload["estimate"]
    assert est["tau"] is None
    assert est["se"] is None
    assert est["ar_kind"] is None


def test_instruments_bad_lam_exits_2(tmp_path):
    csv, _ = _iv_csv(tmp_path)
    result = runner.invoke(
        app,
        ["instruments", str(csv), "--treatment", "T", "--lam", "banana",
         "--out", str(tmp_path / "out")],
    )
    assert result.exit_code == 2
    assert "banana" in result.output
    assert "Traceback" not in result.output


def test_instruments_unknown_column_exits_2(tmp_path):
    csv, _ = _iv_csv(tmp_path)
    result = runner.invoke(
        app,
        ["instruments", str(csv), "--treatment", "ghost",
         "--out", str(tmp_path / "out")],
    )
    assert result.exit_code == 2
    assert "ghost" in result.output
    assert "Traceback" not in result.output


# ---------------------------------------------------------------------------
# natex donors
# ---------------------------------------------------------------------------


def _donors_args(csv, d, out, extra=()):
    return [
        "donors", str(csv), "--outcome", "y", "--unit", "unit", "--time", "time",
        "--treated-unit", d.treated_unit, "--t0", str(d.t0), "--out", str(out),
        *extra,
    ]


def test_donors_end_to_end(tmp_path):
    """Ranked scores, weights, ATT within task-7 tolerance, +1-rank placebo block."""
    csv, d = _sc_csv(tmp_path)
    result = runner.invoke(app, _donors_args(csv, d, tmp_path / "out",
                                             extra=("--n-donors", "8")))
    assert result.exit_code == 0, result.output
    payload = _strict_loads((tmp_path / "out" / "donors.json").read_text())
    assert len(payload["donors"]) == 8
    assert len(payload["scores"]) == 19  # ALL complete candidates, ranked
    assert [s["rank"] for s in payload["scores"]] == list(range(1, 20))
    weights = payload["weights"]
    assert [w["unit"] for w in weights] == payload["donors"]
    assert abs(sum(w["weight"] for w in weights) - 1.0) < 1e-6
    assert abs(payload["att_post"] - d.effect) <= 1.5  # task-7 tolerance
    assert payload["pre_rmspe"] < payload["post_rmspe"]
    assert len(payload["effect_by_time"]) == len(payload["times"]) == 25
    # placebo block: +1-rank arithmetic must be internally consistent
    plc = payload["placebo"]
    ratios = [r["ratio"] for r in plc["ratios"]]
    assert ratios == sorted(ratios, reverse=True)
    assert plc["n_used"] == len(ratios) == 19
    n_ge = sum(r >= plc["ratio_treated"] for r in ratios)
    assert plc["p_value"] == (1 + n_ge) / (plc["n_used"] + 1)
    assert plc["p_value"] == 1.0 / 20.0  # treated most extreme at seed 0


def test_donors_no_placebo_omits_block(tmp_path):
    csv, d = _sc_csv(tmp_path)
    result = runner.invoke(app, _donors_args(csv, d, tmp_path / "out",
                                             extra=("--n-donors", "8", "--no-placebo")))
    assert result.exit_code == 0, result.output
    payload = _strict_loads((tmp_path / "out" / "donors.json").read_text())
    assert "placebo" not in payload
    assert abs(payload["att_post"] - d.effect) <= 1.5


def test_donors_unknown_treated_unit_exits_2(tmp_path):
    csv, d = _sc_csv(tmp_path)
    args = _donors_args(csv, d, tmp_path / "out")
    args[args.index("--treated-unit") + 1] = "Atlantis"
    result = runner.invoke(app, args)
    assert result.exit_code == 2
    assert "Atlantis" in result.output
    assert "Traceback" not in result.output


def test_donors_missing_outcome_exits_2(tmp_path):
    """--outcome is required: typer reports the missing option with exit code 2."""
    csv, d = _sc_csv(tmp_path)
    result = runner.invoke(
        app,
        ["donors", str(csv), "--unit", "unit", "--time", "time",
         "--treated-unit", d.treated_unit, "--t0", str(d.t0)],
    )
    assert result.exit_code == 2
    assert "--outcome" in result.output


def test_donors_unknown_column_exits_2(tmp_path):
    csv, d = _sc_csv(tmp_path)
    args = _donors_args(csv, d, tmp_path / "out")
    args[args.index("--unit") + 1] = "ghost"
    result = runner.invoke(app, args)
    assert result.exit_code == 2
    assert "ghost" in result.output
    assert "Traceback" not in result.output
