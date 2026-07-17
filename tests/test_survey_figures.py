"""Per-family survey figures (phase survey task 7).

Contract under test: executed families get PNG figures under
``out_dir/figures/`` with ``<family>_``-prefixed stems, recorded in
``FamilyResult.figures`` as out_dir-relative posix paths; non-run families
carry a reason starting "no figure:"; a missing matplotlib (probed ONCE per
survey via ``find_spec``) yields the exact extra-install reason for every
executed family while the survey still completes; a raising figure function
is an isolated presentation failure — the family keeps its statistical
verdict and ``no_figure_reason`` records "figure rendering failed: ...".

Stochastic notes: reuses the task-5/6 calibrated DGPs unchanged (synthetic
rdd at q=9/k=25 — verdict pinned only to {credible, null}; kink DGP seed 0
credible with tau ~ 2.19; bunching on the same uniform draw lands
credible-or-null) — no new stochastic gates.
"""

import importlib.util

import numpy as np
import pandas as pd
import pytest

from natex.data.synthetic import make_synthetic
from natex.survey import figures as survey_figures
from natex.survey import survey

_BUDGET = {"q": 9, "k": 25}  # small explicit test budget (plan task 5 convention)


def _synthetic_csv(root):
    """make_synthetic(n=300) binary-treatment CSV (tests/test_survey_runner.py recipe)."""
    ds, _ = make_synthetic(
        n=300, px=3, pz=2, zeta=6.0, kind="binary", rng=np.random.default_rng(0)
    )
    df = ds.df.copy()
    df.insert(df.columns.get_loc("T"), "holiday",
              np.random.default_rng(1).integers(0, 2, len(df)))
    path = root / "synthetic.csv"
    df.to_csv(path, index=False)
    return path


def _kink_df(seed=0, n=800):
    """Piecewise-linear DGP: slope 0.5 below z=2, 2.5 above (task-6 recipe)."""
    rng = np.random.default_rng(seed)
    z = rng.uniform(0.0, 4.0, n)
    y = 1.0 + 0.5 * np.minimum(z, 2.0) + 2.5 * np.maximum(z - 2.0, 0.0)
    return pd.DataFrame({"z": z, "y": y + rng.normal(0.0, 0.3, n)})


def test_executed_families_have_figures(tmp_path):
    pytest.importorskip("matplotlib")
    csv = _synthetic_csv(tmp_path)
    out = tmp_path / "out"
    res = survey(str(csv), rng=np.random.default_rng(0), out_dir=out, budget=_BUDGET)

    rdd = res.families["rdd"]
    assert rdd.status in {"credible", "null"}, rdd.reason
    assert rdd.figures, rdd.no_figure_reason  # executed => nonempty figures map
    for rel in rdd.figures.values():
        assert rel.startswith("figures/rdd_")  # <family>_ stem prefix
        assert rel.endswith(".png")
        assert (out / rel).exists()
    assert rdd.no_figure_reason is None

    # Every non-run family carries an explicit "no figure:" reason.
    for name, fam in res.families.items():
        if fam.status in {"skipped", "needs_input"}:
            assert fam.figures == {}
            assert fam.no_figure_reason.startswith("no figure:"), name
    # kink was never run (no declared cutoff): reason names the cause
    assert res.families["kink"].no_figure_reason.startswith(
        "no figure: family did not run ("
    )


def test_no_matplotlib_reasons(monkeypatch, tmp_path):
    """find_spec("matplotlib") -> None: the executed family gets the exact
    extra-install reason (probe is once-per-survey) and the survey completes."""
    real = importlib.util.find_spec
    monkeypatch.setattr(
        survey_figures, "find_spec",
        lambda name, *a: None if name == "matplotlib" else real(name, *a),
    )
    out = tmp_path / "out"
    res = survey(_kink_df(), rng=np.random.default_rng(0), out_dir=out,
                 cutoffs={"z": 2.0})
    kink = res.families["kink"]
    assert kink.status in {"credible", "null"}, kink.reason  # verdict untouched
    assert kink.figures == {}
    assert kink.no_figure_reason == survey_figures.NO_MPL_REASON
    assert kink.no_figure_reason == (
        'no figure: matplotlib not installed (pip install "natex-discovery[plot]")'
    )
    assert (out / "survey.json").exists()


def test_kink_and_bunching_figures(tmp_path):
    """A declared-cutoff/threshold run produces kink_fit_plot / bunching_hist files."""
    pytest.importorskip("matplotlib")
    out = tmp_path / "out"
    res = survey(_kink_df(), rng=np.random.default_rng(0), out_dir=out,
                 cutoffs={"z": 2.0}, thresholds={"z": 2.0})

    kink = res.families["kink"]
    assert kink.status in {"credible", "null"}, kink.reason
    fit = [p for p in kink.figures.values() if "kink_fit_" in p]
    assert fit, kink.no_figure_reason
    assert (out / fit[0]).exists()

    b = res.families["bunching"]
    assert b.status in {"credible", "null"}, b.reason
    hist = [p for p in b.figures.values() if "bunching_hist_" in p]
    assert hist, b.no_figure_reason
    assert (out / hist[0]).exists()


def test_figure_failure_is_isolated(monkeypatch, tmp_path):
    """A figure function raising ValueError never changes the family verdict:
    figures are presentation only."""
    pytest.importorskip("matplotlib")
    from natex.report import figures as report_figures

    def boom(*args, **kwargs):
        raise ValueError("boom-fig")

    monkeypatch.setattr(report_figures, "kink_fit_plot", boom)
    res = survey(_kink_df(), rng=np.random.default_rng(0),
                 out_dir=tmp_path / "out", cutoffs={"z": 2.0})
    kink = res.families["kink"]
    assert kink.status == "credible", kink.reason  # task-6 pinned seed-0 verdict
    assert kink.figures == {}
    assert kink.no_figure_reason.startswith("figure rendering failed:")
    assert "boom-fig" in kink.no_figure_reason
