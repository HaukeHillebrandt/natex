"""Figures under the plot extra (phase report-paper task 3).

``importorskip`` keeps a core-only install green; the Agg backend is forced
BEFORE any pyplot import so the suite never needs a display. Every figure
function must close its figure (no handle leak) and write PNG (150 dpi) + PDF.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from natex.data.synthetic_kink import make_rkd_synthetic  # noqa: E402
from natex.did.panel import build_panel  # noqa: E402
from natex.did.suddds import suddds_scan  # noqa: E402
from natex.kink import KinkEstimate, regression_kink  # noqa: E402
from natex.rdd.lord3 import lord3_scan  # noqa: E402
from natex.report.bundle import ResultsBundle  # noqa: E402
from natex.report.figures import (  # noqa: E402
    FigurePaths,
    bunching_hist,
    density_hist,
    did_figures,
    discovery_scatter,
    effect_forest,
    kink_fit_plot,
    pretrend_plot,
    rdd_figures,
)
from report_helpers import make_did_bundle, make_rdd_bundle  # noqa: E402


def _assert_saved(paths: FigurePaths) -> None:
    """Contract shared by all four functions: files exist, non-trivial, no leak."""
    assert isinstance(paths, FigurePaths)
    assert paths.png.exists() and paths.pdf.exists()
    assert paths.png.stat().st_size > 2000
    assert paths.pdf.stat().st_size > 500
    assert len(plt.get_fignums()) == 0  # every function closes its figure


# ---------------------------------------------------------------------------
# 1-2. discovery_scatter: 1-D and 2-D, NaN llr dropped (never zeroed)
# ---------------------------------------------------------------------------


def test_discovery_scatter_1d_idempotent(tmp_path):
    rng = np.random.default_rng(0)
    Z = rng.normal(size=(60, 1))
    llr = rng.random(60) * 5.0
    kwargs = dict(
        top_centers=Z[:2], top_normals=np.ones((2, 1)), names=["x0"], out_dir=tmp_path
    )
    p = discovery_scatter(Z, llr, **kwargs)
    _assert_saved(p)
    p2 = discovery_scatter(Z, llr, **kwargs)  # overwrite in place, same paths
    assert p2 == p
    _assert_saved(p2)


def test_discovery_scatter_2d_with_nan_llr(tmp_path):
    rng = np.random.default_rng(0)
    Z = rng.normal(size=(80, 2))
    llr = rng.random(80) * 5.0
    llr[::7] = np.nan  # dropped, never zeroed; must not raise
    p = discovery_scatter(
        Z,
        llr,
        top_centers=Z[:3],
        top_normals=rng.normal(size=(3, 2)),
        names=["x0", "x1"],
        out_dir=tmp_path,
    )
    _assert_saved(p)


# ---------------------------------------------------------------------------
# density_hist: side-split histogram, NaN p-value never rendered as "nan"
# ---------------------------------------------------------------------------


def test_density_hist(tmp_path):
    rng = np.random.default_rng(0)
    s = np.concatenate([rng.normal(-1.0, 1.0, 150), rng.normal(1.2, 0.8, 150)])
    p = density_hist(s, out_dir=tmp_path, p_value=0.031)
    _assert_saved(p)
    _assert_saved(density_hist(s, out_dir=tmp_path, p_value=0.031))  # idempotent
    # NaN p-value and NaN distances must not raise (rendered as an em dash)
    s_nan = np.append(s, np.nan)
    _assert_saved(density_hist(s_nan, out_dir=tmp_path, p_value=float("nan")))


# ---------------------------------------------------------------------------
# bunching_hist: raw values at a DECLARED threshold (phase survey, task 4)
# ---------------------------------------------------------------------------


def test_bunching_hist(tmp_path, monkeypatch):
    """Writes png+pdf (png nonempty); the title names the declared threshold;
    with p_value=None no axis text ever reads 'nan' — missing numbers are
    em-dashed or omitted, never rendered raw."""
    import natex.report.figures as figmod

    captured: dict[str, list[str]] = {}
    orig_save = figmod._save

    def spy(fig, out_dir, stem):
        captured["texts"] = [
            t.get_text()
            for ax in fig.axes
            for t in (ax.title, ax.xaxis.label, ax.yaxis.label, *ax.texts)
        ]
        return orig_save(fig, out_dir, stem)

    monkeypatch.setattr(figmod, "_save", spy)
    rng = np.random.default_rng(0)
    v = np.concatenate([rng.uniform(10.0, 20.0, 300), np.full(60, 15.0)])
    p = bunching_hist(v, 15.0, out_dir=tmp_path, p_value=None, name="income")
    _assert_saved(p)
    assert p.png.name == "bunching_hist.png"
    texts = captured["texts"]
    assert any("15" in t for t in texts if t)  # threshold in the title
    assert not any("nan" in t.lower() for t in texts if t)
    # declared-threshold annotation, NOT the audit-6 search caveat
    assert any("declared threshold" in t for t in texts)
    assert not any("audit" in t.lower() for t in texts)
    # idempotent overwrite; NaN values dropped and NaN p em-dashed, never "nan"
    v_nan = np.append(v, np.nan)
    p2 = bunching_hist(v_nan, 15.0, out_dir=tmp_path, p_value=float("nan"))
    assert p2 == p
    _assert_saved(p2)
    assert not any("nan" in t.lower() for t in captured["texts"] if t)


# ---------------------------------------------------------------------------
# pretrend_plot: reference lines, optional marker sizing by n
# ---------------------------------------------------------------------------


def test_pretrend_plot(tmp_path):
    rng = np.random.default_rng(0)
    times = np.arange(12, dtype=float)
    gaps = rng.normal(0.0, 0.1, 12)
    gaps[8:] += 1.0
    n = rng.integers(5, 50, 12)
    p = pretrend_plot(times, gaps, 8.0, n=n, out_dir=tmp_path)
    _assert_saved(p)
    _assert_saved(pretrend_plot(times, gaps, 8.0, n=n, out_dir=tmp_path))  # idempotent
    _assert_saved(pretrend_plot(times, gaps, 8.0, out_dir=tmp_path))  # n optional


# ---------------------------------------------------------------------------
# 3. effect_forest: NaN-CI row drawn without whiskers; pooled row rendered
# ---------------------------------------------------------------------------


def test_effect_forest_nan_ci_and_pooled(tmp_path):
    args = (
        ["2sls", "wald", "no-ci"],
        [1.2, 1.0, 0.8],
        [0.8, 0.5, float("nan")],
        [1.6, 1.5, float("nan")],
    )
    pooled = ("IVW pooled (indicative)", 1.1, 0.9, 1.3)
    p = effect_forest(*args, pooled=pooled, out_dir=tmp_path)
    _assert_saved(p)
    _assert_saved(effect_forest(*args, pooled=pooled, out_dir=tmp_path))  # idempotent
    _assert_saved(effect_forest(*args, out_dir=tmp_path, stem="forest_nopool"))


# ---------------------------------------------------------------------------
# 4. rdd_figures glue: manifest entries + reload round-trip
# ---------------------------------------------------------------------------


def test_rdd_figures_end_to_end(tmp_path):
    bundle, _report, ds = make_rdd_bundle(tmp_path)
    res = lord3_scan(ds, k=25, rng=np.random.default_rng(0))
    figs = rdd_figures(bundle, ds, res)
    assert set(figs) == {"discovery_scatter", "density_hist", "effect_forest"}
    for paths in figs.values():
        _assert_saved(paths)
    manifest = bundle.results["figures"]
    assert len(manifest) == 3
    for entry in manifest:
        for key in ("png", "pdf"):
            rel = entry[key]
            assert not Path(rel).is_absolute()  # POSIX-relative to the bundle dir
            assert (bundle.out_dir / rel).is_file()
            assert (bundle.figures_dir / Path(rel).name).is_file()  # under figures/
    reloaded = ResultsBundle.load(tmp_path)
    assert reloaded.results["figures"] == manifest


# ---------------------------------------------------------------------------
# 5. did_figures glue: pretrend + forest, manifest updated
# ---------------------------------------------------------------------------


def test_did_figures_end_to_end(tmp_path):
    bundle, _report, ds = make_did_bundle(tmp_path)
    panel = build_panel(ds)
    res = suddds_scan(ds, panel=panel, restarts=2, rng=np.random.default_rng(0))
    figs = did_figures(bundle, panel, res.discoveries[0])
    assert set(figs) == {"pretrend", "effect_forest"}
    for paths in figs.values():
        _assert_saved(paths)
    names = {entry["name"] for entry in bundle.results["figures"]}
    assert names == {"pretrend", "effect_forest"}
    reloaded = ResultsBundle.load(tmp_path)
    assert reloaded.results["figures"] == bundle.results["figures"]


# ---------------------------------------------------------------------------
# 6. kink_fit_plot: scatter + two-sided fits, counterfactual, annotation
# ---------------------------------------------------------------------------


def _rkd_arrays(seed: int):
    data, truth = make_rkd_synthetic(n=400, rng=np.random.default_rng(seed))
    return data.df["running"].to_numpy(), data.df["y"].to_numpy(), truth


def test_kink_fit_plot_saves_and_is_idempotent(tmp_path):
    running, y, _truth = _rkd_arrays(0)
    p = kink_fit_plot(running, y, 0.0, 0.6, tmp_path / "kink_fit")
    _assert_saved(p)
    p2 = kink_fit_plot(running, y, 0.0, 0.6, tmp_path / "kink_fit")  # same paths
    assert p2 == p
    _assert_saved(p2)


def test_kink_fit_plot_estimate_annotation_and_options(tmp_path):
    running, y, truth = _rkd_arrays(1)
    est = regression_kink(
        y, running, policy_kink=truth.policy_kink, cutoff=truth.cutoff, bandwidth=0.6
    )
    assert np.isfinite(est.tau)  # the annotation shows real numbers, not dashes
    p = kink_fit_plot(
        running,
        y,
        truth.cutoff,
        0.6,
        tmp_path / "kink_annotated",
        kernel="uniform",
        donut=0.05,
        estimate=est,
        cutoff_label="reform",
        counterfactual=False,
    )
    _assert_saved(p)


def test_kink_fit_plot_nan_estimate_never_renders_nan(tmp_path):
    running, y, _truth = _rkd_arrays(2)
    nan = float("nan")
    est = KinkEstimate(
        tau=nan,
        se=nan,
        ci=(nan, nan),
        method="sharp_rkd",
        reduced_form=nan,
        reduced_form_se=nan,
        first_stage=nan,
        first_stage_se=nan,
        first_stage_F=nan,
        weak_first_stage=True,
        n_used=0,
        n_by_cell={"left": 0, "right": 0},
    )
    _assert_saved(
        kink_fit_plot(running, y, 0.0, 0.6, tmp_path / "kink_nan", estimate=est)
    )


def test_kink_fit_plot_one_sided_data_still_saves(tmp_path):
    running, y, _truth = _rkd_arrays(3)
    left = running < 0.0  # right cell empty: scatter renders, no fabricated fit
    _assert_saved(
        kink_fit_plot(running[left], y[left], 0.0, 0.6, tmp_path / "kink_left_only")
    )


def test_kink_fit_plot_validates_like_the_estimator(tmp_path):
    running, y, _truth = _rkd_arrays(4)
    with pytest.raises(ValueError, match="kernel"):
        kink_fit_plot(running, y, 0.0, 0.6, tmp_path / "bad", kernel="gaussian")
    with pytest.raises(ValueError, match="bandwidth"):
        kink_fit_plot(running, y, 0.0, -1.0, tmp_path / "bad")


# ---------------------------------------------------------------------------
# 7. import guard: module import is guard-free; calls fail with install hint
# ---------------------------------------------------------------------------


def test_import_guard_names_plot_extra(tmp_path, monkeypatch):
    assert "natex.report.figures" in sys.modules  # module import never needs mpl
    monkeypatch.setitem(sys.modules, "matplotlib", None)
    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", None)
    with pytest.raises(ImportError, match=r"natex-discovery\[plot\]"):
        discovery_scatter(np.zeros((3, 1)), np.zeros(3), out_dir=tmp_path)
