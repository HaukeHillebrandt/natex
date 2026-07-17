"""Per-family survey figures (phase survey task 7) — GLUE ONLY.

Rendering stays in :mod:`natex.report.figures`; this module maps each
executed family's live-object artifacts — small payloads of arrays/objects
the runner stashed while it had them in hand (never re-reading the CSV) —
onto those figure functions. Files land under ``out_dir/figures/`` with
stems prefixed ``<family>_``; the returned map holds ``out_dir``-relative
posix paths to the PNGs (a PDF twin sits beside each).

rdd is the one family whose artifacts carry no drawable discovery arrays:
its renderer performs ONE presentational re-scan of the best config's
dataset (``lord3_scan(ds, k, degree, rng=fam_rng)``); the randomization
test is NOT re-run, so the extra cost is ~1/(q+1) of the family's
statistical run. did likewise rebuilds the panel and re-runs
``suddds_scan`` at the same effective budget to recover the discovery for
the pretrend figure. Neither re-run feeds any statistic — summaries and
verdicts were frozen by the runner before figures start.

Figures are presentation: a rendering failure must never change a family's
statistical verdict, so :func:`render_family_figures` isolates the
per-family renderer behind the NAMED exception list ``_FIGURE_EXCEPTIONS``
(no blanket ``except Exception`` — that boundary is reserved for the
runner's per-family isolation, plan task 5) and reports the failure as
``figure rendering failed: ...``.

matplotlib importability is probed ONCE per survey (the runner calls
:func:`missing_matplotlib_reason` before any rendering); when the extra is
absent every executed family gets the exact install message instead.
"""

from __future__ import annotations

from importlib.util import find_spec
from pathlib import Path

import numpy as np

from natex.report import figures as report_figures

__all__ = ["NO_MPL_REASON", "missing_matplotlib_reason", "render_family_figures"]

NO_MPL_REASON = 'no figure: matplotlib not installed (pip install "natex-discovery[plot]")'

# Presentation-only isolation (module docstring): named, never BaseException,
# never a blanket Exception. LinAlgError/ImportError cover degenerate fits and
# a matplotlib that imports but fails; OSError covers unwritable out_dirs.
_FIGURE_EXCEPTIONS = (
    ValueError, TypeError, KeyError, IndexError, RuntimeError,
    OSError, ImportError, np.linalg.LinAlgError,
)

_MAX_PER_COLUMN = 3  # kink/bunching: at most 3 declared cutoffs/thresholds drawn
_MAX_DEE_ROWS = 15  # dee forest cap
_TOP_M = 5  # rdd scatter: top discoveries starred


def missing_matplotlib_reason() -> str | None:
    """:data:`NO_MPL_REASON` when matplotlib is not importable, else None.

    Probed via ``importlib.util.find_spec`` — the runner calls this ONCE per
    survey and, when the extra is absent, applies the reason to every
    executed family instead of attempting any rendering.
    """
    return None if find_spec("matplotlib") is not None else NO_MPL_REASON


def render_family_figures(
    name: str, artifacts: dict, out_dir: str | Path
) -> tuple[dict[str, str], str | None]:
    """Render one executed family's figures from its live-object artifacts.

    Returns ``(figures, no_figure_reason)``: ``figures`` maps figure name to
    an ``out_dir``-relative posix PNG path under ``figures/``;
    ``no_figure_reason`` is None exactly when rendering completed with at
    least one figure. On an isolated failure, figures saved before the
    failing call stay in the map and the reason records the error.
    """
    if name not in _RENDERERS:
        raise ValueError(f"unknown family {name!r}; known: {sorted(_RENDERERS)}")
    fig_dir = Path(out_dir) / "figures"
    figs: dict[str, str] = {}
    try:
        _RENDERERS[name](artifacts, fig_dir, figs)
    except _FIGURE_EXCEPTIONS as exc:  # presentation-only isolation (module docstring)
        return figs, f"figure rendering failed: {exc}"
    if not figs:
        return figs, f"no figure: nothing to draw for the {name} family"
    return figs, None


def _record(figs: dict[str, str], key: str, paths) -> None:
    """Register a saved figure under its out_dir-relative posix PNG path."""
    figs[key] = f"figures/{paths.png.name}"


def _safe(name: object) -> str:
    """Filesystem-safe stem fragment for a column name."""
    return "".join(ch if (ch.isalnum() or ch in "_-.") else "_" for ch in str(name))


# ---------------------------------------------------------------------------
# per-family renderers (imports lazy: glue must import on a core-only install)
# ---------------------------------------------------------------------------


def _render_rdd(art: dict, fig_dir: Path, figs: dict[str, str]) -> None:
    """Presentational re-scan -> scatter + density; forest from the summary."""
    from natex.rdd.lord3 import lord3_scan
    from natex.validate.placebo import signed_distance

    ds = art["ds"]
    res = lord3_scan(ds, k=art["k"], degree=art["degree"], rng=art["rng"])
    if res.discoveries:
        idx = [d.center_index for d in res.discoveries]
        llr = np.array([d.llr for d in res.discoveries], dtype=float)
        top = res.top(_TOP_M)
        _record(figs, "discovery_scatter", report_figures.discovery_scatter(
            ds.Z[idx], llr,
            top_centers=ds.Z[[d.center_index for d in top]],
            top_normals=np.array([d.normal for d in top], dtype=float),
            names=list(ds.spec.forcing),
            out_dir=fig_dir, stem="rdd_discovery_scatter",
        ))
        _record(figs, "density_hist", report_figures.density_hist(
            signed_distance(ds, res.discoveries[0]),
            out_dir=fig_dir, p_value=art["summary"].get("density_p"),
            stem="rdd_density_hist",
        ))
    effects = art["summary"].get("effects") or {}
    labels: list[str] = []
    tau: list[float] = []
    se: list[float] = []
    lo: list[float] = []
    hi: list[float] = []
    for eff_name in ("2sls", "wald"):
        block = effects.get(eff_name)
        if not isinstance(block, dict):
            continue
        ci = block.get("ci") or [None, None]
        labels.append(eff_name)
        tau.append(report_figures._num(block.get("tau")))
        se.append(report_figures._num(block.get("se")))
        lo.append(report_figures._num(ci[0]))
        hi.append(report_figures._num(ci[1]))
    if labels:
        from natex.report.bundle import ivw_pooled  # lazy: avoids an import cycle

        pooled = ivw_pooled(tau, se)  # indicative — same convention as rdd_figures
        _record(figs, "effect_forest", report_figures.effect_forest(
            labels, tau, lo, hi,
            pooled=("IVW pooled (indicative)", pooled.tau, pooled.ci[0], pooled.ci[1]),
            out_dir=fig_dir, stem="rdd_effect_forest",
        ))


def _render_did(art: dict, fig_dir: Path, figs: dict[str, str]) -> None:
    """Panel rebuild + re-scan -> pretrend; forest from the summary (no pooled
    row: dd/synthetic/gess share the treated cells — did_figures rationale)."""
    from natex.did.effects import period_gaps
    from natex.did.panel import build_panel
    from natex.did.suddds import resolve_default_model, suddds_scan

    ds, b = art["ds"], art["budget"]
    windows = b["windows"]
    if windows is not None:
        windows = tuple(float(w) for w in windows)
    bins, degree = int(b["bins"]), int(b["degree"])
    model = resolve_default_model(b["model"], b["method"])
    panel = build_panel(ds, bins=bins)
    res = suddds_scan(ds, windows=windows, restarts=int(b["restarts"]), model=model,
                      method=b["method"], bins=bins, degree=degree, rng=art["rng"],
                      panel=panel)
    if res.discoveries and panel.y is not None:
        g = period_gaps(panel, res.discoveries[0], "dd")
        _record(figs, "pretrend", report_figures.pretrend_plot(
            g.times, g.gap, g.t0, n=g.n, out_dir=fig_dir, stem="did_pretrend",
        ))
    effects = art["summary"].get("effects") or {}
    labels: list[str] = []
    tau: list[float] = []
    lo: list[float] = []
    hi: list[float] = []
    for eff_name in ("dd", "synthetic", "gess"):
        block = effects.get(eff_name)
        if not isinstance(block, dict):
            continue
        t, s = report_figures._num(block.get("tau")), report_figures._num(block.get("se"))
        labels.append(eff_name)
        tau.append(t)
        lo.append(t - 1.96 * s)
        hi.append(t + 1.96 * s)
    if labels:
        _record(figs, "effect_forest", report_figures.effect_forest(
            labels, tau, lo, hi, out_dir=fig_dir, stem="did_effect_forest",
        ))


def _render_kink(art: dict, fig_dir: Path, figs: dict[str, str]) -> None:
    """kink_fit_plot per usable declared cutoff (max 3), estimate annotated."""
    for item in art.get("cutoffs", [])[:_MAX_PER_COLUMN]:
        col = item["column"]
        _record(figs, f"fit_{col}", report_figures.kink_fit_plot(
            item["running"], item["outcome_values"], item["cutoff"],
            item["bandwidth"], fig_dir / f"kink_fit_{_safe(col)}",
            estimate=item["estimate"],
            cutoff_label=f"declared cutoff on {col}",
        ))


def _render_iv(art: dict, fig_dir: Path, figs: dict[str, str]) -> None:
    """Forest: the 2SLS row plus, when finite, the AR interval (whiskers only
    — AR inverts a test into an interval and has no point estimate)."""
    est = art["estimate"]
    labels = ["2sls"]
    tau = [report_figures._num(est.tau)]
    lo = [report_figures._num(est.ci[0])]
    hi = [report_figures._num(est.ci[1])]
    ar = est.ar_ci
    if ar is not None and np.isfinite(ar[0]) and np.isfinite(ar[1]):
        labels.append("AR (weak-IV robust)")
        tau.append(float("nan"))
        lo.append(float(ar[0]))
        hi.append(float(ar[1]))
    _record(figs, "effect_forest", report_figures.effect_forest(
        labels, tau, lo, hi, out_dir=fig_dir, stem="iv_effect_forest",
    ))


def _render_sc(art: dict, fig_dir: Path, figs: dict[str, str]) -> None:
    """Treated-minus-synthetic gap per period around t0."""
    _record(figs, "gap_plot", report_figures.pretrend_plot(
        art["times"], art["gaps"], art["t0"], out_dir=fig_dir, stem="sc_gap",
        ylabel="treated − synthetic gap",
    ))


def _render_bunching(art: dict, fig_dir: Path, figs: dict[str, str]) -> None:
    """bunching_hist per usable declared threshold (max 3), p annotated."""
    for item in art.get("thresholds", [])[:_MAX_PER_COLUMN]:
        col = item["column"]
        _record(figs, f"hist_{col}", report_figures.bunching_hist(
            item["values"], item["threshold"], out_dir=fig_dir,
            p_value=item["p_value"], name=str(col),
            stem=f"bunching_hist_{_safe(col)}",
        ))


def _render_dee(art: dict, fig_dir: Path, figs: dict[str, str]) -> None:
    """Forest of per-experiment local-2SLS taus (ci = tau ± 1.96 se; NaN rows
    point-only), capped at 15 rows in experiment acceptance order."""
    tau = np.asarray(art["tau"], dtype=float)[:_MAX_DEE_ROWS]
    se = np.asarray(art["se"], dtype=float)[:_MAX_DEE_ROWS]
    if tau.size == 0:
        return
    labels = [f"exp {i}" for i in range(tau.size)]
    _record(figs, "effect_forest", report_figures.effect_forest(
        labels, tau, tau - 1.96 * se, tau + 1.96 * se,
        out_dir=fig_dir, stem="dee_effect_forest",
    ))


_RENDERERS = {
    "rdd": _render_rdd,
    "did": _render_did,
    "kink": _render_kink,
    "iv": _render_iv,
    "sc": _render_sc,
    "bunching": _render_bunching,
    "dee": _render_dee,
}
