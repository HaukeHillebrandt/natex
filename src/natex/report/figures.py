"""Bundle figures under the ``plot`` extra (spec section 7).

Everything here RENDERS numbers already computed by discover/validate/estimate
— no new inference. matplotlib is imported lazily (:func:`_mpl`), so importing
this module works on a core-only install; calling any figure function without
the extra raises ImportError naming ``natex-discovery[plot]``.

Style (``_RC``, applied via ``plt.rc_context`` in every function): Okabe–Ito
colorblind-safe cycle, DejaVu Sans (tabular digits), top/right spines off,
tight bounding boxes. Each function writes ``<stem>.png`` (150 dpi) +
``<stem>.pdf`` into ``out_dir``, closes its figure, and returns
:class:`FigurePaths`. Missing numbers render as an em dash — never "nan".
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from natex.data.spec import Dataset
    from natex.did.panel import CategoricalPanel
    from natex.did.suddds import DiDDiscovery
    from natex.kink.estimate import KinkEstimate
    from natex.rdd.lord3 import LoRD3Result
    from natex.report.bundle import ResultsBundle

__all__ = [
    "FigurePaths",
    "OKABE_ITO",
    "bunching_hist",
    "density_hist",
    "did_figures",
    "discovery_scatter",
    "effect_forest",
    "kink_fit_plot",
    "pretrend_plot",
    "rdd_figures",
]

OKABE_ITO = (
    "#0072B2", "#E69F00", "#009E73", "#D55E00",
    "#CC79A7", "#56B4E9", "#F0E442", "#000000",
)

# rc_context accepts the matplotlibrc string form for the cycler, which keeps
# this dict importable without matplotlib.
_RC = {
    "axes.prop_cycle": f"cycler('color', {list(OKABE_ITO)})",
    "font.family": "DejaVu Sans",  # tabular digits
    "axes.spines.top": False,
    "axes.spines.right": False,
    "savefig.bbox": "tight",
}

_PLOT_EXTRA_MSG = 'figures require the plot extra: pip install "natex-discovery[plot]"'
_ACCENT = OKABE_ITO[3]  # vermillion: cutoffs, top discoveries, pooled row
_BASE = OKABE_ITO[0]  # blue: primary marks


def _mpl():
    """Lazy pyplot import; module import itself never touches matplotlib."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(_PLOT_EXTRA_MSG) from exc
    return plt


@dataclass(frozen=True)
class FigurePaths:
    png: Path
    pdf: Path


def _save(fig, out_dir: str | Path, stem: str) -> FigurePaths:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    png, pdf = out / f"{stem}.png", out / f"{stem}.pdf"
    fig.savefig(png, dpi=150)
    fig.savefig(pdf)
    return FigurePaths(png=png, pdf=pdf)


def _fmt(x, digits: int = 3) -> str:
    """Title-safe number: em dash for missing/non-finite — never 'nan'."""
    if x is None:
        return "—"
    x = float(x)
    return f"{x:.{digits}g}" if np.isfinite(x) else "—"


def _ptp(a: np.ndarray) -> float:
    return float(np.ptp(a)) if a.size else 1.0


# ---------------------------------------------------------------------------
# the four figures
# ---------------------------------------------------------------------------


def discovery_scatter(
    Z,
    llr,
    *,
    top_centers=None,
    top_normals=None,
    names=None,
    out_dir: str | Path,
    stem: str = "discovery_scatter",
) -> FigurePaths:
    """LLR landscape of scored centers in RAW forcing coordinates.

    ``Z`` is (n, d>=1), ``llr`` (n,). d == 1: llr vs the forcing value with
    top centers as vertical cutoff lines. d >= 2: first two dims colored by
    LLR; top discoveries starred, their normals (first two components) drawn
    as arrows scaled to ~10% of the axis span. Non-finite llr entries are
    dropped — never zeroed.
    """
    plt = _mpl()
    Z = np.asarray(Z, dtype=float)
    if Z.ndim == 1:
        Z = Z[:, None]
    llr = np.asarray(llr, dtype=float).ravel()
    if Z.shape[0] != llr.size:
        raise ValueError(f"Z has {Z.shape[0]} rows but llr has {llr.size} entries")
    keep = np.isfinite(llr)
    Zk, lk = Z[keep], llr[keep]
    tc = None if top_centers is None else np.atleast_2d(np.asarray(top_centers, dtype=float))
    tn = None if top_normals is None else np.atleast_2d(np.asarray(top_normals, dtype=float))
    with plt.rc_context(_RC):
        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        try:
            if Z.shape[1] == 1:
                ax.scatter(Zk[:, 0], lk, s=16, color=_BASE, alpha=0.8)
                if tc is not None and tc.size:
                    for j, c in enumerate(tc[:, 0]):
                        ax.axvline(c, color=_ACCENT, linestyle="--", linewidth=1.2,
                                   label="top center" if j == 0 else None)
                    ax.legend(frameon=False, fontsize=8)
                ax.set_xlabel(names[0] if names else "forcing")
                ax.set_ylabel("LLR")
            else:
                sc = ax.scatter(Zk[:, 0], Zk[:, 1], c=lk, cmap="viridis", s=18, alpha=0.9)
                if lk.size:
                    fig.colorbar(sc, ax=ax, label="LLR")
                if tc is not None and tc.size:
                    ax.scatter(tc[:, 0], tc[:, 1], marker="*", s=180, color=_ACCENT,
                               edgecolor="black", linewidths=0.6, zorder=3,
                               label="top discovery")
                    span = max(_ptp(Zk[:, 0]), _ptp(Zk[:, 1]), 1e-12)
                    for c, v in zip(tc, [] if tn is None else tn, strict=False):
                        v2 = np.asarray(v[:2], dtype=float)
                        norm = float(np.hypot(v2[0], v2[1]))
                        if not np.isfinite(norm) or norm <= 0:
                            continue
                        v2 = v2 / norm * 0.1 * span
                        ax.annotate(
                            "", xy=(c[0] + v2[0], c[1] + v2[1]), xytext=(c[0], c[1]),
                            arrowprops={"arrowstyle": "->", "color": _ACCENT, "lw": 1.4},
                        )
                    ax.legend(frameon=False, fontsize=8)
                ax.set_xlabel(names[0] if names else "z0")
                ax.set_ylabel(names[1] if names is not None and len(names) > 1 else "z1")
            ax.set_title("Discovery landscape (LLR over scored centers)", fontsize=10)
            return _save(fig, out_dir, stem)
        finally:
            plt.close(fig)


def density_hist(
    s,
    *,
    out_dir: str | Path,
    cutoff: float = 0.0,
    n_bins: int = 20,
    p_value=None,
    stem: str = "density_hist",
) -> FigurePaths:
    """Signed-distance histogram split at ``cutoff`` (McCrary-style).

    Audit item 6: the density test is valid ONLY for the FROZEN discovered
    geometry — it does not account for the search having selected normal and
    cutoff; that caveat is printed on the axes and in the title. Non-finite
    distances are dropped.
    """
    plt = _mpl()
    s = np.asarray(s, dtype=float).ravel()
    s = s[np.isfinite(s)]
    cutoff = float(cutoff)
    if s.size:
        edges = np.histogram_bin_edges(s, bins=n_bins)
    else:
        edges = np.linspace(-1.0, 1.0, n_bins + 1)
    with plt.rc_context(_RC):
        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        try:
            ax.hist(s[s < cutoff], bins=edges, color=_BASE, alpha=0.85, label="below cutoff")
            ax.hist(s[s >= cutoff], bins=edges, color=OKABE_ITO[1], alpha=0.85,
                    label="at/above cutoff")
            ax.axvline(cutoff, color="black", linestyle="--", linewidth=1.2)
            title = "Signed-distance density (McCrary-style, frozen geometry)"
            if p_value is not None:
                title += f" — p = {_fmt(p_value)}"
            ax.set_title(title, fontsize=10)
            ax.set_xlabel("signed distance to discovered boundary")
            ax.set_ylabel("count")
            ax.legend(frameon=False, fontsize=8)
            ax.text(
                0.02, 0.98,
                "Frozen discovered geometry only;\n"
                "search selection not accounted for (audit 6).",
                transform=ax.transAxes, va="top", fontsize=7, color="0.35",
            )
            return _save(fig, out_dir, stem)
        finally:
            plt.close(fig)


def bunching_hist(
    values,
    threshold: float,
    *,
    out_dir: str | Path,
    n_bins: int = 20,
    p_value=None,
    name: str | None = None,
    stem: str = "bunching_hist",
) -> FigurePaths:
    """Raw-value histogram split at a DECLARED threshold.

    Mirrors :func:`density_hist` styling but carries NO audit-item-6 search
    caveat: the threshold was declared by the user, not searched, and the
    annotation says exactly that. ``p_value`` (typically
    :func:`natex.validate.density.binned_poisson_jump` on
    ``values - threshold``) goes in the title; non-finite values are dropped
    — never zeroed — and missing numbers render as an em dash, never "nan".
    """
    plt = _mpl()
    v = np.asarray(values, dtype=float).ravel()
    v = v[np.isfinite(v)]
    threshold = float(threshold)
    if v.size:
        edges = np.histogram_bin_edges(v, bins=n_bins)
    else:
        edges = np.linspace(threshold - 1.0, threshold + 1.0, n_bins + 1)
    with plt.rc_context(_RC):
        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        try:
            ax.hist(v[v < threshold], bins=edges, color=_BASE, alpha=0.85,
                    label="below threshold")
            ax.hist(v[v >= threshold], bins=edges, color=OKABE_ITO[1], alpha=0.85,
                    label="at/above threshold")
            ax.axvline(threshold, color="black", linestyle="--", linewidth=1.2)
            title = f"Bunching at declared threshold = {_fmt(threshold)}"
            if p_value is not None:
                title += f" — p = {_fmt(p_value)}"
            ax.set_title(title, fontsize=10)
            ax.set_xlabel(name if name else "value")
            ax.set_ylabel("count")
            ax.legend(frameon=False, fontsize=8)
            ax.text(
                0.02, 0.98,
                "Split at the declared threshold — not searched.",
                transform=ax.transAxes, va="top", fontsize=7, color="0.35",
            )
            return _save(fig, out_dir, stem)
        finally:
            plt.close(fig)


def pretrend_plot(
    times,
    gaps,
    t0,
    *,
    n=None,
    out_dir: str | Path,
    stem: str = "pretrend",
) -> FigurePaths:
    """Per-period treated-minus-control gaps around ``t0`` (descriptive).

    Zero reference line, vertical T0 line, pre/post shading; marker area
    scales with the per-period usable count ``n`` when given.
    """
    plt = _mpl()
    times = np.asarray(times, dtype=float).ravel()
    gaps = np.asarray(gaps, dtype=float).ravel()
    if times.size != gaps.size:
        raise ValueError(f"times has {times.size} entries but gaps has {gaps.size}")
    t0 = float(t0)
    if n is not None:
        n_arr = np.asarray(n, dtype=float).ravel()
        top = float(np.nanmax(n_arr)) if n_arr.size else 1.0
        sizes = 20.0 + 80.0 * n_arr / max(top, 1.0)
    else:
        sizes = np.full(times.size, 30.0)
    with plt.rc_context(_RC):
        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        try:
            lo, hi = (float(times.min()), float(times.max())) if times.size else (t0 - 1, t0 + 1)
            pad = 0.02 * max(hi - lo, 1e-12)
            ax.axvspan(min(lo, t0) - pad, t0, color=_BASE, alpha=0.06)
            ax.axvspan(t0, max(hi, t0) + pad, color=_ACCENT, alpha=0.06)
            ax.axhline(0.0, color="0.4", linewidth=1.0)
            ax.axvline(t0, color="black", linestyle="--", linewidth=1.2)
            ax.plot(times, gaps, color=_BASE, linewidth=1.2, zorder=2)
            ax.scatter(times, gaps, s=sizes, color=_BASE, zorder=3)
            ax.set_xlabel("period")
            ax.set_ylabel("treated − control gap (raw y)")
            ax.set_title(f"Per-period gaps around T0 = {_fmt(t0)}", fontsize=10)
            return _save(fig, out_dir, stem)
        finally:
            plt.close(fig)


def effect_forest(
    labels,
    tau,
    lo,
    hi,
    *,
    pooled=None,
    out_dir: str | Path,
    stem: str = "effect_forest",
) -> FigurePaths:
    """Horizontal forest plot with a zero reference line.

    Rows with non-finite lo/hi draw the point WITHOUT whiskers — never a
    fabricated CI. ``pooled = (label, tau, lo, hi)`` is rendered last and
    visually distinct (diamond, accent color).
    """
    plt = _mpl()
    labels = [str(x) for x in labels]
    tau = np.asarray(tau, dtype=float).ravel()
    lo = np.asarray(lo, dtype=float).ravel()
    hi = np.asarray(hi, dtype=float).ravel()
    if not (len(labels) == tau.size == lo.size == hi.size):
        raise ValueError(
            f"labels/tau/lo/hi lengths differ: "
            f"{len(labels)}/{tau.size}/{lo.size}/{hi.size}"
        )
    rows = list(zip(labels, tau, lo, hi, strict=True))
    if pooled is not None:
        plabel, pt, plo, phi = pooled
        rows.append((str(plabel), float(pt), float(plo), float(phi)))
    k = len(rows)
    with plt.rc_context(_RC):
        fig, ax = plt.subplots(figsize=(6.4, 1.2 + 0.5 * max(k, 1)))
        try:
            ax.axvline(0.0, color="0.4", linewidth=1.0)
            for i, (_lab, t, lo_i, hi_i) in enumerate(rows):
                y = k - 1 - i  # first row on top
                is_pooled = pooled is not None and i == k - 1
                color = _ACCENT if is_pooled else _BASE
                if np.isfinite(lo_i) and np.isfinite(hi_i):
                    ax.plot([lo_i, hi_i], [y, y], color=color, linewidth=1.6, zorder=2)
                    for xw in (lo_i, hi_i):
                        ax.plot([xw, xw], [y - 0.12, y + 0.12], color=color, linewidth=1.6)
                if np.isfinite(t):
                    ax.scatter([t], [y], marker="D" if is_pooled else "o",
                               s=55.0 if is_pooled else 40.0, color=color, zorder=3)
            ax.set_yticks([k - 1 - i for i in range(k)], labels=[r[0] for r in rows])
            ax.set_ylim(-0.6, k - 0.4)
            ax.set_xlabel("effect (tau)")
            ax.set_title("Effect estimates", fontsize=10)
            return _save(fig, out_dir, stem)
        finally:
            plt.close(fig)


def kink_fit_plot(
    running,
    outcome,
    cutoff: float,
    bandwidth: float,
    out_stem: str | Path,
    *,
    kernel: str = "triangular",
    donut: float = 0.0,
    estimate: KinkEstimate | None = None,
    cutoff_label: str | None = None,
    counterfactual: bool = True,
) -> FigurePaths:
    """Scatter + two-sided kernel-weighted local-linear kink fit at ``cutoff``.

    Presentational only: the overlay re-runs the SAME weighted local-linear
    fit the kink estimator uses (``natex.kink.estimate`` internals, so kernel
    and design definitions cannot drift) and draws no confidence bands — the
    tau/se corner annotation comes from the ``estimate`` you pass in, never a
    new fit here (non-finite values render as an em dash). The scatter shows
    the padded window: estimation-sample points dark, bandwidth/donut-excluded
    context points light. ``counterfactual=True`` continues the below-cutoff
    fit dashed past the cutoff ("pre-trend continued"). When a side has too
    few points for a line the scatter still renders — no fabricated fit.
    Writes ``<out_stem>.png`` (150 dpi) + ``<out_stem>.pdf``.
    """
    plt = _mpl()
    from natex.kink.estimate import _fit, _kernel_weights, _prepare, _validate_common

    _validate_common(cutoff, bandwidth, 1, kernel, donut, 0.05)
    running_arr = np.asarray(running, dtype=float).ravel()
    outcome_arr = np.asarray(outcome, dtype=float).ravel()
    prepared = _prepare(
        outcome_arr,
        running_arr,
        treatment=None,
        post=None,
        cutoff=cutoff,
        bandwidth=bandwidth,
        degree=1,
        kernel=kernel,
        donut=donut,
        covariates=None,
        clusters=None,
    )
    beta = None
    if all(count >= 2 for count in prepared.n_by_cell.values()):
        try:
            # [a_left, b_left, a_right, b_right] on powers of u = distance/bandwidth
            beta = _fit(prepared, prepared.y).beta
        except (np.linalg.LinAlgError, ValueError):
            beta = None  # degenerate design: scatter-only figure
    distance = running_arr - cutoff
    finite = np.isfinite(distance) & np.isfinite(outcome_arr)
    pad = 1.15 * bandwidth
    inside = finite & (np.abs(distance) <= bandwidth) & (np.abs(distance) >= donut)
    weight = np.zeros_like(distance)
    weight[inside] = _kernel_weights(distance[inside] / bandwidth, kernel)
    context = finite & (np.abs(distance) <= pad) & ~(inside & (weight > 0.0))
    with plt.rc_context(_RC):
        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        try:
            ax.scatter(running_arr[context], outcome_arr[context], s=12,
                       color="0.8", zorder=1)
            ax.scatter(cutoff + prepared.x, prepared.y, s=14, color="0.45",
                       alpha=0.8, zorder=2)
            ax.axvline(cutoff, color="black", linestyle="--", linewidth=1.2,
                       label=cutoff_label)
            if beta is not None:
                left_grid = np.linspace(-bandwidth, 0.0, 64)
                right_grid = np.linspace(0.0, bandwidth, 64)
                a_left, b_left, a_right, b_right = (float(b) for b in beta[:4])
                ax.plot(cutoff + left_grid, a_left + b_left * left_grid / bandwidth,
                        color=_BASE, linewidth=1.8, zorder=3, label="fit below cutoff")
                ax.plot(cutoff + right_grid, a_right + b_right * right_grid / bandwidth,
                        color=OKABE_ITO[1], linewidth=1.8, zorder=3,
                        label="fit at/above cutoff")
                if counterfactual:
                    ax.plot(cutoff + right_grid, a_left + b_left * right_grid / bandwidth,
                            color=_BASE, linewidth=1.4, linestyle="--", zorder=3,
                            label="pre-trend continued")
            if estimate is not None:
                ax.text(0.98, 0.02,
                        f"tau = {_fmt(estimate.tau)}\nse = {_fmt(estimate.se)}",
                        transform=ax.transAxes, ha="right", va="bottom",
                        fontsize=8, color="0.25")
            if any(ax.get_legend_handles_labels()[1]):
                ax.legend(frameon=False, fontsize=8)
            ax.set_xlim(cutoff - pad, cutoff + pad)
            ax.set_xlabel("running")
            ax.set_ylabel("outcome")
            ax.set_title(f"Local-linear kink fit at cutoff = {_fmt(cutoff)}", fontsize=10)
            stem = Path(out_stem)
            return _save(fig, stem.parent, stem.name)
        finally:
            plt.close(fig)


# ---------------------------------------------------------------------------
# bundle glue (needs live objects; registers files in the bundle manifest)
# ---------------------------------------------------------------------------


def _num(x) -> float:
    """JSON-loaded number -> float; None/missing/non-numeric -> NaN, never 0.0."""
    if isinstance(x, (int, float)) and not isinstance(x, bool):
        return float(x)
    return float("nan")


def _best_summary(bundle: ResultsBundle) -> dict:
    """Best config's summary block from the bundle results ({} when absent)."""
    configs = bundle.results.get("configs") or []
    idx = bundle.results.get("best_index")
    if idx is None or not (0 <= int(idx) < len(configs)):
        return {}
    summary = configs[int(idx)].get("summary")
    return summary if isinstance(summary, dict) else {}


def rdd_figures(
    bundle: ResultsBundle,
    dataset: Dataset,
    result: LoRD3Result,
    *,
    top_m: int = 5,
) -> dict[str, FigurePaths]:
    """Standard rdd figures; registers each in the manifest and saves once.

    Needs the live dataset/scan (raw coordinates and memberships are not in
    results.json); the density p-value and the effect rows come from the
    bundle's best config so figures always match the reported numbers. The
    pooled forest row is :func:`natex.report.bundle.ivw_pooled`, labeled
    indicative — the 2SLS and Wald inputs share the same neighborhood.
    """
    from natex.report.bundle import ivw_pooled
    from natex.validate.placebo import signed_distance

    if not result.discoveries:
        raise ValueError("rdd_figures needs at least one discovery")
    out = bundle.figures_dir
    idx = np.array([d.center_index for d in result.discoveries], dtype=int)
    llr = np.array([d.llr for d in result.discoveries], dtype=float)
    top = result.top(top_m)
    summary = _best_summary(bundle)
    figs: dict[str, FigurePaths] = {}
    figs["discovery_scatter"] = discovery_scatter(
        dataset.Z[idx],
        llr,
        top_centers=dataset.Z[[d.center_index for d in top]],
        top_normals=np.array([d.normal for d in top], dtype=float),
        names=list(dataset.spec.forcing),
        out_dir=out,
    )
    figs["density_hist"] = density_hist(
        signed_distance(dataset, result.discoveries[0]),
        out_dir=out,
        p_value=summary.get("density_p"),
    )
    effects = summary.get("effects") or {}
    labels: list[str] = []
    tau: list[float] = []
    se: list[float] = []
    lo: list[float] = []
    hi: list[float] = []
    for name in ("2sls", "wald"):
        block = effects.get(name)
        if not isinstance(block, dict):
            continue
        ci = block.get("ci") or [None, None]
        labels.append(name)
        tau.append(_num(block.get("tau")))
        se.append(_num(block.get("se")))
        lo.append(_num(ci[0]))
        hi.append(_num(ci[1]))
    if labels:
        pooled = ivw_pooled(tau, se)
        figs["effect_forest"] = effect_forest(
            labels, tau, lo, hi,
            pooled=("IVW pooled (indicative)", pooled.tau, pooled.ci[0], pooled.ci[1]),
            out_dir=out,
        )
    for name, paths in figs.items():
        bundle.add_figure(name, paths.png, paths.pdf)
    bundle.save()
    return figs


def did_figures(
    bundle: ResultsBundle,
    panel: CategoricalPanel,
    discovery: DiDDiscovery,
    *,
    control: str = "dd",
) -> dict[str, FigurePaths]:
    """Standard did figures (pretrend + forest); same manifest contract.

    Forest rows come from the bundle's best config (ci = tau ± 1.96 se; NaN
    se -> point without whiskers). NO pooled row: dd/synthetic/gess share the
    treated cells, so pooling them would fabricate precision.
    """
    from natex.did.effects import period_gaps

    out = bundle.figures_dir
    g = period_gaps(panel, discovery, control)
    figs: dict[str, FigurePaths] = {
        "pretrend": pretrend_plot(g.times, g.gap, g.t0, n=g.n, out_dir=out)
    }
    effects = _best_summary(bundle).get("effects") or {}
    labels: list[str] = []
    tau: list[float] = []
    lo: list[float] = []
    hi: list[float] = []
    for name in ("dd", "synthetic", "gess"):
        block = effects.get(name)
        if not isinstance(block, dict):
            continue
        t, s = _num(block.get("tau")), _num(block.get("se"))
        labels.append(name)
        tau.append(t)
        lo.append(t - 1.96 * s)
        hi.append(t + 1.96 * s)
    if labels:
        figs["effect_forest"] = effect_forest(labels, tau, lo, hi, out_dir=out)
    for name, paths in figs.items():
        bundle.add_figure(name, paths.png, paths.pdf)
    bundle.save()
    return figs
