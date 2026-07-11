#!/usr/bin/env python
"""Full ch.6 synthetic discovery benchmark curves (natex phase-3 task 6).

Fig 6.1 analog (``did_curve_magnitude.csv``): precision / recall / F of the
top SuDDDS discovery against the planted subset s_I, versus discontinuity
magnitude zeta (thesis x-axis 0..20), for greedy / wcc / single_delta on the
thesis base config (d=4 dims, V=8 values, 10 periods, s = 2 values x 2 dims).

Fig 6.2 analog (``did_curve_complexity.csv``): the same metrics versus
intervention complexity (s_dims 1..5) at fixed zeta. The complexity sweep
defaults to V=4 and d = max(complexities): with the thesis's V=8 the treated
fraction (s_values/V)^s_dims vanishes to ~0.1% of n at complexity 5, which no
method can recover — the flat thesis Fig 6.2 lines are only reproducible on a
coarser value grid (matching the CI-small config).

``--exhaustive-max-values`` defaults to 0, forcing every method through its
heuristic priority branch — the thesis-faithful comparison, since the repaired
exact per-dimension enumeration (audit 16) makes greedy and wcc coincide at
these cardinalities. Set it >= V to benchmark the exact branch instead.

The synthetic DGP has no unit column, so ``build_panel``'s profile-id unit
fallback saturates the default background at the thesis base config (V=8:
4096 profiles vs n=2000 records — per-profile intercepts absorb the planted
jump). The script therefore fits one shared background per dataset WITHOUT
unit effects and passes it to every scan (the plan's precomputed
panel/background composition).

Expected qualitative shapes (thesis 6.4.2): all metrics rise with zeta and
are roughly flat in complexity. Two documented deviations from the printed
figures: greedy does NOT lag wcc here (the audit-13/16 repaired subset step
already fixes the premature termination the thesis attributes to greedy), and
single_delta sits BELOW the double-beta methods on the V=8 base config — the
audit-15 corrected statistic profiles mu_i per covariate profile and needs
>= 2 in-window records on both sides per profile, which singleton profiles
cannot supply (it is competitive on the coarser V=4 complexity sweep).

Writes CSVs (and PNG line charts when matplotlib — optional ``plot`` extra —
is importable; otherwise plotting is silently skipped) into
``benchmarks/out/``. Run e.g.::

    uv run python benchmarks/run_did_curves.py --small
"""

from __future__ import annotations

import argparse
import importlib.util
import math
from pathlib import Path

import numpy as np
import pandas as pd

from natex.data.synthetic_did import make_did_synthetic
from natex.did.background import fit_did_background
from natex.did.metrics import subset_precision_recall
from natex.did.panel import build_panel
from natex.did.suddds import suddds_scan

METHODS = ("greedy", "wcc", "single_delta")
DEFAULT_ZETAS = (0.0, 2.5, 5.0, 7.5, 10.0, 12.5, 15.0, 17.5, 20.0)
DEFAULT_COMPLEXITIES = (1, 2, 3, 4, 5)

CURVE_COLUMNS = [
    "method",
    "precision_mean",
    "precision_se",
    "recall_mean",
    "recall_se",
    "f_mean",
    "f_se",
    "t0_hit_rate",
    "llr_mean",
    "n_empty",
    "n_experiments",
]

# Validated categorical palette (dataviz reference instance), fixed slot order.
_SERIES_COLORS = ["#2a78d6", "#1baf7a", "#eda100", "#008300", "#4a3aa7", "#e34948"]
_SURFACE, _GRID, _INK, _MUTED = "#fcfcfb", "#e1e0d9", "#0b0b0b", "#898781"

_NAN = float("nan")


def _have_matplotlib() -> bool:
    return importlib.util.find_spec("matplotlib") is not None


def _nanmean(values: list[float]) -> float:
    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    return float(finite.mean()) if finite.size else _NAN


def _nanse(values: list[float]) -> float:
    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size < 2:
        return _NAN
    return float(finite.std(ddof=1) / math.sqrt(finite.size))


def _run_cell(ds, panel, background, truth, method: str, rng: np.random.Generator,
              args) -> dict[str, float]:
    """One (dataset, method) discovery: top-1 subset metrics vs the truth.

    An empty result (no qualifying cutoff anywhere) yields NaN metrics — never
    0.0 — and counts as a t0 miss.
    """
    res = suddds_scan(
        ds,
        windows=tuple(args.windows),
        restarts=args.restarts,
        method=method,
        exhaustive_max_values=args.exhaustive_max_values,
        rng=rng,
        panel=panel,
        background=background,
    )
    if not res.discoveries:
        return {"precision": _NAN, "recall": _NAN, "f": _NAN, "t0_hit": 0.0, "llr": _NAN,
                "empty": 1.0}
    top = res.discoveries[0]
    precision, recall, f = subset_precision_recall(top.mask, truth.record_mask)
    return {
        "precision": precision,
        "recall": recall,
        "f": f,
        "t0_hit": float(top.t0 == truth.t0),
        "llr": float(top.llr),
        "empty": 0.0,
    }


def discovery_curve(points: list[dict], key: str, methods: tuple[str, ...], args) -> pd.DataFrame:
    """Sweep DGP configs; one row per (sweep point, method).

    Reproducibility contract (mirrors run_nig_curve.py): every experiment
    draws from child generators spawned from ``default_rng(seed)`` keyed by
    fixed (sweep-point, experiment) indices; each (dataset, method) cell gets
    its own spawned stream, so rows are independent of evaluation order.
    """
    children = np.random.default_rng(args.seed).spawn(len(points) * args.n_experiments)
    rows: list[dict] = []
    for ip, point in enumerate(points):
        cells: dict[str, list[dict[str, float]]] = {m: [] for m in methods}
        for ie in range(args.n_experiments):
            streams = children[ip * args.n_experiments + ie].spawn(1 + len(methods))
            ds, truth = make_did_synthetic(rng=streams[0], **point["dgp"])
            # Shared per-dataset panel + no-unit-effects background (see docstring).
            bins = args.bins if args.bins is not None else max(args.V, args.complexity_v)
            panel = build_panel(ds, bins=bins)
            background = fit_did_background(panel, model="auto", unit_effects=False)
            for im, method in enumerate(methods):
                cells[method].append(
                    _run_cell(ds, panel, background, truth, method, streams[1 + im], args)
                )
        for method in methods:
            cell = cells[method]
            rows.append(
                {
                    key: point[key],
                    "method": method,
                    "precision_mean": _nanmean([c["precision"] for c in cell]),
                    "precision_se": _nanse([c["precision"] for c in cell]),
                    "recall_mean": _nanmean([c["recall"] for c in cell]),
                    "recall_se": _nanse([c["recall"] for c in cell]),
                    "f_mean": _nanmean([c["f"] for c in cell]),
                    "f_se": _nanse([c["f"] for c in cell]),
                    "t0_hit_rate": float(np.mean([c["t0_hit"] for c in cell])),
                    "llr_mean": _nanmean([c["llr"] for c in cell]),
                    "n_empty": int(sum(c["empty"] for c in cell)),
                    "n_experiments": args.n_experiments,
                }
            )
    return pd.DataFrame(rows, columns=[key, *CURVE_COLUMNS])


def _styled_axes(plt, title: str, xlabel: str, ylabel: str):
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
    fig.patch.set_facecolor(_SURFACE)
    ax.set_facecolor(_SURFACE)
    ax.grid(True, color=_GRID, linewidth=0.8)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(colors=_MUTED, labelsize=9)
    ax.set_title(title, color=_INK, fontsize=11, loc="left")
    ax.set_xlabel(xlabel, color=_MUTED, fontsize=9)
    ax.set_ylabel(ylabel, color=_MUTED, fontsize=9)
    return fig, ax


def _plot_metric(df: pd.DataFrame, key: str, metric: str, title: str, xlabel: str,
                 path: Path) -> None:
    """One-axis line chart of ``metric`` vs ``key``, one series per method."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = _styled_axes(plt, title, xlabel, metric)
    for slot, method in enumerate(m for m in METHODS if m in set(df["method"])):
        sub = df[df["method"] == method].sort_values(key)
        ax.plot(
            sub[key],
            sub[metric],
            color=_SERIES_COLORS[slot % len(_SERIES_COLORS)],
            linewidth=2,
            marker="o",
            markersize=6,
            label=method,
        )
    ax.set_ylim(-0.02, 1.02)
    ax.legend(frameon=False, fontsize=9, labelcolor=_INK)
    fig.tight_layout()
    fig.savefig(path, facecolor=_SURFACE)
    plt.close(fig)
    print(f"wrote {path}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--zetas", type=float, nargs="+", default=list(DEFAULT_ZETAS))
    parser.add_argument("--complexities", type=int, nargs="+",
                        default=list(DEFAULT_COMPLEXITIES),
                        help="s_dims grid for the Fig 6.2 analog")
    parser.add_argument("--complexity-zeta", type=float, default=10.0,
                        help="fixed magnitude for the complexity sweep (thesis: 10)")
    parser.add_argument("--methods", nargs="+", default=list(METHODS), choices=METHODS)
    parser.add_argument("--n-experiments", type=int, default=20)
    parser.add_argument("--n", type=int, default=2000)
    parser.add_argument("--d", type=int, default=4)
    parser.add_argument("--V", type=int, default=8)
    parser.add_argument("--complexity-v", type=int, default=4,
                        help="value-grid size for the complexity sweep (see module docstring)")
    parser.add_argument("--periods", type=int, default=10)
    parser.add_argument("--tau", type=float, default=10.0)
    parser.add_argument("--s-values", type=int, default=2)
    parser.add_argument("--s-dims", type=int, default=2,
                        help="intervention dims for the magnitude sweep (thesis base: 2)")
    parser.add_argument("--windows", type=float, nargs="+", default=[3.0])
    parser.add_argument("--restarts", type=int, default=8)
    parser.add_argument("--bins", type=int, default=None,
                        help="panel bins (default: max(V, complexity V) so values stay categorical)")
    parser.add_argument("--exhaustive-max-values", type=int, default=0,
                        help="0 (default) forces the heuristic priority branches "
                             "(thesis-faithful); >= V benchmarks the repaired exact branch")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--small", action="store_true",
                        help="quick pass: 3 experiments, n=800, coarse grids")
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "out")
    args = parser.parse_args(argv)

    if args.small:
        args.zetas = [0.0, 8.0, 16.0]
        args.complexities = [1, 2, 3]
        args.n_experiments = 3
        args.n = 1200

    args.out.mkdir(parents=True, exist_ok=True)
    plots = _have_matplotlib()  # optional extra: silently skip plots when absent
    methods = tuple(args.methods)

    # Fig 6.1 analog: magnitude sweep on the thesis base config.
    magnitude_points = [
        {
            "zeta": float(z),
            "dgp": dict(n=args.n, d=args.d, V=args.V, periods=args.periods, zeta=float(z),
                        tau=args.tau, s_dims=args.s_dims, s_values=args.s_values),
        }
        for z in args.zetas
    ]
    magnitude = discovery_curve(magnitude_points, "zeta", methods, args)
    path = args.out / "did_curve_magnitude.csv"
    magnitude.to_csv(path, index=False)
    print(f"wrote {path}")

    # Fig 6.2 analog: complexity sweep (coarser grid; d lifted to fit s_dims).
    d_c = max(args.d, max(args.complexities))
    complexity_points = [
        {
            "s_dims": int(s),
            "dgp": dict(n=args.n, d=d_c, V=args.complexity_v, periods=args.periods,
                        zeta=args.complexity_zeta, tau=args.tau, s_dims=int(s),
                        s_values=min(args.s_values, args.complexity_v)),
        }
        for s in args.complexities
    ]
    complexity = discovery_curve(complexity_points, "s_dims", methods, args)
    path = args.out / "did_curve_complexity.csv"
    complexity.to_csv(path, index=False)
    print(f"wrote {path}")

    if plots:
        for metric, label in (("precision_mean", "precision"), ("recall_mean", "recall"),
                              ("f_mean", "F-score")):
            _plot_metric(magnitude, "zeta", metric,
                         f"Top-1 subset {label} vs zeta (Fig 6.1 analog)",
                         "zeta (discontinuity magnitude)",
                         args.out / f"did_magnitude_{label.split('-')[0].lower()}.png")
            _plot_metric(complexity, "s_dims", metric,
                         f"Top-1 subset {label} vs complexity (Fig 6.2 analog)",
                         "s_dims (intervention complexity)",
                         args.out / f"did_complexity_{label.split('-')[0].lower()}.png")


if __name__ == "__main__":
    main()
