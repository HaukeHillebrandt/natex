#!/usr/bin/env python
"""Full KDD-2018 ch.5 synthetic benchmark curves (natex phase-2 task 7).

Paper protocol: 50 experiments per zeta, n = 1000, k = 50, tau = 5; zeta swept
up to 2.5 for real-valued T and to ~5 for binary T; polynomial orders 1/2/4;
label-noise protocol P(T_rho = T) = rho with 25 experiments per rho. Expected
qualitative shapes: NIG and power increase with zeta; orders 2/4 mildly worse
than 1; Bernoulli >= Normal NIG on binary T; tau-hat overestimates at low zeta.

Writes ``benchmarks/out/nig_curve_{kind}.csv`` (and ``label_noise.csv`` with
``--label-noise``). If matplotlib is importable (optional ``plot`` extra), PNG
line charts are written next to the CSVs; otherwise plotting is silently
skipped. Run e.g.::

    uv run python benchmarks/run_nig_curve.py --kind both --label-noise
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import pandas as pd

from natex.benchmarks import label_noise_curve, nig_power_curve

DEFAULT_ZETAS = {
    "real": (0.0, 0.5, 1.0, 1.5, 2.0, 2.5),
    "binary": (0.0, 1.0, 2.0, 3.0, 4.0, 5.0),
}
DEFAULT_MODELS = {"real": ("auto",), "binary": ("normal", "bernoulli")}

# Validated categorical palette (dataviz reference instance), fixed slot order.
_SERIES_COLORS = ["#2a78d6", "#1baf7a", "#eda100", "#008300", "#4a3aa7", "#e34948"]
_SURFACE, _GRID, _INK, _MUTED = "#fcfcfb", "#e1e0d9", "#0b0b0b", "#898781"


def _have_matplotlib() -> bool:
    return importlib.util.find_spec("matplotlib") is not None


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


def _plot_metric(df: pd.DataFrame, metric: str, title: str, path: Path) -> None:
    """One-axis line chart of `metric` vs zeta, one series per (model, degree)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = _styled_axes(plt, title, "zeta (discontinuity strength)", metric)
    series = sorted(df.groupby(["model", "degree"]).groups)  # fixed order, never cycled
    for slot, (model, degree) in enumerate(series):
        sub = df[(df["model"] == model) & (df["degree"] == degree)].sort_values("zeta")
        color = _SERIES_COLORS[slot % len(_SERIES_COLORS)]
        ax.plot(
            sub["zeta"],
            sub[metric],
            color=color,
            linewidth=2,
            marker="o",
            markersize=6,
            label=f"{model}, degree {degree}",
        )
    ax.legend(frameon=False, fontsize=9, labelcolor=_INK)
    fig.tight_layout()
    fig.savefig(path, facecolor=_SURFACE)
    plt.close(fig)
    print(f"wrote {path}")


def _plot_label_noise(df: pd.DataFrame, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = _styled_axes(plt, "Label-noise protocol: top-1 NIG vs rho", "rho = P(T_rho = T)", "NIG")
    sub = df.sort_values("rho")
    ax.plot(sub["rho"], sub["nig_mean"], color=_SERIES_COLORS[0], linewidth=2, marker="o", markersize=6)
    fig.tight_layout()
    fig.savefig(path, facecolor=_SURFACE)
    plt.close(fig)
    print(f"wrote {path}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--kind", choices=("real", "binary", "both"), default="both")
    parser.add_argument("--zetas", type=float, nargs="+", default=None,
                        help="zeta grid (default: per-kind paper grid)")
    parser.add_argument("--degrees", type=int, nargs="+", default=(1, 2, 4))
    parser.add_argument("--models", nargs="+", default=None,
                        help="treatment models (default: auto for real; normal+bernoulli for binary)")
    parser.add_argument("--n-experiments", type=int, default=50)
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--k", type=int, default=50)
    parser.add_argument("--Q", type=int, default=99)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--tau", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--label-noise", action="store_true",
                        help="also run the label-noise curve (label_noise.csv)")
    parser.add_argument("--rhos", type=float, nargs="+",
                        default=(0.5, 0.6, 0.7, 0.8, 0.9, 1.0))
    parser.add_argument("--noise-experiments", type=int, default=25)
    parser.add_argument("--noise-n", type=int, default=2000)
    parser.add_argument("--noise-k", type=int, default=50)
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "out")
    args = parser.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)
    plots = _have_matplotlib()  # optional extra: silently skip plots when absent

    kinds = ("real", "binary") if args.kind == "both" else (args.kind,)
    for kind in kinds:
        df = nig_power_curve(
            kind,
            zetas=args.zetas if args.zetas is not None else DEFAULT_ZETAS[kind],
            n_experiments=args.n_experiments,
            n=args.n,
            k=args.k,
            degrees=tuple(args.degrees),
            models=tuple(args.models) if args.models is not None else DEFAULT_MODELS[kind],
            Q=args.Q,
            alpha=args.alpha,
            tau=args.tau,
            seed=args.seed,
        )
        csv_path = args.out / f"nig_curve_{kind}.csv"
        df.to_csv(csv_path, index=False)
        print(f"wrote {csv_path}")
        if plots:
            _plot_metric(df, "nig_mean", f"Top-1 NIG vs zeta ({kind} T)",
                         args.out / f"nig_curve_{kind}.png")
            _plot_metric(df, "power", f"Power at alpha={args.alpha} vs zeta ({kind} T)",
                         args.out / f"power_curve_{kind}.png")

    if args.label_noise:
        noise = label_noise_curve(
            rhos=args.rhos,
            n_experiments=args.noise_experiments,
            n=args.noise_n,
            k=args.noise_k,
            seed=args.seed,
        )
        noise_path = args.out / "label_noise.csv"
        noise.to_csv(noise_path, index=False)
        print(f"wrote {noise_path}")
        if plots:
            _plot_label_noise(noise, args.out / "label_noise.png")


if __name__ == "__main__":
    main()
