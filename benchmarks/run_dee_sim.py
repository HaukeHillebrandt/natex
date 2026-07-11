#!/usr/bin/env python
"""Scaled DEE simulation-1 replication benchmark (natex phase-4 task 9).

Paper analog (Jakubowski et al. 2023, simulation 1), scaled: for each seed a
fresh ``make_dee_synthetic`` draw (GP-sampled CATE and bias surfaces), a
LoRD3 scan, the full ``dee_debias`` pipeline, and truth-grid MSEs of the four
estimators -- raw observational T-learner, GP-debiased, direct
CATE-extrapolation GP, and the stacked mixture. The full run is 20 seeds x
lengthscales {0.2, 0.5}^2 at n=4000 and reproduces the qualitative sim-1
claim: debiased/mixture beats the raw observational estimator in median grid
MSE.

Documented deviations from the paper defaults:

- ``type_probs`` defaults to (0.1, 0.4, 0.4, 0.1) -- 0.4 compliers per
  boundary. The uniform (0.25,)*4 types leave the local 2SLS too weak at
  these scaled-down n to separate tau from the confounded contrast
  (calibration table in tests/test_dee_debias.py); the harness exposes the
  knob so the paper's uniform mix can still be run.
- The row dict carries ``m_prime`` (the VKNN candidate count actually used)
  in addition to the plan's keys, so the ``q_null`` path is auditable:
  ``q_null > 0`` replaces the fixed ``m_prime`` with ``select_m_prime`` on
  the phase-1 fitted-null Monte Carlo (randomization test sharing ONE scan
  geometry with the observed scan -- the phase-2 shared-geometry path).

Reproducibility: one root ``default_rng(seed)`` spawns the three independent
streams (data, null replicas, pipeline); identical arguments => bitwise
identical row.

Writes ``dee_sim1.csv`` (columns exactly ``RESULT_COLUMNS``) and, when
matplotlib (optional ``plot`` extra) is importable, a per-config MSE box plot
``dee_sim1_mse.png`` into ``benchmarks/out/``; without matplotlib plotting is
silently skipped. Run e.g.::

    uv run python benchmarks/run_dee_sim.py --small
"""

from __future__ import annotations

import argparse
import importlib.util
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from natex.data.synthetic_dee import make_dee_synthetic
from natex.dee.debias import dee_debias
from natex.dee.vknn import select_m_prime
from natex.rdd.lord3 import lord3_scan
from natex.scan.geometry import build_geometry
from natex.validate.randomization import randomization_test

DEFAULT_TYPE_PROBS = (0.1, 0.4, 0.4, 0.1)
DEFAULT_LENGTHSCALES = (0.2, 0.5)

RESULT_COLUMNS = [
    "seed",
    "cate_ls",
    "bias_ls",
    "n_experiments",
    "m_prime",
    "w_debias",
    "mse_raw",
    "mse_debiased",
    "mse_direct",
    "mse_mixture",
]

ESTIMATORS = ("raw", "debiased", "direct", "mixture")

# Validated categorical palette (dataviz reference instance), fixed slot order
# raw / debiased / direct / mixture; same instance as run_did_curves.py.
_SERIES_COLORS = ["#2a78d6", "#1baf7a", "#eda100", "#4a3aa7"]
_SURFACE, _GRID, _INK, _MUTED = "#fcfcfb", "#e1e0d9", "#0b0b0b", "#898781"

_NAN = float("nan")


def _have_matplotlib() -> bool:
    return importlib.util.find_spec("matplotlib") is not None


def run_dee_replication(
    seed: int,
    n: int,
    cate_ls: float,
    bias_ls: float,
    *,
    k: int = 100,
    m_prime: int | None = 40,
    q_null: int = 0,
    k_prime: int = 400,
    t_side: int = 25,
    grid: int = 25,
    weighting: str = "stacking",
    type_probs: tuple[float, float, float, float] = DEFAULT_TYPE_PROBS,
) -> dict:
    """One seeded sim-1 replication; returns the RESULT_COLUMNS row dict.

    ``q_null > 0`` switches M' selection to the randomization-test null
    (``select_m_prime``; the ``m_prime`` argument is then ignored); the
    Q=q_null replica scans share the observed scan's geometry. MSEs are
    truth-grid MSEs against ``DEETruth.cate_query``; a degenerate pipeline
    (< 3 usable experiments) yields NaN MSEs -- never 0.0.
    """
    data_rng, null_rng, pipe_rng = np.random.default_rng(seed).spawn(3)
    ds, truth = make_dee_synthetic(
        n,
        cate_lengthscale=cate_ls,
        bias_lengthscale=bias_ls,
        grid=grid,
        type_probs=type_probs,
        rng=data_rng,
    )
    geometry = build_geometry(ds.Z_std, k)
    scan = lord3_scan(ds, k=k, model="bernoulli", geometry=geometry)
    if q_null > 0:
        report = randomization_test(
            ds, scan, Q=q_null, rng=null_rng, scan_kwargs={"k": k}, geometry=geometry
        )
        m_prime_used = select_m_prime(scan, report.null_max_llrs)
    else:
        if m_prime is None:
            raise ValueError("m_prime is required when q_null == 0")
        m_prime_used = int(m_prime)

    res = dee_debias(
        ds,
        truth.query,
        scan,
        m_prime=m_prime_used,
        k_prime=k_prime,
        t_side=t_side,
        weighting=weighting,
        rng=pipe_rng,
    )

    def _mse(estimate: np.ndarray) -> float:
        return float(np.mean((np.asarray(estimate, dtype=float) - truth.cate_query) ** 2))

    return {
        "seed": int(seed),
        "cate_ls": float(cate_ls),
        "bias_ls": float(bias_ls),
        "n_experiments": int(res.diagnostics["n_experiments_used"]),
        "m_prime": int(m_prime_used),
        "w_debias": float(res.weights.w_debias),
        "mse_raw": _mse(res.cate_raw),
        "mse_debiased": _mse(res.cate_debiased),
        "mse_direct": _mse(res.cate_direct),
        "mse_mixture": _mse(res.mixture.mean) if res.mixture is not None else _NAN,
    }


def run_dee_grid(
    seeds: Sequence[int], lengthscales: Sequence[float], n: int, **kw
) -> pd.DataFrame:
    """Replications over the (cate_ls, bias_ls) product grid x seeds.

    One row per replication, columns exactly ``RESULT_COLUMNS``; ``kw`` is
    forwarded to ``run_dee_replication`` verbatim. Rows only depend on their
    own (seed, config) arguments, so the frame is independent of sweep order.
    """
    rows = [
        run_dee_replication(int(seed), n, float(cate_ls), float(bias_ls), **kw)
        for cate_ls in lengthscales
        for bias_ls in lengthscales
        for seed in seeds
    ]
    return pd.DataFrame(rows, columns=RESULT_COLUMNS)


def _plot_mse_boxes(df: pd.DataFrame, path: Path) -> None:
    """Per-config MSE box plot: one group per (cate_ls, bias_ls), one box per
    estimator in fixed slot order, log-MSE axis (MSEs span decades)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    configs = sorted(set(zip(df["cate_ls"], df["bias_ls"], strict=True)))
    fig, ax = plt.subplots(figsize=(1.6 + 2.1 * len(configs), 4.5), dpi=150)
    fig.patch.set_facecolor(_SURFACE)
    ax.set_facecolor(_SURFACE)
    ax.grid(True, axis="y", color=_GRID, linewidth=0.8)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(colors=_MUTED, labelsize=9)

    width, gap = 0.18, 0.02
    for ic, (cls, bls) in enumerate(configs):
        sub = df[(df["cate_ls"] == cls) & (df["bias_ls"] == bls)]
        for slot, est in enumerate(ESTIMATORS):
            vals = sub[f"mse_{est}"].to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                continue
            pos = ic + (slot - (len(ESTIMATORS) - 1) / 2) * (width + gap)
            box = ax.boxplot(
                [vals], positions=[pos], widths=width, patch_artist=True,
                medianprops={"color": _INK, "linewidth": 1.4},
                whiskerprops={"color": _MUTED}, capprops={"color": _MUTED},
                flierprops={"marker": "o", "markersize": 3,
                            "markerfacecolor": _MUTED, "markeredgecolor": "none"},
            )
            for patch in box["boxes"]:
                patch.set_facecolor(_SERIES_COLORS[slot])
                patch.set_edgecolor(_SURFACE)
                patch.set_linewidth(2)
    ax.set_yscale("log")
    ax.set_xticks(range(len(configs)))
    ax.set_xticklabels([f"lc={c}, lb={b}" for c, b in configs], color=_INK, fontsize=9)
    ax.set_title("DEE sim-1 (scaled): truth-grid MSE by estimator",
                 color=_INK, fontsize=11, loc="left")
    ax.set_xlabel("surface lengthscales (cate, bias)", color=_MUTED, fontsize=9)
    ax.set_ylabel("grid MSE (log scale)", color=_MUTED, fontsize=9)
    ax.legend(
        handles=[Patch(facecolor=_SERIES_COLORS[i], label=e)
                 for i, e in enumerate(ESTIMATORS)],
        frameon=False, fontsize=9, labelcolor=_INK, ncols=len(ESTIMATORS),
        loc="upper left", bbox_to_anchor=(0.0, -0.12),
    )
    fig.tight_layout()
    fig.savefig(path, facecolor=_SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n-seeds", type=int, default=20)
    parser.add_argument("--seed0", type=int, default=0, help="first seed (seeds are contiguous)")
    parser.add_argument("--lengthscales", type=float, nargs="+",
                        default=list(DEFAULT_LENGTHSCALES),
                        help="surface lengthscale grid; configs are the product set")
    parser.add_argument("--n", type=int, default=4000)
    parser.add_argument("--k", type=int, default=100)
    parser.add_argument("--m-prime", type=int, default=40)
    parser.add_argument("--q-null", type=int, default=0,
                        help="> 0 selects M' from a Q=q_null randomization-test null")
    parser.add_argument("--k-prime", type=int, default=400)
    parser.add_argument("--t-side", type=int, default=25)
    parser.add_argument("--grid", type=int, default=25)
    parser.add_argument("--weighting", default="stacking", choices=("stacking", "loo", "mll"))
    parser.add_argument("--small", action="store_true",
                        help="quick pass: 3 seeds, n=1200, CI-small pipeline knobs")
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "out")
    args = parser.parse_args(argv)

    if args.small:
        args.n_seeds = 3
        args.lengthscales = [0.5]
        args.n = 1200
        args.k = 50
        args.m_prime = 25
        args.k_prime = 250
        args.t_side = 15
        args.grid = 12

    args.out.mkdir(parents=True, exist_ok=True)
    seeds = range(args.seed0, args.seed0 + args.n_seeds)
    df = run_dee_grid(
        seeds,
        args.lengthscales,
        args.n,
        k=args.k,
        m_prime=args.m_prime,
        q_null=args.q_null,
        k_prime=args.k_prime,
        t_side=args.t_side,
        grid=args.grid,
        weighting=args.weighting,
    )
    path = args.out / "dee_sim1.csv"
    df.to_csv(path, index=False)
    print(f"wrote {path}")

    medians = df.groupby(["cate_ls", "bias_ls"])[
        [f"mse_{e}" for e in ESTIMATORS]
    ].median()
    print("median grid MSE per config:")
    print(medians.to_string())

    if _have_matplotlib():  # optional extra: silently skip plots when absent
        _plot_mse_boxes(df, args.out / "dee_sim1_mse.png")


if __name__ == "__main__":
    main()
