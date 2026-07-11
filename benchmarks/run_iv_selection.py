#!/usr/bin/env python
"""IV instrument-selection / SC donor-recovery benchmark (natex phase-5 task 10).

IV block — BCCH exponential-design DGP (``make_iv_synthetic``) over a
concentration grid mu^2 in {8, 30, 80, 180, 400} x 20 seeds at
(n=500, p=50, s=5). Per (seed, mu2) cell:

- full-sample plug-in Lasso selection (``select_instruments``; never reads
  ``y``): precision/recall vs ``true_support`` and recall of the top-3 true
  instruments;
- post-Lasso 2SLS on the selected columns (``iv_2sls``): |tau_hat - tau|,
  the AR set kind (``ar_bounded = 1`` iff kind == "interval"; "empty" is a
  rejection, not a usable finite CI) and whether the Wald CI is finite —
  the Wald CI stays finite even when the first stage is weak, the honesty
  gap the AR column exists to show;
- naive OLS of y on T: |tau_hat - tau| (plim bias endog/(pi' Sigma pi + 1));
- honest pipeline (``discover_instruments``, select on half A / estimate on
  half B): does the estimation-half AR set cover the true tau (all four set
  kinds handled: interval/disjoint membership, unbounded => covered,
  empty => not covered), and does the Wald CI.

SC block — factor-model DGP (``make_sc_synthetic``) over
noise in {0.25, 0.5, 1.0} x 20 seeds: top-``n_donors`` pre-trend donor
selection (``select_donors``), fraction of true donors recovered, summed
simplex weight on the true donors, and |ATT - effect|.

NaN policy (never 0.0 on failure): empty selection => NaN precision, NaN
biases, NaN ar/wald fields; NaN ATT propagates to NaN |ATT - effect|.

Reproducibility: one root ``default_rng(seed)`` spawns the (data, selection,
split) streams; plug-in selection and donor selection are themselves
rng-free, so identical arguments => bitwise-identical row.

Writes ``iv_selection.csv`` (IV block) and ``sc_recovery.csv`` (SC block)
and, when matplotlib (optional ``plot`` extra) is importable, a two-panel
mean-rate curve figure ``iv_selection.png`` into ``benchmarks/out/``;
without matplotlib plotting is silently skipped. Run e.g.::

    uv run python benchmarks/run_iv_selection.py --small
"""

from __future__ import annotations

import argparse
import importlib.util
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from natex.data.synthetic_iv import make_iv_synthetic
from natex.data.synthetic_sc import make_sc_synthetic
from natex.estimate.iv2sls import IVEstimate, iv_2sls
from natex.iv.donors import select_donors, unit_time_matrix
from natex.iv.pipeline import discover_instruments
from natex.iv.search import select_instruments

DEFAULT_MU2_GRID = (8.0, 30.0, 80.0, 180.0, 400.0)
DEFAULT_NOISE_GRID = (0.25, 0.5, 1.0)

IV_RESULT_COLUMNS = [
    "seed",
    "mu2",
    "n",
    "p",
    "s",
    "concentration",
    "n_selected",
    "precision",
    "recall",
    "recall_top3",
    "first_stage_F",
    "weak",
    "bias_2sls",
    "bias_ols",
    "ar_kind",
    "ar_bounded",
    "wald_finite",
    "honest_ar_kind",
    "honest_ar_covers",
    "honest_wald_covers",
]

SC_RESULT_COLUMNS = [
    "seed",
    "noise",
    "n_units",
    "n_pre",
    "n_post",
    "k_true",
    "effect",
    "n_donors",
    "n_true_recovered",
    "donor_recovery",
    "weight_on_true",
    "att_post",
    "abs_att_error",
    "pre_rmspe",
    "post_rmspe",
]

# Validated categorical palette (dataviz reference instance), fixed slot order
# precision / recall / ar-bounded / honest-AR-coverage (IV panel) and
# donor-recovery / |ATT err| (SC panel reuses slots 1 and 3); same instance
# as run_dee_sim.py / run_did_curves.py.
_SERIES_COLORS = ["#2a78d6", "#1baf7a", "#eda100", "#4a3aa7"]
_SURFACE, _GRID, _INK, _MUTED = "#fcfcfb", "#e1e0d9", "#0b0b0b", "#898781"

_NAN = float("nan")


def _have_matplotlib() -> bool:
    return importlib.util.find_spec("matplotlib") is not None


def _wald_covers(est: IVEstimate, tau: float) -> float:
    """1.0/0.0 if the Wald CI is finite and does/does not cover tau; NaN otherwise."""
    lo, hi = est.ci
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return _NAN
    return float(lo <= tau <= hi)


def _ar_covers(est: IVEstimate, tau: float) -> float:
    """Does the AR confidence set contain tau? Handles all four set kinds
    (napkin: coverage helpers must): interval/disjoint by membership,
    unbounded = whole line => 1.0, empty => 0.0; NaN when no set was computed."""
    kind = est.ar_kind
    if kind is None:
        return _NAN
    if kind == "interval":
        lo, hi = est.ar_ci
        return float(lo <= tau <= hi)
    if kind == "unbounded":
        return 1.0
    if kind == "empty":
        return 0.0
    rays = est.extras["ar_rays"]  # kind == "disjoint"
    return float(any(lo <= tau <= hi for lo, hi in rays))


def _ols_slope(y: np.ndarray, t: np.ndarray) -> float:
    """Naive OLS slope of y on [1, T] — the endogeneity-biased comparator."""
    coef, *_ = np.linalg.lstsq(np.c_[np.ones(t.size), t], y, rcond=None)
    return float(coef[1])


def run_iv_replication(
    seed: int,
    mu2: float,
    *,
    n: int = 500,
    p: int = 50,
    s: int = 5,
    rho_z: float = 0.5,
    endog: float = 0.6,
    tau: float = 1.0,
    lam: float | str = "plugin",
    alpha: float = 0.05,
    frac_discovery: float = 0.5,
) -> dict:
    """One seeded IV cell; returns the IV_RESULT_COLUMNS row dict.

    Selection runs on the full sample (reads only (T, pool)); 2SLS/OLS/AR are
    full-sample too, so ``bias_2sls`` is the post-Lasso benchmark quantity.
    The honest_* fields come from a separate ``discover_instruments`` run
    (select on the discovery half, estimate + AR on the estimation half),
    whose split stream is spawned from the same root rng.
    """
    data_rng, sel_rng, split_rng = np.random.default_rng(seed).spawn(3)
    ds = make_iv_synthetic(
        n=n, p=p, s=s, mu2=mu2, rho_z=rho_z, endog=endog, tau=tau, rng=data_rng
    )
    t_vec = ds.df["T"].to_numpy(dtype=float)
    y = ds.df["y"].to_numpy(dtype=float)
    z = ds.df[ds.pool_names].to_numpy(dtype=float)

    search = select_instruments(t_vec, z, pool_names=ds.pool_names, lam=lam, rng=sel_rng)
    selected = set(search.selected)
    true = set(ds.true_support)
    top3 = ds.true_support[: min(3, s)]
    n_selected = len(selected)
    precision = len(selected & true) / n_selected if n_selected else _NAN
    recall = len(selected & true) / len(true)
    recall_top3 = len(selected & set(top3)) / len(top3)

    bias_2sls = _NAN
    ar_kind: str | None = None
    ar_bounded = _NAN
    wald_finite = _NAN
    if n_selected:
        cols = [ds.pool_names.index(name) for name in search.selected]
        est = iv_2sls(y, t_vec, z[:, cols], alpha=alpha)
        bias_2sls = abs(est.tau - tau)
        ar_kind = est.ar_kind
        if ar_kind is not None:
            ar_bounded = float(ar_kind == "interval")
        wald_finite = float(np.isfinite(est.ci[0]) and np.isfinite(est.ci[1]))

    bias_ols = abs(_ols_slope(y, t_vec) - tau)

    disc = discover_instruments(
        ds.df,
        "T",
        ds.pool_names,
        outcome="y",
        honest=True,
        frac_discovery=frac_discovery,
        lam=lam,
        rng=split_rng,
    )
    est_h = disc.estimate
    return {
        "seed": int(seed),
        "mu2": float(mu2),
        "n": int(n),
        "p": int(p),
        "s": int(s),
        "concentration": float(ds.concentration),
        "n_selected": n_selected,
        "precision": precision,
        "recall": recall,
        "recall_top3": recall_top3,
        "first_stage_F": float(search.first_stage_F),
        "weak": float(search.weak),
        "bias_2sls": bias_2sls,
        "bias_ols": bias_ols,
        "ar_kind": ar_kind,
        "ar_bounded": ar_bounded,
        "wald_finite": wald_finite,
        "honest_ar_kind": est_h.ar_kind,
        "honest_ar_covers": _ar_covers(est_h, tau),
        "honest_wald_covers": _wald_covers(est_h, tau),
    }


def run_iv_grid(seeds: Sequence[int], mu2_grid: Sequence[float], **kw) -> pd.DataFrame:
    """IV replications over the mu2 grid x seeds; columns exactly IV_RESULT_COLUMNS.

    ``kw`` is forwarded to ``run_iv_replication`` verbatim; rows depend only
    on their own (seed, mu2), so the frame is independent of sweep order.
    """
    rows = [
        run_iv_replication(int(seed), float(mu2), **kw)
        for mu2 in mu2_grid
        for seed in seeds
    ]
    return pd.DataFrame(rows, columns=IV_RESULT_COLUMNS)


def run_sc_replication(
    seed: int,
    noise: float,
    *,
    n_units: int = 20,
    n_pre: int = 15,
    n_post: int = 10,
    n_factors: int = 2,
    k_true: int = 3,
    effect: float = 10.0,
    n_donors: int = 8,
    scoring: str = "rmse",
) -> dict:
    """One seeded SC cell; returns the SC_RESULT_COLUMNS row dict.

    Donor selection/weighting reads only pre-t0 outcomes (method property);
    recovery = fraction of the DGP's true donors inside the selected pool.
    """
    rng = np.random.default_rng(seed)
    ds = make_sc_synthetic(
        n_units=n_units,
        n_pre=n_pre,
        n_post=n_post,
        n_factors=n_factors,
        k_true=k_true,
        effect=effect,
        noise=noise,
        rng=rng,
    )
    y_mat, units, times = unit_time_matrix(ds.df, "unit", "time", "y")
    res = select_donors(
        y_mat, units, times, ds.treated_unit, ds.t0, n_donors=n_donors, scoring=scoring
    )
    true = set(ds.true_donors)
    n_true_recovered = len(true & set(res.donors))
    weight_on_true = float(
        sum(w for donor, w in zip(res.donors, res.weights, strict=True) if donor in true)
    )
    att = float(res.att_post)
    return {
        "seed": int(seed),
        "noise": float(noise),
        "n_units": int(n_units),
        "n_pre": int(n_pre),
        "n_post": int(n_post),
        "k_true": int(k_true),
        "effect": float(effect),
        "n_donors": int(n_donors),
        "n_true_recovered": n_true_recovered,
        "donor_recovery": n_true_recovered / k_true,
        "weight_on_true": weight_on_true,
        "att_post": att,
        "abs_att_error": abs(att - effect),  # NaN att propagates, never 0.0
        "pre_rmspe": float(res.pre_rmspe),
        "post_rmspe": float(res.post_rmspe),
    }


def run_sc_grid(seeds: Sequence[int], noise_grid: Sequence[float], **kw) -> pd.DataFrame:
    """SC replications over the noise grid x seeds; columns exactly SC_RESULT_COLUMNS."""
    rows = [
        run_sc_replication(int(seed), float(noise), **kw)
        for noise in noise_grid
        for seed in seeds
    ]
    return pd.DataFrame(rows, columns=SC_RESULT_COLUMNS)


def _plot_curves(iv: pd.DataFrame, sc: pd.DataFrame, path: Path) -> None:
    """Two-panel mean-rate curves: IV metrics vs mu2 (log x), SC metrics vs
    noise. All series are rates in [0, 1] (|ATT err| is scaled by 1/effect
    into a rate-like error fraction), one y axis per panel."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.2), dpi=150)
    fig.patch.set_facecolor(_SURFACE)

    # precision is dashed and drawn on top (zorder): it and "AR set bounded"
    # both saturate at 1.0 on the default grid, and a dash-on-solid overlap
    # keeps both series visible.
    iv_series = [
        ("precision", "precision", 0, "--", 4),
        ("recall", "recall", 1, "-", 2),
        ("ar_bounded", "AR set bounded", 2, "-", 2),
        ("honest_ar_covers", "honest AR coverage", 3, "-", 2),
    ]
    g1 = iv.groupby("mu2")
    for col, label, slot, style, zorder in iv_series:
        means = g1[col].mean()  # NaN cells (empty selections) skipped per-cell
        ax1.plot(
            means.index,
            means.to_numpy(),
            color=_SERIES_COLORS[slot],
            linewidth=2,
            linestyle=style,
            marker="o",
            markersize=5,
            zorder=zorder,
            label=label,
        )
    ax1.set_xscale("log")
    ax1.set_xlabel("concentration mu^2 (log scale)", color=_MUTED, fontsize=9)
    ax1.set_ylabel("mean rate", color=_MUTED, fontsize=9)
    ax1.set_title("IV selection and inference vs concentration", color=_INK, fontsize=10, loc="left")

    effect = float(sc["effect"].iloc[0])  # |ATT err| / effect is a rate-like fraction
    g2 = sc.groupby("noise")
    ax2.plot(
        g2["donor_recovery"].mean().index,
        g2["donor_recovery"].mean().to_numpy(),
        color=_SERIES_COLORS[1],
        linewidth=2,
        marker="o",
        markersize=5,
        label="donor recovery",
    )
    ax2.plot(
        g2["abs_att_error"].mean().index,
        g2["abs_att_error"].mean().to_numpy() / effect,
        color=_SERIES_COLORS[3],
        linewidth=2,
        marker="o",
        markersize=5,
        label=f"|ATT err| / {effect:g}",
    )
    ax2.set_xlabel("outcome noise sd", color=_MUTED, fontsize=9)
    ax2.set_ylabel("mean rate", color=_MUTED, fontsize=9)
    ax2.set_title("SC donor recovery vs noise", color=_INK, fontsize=10, loc="left")

    for ax in (ax1, ax2):
        ax.set_facecolor(_SURFACE)
        ax.grid(True, axis="y", color=_GRID, linewidth=0.8)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(colors=_MUTED, labelsize=9)
        ax.set_ylim(-0.02, 1.05)
        ax.legend(frameon=False, fontsize=9, labelcolor=_INK, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, facecolor=_SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n-seeds", type=int, default=20)
    parser.add_argument("--seed0", type=int, default=0, help="first seed (seeds are contiguous)")
    parser.add_argument("--mu2-grid", type=float, nargs="+", default=list(DEFAULT_MU2_GRID))
    parser.add_argument("--noise-grid", type=float, nargs="+", default=list(DEFAULT_NOISE_GRID))
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--p", type=int, default=50)
    parser.add_argument("--s", type=int, default=5)
    parser.add_argument("--endog", type=float, default=0.6)
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--n-donors", type=int, default=8)
    parser.add_argument("--sc-effect", type=float, default=10.0)
    parser.add_argument("--small", action="store_true",
                        help="quick pass: 3 seeds, mu2 {8, 180}, noise {0.5}")
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "out")
    args = parser.parse_args(argv)

    if args.small:
        args.n_seeds = 3
        args.mu2_grid = [8.0, 180.0]
        args.noise_grid = [0.5]

    args.out.mkdir(parents=True, exist_ok=True)
    seeds = range(args.seed0, args.seed0 + args.n_seeds)

    iv = run_iv_grid(
        seeds, args.mu2_grid, n=args.n, p=args.p, s=args.s, endog=args.endog, tau=args.tau
    )
    iv_path = args.out / "iv_selection.csv"
    iv.to_csv(iv_path, index=False)
    print(f"wrote {iv_path}")
    print("IV means per mu2 (NaN cells skipped):")
    print(
        iv.groupby("mu2")[
            ["precision", "recall", "recall_top3", "bias_2sls", "bias_ols",
             "ar_bounded", "wald_finite", "honest_ar_covers", "honest_wald_covers"]
        ].mean().to_string()
    )

    sc = run_sc_grid(seeds, args.noise_grid, n_donors=args.n_donors, effect=args.sc_effect)
    sc_path = args.out / "sc_recovery.csv"
    sc.to_csv(sc_path, index=False)
    print(f"wrote {sc_path}")
    print("SC means per noise:")
    print(
        sc.groupby("noise")[
            ["donor_recovery", "weight_on_true", "abs_att_error"]
        ].mean().to_string()
    )

    if _have_matplotlib():  # optional extra: silently skip plots when absent
        _plot_curves(iv, sc, args.out / "iv_selection.png")


if __name__ == "__main__":
    main()
