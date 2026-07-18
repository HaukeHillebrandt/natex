"""Regenerate fig1 (PDF + PNG) for the btos-rewording-rdd mini-paper.

Reads the frozen analysis outputs of the run of record
(``/Users/haukehillebrandt/dev/study-runs/btos-rewording-rdd``, natex v0.2.0,
seed 0) — pass a different location with ``--results-root``. The underlying
Census BTOS workbooks are NOT committed to this repository; only the derived
files written by the analysis pass are read here:

- ``panel.csv``       the spliced 71-wave national AI-use panel
- ``rd_results.json`` headline RD estimate, bandwidth grid, placebo-cutoff grid

The headline estimate is asserted against the numbers of record before a
figure is written, and the SE-weighted local-linear fit drawn in panel (a) is
recomputed here from ``panel.csv`` and asserted against the same numbers, so a
drifted results vintage fails loudly instead of silently redrawing different
numbers.

Deterministic: pure file reads, no RNG, and ``SOURCE_DATE_EPOCH`` is pinned so
the PDF/PNG bytes are reproducible.

Run (from the repo root):

    uv run python papers/btos-rewording-rdd/figures/make_fig.py \
        --results-root /Users/haukehillebrandt/dev/study-runs/btos-rewording-rdd
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("SOURCE_DATE_EPOCH", "0")  # reproducible figure bytes

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Okabe-Ito, colorblind-safe; hues assigned in fixed order by entity.
BLUE = "#0072B2"        # old-wording series
ORANGE = "#E69F00"      # new-wording series
VERMILLION = "#D55E00"  # cutoff / true-cutoff estimate
GRAY = "#666666"        # placebo cutoffs
INK = "#222222"

T_C = 2025.838356        # 2025-11-03, reference-period start of wave 202524
H = 1.0                  # headline bandwidth (years)
GAP_LO, GAP_HI = 2025.7027397260274, 2025.8561643835617  # no-data ref midpoints

# Numbers of record (papers/btos-rewording-rdd/main.tex, analysis 2026-07).
RECORD_TAU, RECORD_SE_HC1, RECORD_SE_HAC = 6.0394835180469855, 0.2634901414016464, 0.37137523987750354
RECORD_PRE_LEVEL, RECORD_POST_LEVEL = 10.922100279234769, 16.961583797281754
RECORD_PLACEBO_MAX = 1.2981494363545778
ATOL = 5e-6


def wls_side(df: pd.DataFrame, lo: float, hi: float) -> np.ndarray:
    """SE-weighted (w=1/se^2) linear fit of ai_use on (t - T_C) over [lo, hi)."""
    side = df[(df["t"] - T_C >= lo) & (df["t"] - T_C < hi)]
    x = side["t"].values - T_C
    y = side["ai_use"].values
    w = 1.0 / side["se"].values ** 2
    X = np.column_stack([np.ones_like(x), x])
    return np.linalg.solve(X.T @ (X * w[:, None]), (X * w[:, None]).T @ y)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--results-root",
        type=Path,
        default=Path("/Users/haukehillebrandt/dev/study-runs/btos-rewording-rdd"),
    )
    ap.add_argument("--skip-checks", action="store_true")
    args = ap.parse_args()

    panel = pd.read_csv(args.results_root / "panel.csv")
    res = json.loads((args.results_root / "rd_results.json").read_text())

    # --- assert the results vintage against the numbers of record ----------
    head = res["headline"]
    beta_pre = wls_side(panel, -H, 0.0)   # post side uses r >= 0 exactly as analyze.py
    beta_post = wls_side(panel, 0.0, H + 1e-9)
    tau_refit = beta_post[0] - beta_pre[0]
    checks = [
        ("headline tau (json)", head["tau"], RECORD_TAU),
        ("headline se HC1 (json)", head["se"], RECORD_SE_HC1),
        ("tau refit from panel.csv", tau_refit, RECORD_TAU),
        ("pre level at cutoff (refit)", beta_pre[0], RECORD_PRE_LEVEL),
        ("post level at cutoff (refit)", beta_post[0], RECORD_POST_LEVEL),
        ("max |placebo tau| (json)", res["placebo"]["max_abs_tau"], RECORD_PLACEBO_MAX),
    ]
    ok = True
    for name, got, want in checks:
        good = np.isclose(got, want, atol=ATOL)
        ok &= bool(good)
        print(f"[{name}] {got:+.6f} vs record {want:+.6f} {'OK' if good else 'MISMATCH'}")
    if not ok and not args.skip_checks:
        sys.exit(
            "estimates do not match the numbers of record — wrong results "
            "vintage? (--skip-checks to override)"
        )

    plt.rcParams.update(
        {
            "font.size": 8.5,
            "axes.edgecolor": GRAY,
            "axes.linewidth": 0.6,
            "axes.labelcolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "text.color": INK,
            "axes.grid": True,
            "grid.color": "#dddddd",
            "grid.linewidth": 0.5,
            "legend.frameon": False,
            "pdf.fonttype": 42,
        }
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.6, 2.9), constrained_layout=True)

    # --- (a) spliced series with local-linear fits at the cutoff -----------
    old = panel[panel["new_wording"] == 0]
    new = panel[panel["new_wording"] == 1]
    ax1.axvspan(GAP_LO, GAP_HI, color="#eeeeee", zorder=0)
    ax1.axvline(T_C, color=VERMILLION, lw=0.9, zorder=1)
    ax1.plot(
        old["t"], old["ai_use"], color=BLUE, lw=0.9, marker="o", ms=2.2,
        label="Old wording (producing goods or services)",
    )
    ax1.plot(
        new["t"], new["ai_use"], color=ORANGE, lw=0.9, marker="o", ms=2.2,
        label="New wording (any business function)",
    )
    xs_pre = np.linspace(T_C - H, GAP_LO, 40)
    ax1.plot(xs_pre, beta_pre[0] + beta_pre[1] * (xs_pre - T_C), color=INK, lw=1.1)
    xs_gap = np.linspace(GAP_LO, T_C, 10)
    ax1.plot(xs_gap, beta_pre[0] + beta_pre[1] * (xs_gap - T_C), color=INK, lw=1.1, ls="--")
    xs_post = np.linspace(T_C, T_C + 0.65, 40)
    ax1.plot(xs_post, beta_post[0] + beta_post[1] * (xs_post - T_C), color=INK, lw=1.1)
    ax1.annotate(
        "", xy=(T_C, beta_post[0]), xytext=(T_C, beta_pre[0]),
        arrowprops=dict(arrowstyle="<->", color=VERMILLION, lw=1.2),
    )
    ax1.text(T_C - 0.06, 13.4, "+6.04 pp", color=VERMILLION, fontsize=8, ha="right")
    ax1.text(
        GAP_LO - 0.03, 0.8, "shutdown gap", rotation=90, va="bottom", ha="right",
        color=GRAY, fontsize=6.5,
    )
    ax1.set_xlabel("Reference-period midpoint (year)")
    ax1.set_ylabel("Share using AI (%)")
    ax1.set_ylim(0, 23.5)
    ax1.legend(
        loc="upper left", fontsize=6.6, handlelength=1.4,
        frameon=True, facecolor="white", edgecolor="none", framealpha=1.0,
    )
    ax1.set_title("(a) BTOS AI use across the rewording", fontsize=9, loc="left")

    # --- (b) placebo-cutoff grid vs. the true cutoff ------------------------
    grid = res["placebo"]["grid"]
    cs = np.array([g["c"] for g in grid])
    taus = np.array([g["tau"] for g in grid])
    ax2.axhline(0.0, color=INK, lw=0.7)
    ax2.plot(
        cs, taus, ls="none", marker="o", ms=3.6, mfc="white", mec=GRAY, mew=1.0,
        label="49 placebo cutoffs",
    )
    lo = RECORD_TAU - 1.96 * RECORD_SE_HAC
    hi = RECORD_TAU + 1.96 * RECORD_SE_HAC
    ax2.plot([T_C, T_C], [lo, hi], color=VERMILLION, lw=1.2)
    ax2.plot(
        [T_C], [RECORD_TAU], ls="none", marker="o", ms=6, mfc=VERMILLION,
        mec=VERMILLION, label="True cutoff (95% HAC CI)",
    )
    ax2.text(
        2023.95, 5.1,
        "max |placebo| = 1.30 pp\n0 of 49 $\\geq$ true jump\nrandomization p = 0.02",
        fontsize=6.8, color=GRAY, va="top",
    )
    ax2.set_xlabel("Cutoff placed at (year)")
    ax2.set_ylabel("RD jump (pp)")
    ax2.set_ylim(-2.2, 7.6)
    ax2.legend(loc="upper left", fontsize=6.6, handlelength=1.2)
    ax2.set_title("(b) Placebo-cutoff calibration", fontsize=9, loc="left")

    out = Path(__file__).resolve().parent
    fig.savefig(out / "fig1.pdf")
    fig.savefig(out / "fig1.png", dpi=200)
    print(f"wrote {out / 'fig1.pdf'} and fig1.png")


if __name__ == "__main__":
    main()
