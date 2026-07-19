"""Regenerate fig1 (PDF + PNG) for the aei-btos-state-gradient mini-paper.

Reads the frozen analysis outputs of the run of record
(``/Users/haukehillebrandt/dev/study-runs/aei-btos-state-gradient``, natex
v0.2.0, seed 0) — pass a different location with ``--results-root``. The
underlying Census BTOS workbooks and the AEI state release are NOT committed
to this repository; only the small derived outputs written by the analysis
pass are read here:

- ``out/state_slopes_w0.5.csv``    per-state pre/post slopes and slope change
                                   at R1 (w=0.5, 36 states with >=6 waves/side)
- ``out/gradient_results.json``    the gradient battery of record
- ``out/placebo_grid.csv``         gradient estimates at placebo cutoffs

The headline gradient and the significant-placebo numbers are asserted
against the numbers of record before a figure is written, so a drifted
results vintage fails loudly instead of silently redrawing different numbers.

Deterministic: pure file reads, no RNG, and ``SOURCE_DATE_EPOCH`` is pinned
so the PDF/PNG bytes are reproducible.

Run (from the repo root):

    uv run python papers/aei-btos-state-gradient/figures/make_fig.py \
        --results-root /Users/haukehillebrandt/dev/study-runs/aei-btos-state-gradient
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
BLUE = "#0072B2"       # states (AI-adoption outcome)
ORANGE = "#E69F00"     # remote-work placebo outcome
VERMILLION = "#D55E00" # R1 headline gradient / fit line
GRAY = "#666666"       # placebo cutoffs
INK = "#222222"

# Numbers of record (papers/aei-btos-state-gradient/main.tex, analysis 2026-07).
RECORD_B = 1.634873636424917       # +1.635 pp/yr per AEI index unit
RECORD_SE_HC1 = 1.4583286630597463
RECORD_P_PERM = 0.2565743425657434
RECORD_Q6W_B = 4.3917490043081004  # the significant PLACEBO (+4.39, p 0.045)
RECORD_Q6W_P = 0.0451954804519548
ATOL = 5e-6


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--results-root",
        type=Path,
        default=Path("/Users/haukehillebrandt/dev/study-runs/aei-btos-state-gradient"),
    )
    ap.add_argument("--skip-checks", action="store_true")
    args = ap.parse_args()

    out_dir = args.results_root / "out"
    slopes = pd.read_csv(out_dir / "state_slopes_w0.5.csv")
    grad = json.loads((out_dir / "gradient_results.json").read_text())
    placebo = pd.read_csv(out_dir / "placebo_grid.csv")

    head = grad["confirm_w0.5_unweighted"]
    q6w = grad["q6_placebo_w0.5_weighted"]
    q6 = grad["q6_placebo_w0.5"]

    # Recompute the headline OLS gradient from the per-state slope file.
    n = len(slopes)
    x = slopes["aei"].to_numpy(float)
    y = slopes["dslope"].to_numpy(float)
    design = np.column_stack([np.ones(n), x])
    coef = np.linalg.lstsq(design, y, rcond=None)[0]

    checks = [
        ("headline b (json)", head["b"], RECORD_B),
        ("headline se_hc1", head["se_hc1"], RECORD_SE_HC1),
        ("headline p_perm", head["p_perm"], RECORD_P_PERM),
        ("headline b (recomputed)", coef[1], RECORD_B),
        ("q6 weighted b", q6w["b"], RECORD_Q6W_B),
        ("q6 weighted p_perm", q6w["p_perm"], RECORD_Q6W_P),
        ("n states", float(n), 36.0),
    ]
    ok = True
    for name, got, want in checks:
        good = bool(np.isclose(got, want, atol=ATOL))
        ok &= good
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

    # --- (a) slope change vs AEI exposure at R1 ---------------------------
    ax1.scatter(x, y, s=18, facecolors="white", edgecolors=BLUE, linewidths=1.1, zorder=3)
    xs = np.linspace(x.min() - 0.03, x.max() + 0.03, 50)
    ax1.plot(xs, coef[0] + coef[1] * xs, color=VERMILLION, lw=1.4, zorder=2)
    for st in ("CA", "CO"):
        row = slopes[slopes["state"] == st]
        if len(row) == 1:
            ax1.annotate(
                st, (float(row["aei"].iloc[0]), float(row["dslope"].iloc[0])),
                textcoords="offset points", xytext=(4, 3), fontsize=6.8, color=GRAY,
            )
    ax1.text(
        0.97, 0.03,
        "b = +1.63 pp/yr per index unit\npermutation p = 0.257 (B = 10,000)",
        transform=ax1.transAxes, fontsize=6.8, color=GRAY, va="bottom", ha="right",
    )
    ax1.set_xlabel("AEI usage-per-capita index (Apr–Jun 2026)")
    ax1.set_ylabel("Slope change at R1 (pp/yr)")
    ax1.set_title("(a) State slope change at R1 vs. AEI exposure", fontsize=9, loc="left")

    # --- (b) the same gradient across cutoffs and outcomes ----------------
    pg = placebo.set_index("cutoff")
    items = [
        # (label, b, se_hc1, p_perm, color)
        ("2024.30", *(float(pg.loc[2024.3, c]) for c in ("b_unw", "se_unw", "p_perm_unw")), GRAY),
        ("2024.50", *(float(pg.loc[2024.5, c]) for c in ("b_unw", "se_unw", "p_perm_unw")), GRAY),
        (
            "o1\n2024.70",
            *(float(pg.loc[2024.6967, c]) for c in ("b_unw", "se_unw", "p_perm_unw")),
            GRAY,
        ),
        ("R1\n2025.055", head["b"], head["se_hc1"], head["p_perm"], VERMILLION),
        ("Q6\nat R1", q6["b"], q6["se_hc1"], q6["p_perm"], ORANGE),
        ("Q6 wtd\nat R1", q6w["b"], q6w["se_hc1"], q6w["p_perm"], ORANGE),
    ]
    xs2 = np.arange(len(items))
    for xi, (label, b, se, p, color) in zip(xs2, items):
        lo, hi = b - 1.96 * se, b + 1.96 * se
        sig = p < 0.05
        ax2.plot([xi, xi], [lo, hi], color=color, lw=1.2)
        ax2.plot(
            [xi], [b],
            marker="o", ms=6, mfc=color if sig else "white",
            mec=color, mew=1.2, color=color,
        )
    ax2.axhline(0.0, color=INK, lw=0.7)
    ax2.set_xticks(xs2)
    ax2.set_xticklabels([it[0] for it in items], fontsize=6.8)
    ax2.set_ylabel("Gradient (pp/yr per index unit)")
    ax2.set_ylim(-4.7, 9.7)
    ax2.set_title("(b) Gradient at R1 vs. placebos", fontsize=9, loc="left")
    ax2.text(
        0.03, 0.97,
        "filled = permutation p < 0.05\nbars: 95% HC1 intervals",
        transform=ax2.transAxes, fontsize=6.8, color=GRAY, va="top",
    )

    out = Path(__file__).resolve().parent
    fig.savefig(out / "fig1.pdf")
    fig.savefig(out / "fig1.png", dpi=200)
    print(f"wrote {out / 'fig1.pdf'} and fig1.png")


if __name__ == "__main__":
    main()
