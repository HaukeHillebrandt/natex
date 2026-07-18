"""Regenerate fig1 (PDF + PNG) for the btos-sector-did-r1 mini-paper.

Reads the frozen analysis outputs of the run of record
(``/Users/haukehillebrandt/dev/study-runs/btos-sector-did-r1``, natex v0.2.0,
seed 0) — pass a different location with ``--results-root``. The underlying
Census BTOS workbooks are NOT committed to this repository; only the two small
derived CSVs written by the analysis pass are read here:

- ``group_means_by_wave.csv``  exposed / unexposed mean AI-use share per wave
- ``kink_grid.csv``            the difference-in-kinks estimate grid

The primary estimate is asserted against the numbers of record before a figure
is written, so a drifted results vintage fails loudly instead of silently
redrawing different numbers.

Deterministic: pure file reads, no RNG, and ``SOURCE_DATE_EPOCH`` is pinned so
the PDF/PNG bytes are reproducible.

Run (from the repo root):

    uv run python papers/btos-sector-did-r1/figures/make_fig.py \
        --results-root /Users/haukehillebrandt/dev/study-runs/btos-sector-did-r1
"""

from __future__ import annotations

import argparse
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
BLUE = "#0072B2"       # exposed group
ORANGE = "#E69F00"     # unexposed group
VERMILLION = "#D55E00" # R1 cutoff / highlighted estimate
GRAY = "#666666"       # placebo cutoffs
INK = "#222222"

R1 = 2025.055    # DeepSeek-R1 2025-01-20 (first post wave 202503)
O1 = 2024.6967   # OpenAI o1 2024-09-12

# Numbers of record (papers/btos-sector-did-r1/main.tex, analysis 2026-07).
RECORD_TAU, RECORD_SE = 7.296850157361588, 2.7966906576734076
ATOL = 5e-6

# Rows of kink_grid.csv drawn in panel (b): (run key, label, is R1, is o1)
GRID_ROWS = [
    ("kink_r1_plc_2024p30", "2024.30", False, False),
    ("kink_r1_plc_2024p50", "2024.50", False, False),
    ("kink_r1_plc_o1_bw05", "o1\n2024.70", False, True),
    ("kink_r1_plc_2024p85", "2024.85", False, False),
    ("kink_r1_bw05_tri_donut", "R1\n2025.055", True, False),
    ("kink_r1_plc_2025p30", "2025.30", False, False),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--results-root",
        type=Path,
        default=Path("/Users/haukehillebrandt/dev/study-runs/btos-sector-did-r1"),
    )
    ap.add_argument("--skip-checks", action="store_true")
    args = ap.parse_args()

    means = pd.read_csv(args.results_root / "group_means_by_wave.csv")
    grid = pd.read_csv(args.results_root / "kink_grid.csv", comment="#")
    grid = grid.set_index("run")

    tau = float(grid.loc["kink_r1_bw05_tri_donut", "tau"])
    se = float(grid.loc["kink_r1_bw05_tri_donut", "se"])
    ok = np.isclose(tau, RECORD_TAU, atol=ATOL) and np.isclose(se, RECORD_SE, atol=ATOL)
    print(f"[primary DiK] tau={tau:+.6f} se={se:.6f} {'OK' if ok else 'MISMATCH'}")
    if not ok and not args.skip_checks:
        sys.exit(
            "primary estimate does not match the numbers of record — wrong "
            "results vintage? (--skip-checks to override)"
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

    # --- (a) group means by wave ------------------------------------------
    ax1.plot(means["t"], means["exp_mean"], color=BLUE, lw=1.4, label="AI-exposed (51, 52, 54)")
    ax1.plot(
        means["t"], means["unexp_mean"], color=ORANGE, lw=1.4,
        label="Unexposed (11, 21, 23, 72)",
    )
    ax1.axvline(R1, color=VERMILLION, lw=0.9, ls="-", zorder=0)
    ax1.axvline(O1, color=GRAY, lw=0.9, ls="--", zorder=0)
    ax1.text(R1 + 0.02, 26.2, "R1", color=VERMILLION, fontsize=8)
    ax1.text(O1 + 0.02, 18.0, "o1", color=GRAY, fontsize=8)
    ax1.set_xlabel("Collection-period midpoint (year)")
    ax1.set_ylabel("Share using AI (%)")
    ax1.set_ylim(0, 28)
    ax1.legend(
        loc="upper left", fontsize=7.5, handlelength=1.4,
        frameon=True, facecolor="white", edgecolor="none", framealpha=1.0,
    )
    ax1.set_title("(a) BTOS AI use by exposure group", fontsize=9, loc="left")

    # --- (b) DiK estimates across cutoffs ---------------------------------
    xs = np.arange(len(GRID_ROWS))
    for x, (run, label, is_r1, is_o1) in zip(xs, GRID_ROWS):
        row = grid.loc[run]
        color = VERMILLION if is_r1 else GRAY
        sig = bool(row["sig_5pct"] is True or str(row["sig_5pct"]) == "True")
        lo, hi = float(row["ci_lo"]), float(row["ci_hi"])
        ax2.plot([x, x], [lo, hi], color=color, lw=1.2)
        ax2.plot(
            [x], [float(row["tau"])],
            marker="o", ms=6, mfc=color if sig else "white",
            mec=color, mew=1.2, color=color,
        )
    ax2.axhline(0.0, color=INK, lw=0.7)
    ax2.set_xticks(xs)
    ax2.set_xticklabels([r[1] for r in GRID_ROWS], fontsize=6.8)
    ax2.set_ylabel("DiK slope change (pp/yr)")
    ax2.set_ylim(-16.5, 22.5)
    ax2.set_title("(b) DiK at R1 vs. placebo cutoffs", fontsize=9, loc="left")
    ax2.text(
        0.24, 0.97,
        "filled = significant at 5% (CR1, 7 clusters)\nbw 0.5 yr, triangular, donut 1 wave",
        transform=ax2.transAxes, fontsize=6.8, color=GRAY, va="top",
    )

    out = Path(__file__).resolve().parent
    fig.savefig(out / "fig1.pdf")
    fig.savefig(out / "fig1.png", dpi=200)
    print(f"wrote {out / 'fig1.pdf'} and fig1.png")


if __name__ == "__main__":
    main()
