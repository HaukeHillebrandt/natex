"""Regenerate fig1 (PDF + PNG) for the btos-spliced-r1-extension mini-paper.

Reads the frozen analysis outputs of the run of record
(``/Users/haukehillebrandt/dev/study-runs/btos-spliced-r1-extension``, natex
v0.2.0, deterministic — the kink CLI has no RNG) — pass a different location
with ``--results-root``. The underlying Census BTOS workbooks are NOT committed
to this repository; only the two small derived CSVs written by the analysis
pass are read here:

- ``group_gap_by_wave.csv``  spliced exposed / unexposed mean AI-use share per wave
- ``kink_grid.csv``          the difference-in-kinks estimate grid

The headline estimates are asserted against the numbers of record before a
figure is written, so a drifted results vintage fails loudly instead of
silently redrawing different numbers.

Deterministic: pure file reads, no RNG, and ``SOURCE_DATE_EPOCH`` is pinned so
the PDF/PNG bytes are reproducible.

Run (from the repo root):

    uv run python papers/btos-spliced-r1-extension/figures/make_fig.py \
        --results-root /Users/haukehillebrandt/dev/study-runs/btos-spliced-r1-extension
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
VERMILLION = "#D55E00" # R1 cutoff / highlighted estimates
GRAY = "#666666"       # placebo cutoffs
INK = "#222222"

R1 = 2025.055        # DeepSeek-R1 2025-01-20 (first post wave 202503)
O1 = 2024.6967       # OpenAI o1 2024-09-12
SPLICE = 2025.845    # first new-wording wave 202524 (rewording cutoff)
GAP_LO, GAP_HI = 2025.78, 2025.92  # federal-shutdown no-data gap (202521-23)

# Numbers of record (papers/btos-spliced-r1-extension/main.tex, analysis 2026-07).
RECORDS = {  # (panel, cutoff, bw) -> (tau, se)
    ("nat", 2025.055, 1.0): (-0.1646094497022803, 3.4366293643223282),
    ("nat", 2025.055, 0.5): (7.271369545612325, 2.814520435219821),
    ("nat", 2025.6, 1.0): (-9.082122157688243, 2.489597503644073),
}
ATOL = 5e-6
N_WAVES = 71  # 54 old-wording + 17 new-wording waves

# Rows of kink_grid.csv drawn in panel (b): (panel, cutoff, bw, label, is R1)
GRID_ROWS = [
    ("nat", 2024.3, 1.0, "2024.30", False),
    ("nat", 2024.5, 1.0, "2024.50", False),
    ("nat", 2024.6967, 1.0, "o1 2024.70", False),
    ("nat", 2024.85, 1.0, "2024.85", False),
    ("nat", 2025.055, 0.5, "R1 repl. bw 0.5", True),
    ("nat", 2025.055, 1.0, "R1 ext. bw 1.0", True),
    ("nat", 2025.055, 1.5, "R1 ext. bw 1.5", True),
    ("nat", 2025.3, 1.0, "2025.30", False),
    ("nat", 2025.6, 1.0, "2025.60", False),
    ("nat", 2025.845, 1.0, "splice 2025.845", False),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--results-root",
        type=Path,
        default=Path("/Users/haukehillebrandt/dev/study-runs/btos-spliced-r1-extension"),
    )
    ap.add_argument("--skip-checks", action="store_true")
    args = ap.parse_args()

    means = pd.read_csv(args.results_root / "group_gap_by_wave.csv")
    grid = pd.read_csv(args.results_root / "kink_grid.csv")
    grid = grid.set_index(["panel", "cutoff", "bw"])

    ok = len(means) == N_WAVES
    print(f"[panel] {len(means)} waves {'OK' if ok else 'MISMATCH'}")
    for key, (rtau, rse) in RECORDS.items():
        tau, se = float(grid.loc[key, "tau"]), float(grid.loc[key, "se"])
        match = np.isclose(tau, rtau, atol=ATOL) and np.isclose(se, rse, atol=ATOL)
        ok = ok and match
        print(f"[{key}] tau={tau:+.6f} se={se:.6f} {'OK' if match else 'MISMATCH'}")
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

    # --- (a) spliced group means by wave, line broken at the no-data gap ----
    old = means[means["t"] < GAP_LO]
    new = means[means["t"] > GAP_HI]
    for seg, lab_exp, lab_unexp in ((old, "AI-exposed (51, 52, 54)", "Unexposed (11, 21, 23, 72)"),
                                    (new, None, None)):
        ax1.plot(seg["t"], seg["outcome_exposed"], color=BLUE, lw=1.4, label=lab_exp)
        ax1.plot(seg["t"], seg["outcome_unexposed"], color=ORANGE, lw=1.4, label=lab_unexp)
    ax1.axvspan(GAP_LO, GAP_HI, color="#eeeeee", zorder=0)
    ax1.axvline(R1, color=VERMILLION, lw=0.9, ls="-", zorder=0)
    ax1.axvline(O1, color=GRAY, lw=0.9, ls="--", zorder=0)
    ax1.axvline(SPLICE, color=GRAY, lw=0.9, ls=":", zorder=0)
    ax1.text(R1 + 0.03, 10.8, "R1", color=VERMILLION, fontsize=8)
    ax1.text(O1 + 0.03, 21.5, "o1", color=GRAY, fontsize=8)
    ax1.text(SPLICE - 0.30, 8.6, "new wording\n$\\div$ 1.553", color=GRAY, fontsize=6.4)
    ax1.set_xlabel("Collection-period midpoint (year)")
    ax1.set_ylabel("Share using AI (%, spliced)")
    ax1.set_ylim(0, 29)
    ax1.tick_params(labelsize=7.5)
    ax1.legend(
        loc="upper left", fontsize=7.5, handlelength=1.4,
        frameon=True, facecolor="white", edgecolor="none", framealpha=1.0,
    )
    ax1.set_title("(a) Spliced BTOS AI use by exposure group", fontsize=9, loc="left")

    # --- (b) DiK estimates across cutoffs and bandwidths -------------------
    xs = np.arange(len(GRID_ROWS))
    for x, (panel, cutoff, bw, label, is_r1) in zip(xs, GRID_ROWS):
        row = grid.loc[(panel, cutoff, bw)]
        color = VERMILLION if is_r1 else GRAY
        sig = float(row["p"]) < 0.05
        lo, hi = float(row["ci_lo"]), float(row["ci_hi"])
        ax2.plot([x, x], [lo, hi], color=color, lw=1.2)
        ax2.plot(
            [x], [float(row["tau"])],
            marker="o", ms=6, mfc=color if sig else "white",
            mec=color, mew=1.2, color=color,
        )
    ax2.axhline(0.0, color=INK, lw=0.7)
    ax2.set_xticks(xs)
    ax2.set_xticklabels(
        [r[3] for r in GRID_ROWS], fontsize=6.4, rotation=45, ha="right",
        rotation_mode="anchor",
    )
    ax2.tick_params(axis="y", labelsize=7.5)
    ax2.set_ylabel("DiK slope change (pp/yr)")
    ax2.set_ylim(-17.5, 17.0)
    ax2.set_title("(b) DiK at R1 vs. placebo cutoffs", fontsize=9, loc="left")
    ax2.text(
        0.02, 0.97,
        "filled = significant at 5% (CR1, 6-7 clusters)\nnational splice; "
        "triangular, donut 1 wave",
        transform=ax2.transAxes, fontsize=6.4, color=GRAY, va="top",
    )

    out = Path(__file__).resolve().parent
    fig.savefig(out / "fig1.pdf")
    fig.savefig(out / "fig1.png", dpi=200)
    print(f"wrote {out / 'fig1.pdf'} and fig1.png")


if __name__ == "__main__":
    main()
