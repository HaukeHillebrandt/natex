"""Regenerate fig1 (PDF + PNG) for the prop99-validation-writeup mini-paper.

Reads the frozen derived outputs of the analysis pass of record
(``/Users/haukehillebrandt/dev/study-runs/prop99-validation-writeup``, written
by ``prep_fig_data.py`` in that directory from the deterministic natex
synthetic-control donor path; natex v0.2.0, no rng anywhere in the path) —
pass a different location with ``--results-root``. The underlying ADH smoking
panel is NOT committed to this repository; only three small derived CSVs are
read here:

- ``trajectory.csv``      year, California, synthetic California (full pool), gap
- ``weights.csv``         full-pool simplex donor weights
- ``placebo_ratios.csv``  post/pre RMSPE ratio for all 39 states (0 skipped)

The headline numbers are asserted against the numbers of record before a
figure is written, so a drifted results vintage fails loudly instead of
silently redrawing different numbers.

Deterministic: pure file reads, no RNG, and ``SOURCE_DATE_EPOCH`` is pinned so
the PDF/PNG bytes are reproducible.

Run (from the repo root):

    uv run python papers/prop99-validation-writeup/figures/make_fig.py \
        --results-root /Users/haukehillebrandt/dev/study-runs/prop99-validation-writeup
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
BLUE = "#0072B2"        # California (treated unit)
VERMILLION = "#D55E00"  # synthetic California (counterfactual)
GRAY = "#666666"        # placebo cloud / annotations
INK = "#222222"

T0 = 1989

# Numbers of record (tests/backtests/test_prop99_donors.py, run of record
# 2026-07-11, re-verified 2026-07-18).
RECORD_ATT = -19.514
RECORD_PRE_RMSPE = 1.656
RECORD_POST_RMSPE_MIN = 10.0
RECORD_ADH5_WEIGHT = 0.955
RECORD_RATIO = 12.440
RECORD_RANK = 3  # of 39
ADH_FIVE = frozenset({"Colorado", "Connecticut", "Montana", "Nevada", "Utah"})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--results-root",
        type=Path,
        default=Path("/Users/haukehillebrandt/dev/study-runs/prop99-validation-writeup"),
    )
    ap.add_argument("--skip-checks", action="store_true")
    args = ap.parse_args()

    traj = pd.read_csv(args.results_root / "trajectory.csv")
    wts = pd.read_csv(args.results_root / "weights.csv")
    plc = pd.read_csv(args.results_root / "placebo_ratios.csv")

    post = traj["year"] >= T0
    pre = ~post
    att = float(traj.loc[post, "gap"].mean())
    pre_rmspe = float(np.sqrt(np.mean(traj.loc[pre, "gap"] ** 2)))
    post_rmspe = float(np.sqrt(np.mean(traj.loc[post, "gap"] ** 2)))
    adh_weight = float(wts.loc[wts["donor"].isin(ADH_FIVE), "weight"].sum())
    treated = plc[plc["is_treated"]]
    ratio_treated = float(treated["ratio"].iloc[0])
    rank = int((plc["ratio"] >= ratio_treated).sum())  # California counts itself
    checks = [
        (len(traj) == 31 and len(plc) == 39, f"panel shape {len(traj)}y/{len(plc)}u"),
        (np.isclose(att, RECORD_ATT, atol=5e-4), f"ATT {att:.4f}"),
        (np.isclose(pre_rmspe, RECORD_PRE_RMSPE, atol=5e-4), f"pre-RMSPE {pre_rmspe:.4f}"),
        (post_rmspe > RECORD_POST_RMSPE_MIN, f"post-RMSPE {post_rmspe:.4f}"),
        (np.isclose(adh_weight, RECORD_ADH5_WEIGHT, atol=5e-4), f"ADH-five weight {adh_weight:.4f}"),
        (np.isclose(ratio_treated, RECORD_RATIO, atol=5e-4), f"treated ratio {ratio_treated:.4f}"),
        (rank == RECORD_RANK, f"placebo rank {rank}/39"),
    ]
    for ok, msg in checks:
        print(f"[{'OK' if ok else 'MISMATCH'}] {msg}")
    if not all(ok for ok, _ in checks) and not args.skip_checks:
        sys.exit(
            "figure inputs do not match the numbers of record — wrong results "
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

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.6, 2.6), constrained_layout=True)

    # --- (a) California vs synthetic California ---------------------------
    ax1.plot(traj["year"], traj["california"], color=BLUE, lw=1.5, label="California")
    ax1.plot(
        traj["year"], traj["synthetic"], color=VERMILLION, lw=1.5, ls="--",
        label="synthetic California",
    )
    ax1.axvline(T0, color=GRAY, lw=0.9, ls=":", zorder=0)
    ax1.text(T0 - 0.6, 137, "Prop 99\n(1989)", color=GRAY, fontsize=6.8, ha="right")
    ax1.annotate(
        "ATT 1989–2000:\n$-19.5$ packs",
        xy=(1996, 71), xytext=(1988.3, 52), fontsize=6.8, color=INK,
        arrowprops={"arrowstyle": "-", "color": GRAY, "lw": 0.7},
    )
    ax1.text(
        1970.7, 43,
        "donor weights: UT .39, MT .23,\nNV .20, CT .11 (ADH five: .955)",
        fontsize=6.2, color=GRAY, va="bottom",
    )
    ax1.set_xlabel("Year")
    ax1.set_ylabel("Cigarette sales (packs per capita)")
    ax1.set_ylim(35, 145)
    ax1.legend(loc="upper right", fontsize=6.8, handlelength=1.6,
               labelspacing=0.3, borderaxespad=0.2)
    ax1.set_title("(a) Trajectories (full-pool simplex fit)", fontsize=8.5, loc="left")

    # --- (b) in-space placebo: post/pre RMSPE ratios ----------------------
    plc_sorted = plc.sort_values("ratio", ascending=True).reset_index(drop=True)
    ypos = np.arange(1, len(plc_sorted) + 1)  # rank 39 (smallest) at bottom
    is_ca = plc_sorted["is_treated"].to_numpy()
    ax2.hlines(ypos, 0, plc_sorted["ratio"], color="#cccccc", lw=0.7, zorder=1)
    ax2.plot(
        plc_sorted.loc[~is_ca, "ratio"], ypos[~is_ca],
        ls="none", marker="o", ms=3.2, mfc=GRAY, mec="none", alpha=0.75,
        label="placebo state (38)",
    )
    ax2.plot(
        plc_sorted.loc[is_ca, "ratio"], ypos[is_ca],
        ls="none", marker="o", ms=6.0, mfc=BLUE, mec=INK, mew=0.6,
        label="California",
    )
    for name in ("Missouri", "Virginia"):
        row = plc_sorted[plc_sorted["unit"] == name]
        ax2.text(
            float(row["ratio"].iloc[0]) - 0.4, float(ypos[row.index[0]]),
            name, fontsize=6.2, color=GRAY, ha="right", va="center",
        )
    ca_row = plc_sorted[is_ca]
    ax2.annotate(
        "California: ratio 12.44,\nrank 3/39 ($p = 0.077$)",
        xy=(float(ca_row["ratio"].iloc[0]), float(ypos[ca_row.index[0]])),
        xytext=(13.4, 27.5), fontsize=6.8, color=BLUE,
        arrowprops={"arrowstyle": "-", "color": BLUE, "lw": 0.7},
    )
    ax2.set_xlabel("Post/pre RMSPE ratio")
    ax2.set_ylabel("State (ranked by ratio)")
    ax2.set_yticks([1, 10, 20, 30, 39])
    ax2.set_yticklabels(["39", "30", "20", "10", "1"])  # rank 1 = most extreme, top
    ax2.set_xlim(0, 25.5)
    ax2.legend(loc="lower right", fontsize=6.8, handlelength=1.1,
               labelspacing=0.3, borderaxespad=0.2)
    ax2.set_title("(b) Placebo ratios (all 38 usable, 0 skipped)",
                  fontsize=8.5, loc="left")

    out = Path(__file__).resolve().parent
    fig.savefig(out / "fig1.pdf")
    fig.savefig(out / "fig1.png", dpi=200)
    print(f"wrote {out / 'fig1.pdf'} and fig1.png")


if __name__ == "__main__":
    main()
