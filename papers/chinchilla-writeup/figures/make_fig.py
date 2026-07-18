"""Regenerate fig1 (PDF + PNG) for the chinchilla-writeup mini-paper.

Reads the frozen derived outputs of the analysis pass of record
(``/Users/haukehillebrandt/dev/study-runs/chinchilla-writeup``, written by
``prep_fig_data.py`` in that directory from the natex LoRD3 run of record in
``/Users/haukehillebrandt/dev/epoch-data/natex-runs/chinchilla``; natex
v0.2.0, seed 0) — pass a different location with ``--results-root``. The
underlying Epoch AI models panel is NOT committed to this repository; only
three small derived CSVs are read here:

- ``scatter.csv``         date, log10 tokens/param, fine-tune-artifact flag
- ``monthly_median.csv``  monthly median of log10 tokens/param
- ``llr_profile.csv``     LoRD3 LLR per candidate center (k=50, degree 1)

The headline numbers are asserted against the numbers of record before a
figure is written, so a drifted results vintage fails loudly instead of
silently redrawing different numbers.

Deterministic: pure file reads, no RNG, and ``SOURCE_DATE_EPOCH`` is pinned so
the PDF/PNG bytes are reproducible.

Run (from the repo root):

    uv run python papers/chinchilla-writeup/figures/make_fig.py \
        --results-root /Users/haukehillebrandt/dev/study-runs/chinchilla-writeup
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("SOURCE_DATE_EPOCH", "0")  # reproducible figure bytes

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Okabe-Ito, colorblind-safe; hues assigned in fixed order by entity.
BLUE = "#0072B2"        # the known truth (Chinchilla, 2022-03-29) + median line
VERMILLION = "#D55E00"  # the discovered break / artifact entity
ORANGE = "#E69F00"      # secondary discovered cluster (over-training era)
GRAY = "#666666"        # data cloud / profile
INK = "#222222"

TRUTH = pd.Timestamp("2022-03-29")       # Hoffmann et al. arXiv posting
DISCOVERED = pd.Timestamp("2022-09-22")  # top LoRD3 center, day 1360

# Numbers of record (papers/chinchilla-writeup/main.tex, analysis 2026-07).
RECORD_TOP_LLR = 8.687416391870288
RECORD_TRUTH_BEST_LLR = 1.39  # 2022-04-15, atol 0.01
RECORD_N_MODELS = 549
RECORD_N_CANDIDATES = 547
RECORD_N_ARTIFACT = 7
RECORD_MEDIAN_PRE, RECORD_MEDIAN_POST = 1.69, 29.17  # tokens/param, n=185/364


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--results-root",
        type=Path,
        default=Path("/Users/haukehillebrandt/dev/study-runs/chinchilla-writeup"),
    )
    ap.add_argument("--skip-checks", action="store_true")
    args = ap.parse_args()

    scat = pd.read_csv(args.results_root / "scatter.csv", parse_dates=["date"])
    mm = pd.read_csv(args.results_root / "monthly_median.csv", parse_dates=["date"])
    prof = pd.read_csv(args.results_root / "llr_profile.csv", parse_dates=["center_date"])

    top = prof.loc[prof["llr"].idxmax()]
    tp = 10 ** scat["log10_tokens_per_param"]
    med_pre = float(tp[scat["date"] < TRUTH].median())
    med_post = float(tp[scat["date"] >= TRUTH].median())
    win = prof[(prof["center_date"] - TRUTH).abs() <= pd.Timedelta(days=45)]
    truth_best = win.loc[win["llr"].idxmax()]
    checks = [
        (len(scat) == RECORD_N_MODELS, f"n models {len(scat)}"),
        (len(prof) == RECORD_N_CANDIDATES, f"n candidates {len(prof)}"),
        (int(scat["is_finetune_artifact"].sum()) == RECORD_N_ARTIFACT, "artifact rows"),
        (np.isclose(top["llr"], RECORD_TOP_LLR, atol=1e-6), f"top LLR {top['llr']}"),
        (top["center_date"] == DISCOVERED, f"top center {top['center_date']}"),
        (np.isclose(truth_best["llr"], RECORD_TRUTH_BEST_LLR, atol=0.01),
         f"truth-window best LLR {truth_best['llr']}"),
        (np.isclose(med_pre, RECORD_MEDIAN_PRE, atol=0.01), f"median pre {med_pre}"),
        (np.isclose(med_post, RECORD_MEDIAN_POST, atol=0.01), f"median post {med_post}"),
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

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(6.6, 3.0), sharex=True, constrained_layout=True
    )

    # --- (a) tokens/param panel -------------------------------------------
    base = scat[~scat["is_finetune_artifact"]]
    art = scat[scat["is_finetune_artifact"]]
    ax1.plot(
        base["date"], base["log10_tokens_per_param"],
        ls="none", marker="o", ms=2.0, mfc=GRAY, mec="none", alpha=0.45,
        label="language model (n=549)",
    )
    ax1.plot(
        art["date"], art["log10_tokens_per_param"],
        ls="none", marker="v", ms=4.5, mfc="white", mec=VERMILLION, mew=1.1,
        label="fine-tune artifact row",
    )
    ax1.plot(
        mm["date"], mm["median_log10_tp"], color=BLUE, lw=1.4,
        label="monthly median",
    )
    ax1.axvline(TRUTH, color=BLUE, lw=0.9, ls="--", zorder=0)
    ax1.axvline(DISCOVERED, color=VERMILLION, lw=0.9, ls="-", zorder=0)
    ax1.text(TRUTH - pd.Timedelta(days=25), 3.0, "Chinchilla",
             color=BLUE, fontsize=6.8, ha="right")
    ax1.text(DISCOVERED + pd.Timedelta(days=25), 3.9, "top\ndiscovery",
             color=VERMILLION, fontsize=6.8)
    ax1.set_ylabel(r"$\log_{10}$ tokens per parameter")
    ax1.set_ylim(-5.9, 4.9)  # keep LMSI-PaLM (t/p 3.6e-06) in frame
    ax1.legend(loc="lower left", fontsize=6.4, handlelength=1.1,
               labelspacing=0.3, borderaxespad=0.2,
               frameon=True, facecolor="white", edgecolor="none", framealpha=0.9)
    ax1.set_title("(a) Tokens per parameter", fontsize=8.5, loc="left")

    # --- (b) LoRD3 LLR profile --------------------------------------------
    ax2.plot(
        prof["center_date"], prof["llr"],
        ls="none", marker="o", ms=2.2, mfc=GRAY, mec="none", alpha=0.55,
        label="candidate (547)",
    )
    ax2.axvline(TRUTH, color=BLUE, lw=0.9, ls="--", zorder=0)
    ax2.axvline(DISCOVERED, color=VERMILLION, lw=0.9, ls="-", zorder=0)
    ax2.plot([top["center_date"]], [top["llr"]], marker="o", ms=6.5,
             mfc=VERMILLION, mec=INK, mew=0.6, ls="none")
    ax2.plot([truth_best["center_date"]], [truth_best["llr"]], marker="o", ms=6.5,
             mfc="white", mec=BLUE, mew=1.3, ls="none")
    ax2.text(
        pd.Timestamp("2019-02-01"), 9.55,
        f"top discovery: LLR {top['llr']:.2f}, p $\\leq$ 0.01",
        fontsize=6.8, color=VERMILLION, va="top",
    )
    ax2.annotate(
        "best near truth:\nLLR 1.39,\nrank 158/547",
        xy=(truth_best["center_date"], truth_best["llr"]),
        xytext=(pd.Timestamp("2019-10-01"), 5.6), fontsize=6.8, color=BLUE,
        arrowprops={"arrowstyle": "-", "color": BLUE, "lw": 0.7},
    )
    ax2.annotate(
        "over-training-\nera onset",
        xy=(pd.Timestamp("2023-07-05"), 7.2),
        xytext=(pd.Timestamp("2023-12-15"), 7.9), fontsize=6.8, color=GRAY,
        arrowprops={"arrowstyle": "-", "color": GRAY, "lw": 0.7},
    )
    ax2.set_xlabel("Candidate break date")
    ax2.set_ylabel("LoRD3 LLR")
    ax2.set_ylim(0, 9.8)
    ax2.set_title("(b) Scan profile (k=50, degree 1)", fontsize=8.5, loc="left")
    for ax in (ax1, ax2):
        ax.xaxis.set_major_locator(mdates.YearLocator(2))
        ax.xaxis.set_minor_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax1.set_xlabel("Release date")

    out = Path(__file__).resolve().parent
    fig.savefig(out / "fig1.pdf")
    fig.savefig(out / "fig1.png", dpi=200)
    print(f"wrote {out / 'fig1.pdf'} and fig1.png")


if __name__ == "__main__":
    main()
