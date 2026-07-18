"""Regenerate fig1 (PDF + PNG) for the euact-bunching-writeup mini-paper.

Reads the frozen analysis outputs of the run of record
(``/Users/haukehillebrandt/dev/epoch-data/natex-runs/euact-bunching``,
analysis pass 2026-07, fully deterministic -- no RNG anywhere in the
pipeline) -- pass a different location with ``--results-root``. The Epoch AI
source dataset is NOT committed to this repository; only the two small files
written by the analysis pass are read here:

- ``compute.csv``          model, date, log10_compute (719 rows)
- ``density_results.json`` all density-test / Fisher / interaction numbers

The headline estimates are asserted against the numbers of record before a
figure is written, so a drifted results vintage fails loudly instead of
silently redrawing different numbers.

Deterministic: pure file reads, no RNG, and ``SOURCE_DATE_EPOCH`` is pinned so
the PDF/PNG bytes are reproducible.

Run (from the repo root):

    uv run python papers/euact-bunching-writeup/figures/make_fig.py \
        --results-root /Users/haukehillebrandt/dev/epoch-data/natex-runs/euact-bunching
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
from scipy import stats

# Okabe-Ito, colorblind-safe; color follows the entity (side of the statutory
# line) in every panel. Validated (dataviz six checks): CVD dE 91.9, all PASS.
BLUE = "#0072B2"        # mass BELOW the 1e25 line
VERMILLION = "#D55E00"  # mass AT/ABOVE the 1e25 line
INK = "#222222"
MUTED = "#666666"       # annotation text / event lines only, never a series

THR = 25.0
ACT = pd.Timestamp("2024-08-01")    # AI Act entry into force
GPAI = pd.Timestamp("2025-08-02")   # GPAI/systemic-risk obligations apply

# Numbers of record (papers/euact-bunching-writeup/main.tex, analysis 2026-07).
RECORD_FISHER_TABLE = [[13, 8], [47, 5]]
RECORD_FISHER_OR, RECORD_FISHER_P = 0.173, 0.00715
RECORD_THETA_POST, RECORD_P_POST = -1.334, 0.129
RECORD_EXPECTED_ABOVE, RECORD_DEFICIT_PCT = 28.9, 82.7
# below:above per half-year in the +-0.5-dex window (numbers of record)
RECORD_HALFYEARS = [
    ("2023H1", 2, 2), ("2023H2", 3, 1), ("2024H1", 6, 4), ("2024H2", 11, 3),
    ("2025H1", 17, 1), ("2025H2", 15, 1), ("2026H1", 6, 1),
]
ATOL = 5e-4


def halfyear_counts(df: pd.DataFrame) -> list[tuple[str, int, int]]:
    w = df[(df.log10_compute >= THR - 0.5) & (df.log10_compute <= THR + 0.5)].copy()
    key = w.date.dt.year.astype(str) + np.where(w.date.dt.month <= 6, "H1", "H2")
    out = []
    for hy in sorted(key.unique()):
        s = w[key == hy]
        out.append((hy, int((s.log10_compute < THR).sum()),
                    int((s.log10_compute >= THR).sum())))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--results-root",
        type=Path,
        default=Path("/Users/haukehillebrandt/dev/epoch-data/natex-runs/euact-bunching"),
    )
    ap.add_argument("--skip-checks", action="store_true")
    args = ap.parse_args()

    df = pd.read_csv(args.results_root / "compute.csv", parse_dates=["date"])
    res = json.loads((args.results_root / "density_results.json").read_text())

    pre = df[df.date < ACT]
    post = df[df.date >= ACT]

    # ---- assert the headline estimates against the numbers of record -------
    if not args.skip_checks:
        tab = []
        for s in (pre, post):
            w = s[(s.log10_compute >= THR - 0.5) & (s.log10_compute <= THR + 0.5)]
            tab.append([int((w.log10_compute < THR).sum()),
                        int((w.log10_compute >= THR).sum())])
        assert tab == RECORD_FISHER_TABLE, f"Fisher table drifted: {tab}"
        odds, p = stats.fisher_exact(tab, alternative="two-sided")
        assert abs(odds - RECORD_FISHER_OR) < ATOL, f"Fisher OR drifted: {odds}"
        assert abs(p - RECORD_FISHER_P) < ATOL, f"Fisher p drifted: {p}"
        dpost = res["primary"]["thr25.0_post_act"]
        assert abs(dpost["theta"] - RECORD_THETA_POST) < ATOL, dpost
        assert abs(dpost["p"] - RECORD_P_POST) < ATOL, dpost
        expected = tab[1][0] * (tab[0][1] / tab[0][0])
        assert abs(expected - RECORD_EXPECTED_ABOVE) < 0.05, expected
        assert abs((1 - tab[1][1] / expected) * 100 - RECORD_DEFICIT_PCT) < 0.05
        hy = halfyear_counts(df)
        assert hy == list(map(tuple, RECORD_HALFYEARS)), f"half-years drifted: {hy}"
        print("numbers-of-record checks: all passed", file=sys.stderr)

    # ---- draw --------------------------------------------------------------
    plt.rcParams.update({
        "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5,
        "axes.edgecolor": MUTED, "axes.linewidth": 0.7,
        "xtick.color": INK, "ytick.color": INK, "text.color": INK,
        "axes.labelcolor": INK, "figure.facecolor": "white",
    })
    fig = plt.figure(figsize=(9.2, 3.6))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.15, 1.0], hspace=0.14, wspace=0.26)
    ax_pre = fig.add_subplot(gs[0, 0])
    ax_post = fig.add_subplot(gs[1, 0], sharex=ax_pre, sharey=ax_pre)
    ax_t = fig.add_subplot(gs[:, 1])

    # (a) histograms of log10 compute, pre vs post, bars colored by side of 25.0
    bins = np.arange(23.0, 27.01, 0.25)
    for ax, sub, label in ((ax_pre, pre, "pre-Act (2023-01 .. 2024-07)"),
                           (ax_post, post, "post-Act (2024-08 .. 2026-06)")):
        x = sub.log10_compute[(sub.log10_compute >= bins[0])
                              & (sub.log10_compute <= bins[-1])]
        counts, _ = np.histogram(x, bins=bins)
        centers = (bins[:-1] + bins[1:]) / 2
        colors = [BLUE if c < THR else VERMILLION for c in centers]
        ax.bar(centers, counts, width=0.23, color=colors, edgecolor="white",
               linewidth=0.4)
        ax.axvline(THR, color=INK, lw=0.9, ls="--", zorder=5)
        ax.text(0.02, 0.86, label, transform=ax.transAxes, fontsize=8)
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="#dddddd", lw=0.5)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    plt.setp(ax_pre.get_xticklabels(), visible=False)
    ax_pre.set_ylim(0, 34)
    ax_post.set_xlabel(r"$\log_{10}$ training compute (FLOP)")
    ax_pre.set_ylabel("models")
    ax_post.set_ylabel("models")
    ax_pre.set_title("(a) Model density around the statutory line", loc="left",
                     fontsize=9)
    ax_pre.text(THR + 0.06, 26, "1e25 FLOP\nArt. 51(2)", fontsize=7, color=INK)
    ax_post.annotate(
        "(25.0, 25.5]: observed 5\nexpected 28.9  (−83%)",
        xy=(25.25, 3), xytext=(25.6, 16), fontsize=7.5, color=INK,
        arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.7),
    )

    # (b) below:above counts per half-year in the +-0.5-dex window
    hy = RECORD_HALFYEARS
    xs = np.arange(len(hy))
    below = [b for _, b, _ in hy]
    above = [a for _, _, a in hy]
    ax_t.bar(xs - 0.21, below, width=0.38, color=BLUE,
             label="[24.5, 25.0)  below")
    ax_t.bar(xs + 0.21, above, width=0.38, color=VERMILLION,
             label="[25.0, 25.5]  at/above")
    for x, a in zip(xs, above):
        ax_t.text(x + 0.21, a + 0.35, str(a), ha="center", fontsize=7,
                  color=VERMILLION)
    # event lines: Act in force (2024-08-01, one month into 2024H2) and GPAI
    # applicability (2025-08-02, start of 2025H2)
    for xpos, ytxt, lab in ((2.53, 12.6, "Act in force\n2024-08"),
                            (4.5, 18.0, "GPAI applies\n2025-08")):
        ax_t.axvline(xpos, color=MUTED, lw=0.9, ls=":")
        ax_t.text(xpos + 0.08, ytxt, lab, fontsize=7, color=MUTED)
    ax_t.set_xticks(xs, [h[2:] for h, _, _ in hy], fontsize=7.5)
    ax_t.set_yticks([0, 5, 10, 15])
    ax_t.set_ylim(0, 19.5)
    ax_t.set_ylabel("models in the $\\pm$0.5-dex window")
    ax_t.set_title("(b) The above-line count collapses from 2025H1", loc="left",
                   fontsize=9)
    ax_t.legend(frameon=False, fontsize=7.5, loc="upper left")
    ax_t.set_axisbelow(True)
    ax_t.grid(axis="y", color="#dddddd", lw=0.5)
    for s in ("top", "right"):
        ax_t.spines[s].set_visible(False)

    out = Path(__file__).resolve().parent
    fig.savefig(out / "fig1.pdf", bbox_inches="tight")
    fig.savefig(out / "fig1.png", dpi=200, bbox_inches="tight")
    print(f"wrote {out / 'fig1.pdf'} and {out / 'fig1.png'}", file=sys.stderr)


if __name__ == "__main__":
    main()
