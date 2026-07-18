"""Regenerate fig1 (PDF + PNG) for the capex-dik-chatgpt mini-paper.

Reads the frozen analysis outputs of the run of record
(``/Users/haukehillebrandt/dev/study-runs/capex-dik-chatgpt``, natex v0.2.0)
— pass a different location with ``--results-root``. The underlying SEC EDGAR
XBRL pulls are NOT committed to this repository; only the derived panel and
the JSON estimate files written by the analysis pass are read here:

- ``panel_agg.csv``               46 quarters x 2 groups, capex in $bn/quarter
- ``out/main_bw3_tri/kink.json``  the main group difference-in-kinks CLI run
- ``out/diagnostics.json``        pre-only placebo-cutoff grid (bw 2.0)
- ``out/followups.json``          placebo calibration at the matched bandwidth

The headline estimates are asserted against the numbers of record before a
figure is written, so a drifted results vintage fails loudly instead of
silently redrawing different numbers.

Deterministic: pure file reads, no RNG, and ``SOURCE_DATE_EPOCH`` is pinned so
the PDF/PNG bytes are reproducible.

Run (from the repo root):

    uv run python papers/capex-dik-chatgpt/figures/make_fig.py \
        --results-root /Users/haukehillebrandt/dev/study-runs/capex-dik-chatgpt
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
BLUE = "#0072B2"        # big-4 hyperscaler aggregate
ORANGE = "#E69F00"      # non-AI capital-intensive control aggregate
VERMILLION = "#D55E00"  # ChatGPT cutoff / main estimate
GRAY = "#666666"        # placebo cutoffs / BIS event
INK = "#222222"

CHATGPT = 2022.913  # 2022-11-30
BIS = 2022.767      # 2022-10-07 export controls

# Numbers of record (papers/capex-dik-chatgpt/main.tex, analysis 2026-07).
RECORD_TAU, RECORD_SE = 0.13958501011891217, 0.03021611654566233
RECORD_TAU_BW2, RECORD_Z_BW2 = 0.17781759746777703, 4.348048133575059
RECORD_N_PLACEBO, RECORD_N_REJECT = 15, 7
ATOL = 5e-6


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--results-root",
        type=Path,
        default=Path("/Users/haukehillebrandt/dev/study-runs/capex-dik-chatgpt"),
    )
    ap.add_argument("--skip-checks", action="store_true")
    args = ap.parse_args()

    panel = pd.read_csv(args.results_root / "panel_agg.csv")
    with open(args.results_root / "out" / "main_bw3_tri" / "kink.json") as fh:
        kink = json.load(fh)
    with open(args.results_root / "out" / "diagnostics.json") as fh:
        diag = json.load(fh)
    with open(args.results_root / "out" / "followups.json") as fh:
        follow = json.load(fh)

    tau = float(kink["estimate"]["tau"])
    se = float(kink["estimate"]["se"])
    cal = follow["placebo_calibration_bw2"]
    placebos = diag["placebo_pre_only"]["placebos"]
    n_reject = sum(1 for p in placebos if p["p_value"] < 0.05)
    ok = (
        np.isclose(tau, RECORD_TAU, atol=ATOL)
        and np.isclose(se, RECORD_SE, atol=ATOL)
        and np.isclose(float(cal["main_tau"]), RECORD_TAU_BW2, atol=ATOL)
        and np.isclose(float(cal["main_z"]), RECORD_Z_BW2, atol=ATOL)
        and len(placebos) == RECORD_N_PLACEBO
        and n_reject == RECORD_N_REJECT
    )
    print(
        f"[main DiK] tau={tau:+.6f} se={se:.6f} "
        f"[bw2] tau={float(cal['main_tau']):+.6f} z={float(cal['main_z']):.4f} "
        f"[placebos] {n_reject}/{len(placebos)} reject {'OK' if ok else 'MISMATCH'}"
    )
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

    # --- (a) aggregate capex by group -------------------------------------
    big4 = panel[panel["group"] == 1].sort_values("t_years")
    ctrl = panel[panel["group"] == 0].sort_values("t_years")
    ax1.plot(
        big4["t_years"], big4["capex_bn"], color=BLUE, lw=1.4,
        label="Big-4 hyperscalers",
    )
    ax1.plot(
        ctrl["t_years"], ctrl["capex_bn"], color=ORANGE, lw=1.4,
        label="Non-AI capital-intensive",
    )
    ax1.set_yscale("log")
    ax1.set_yticks([10, 20, 40, 80])
    ax1.set_yticklabels(["10", "20", "40", "80"])
    ax1.axvline(CHATGPT, color=VERMILLION, lw=0.9, ls="-", zorder=0)
    ax1.axvline(BIS, color=GRAY, lw=0.9, ls="--", zorder=0)
    ax1.text(CHATGPT + 0.12, 9.0, "ChatGPT", color=VERMILLION, fontsize=7.5)
    ax1.text(BIS - 0.12, 110.0, "BIS", color=GRAY, fontsize=7.5, ha="right")
    ax1.set_xlabel("Quarter midpoint (year)")
    ax1.set_ylabel("Capex (\\$bn/quarter, log scale)")
    ax1.legend(
        loc="upper left", fontsize=7.5, handlelength=1.4,
        frameon=True, facecolor="white", edgecolor="none", framealpha=1.0,
    )
    ax1.set_title("(a) Quarterly capex by group", fontsize=9, loc="left")

    # --- (b) DiK at ChatGPT vs pre-period placebo cutoffs (bw 2.0) --------
    for p in placebos:
        x = float(p["cutoff"])
        est, pse = float(p["estimate"]), float(p["se"])
        sig = float(p["p_value"]) < 0.05
        ax2.plot([x, x], [est - 1.96 * pse, est + 1.96 * pse], color=GRAY, lw=1.1)
        ax2.plot(
            [x], [est], marker="o", ms=4.5, mfc=GRAY if sig else "white",
            mec=GRAY, mew=1.1, color=GRAY,
        )
    m_tau = float(cal["main_tau"])
    m_se = m_tau / float(cal["main_z"])
    ax2.plot(
        [CHATGPT, CHATGPT], [m_tau - 1.96 * m_se, m_tau + 1.96 * m_se],
        color=VERMILLION, lw=1.3,
    )
    ax2.plot(
        [CHATGPT], [m_tau], marker="o", ms=5.5, mfc=VERMILLION,
        mec=VERMILLION, mew=1.3, color=VERMILLION,
    )
    ax2.text(
        CHATGPT, m_tau + 1.96 * m_se + 0.02, "ChatGPT", color=VERMILLION,
        fontsize=7.5, ha="center",
    )
    ax2.axhline(0.0, color=INK, lw=0.7)
    ax2.set_xlabel("Cutoff (year)")
    ax2.set_ylabel("DiK slope change (dex/yr)")
    ax2.set_xlim(2016.6, 2023.6)
    ax2.set_ylim(-0.45, 0.45)
    ax2.set_title("(b) DiK vs. non-event placebo cutoffs", fontsize=9, loc="left")
    ax2.text(
        0.03, 0.03,
        f"filled = nominal $p<0.05$ (HC1); {n_reject}/{len(placebos)} placebos reject\n"
        "bw 2.0 yr, triangular, donut 1 qtr",
        transform=ax2.transAxes, fontsize=6.8, color=GRAY, va="bottom",
    )

    out = Path(__file__).resolve().parent
    fig.savefig(out / "fig1.pdf")
    fig.savefig(out / "fig1.png", dpi=200)
    print(f"wrote {out / 'fig1.pdf'} and fig1.png")


if __name__ == "__main__":
    main()
