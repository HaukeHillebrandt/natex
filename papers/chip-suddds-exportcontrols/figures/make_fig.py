"""Regenerate fig1 (PDF + PNG) for the chip-suddds-exportcontrols mini-paper.

Reads the frozen analysis outputs of the run of record
(``/Users/haukehillebrandt/dev/study-runs/chip-suddds-exportcontrols``, natex
v0.2.0, seed 0) — pass a different location with ``--results-root``. The
underlying Epoch AI chip datasets are NOT committed to this repository; only
the small derived files written by the analysis pass are read here:

- ``panel_sales_oct23.csv`` (+ ``_aliases.csv``)  the 7 x 16 Oct-2023 panel
- ``panel_sales_h20.csv``   (+ ``_aliases.csv``)  the 20 x 8 H20 panel
- ``effects.json``                                 TWFE effect leg of record
- ``out_panel_sales_oct23_bern/results.json``      headline scan of record

The headline estimates (Oct-23 scan LLR/date and the matched H20 TWFE tau/se)
are asserted against the numbers of record before a figure is written, so a
drifted results vintage fails loudly instead of silently redrawing different
numbers. Per the collection convention, only exactly-reproducible statistics
are asserted (scan LLR to 1e-6, break quarter, effect tau/se) — never
tie-convention-dependent ranks.

Deterministic: pure file reads, no RNG, and ``SOURCE_DATE_EPOCH`` is pinned so
the PDF/PNG bytes are reproducible.

Run (from the repo root):

    uv run python papers/chip-suddds-exportcontrols/figures/make_fig.py \
        --results-root /Users/haukehillebrandt/dev/study-runs/chip-suddds-exportcontrols
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
from matplotlib.lines import Line2D

# Okabe-Ito, colorblind-safe; hues assigned in fixed order by entity.
# Palette validated (dataviz six-checks validator, light surface): the
# chromatic identity trio passes; #E69F00's contrast WARN is relieved by the
# direct "H800" label; gray is deliberate muted context, not an identity slot.
VERMILLION = "#D55E00"  # primary treated SKU (A800, H20)
ORANGE = "#E69F00"      # second treated SKU (H800)
BLUE = "#0072B2"        # event-study estimate
GRAY = "#666666"        # matched control chips (context)
FAINT = "#bbbbbb"       # unmatched control chips (context)
INK = "#222222"

# Numbers of record (papers/chip-suddds-exportcontrols/main.tex, analysis 2026-07).
RECORD_OCT23_LLR = 4.895955314788866   # Bernoulli/wcc scan, max LLR
RECORD_OCT23_T0 = 8095                 # 2023Q4 (t = year*4 + quarter-1)
RECORD_OCT23_W = 4
RECORD_OCT23_P = 0.03                  # Q=99 permutation
RECORD_TAU, RECORD_SE = -2.7559, 0.6517  # matched H20 TWFE, CR1 (4 dp of record)
RECORD_ES_2025Q4 = -4.0714             # event-study endpoint
ATOL_LLR, ATOL_EFF = 1e-6, 1e-4

OCT23_BAN, H20_BAN = 8094.5, 8100.5    # quarter boundaries of the two rounds


def qlab(q: int) -> str:
    return f"{q // 4}Q{q % 4 + 1}"


def load_panel(root: Path, name: str) -> pd.DataFrame:
    df = pd.read_csv(root / f"{name}.csv")
    aliases = pd.read_csv(root / f"{name}_aliases.csv")
    return df.merge(aliases, on="chip_id")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--results-root",
        type=Path,
        default=Path("/Users/haukehillebrandt/dev/study-runs/chip-suddds-exportcontrols"),
    )
    ap.add_argument("--skip-checks", action="store_true")
    args = ap.parse_args()
    root = args.results_root

    oct23 = load_panel(root, "panel_sales_oct23")
    h20 = load_panel(root, "panel_sales_h20")
    eff = json.loads((root / "effects.json").read_text())
    scan = json.loads((root / "out_panel_sales_oct23_bern" / "results.json").read_text())

    disc = scan["did"]["discoveries"][0]
    matched = eff["matched_sales_h20"]
    es = eff["event_study_h20_matched"]
    ok = (
        np.isclose(disc["llr"], RECORD_OCT23_LLR, atol=ATOL_LLR)
        and int(disc["t0"]) == RECORD_OCT23_T0
        and int(disc["window"]) == RECORD_OCT23_W
        and np.isclose(scan["did"]["scan"]["p_value"], RECORD_OCT23_P, atol=1e-9)
        and np.isclose(matched["tau"], RECORD_TAU, atol=ATOL_EFF)
        and np.isclose(matched["se_cr1"], RECORD_SE, atol=ATOL_EFF)
        and np.isclose(es["2025Q4"], RECORD_ES_2025Q4, atol=ATOL_EFF)
    )
    print(
        f"[record] oct23 scan LLR={disc['llr']:.6f} t0={qlab(int(disc['t0']))} "
        f"W={int(disc['window'])} p={scan['did']['scan']['p_value']}; "
        f"matched H20 tau={matched['tau']:+.4f} se={matched['se_cr1']:.4f} "
        f"{'OK' if ok else 'MISMATCH'}"
    )
    if not ok and not args.skip_checks:
        sys.exit(
            "headline estimates do not match the numbers of record — wrong "
            "results vintage? (--skip-checks to override)"
        )

    plt.rcParams.update(
        {
            "font.size": 7.5,
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

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(6.8, 2.55), constrained_layout=True)
    for ax in (ax1, ax2, ax3):
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=6.3)

    # --- (a) Oct-2023 round: A800 + H800 vs incumbents ---------------------
    for chip, g in oct23.groupby("chip"):
        g = g.sort_values("t")
        if chip == "A800":
            ax1.plot(g["t"], g["outcome"], color=VERMILLION, lw=1.6, zorder=5)
            i = int(g["outcome"].idxmax())
            ax1.annotate("A800", (g.loc[i, "t"] - 1.6, g.loc[i, "outcome"] + 0.25),
                         color=VERMILLION, fontsize=6.5, fontweight="bold")
        elif chip == "H800":
            ax1.plot(g["t"], g["outcome"], color=ORANGE, lw=1.6, zorder=5)
            i = int(g["outcome"].idxmax())
            ax1.annotate("H800", (g.loc[i, "t"] + 0.3, g.loc[i, "outcome"] + 0.15),
                         color=ORANGE, fontsize=6.5, fontweight="bold")
        else:
            ax1.plot(g["t"], g["outcome"], color=GRAY, lw=0.9, alpha=0.6)
    ax1.axvline(OCT23_BAN, color=INK, lw=0.8, ls="--", zorder=1)
    ax1.set_xticks(range(8088, 8104, 4))
    ax1.set_xticklabels([qlab(q) for q in range(8088, 8104, 4)])
    ax1.set_ylabel("asinh(kH100e shipped / quarter)", fontsize=7)
    ax1.set_title("(a) Oct-2023 ban (p = 0.030)", fontsize=7.6, loc="left")
    ax1.legend(
        handles=[
            Line2D([], [], color=VERMILLION, lw=1.6, label="treated (banned SKU)"),
            Line2D([], [], color=GRAY, lw=0.9, label="control chips"),
        ],
        fontsize=5.8, loc="upper left", handlelength=1.3,
        frameon=True, facecolor="white", edgecolor="none", framealpha=0.9,
    )

    # --- (b) H20 round: matched vs unmatched controls ----------------------
    matched_names = set(eff["matched_controls_h20"])
    for chip, g in h20.groupby("chip"):
        g = g.sort_values("t")
        if chip == "H20":
            ax2.plot(g["t"], g["outcome"], color=VERMILLION, lw=1.6, zorder=5)
            i = int(g["outcome"].idxmax())
            ax2.annotate("H20", (g.loc[i, "t"] - 1.3, g.loc[i, "outcome"] + 0.22),
                         color=VERMILLION, fontsize=6.5, fontweight="bold")
        elif chip in matched_names:
            ax2.plot(g["t"], g["outcome"], color=GRAY, lw=0.9, alpha=0.8)
        else:
            ax2.plot(g["t"], g["outcome"], color=FAINT, lw=0.8, ls=":", alpha=0.9)
    ax2.axvline(H20_BAN, color=INK, lw=0.8, ls="--", zorder=1)
    ax2.set_xticks(range(8096, 8104, 2))
    ax2.set_xticklabels([qlab(q) for q in range(8096, 8104, 2)])
    ax2.set_title("(b) H20 round (p = 0.44)", fontsize=7.6, loc="left")
    ax2.legend(
        handles=[
            Line2D([], [], color=VERMILLION, lw=1.6, label="H20 (treated)"),
            Line2D([], [], color=GRAY, lw=0.9, label="matched control"),
            Line2D([], [], color=FAINT, lw=0.8, ls=":", label="unmatched (birth/death)"),
        ],
        fontsize=5.8, loc="lower left", handlelength=1.3,
        frameon=True, facecolor="white", edgecolor="none", framealpha=0.9,
    )

    # --- (c) event study with the placebo-in-space envelope ----------------
    qs = list(range(8096, 8104))
    vals = [es[qlab(q)] for q in qs]
    env = max(abs(t) for t in matched["placebo_taus"])
    ax3.axhspan(-env, env, color="#dddddd", alpha=0.55, lw=0, zorder=0)
    ax3.axhline(0.0, color=GRAY, lw=0.7)
    ax3.axvline(H20_BAN, color=INK, lw=0.8, ls="--", zorder=1)
    ax3.plot(qs, vals, color=BLUE, lw=1.6, marker="o", ms=3.5, zorder=5)
    ax3.annotate(
        "placebo envelope\n(max |tau|, 6 controls)",
        (8096.1, -2.35), fontsize=5.8, color=GRAY,
    )
    ax3.annotate(
        f"TWFE tau = {matched['tau']:.2f}\nCR1 t = {matched['t_cr1']:.2f}\n"
        f"placebo p = 0.143 (floor)",
        (8096.1, -4.25), fontsize=6.0, color=INK, va="bottom",
    )
    ax3.set_xticks(range(8096, 8104, 2))
    ax3.set_xticklabels([qlab(q) for q in range(8096, 8104, 2)])
    ax3.set_ylabel("DiD coefficient (asinh units)", fontsize=7)
    ax3.set_title("(c) H20 $-$ matched controls", fontsize=7.6, loc="left")

    out = Path(__file__).resolve().parent
    fig.savefig(out / "fig1.pdf")
    fig.savefig(out / "fig1.png", dpi=200)
    print(f"wrote {out / 'fig1.pdf'} and fig1.png")


if __name__ == "__main__":
    main()
