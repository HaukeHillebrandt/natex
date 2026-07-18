"""Regenerate fig1 (PDF + PNG) for the export-controls-three-leg mini-paper.

Reads the frozen analysis outputs of the runs of record:

- Leg 1 (SuDDDS DiD, country panel):
  ``/Users/haukehillebrandt/dev/epoch-data/natex-runs/export-controls``
  (``panel.csv``, ``placebo_battery.json``; natex, seed 7, q=99)
- Legs 2-3 (group difference-in-kinks, chip stocks):
  ``/Users/haukehillebrandt/dev/epoch-data/kink-runs/china_chip_dik``
  (``input_legal.csv``, ``input_total.csv``, ``diagnostics.json``,
  ``out_legal_bw548_tri/kink.json``; seed 20260716, estimator deterministic)

Pass different locations with ``--did-root`` / ``--dik-root``. The underlying
Epoch AI datasets are NOT committed to this repository; only the small derived
CSV/JSON files written by the analysis passes are read here.

The headline estimates are asserted against the numbers of record before a
figure is written, so a drifted results vintage fails loudly instead of
silently redrawing different numbers.

Deterministic: pure file reads, no RNG, and ``SOURCE_DATE_EPOCH`` is pinned so
the PDF/PNG bytes are reproducible.

Run (from the natex repo root):

    uv run python papers/export-controls-three-leg/figures/make_fig.py
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("SOURCE_DATE_EPOCH", "0")  # reproducible figure bytes

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

# Okabe-Ito, colorblind-safe; hues assigned in fixed order by entity.
BLUE = "#0072B2"        # China (legal series in panel b)
SKY = "#56B4E9"         # China incl. smuggled
ORANGE = "#E69F00"      # US (a) / hyperscaler control group (b)
GREEN = "#009E73"       # EU
PURPLE = "#CC79A7"      # Other
VERMILLION = "#D55E00"  # Oct-2023 cutoff / headline DiK estimate
GRAY = "#666666"        # placebo cutoffs
INK = "#222222"

DID_ROOT = Path("/Users/haukehillebrandt/dev/epoch-data/natex-runs/export-controls")
DIK_ROOT = Path("/Users/haukehillebrandt/dev/epoch-data/kink-runs/china_chip_dik")

ROUND1 = 2022 + 280 / 365.0  # 2022-10-07 first export-control round
ROUND2 = 2023 + 290 / 365.0  # 2023-10-17 second round (loophole closed)

# Numbers of record (papers/export-controls-three-leg/main.tex, analysis 2026-07).
RECORD_DIK_TAU, RECORD_DIK_SE = -0.0015403737041809703, 0.0004670948115061569
RECORD_DD_TAU, RECORD_DD_SE = 3.214141414141414, 1.1567454191509172
RECORD_PLC_OCT22 = 0.0018203357335959773
ATOL = 5e-10


def _check(name: str, got: float, want: float) -> None:
    if not math.isclose(got, want, rel_tol=0.0, abs_tol=ATOL):
        sys.exit(f"numbers-of-record check failed for {name}: got {got!r}, want {want!r}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--did-root", type=Path, default=DID_ROOT)
    ap.add_argument("--dik-root", type=Path, default=DIK_ROOT)
    args = ap.parse_args()

    here = Path(__file__).resolve().parent

    # ---- load + verify the numbers of record -------------------------------
    panel = pd.read_csv(args.did_root / "panel.csv")
    battery = json.loads((args.did_root / "placebo_battery.json").read_text())
    legal = pd.read_csv(args.dik_root / "input_legal.csv")
    total = pd.read_csv(args.dik_root / "input_total.csv")
    diag = json.loads((args.dik_root / "diagnostics.json").read_text())
    kink = json.loads((args.dik_root / "out_legal_bw548_tri" / "kink.json").read_text())

    china_dd = next(
        r for r in battery["space"] if r["unit"] == "China" and r["label"] == "2022Q4"
    )
    _check("DD tau (China, 2022Q4)", china_dd["all_units"]["tau"], RECORD_DD_TAU)
    _check("DD se (China, 2022Q4)", china_dd["all_units"]["se"], RECORD_DD_SE)
    _check("DiK tau (legal, bw548 tri)", kink["estimate"]["tau"], RECORD_DIK_TAU)
    _check("DiK se (legal, bw548 tri)", kink["estimate"]["se"], RECORD_DIK_SE)
    placebos = diag["placebo_kinks_legal_bw548"]["placebos"]
    oct22 = next(p for p in placebos if p["cutoff"] == -375.0)
    _check("placebo DiK (Oct-2022 round)", oct22["estimate"], RECORD_PLC_OCT22)

    # ---- figure ------------------------------------------------------------
    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.titlesize": 8.5,
            "axes.labelsize": 8,
            "axes.edgecolor": INK,
            "axes.linewidth": 0.7,
            "xtick.color": INK,
            "ytick.color": INK,
            "text.color": INK,
            "axes.labelcolor": INK,
            "legend.frameon": False,
        }
    )
    fig, (ax_a, ax_b, ax_c) = plt.subplots(
        1, 3, figsize=(10.8, 3.3), constrained_layout=True
    )

    # (a) Leg 1: big-run counts by country group ----------------------------
    year = 2020.0 + (panel["quarter_idx"] + 0.5) / 4.0
    for unit, color in [("China", BLUE), ("US", ORANGE), ("EU", GREEN), ("Other", PURPLE)]:
        m = panel["unit"] == unit
        ax_a.plot(
            year[m], panel.loc[m, "n_ge_1e24"], color=color, lw=1.6, label=unit,
            marker="o", ms=2.2, mew=0,
        )
    ax_a.axvspan(2026.0, float(year.max()) + 0.125, color="0.88", zorder=0)
    ax_a.text(2026.02, ax_a.get_ylim()[1] * 0.97, "reporting\nlag", fontsize=6.5,
              color=GRAY, va="top")
    for x, lbl, ha in [(ROUND1, "Oct 2022\ncontrols", "right"),
                       (ROUND2, "Oct 2023\ncontrols", "left")]:
        ax_a.axvline(x, color=INK, lw=0.8, ls=(0, (4, 3)))
        ax_a.text(x + (0.06 if ha == "left" else -0.06), 15.5, lbl, fontsize=6.5,
                  color=INK, ha=ha)
    ax_a.set_title("(a) Training runs $\\geq 10^{24}$ FLOP per quarter")
    ax_a.set_ylabel("runs per quarter")
    ax_a.set_xlabel("year")
    ax_a.legend(loc="upper left", fontsize=7, handlelength=1.4)

    # (b) Legs 2-3: ln chip stock around the Oct-2023 cutoff ----------------
    hyper = legal[legal["china"] == 0].sort_values("days")
    ch_legal = legal[legal["china"] == 1].sort_values("days")
    ch_total = total[total["china"] == 1].sort_values("days")
    ax_b.plot(hyper["days"], hyper["ln_stock"], color=ORANGE, lw=1.6, marker="o",
              ms=2.6, mew=0, label="hyperscalers (control)")
    ax_b.plot(ch_total["days"], ch_total["ln_stock"], color=SKY, lw=1.4, ls="--",
              marker="s", ms=2.6, mew=0, label="China, total (incl. smuggled)")
    ax_b.plot(ch_legal["days"], ch_legal["ln_stock"], color=BLUE, lw=1.6, marker="o",
              ms=2.6, mew=0, label="China, legal")
    ax_b.axvline(0, color=VERMILLION, lw=1.0, ls=(0, (4, 3)))
    ax_b.text(25, 16.35, "2023-10-17\ncontrols", fontsize=6.5, color=VERMILLION,
              va="top")
    ax_b.set_title("(b) AI-chip stock, ln H100e")
    ax_b.set_xlabel("days since 2023-10-17")
    ax_b.set_ylabel("ln cumulative H100e stock")
    ax_b.legend(loc="lower right", fontsize=7, handlelength=1.6)

    # (c) Leg 2 placebo cutoffs vs the headline DiK -------------------------
    est0, se0 = kink["estimate"]["tau"], kink["estimate"]["se"]
    xs = [p["cutoff"] for p in placebos]
    ys = [p["estimate"] for p in placebos]
    es = [1.96 * p["se"] for p in placebos]
    ax_c.axhline(0, color="0.75", lw=0.7)
    ax_c.errorbar(xs, ys, yerr=es, fmt="o", color=GRAY, ecolor=GRAY, ms=4.5,
                  elinewidth=1.1, capsize=2.4, label="placebo cutoffs")
    ax_c.errorbar([0.0], [est0], yerr=[1.96 * se0], fmt="D", color=VERMILLION,
                  ecolor=VERMILLION, ms=5.5, elinewidth=1.4, capsize=2.6,
                  label="Oct-2023 cutoff (headline)")
    ax_c.annotate("Oct-2022 round\n(loophole): null", xy=(-375, ys[0]),
                  xytext=(-355, 0.0042), fontsize=6.5, color=INK,
                  arrowprops=dict(arrowstyle="-", color=GRAY, lw=0.7))
    ax_c.set_title("(c) Group DiK: China legal vs hyperscalers")
    ax_c.set_xlabel("cutoff placement, days relative to 2023-10-17")
    ax_c.set_ylabel("DiK, ln H100e per day (95% CI)")
    ax_c.legend(loc="upper right", fontsize=7, handlelength=1.4)

    for ax in (ax_a, ax_b, ax_c):
        ax.grid(color="0.92", lw=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    fig.savefig(here / "fig1.pdf")
    fig.savefig(here / "fig1.png", dpi=200)
    print(f"wrote {here / 'fig1.pdf'} and {here / 'fig1.png'}")


if __name__ == "__main__":
    main()
