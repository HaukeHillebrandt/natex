"""Regenerate fig1 (PDF + PNG) for the semis-event-studies-writeup mini-paper.

Reads the frozen analysis outputs of the runs of record:

- ``/Users/haukehillebrandt/dev/study-runs/semis-event-studies-writeup/``
  (``nvda_minus_soxx_weekly.csv`` -- the weekly NVDA-minus-SOXX relative log
  adjusted-close price derived from the yfinance weeklies, 784 weeks
  2011--2026; ``kink_sensitivity.json`` -- headline and bandwidth-sensitivity
  estimates copied from the three natex survey runs
  ``out/natex_survey_semis{,_2022,_2023_nvda}``, natex v0.2.0, seed 0).

Pass a different location with ``--root``. The underlying per-ticker price
CSVs are NOT committed to this repository; only the small derived CSV/JSON
files written by the analysis pass are read here.

The headline estimates are asserted against the numbers of record before a
figure is written, so a drifted results vintage fails loudly instead of
silently redrawing different numbers.

Deterministic: pure file reads, no RNG, and ``SOURCE_DATE_EPOCH`` is pinned so
the PDF/PNG bytes are reproducible.

Run (from the natex repo root):

    uv run python papers/semis-event-studies-writeup/figures/make_fig.py
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
import numpy as np
import pandas as pd

# Okabe-Ito, colorblind-safe; hues assigned in fixed order by entity.
BLUE = "#0072B2"        # NVDA-minus-SOXX series / DeepSeek-cutoff kink run
ORANGE = "#E69F00"      # BIS-2022 kink run
GREEN = "#009E73"       # BIS-2023 NVDA-only kink run
VERMILLION = "#D55E00"  # DeepSeek cutoff / post-cutoff fit
GRAY = "#666666"
INK = "#222222"

ROOT = Path("/Users/haukehillebrandt/dev/study-runs/semis-event-studies-writeup")

T_BIS22 = 2022.7644     # 2022-10-07 first BIS export-control round
T_BIS23 = 2023.7918     # 2023-10-17 second round
T_DEEPSEEK = 2025.0712  # 2025-01-27 DeepSeek crash week
T_CHATGPT = 2022 + 333 / 365.0  # 2022-11-30 ChatGPT release
HALF_WIN = 0.5          # +/-26wk local OLS window (years)

# Numbers of record (papers/semis-event-studies-writeup/main.tex, analysis 2026-07).
RECORD_NVDA_RET, RECORD_SOXX_RET = -0.172109, -0.055767  # one-week log returns
RECORD_REL_BREAK = -0.116342                             # relative level break
RECORD_SLOPES = {  # cutoff -> (pre, post) local +/-26wk OLS slopes, log/yr, 2dp
    T_BIS22: (-0.46, 1.00),
    T_BIS23: (0.61, 1.02),
    T_DEEPSEEK: (0.52, 0.39),
}
RECORD_KINKS = {  # label in kink_sensitivity.json -> (tau, se)
    "DeepSeek 2025.0712": (0.47556331053361206, 0.08390036527760769),
    "BIS-2022 2022.7644": (0.1305325332720872, 0.05700826230945853),
    "BIS-2023 NVDA-only 2023.7918": (0.1534252809704366, 0.04687259087173773),
}
ATOL_EXACT = 5e-10
ATOL_CSV = 1e-5
ATOL_SLOPE = 5e-3


def _check(name: str, got: float, want: float, atol: float = ATOL_EXACT) -> None:
    if not math.isclose(got, want, rel_tol=0.0, abs_tol=atol):
        sys.exit(f"numbers-of-record check failed for {name}: got {got!r}, want {want!r}")


def _local_slopes(df: pd.DataFrame, cutoff: float) -> tuple[float, float]:
    """Pre/post +/-26wk OLS slopes of the relative log price (convention of record:
    pre = [cutoff-0.5, cutoff), post = [cutoff, cutoff+0.5], cutoff week in post)."""
    pre = df[(df.t >= cutoff - HALF_WIN) & (df.t < cutoff)]
    post = df[(df.t >= cutoff) & (df.t <= cutoff + HALF_WIN)]
    s_pre = float(np.polyfit(pre.t, pre.rel_log_nvda_minus_soxx, 1)[0])
    s_post = float(np.polyfit(post.t, post.rel_log_nvda_minus_soxx, 1)[0])
    return s_pre, s_post


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=ROOT)
    args = ap.parse_args()

    here = Path(__file__).resolve().parent

    # ---- load + verify the numbers of record -------------------------------
    df = pd.read_csv(args.root / "nvda_minus_soxx_weekly.csv")
    sens = json.loads((args.root / "kink_sensitivity.json").read_text())

    crash = df[df.date == "2025-01-27"].iloc[0]
    prev = df[df.date == "2025-01-20"].iloc[0]
    _check("NVDA one-week log return", crash.log_nvda - prev.log_nvda,
           RECORD_NVDA_RET, ATOL_CSV)
    _check("SOXX one-week log return", crash.log_soxx - prev.log_soxx,
           RECORD_SOXX_RET, ATOL_CSV)
    _check("relative level break",
           crash.rel_log_nvda_minus_soxx - prev.rel_log_nvda_minus_soxx,
           RECORD_REL_BREAK, ATOL_CSV)
    slopes = {c: _local_slopes(df, c) for c in RECORD_SLOPES}
    for c, (want_pre, want_post) in RECORD_SLOPES.items():
        _check(f"pre slope at {c}", slopes[c][0], want_pre, ATOL_SLOPE)
        _check(f"post slope at {c}", slopes[c][1], want_post, ATOL_SLOPE)
    for label, (want_tau, want_se) in RECORD_KINKS.items():
        _check(f"kink tau ({label})", sens[label]["headline"]["tau"], want_tau)
        _check(f"kink se ({label})", sens[label]["headline"]["se"], want_se)

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

    # (a) full relative series with the three cutoffs -----------------------
    ax_a.plot(df.t, df.rel_log_nvda_minus_soxx, color=BLUE, lw=1.1)
    for x, lbl in [(T_BIS22, "BIS Oct-22"), (T_BIS23, "BIS Oct-23"),
                   (T_DEEPSEEK, "DeepSeek Jan-25")]:
        color = VERMILLION if x == T_DEEPSEEK else INK
        ax_a.axvline(x, color=color, lw=0.8, ls=(0, (4, 3)))
        ax_a.text(x - 0.13, -4.1, lbl, fontsize=6.5, color=color,
                  rotation=90, ha="right", va="bottom")
    ax_a.plot([T_CHATGPT], [df.loc[(df.t - T_CHATGPT).abs().idxmin(),
                                   "rel_log_nvda_minus_soxx"]],
              marker="v", ms=5, color=GRAY, mew=0, ls="none")
    ax_a.annotate("ChatGPT\n(+8 wk)", xy=(T_CHATGPT, -2.55), xytext=(2016.8, -1.7),
                  fontsize=6.5, color=GRAY,
                  arrowprops=dict(arrowstyle="-", color=GRAY, lw=0.7))
    ax_a.set_title("(a) NVDA $-$ SOXX, weekly relative log price")
    ax_a.set_xlabel("year")
    ax_a.set_ylabel("ln(NVDA) $-$ ln(SOXX), adj. close")

    # (b) DeepSeek zoom: level break, not kink ------------------------------
    m = (df.t >= T_DEEPSEEK - HALF_WIN) & (df.t <= T_DEEPSEEK + HALF_WIN)
    z = df[m]
    ax_b.plot(z.t, z.rel_log_nvda_minus_soxx, color=BLUE, lw=1.3, marker="o",
              ms=2.4, mew=0, label="NVDA $-$ SOXX")
    for lo, hi, (a1, b1), color, lbl in [
        (T_DEEPSEEK - HALF_WIN, T_DEEPSEEK, np.polyfit(
            z[z.t < T_DEEPSEEK].t, z[z.t < T_DEEPSEEK].rel_log_nvda_minus_soxx, 1
        )[::-1], INK, "pre fit ($+0.52$/yr)"),
        (T_DEEPSEEK, T_DEEPSEEK + HALF_WIN, np.polyfit(
            z[z.t >= T_DEEPSEEK].t, z[z.t >= T_DEEPSEEK].rel_log_nvda_minus_soxx, 1
        )[::-1], VERMILLION, "post fit ($+0.39$/yr)"),
    ]:
        xs = np.array([lo, hi])
        ax_b.plot(xs, a1 + b1 * xs, color=color, lw=1.4, ls="--", label=lbl)
    ax_b.axvline(T_DEEPSEEK, color=VERMILLION, lw=0.8, ls=(0, (4, 3)))
    y0 = prev.rel_log_nvda_minus_soxx
    y1 = crash.rel_log_nvda_minus_soxx
    ax_b.annotate(
        "", xy=(prev.t - 0.006, y1), xytext=(prev.t - 0.006, y0),
        arrowprops=dict(arrowstyle="->", color=INK, lw=1.0),
    )
    ax_b.text(prev.t - 0.018, y1 - 0.012, "$-11.6$ log-pts\nin one week",
              fontsize=6.5, color=INK, ha="right", va="top")
    ax_b.set_title("(b) DeepSeek week: a level break, not a kink")
    ax_b.set_xlabel("year")
    ax_b.set_ylabel("ln(NVDA) $-$ ln(SOXX)")
    ax_b.legend(loc="lower left", fontsize=6.5, handlelength=1.6)

    # (c) declared kinks: bandwidth sensitivity -----------------------------
    ax_c.axhline(0, color="0.75", lw=0.7)
    series = [
        ("DeepSeek 2025.0712", "DeepSeek cutoff (5 tickers)", BLUE, 0.94),
        ("BIS-2022 2022.7644", "BIS-2022 cutoff (5 tickers)", ORANGE, 1.0),
        ("BIS-2023 NVDA-only 2023.7918", "BIS-2023 cutoff (NVDA only)", GREEN, 1.06),
    ]
    for label, legend, color, jitter in series:
        rows = sens[label]["sensitivity"]
        bw0 = sens[label]["headline"]["bandwidth"]
        xs = [r["bandwidth"] * jitter for r in rows]
        ys = [r["tau"] for r in rows]
        es = [1.96 * r["se"] for r in rows]
        ax_c.errorbar(xs, ys, yerr=es, fmt="o", color=color, ecolor=color, ms=3.8,
                      elinewidth=1.1, capsize=2.2, lw=0.9, ls="-", label=legend)
        k = next(i for i, r in enumerate(rows) if r["bandwidth"] == bw0)
        ax_c.plot([xs[k]], [ys[k]], marker="D", ms=5.5, color=color, mew=0)
    ax_c.annotate('BIS-2022 at half\nbandwidth: $+0.83$', xy=(1.876, 0.832),
                  xytext=(2.6, 0.86), fontsize=6.5, color=INK,
                  arrowprops=dict(arrowstyle="-", color=GRAY, lw=0.7))
    ax_c.set_xscale("log")
    ax_c.set_xticks([2, 3, 4, 6, 8, 12])
    ax_c.set_xticklabels(["2", "3", "4", "6", "8", "12"])
    ax_c.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax_c.set_title("(c) Declared kinks: bandwidth sensitivity")
    ax_c.set_xlabel("bandwidth (years, log scale; diamond = CV headline)")
    ax_c.set_ylabel(r"kink $\hat\tau$, $\Delta$slope (log/yr, 95% CI)")
    ax_c.legend(loc="upper right", fontsize=6.5, handlelength=1.4)

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
