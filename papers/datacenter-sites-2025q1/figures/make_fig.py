"""Regenerate fig1 (PDF + PNG) for the datacenter-sites-2025q1 mini-paper.

Reads the frozen analysis outputs of the run of record
(``/Users/haukehillebrandt/dev/study-runs/datacenter-sites-2025q1``, analysis
pass 2026-07-19, seed 0 everywhere) -- pass a different location with
``--results-root``. The Epoch AI source dataset is NOT committed to this
repository; only the small files written by the analysis pass are read here:

- ``panels/baseline/aggregate.csv``   quarterly tracked additions/levels by group
- ``decomposition.csv``               per-site post-2025Q1 contribution table
- ``results_summary.json``            placebo grid / TWFE / permutation numbers
- ``out_scan_neocloud/results.json``  seeded natex SuDDDS scan (checked only)

The headline estimates are asserted against the numbers of record before a
figure is written, so a drifted results vintage fails loudly instead of
silently redrawing different numbers.

Deterministic: pure file reads, no RNG, and ``SOURCE_DATE_EPOCH`` is pinned so
the PDF/PNG bytes are reproducible.

Run (from the repo root):

    uv run python papers/datacenter-sites-2025q1/figures/make_fig.py \
        --results-root /Users/haukehillebrandt/dev/study-runs/datacenter-sites-2025q1
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

# Okabe-Ito, colorblind-safe; color follows the entity: blue = other
# (hyperscaler) sites, orange = the 21-site neocloud group, vermillion =
# the claimed 2025Q1 break date in the placebo-grid panel (a highlight, not a
# series; candidates are neutral gray). Validated (dataviz six checks): CVD
# deltaE 109.5 (protan) / 121.7 (deutan) for the blue/orange pair, PASS; text
# always wears ink, never a series color.
BLUE = "#0072B2"        # other (hyperscaler) sites
ORANGE = "#E69F00"      # neocloud group (21 sites)
VERMILLION = "#D55E00"  # the claimed break date 2025Q1 in panel (c)
GRAY = "#b8b8c0"        # placebo candidate dates in panel (c)
INK = "#222222"
MUTED = "#666666"       # annotation text / event lines only, never a series

BREAK_Q = "2025Q1"

# Numbers of record (papers/datacenter-sites-2025q1/main.tex, run 2026-07-19).
RECORD_PRE_MEAN, RECORD_POST_MEAN, RECORD_RATIO = 296.8, 1516.867, 5.1107
RECORD_MIN_POST, RECORD_MAX_PRE = 537.0, 448.0
RECORD_GROWTH_PRE, RECORD_GROWTH_POST = 0.40388, 0.32730
RECORD_S1_OBS, RECORD_S1_RANK, RECORD_S1_ARGMAX = 1.48802, 3, "2025Q4"
RECORD_S1_ARGMAX_VAL = 1.66544
RECORD_S2_OBS, RECORD_S2_RANK, RECORD_S2_ARGMAX = 1354.355, 6, "2025Q4"
RECORD_S3_OBS = -0.10024  # and negative at all 6 candidate dates
RECORD_TAU, RECORD_T, RECORD_PERM_P, RECORD_PERM_SD = 0.09759, 0.6993, 0.560, 0.1500
RECORD_SCAN_LLR, RECORD_SCAN_T0, RECORD_SCAN_W, RECORD_SCAN_P = 24.114210253191732, 8100, 3, 0.010
RECORD_POST_TOTAL_MW, RECORD_NEO_POST_MW = 9101.2, 2158.0
RECORD_NEO_SHARE, RECORD_TOP5, RECORD_TOP10 = 0.2371, 0.3393, 0.4910
RECORD_TOP_SITE, RECORD_TOP_SITE_MW = "Anthropic-Amazon New Carlisle", 910.0
ATOL = 5e-4


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--results-root",
        type=Path,
        default=Path("/Users/haukehillebrandt/dev/study-runs/datacenter-sites-2025q1"),
    )
    ap.add_argument("--skip-checks", action="store_true")
    args = ap.parse_args()

    agg = pd.read_csv(args.results_root / "panels" / "baseline" / "aggregate.csv")
    dec = pd.read_csv(args.results_root / "decomposition.csv")
    res = json.loads((args.results_root / "results_summary.json").read_text())["baseline"]
    scan = json.loads(
        (args.results_root / "out_scan_neocloud" / "results.json").read_text()
    )["did"]

    ab = res["aggregate_break"]
    d = ab["descriptive"]
    twfe = res["twfe_neocloud"]
    tot = res["decomposition_totals"]

    # ---- assert the headline estimates against the numbers of record -------
    if not args.skip_checks:
        assert abs(d["pre_mean_add_mw"] - RECORD_PRE_MEAN) < 0.05, d
        assert abs(d["post_mean_add_mw"] - RECORD_POST_MEAN) < 0.05, d
        assert abs(d["ratio"] - RECORD_RATIO) < ATOL, d
        assert d["complete_separation"] is True
        assert abs(d["min_post_add"] - RECORD_MIN_POST) < ATOL
        assert abs(d["max_pre_add"] - RECORD_MAX_PRE) < ATOL
        assert abs(d["mean_qtrly_loggrowth_pre"] - RECORD_GROWTH_PRE) < ATOL
        assert abs(d["mean_qtrly_loggrowth_post"] - RECORD_GROWTH_POST) < ATOL
        s1, s2, s3 = ab["S1_logadd_shift"], ab["S2_level_kink_mw"], ab["S3_loglevel_kink"]
        assert abs(s1["observed"] - RECORD_S1_OBS) < ATOL, s1
        assert (s1["rank_of_2025Q1"], s1["argmax"]) == (RECORD_S1_RANK, RECORD_S1_ARGMAX)
        assert abs(ab["stats"][RECORD_S1_ARGMAX]["S1_logadd_shift"]
                   - RECORD_S1_ARGMAX_VAL) < ATOL
        assert abs(s2["observed"] - RECORD_S2_OBS) < 0.05, s2
        assert (s2["rank_of_2025Q1"], s2["argmax"]) == (RECORD_S2_RANK, RECORD_S2_ARGMAX)
        assert abs(s3["observed"] - RECORD_S3_OBS) < ATOL, s3
        assert all(v["S3_loglevel_kink"] < 0 for v in ab["stats"].values())
        assert abs(twfe["tau"] - RECORD_TAU) < ATOL, twfe
        assert abs(twfe["t"] - RECORD_T) < ATOL, twfe
        assert abs(twfe["perm_all"]["p"] - RECORD_PERM_P) < ATOL, twfe
        assert abs(twfe["perm_all"]["perm_tau_sd"] - RECORD_PERM_SD) < ATOL, twfe
        assert abs(scan["scan"]["observed_max_llr"] - RECORD_SCAN_LLR) < 1e-9
        assert abs(scan["scan"]["p_value"] - RECORD_SCAN_P) < ATOL
        disc = scan["discoveries"][0]
        assert (int(disc["t0"]), int(disc["window"])) == (RECORD_SCAN_T0, RECORD_SCAN_W)
        assert abs(tot["total_post_mw"] - RECORD_POST_TOTAL_MW) < 0.05
        assert abs(tot["neocloud_post_mw"] - RECORD_NEO_POST_MW) < 0.05
        assert abs(tot["neocloud_post_mw"] / tot["total_post_mw"] - RECORD_NEO_SHARE) < ATOL
        assert abs(tot["top5_share_post"] - RECORD_TOP5) < ATOL
        assert abs(tot["top10_share_post"] - RECORD_TOP10) < ATOL
        assert dec.iloc[0]["site"] == RECORD_TOP_SITE
        assert abs(dec.iloc[0]["mw_post"] - RECORD_TOP_SITE_MW) < ATOL
        print("numbers-of-record checks: all passed", file=sys.stderr)

    # ---- draw --------------------------------------------------------------
    plt.rcParams.update({
        "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5,
        "axes.edgecolor": MUTED, "axes.linewidth": 0.7,
        "xtick.color": INK, "ytick.color": INK, "text.color": INK,
        "axes.labelcolor": INK, "figure.facecolor": "white",
    })
    fig, axes = plt.subplots(2, 2, figsize=(9.6, 6.2))
    fig.subplots_adjust(hspace=0.52, wspace=0.38, top=0.95, bottom=0.09,
                        left=0.08, right=0.98)
    for ax in axes.flat:
        ax.set_axisbelow(True)
        ax.grid(axis="y" if ax is not axes[1, 1] else "x",
                color="#dddddd", lw=0.5)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    qlabels = agg["quarter"].tolist()
    x = np.arange(len(qlabels))
    i0 = qlabels.index(BREAK_Q)

    # (a) quarterly additions, stacked neocloud vs other
    ax = axes[0, 0]
    ax.bar(x, agg["other_add_mw"], width=0.62, color=BLUE, edgecolor="white",
           linewidth=0.6, label="other (hyperscaler) sites")
    ax.bar(x, agg["neo_add_mw"], width=0.62, bottom=agg["other_add_mw"],
           color=ORANGE, edgecolor="white", linewidth=0.6,
           label="neocloud group (21 sites)")
    ax.axvline(i0 - 0.5, color=INK, lw=0.9, ls="--", zorder=5)
    ax.text(i0 - 0.38, 3350, "2025Q1\n(Stargate, R1)", fontsize=7, color=INK)
    ax.set_xticks(x, qlabels, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("IT-power additions (MW/quarter)")
    ax.set_title("(a) Additions rise ~5x after 2025Q1 — in both groups",
                 loc="left", fontsize=9)
    ax.legend(frameon=False, fontsize=7.5, loc="upper left")

    # (b) log tracked capacity: near-linear, decelerating growth
    ax = axes[0, 1]
    ax.plot(x, np.log(agg["level_mw"]), color=BLUE, lw=2, marker="o",
            markersize=4)
    ax.axvline(i0 - 0.5, color=INK, lw=0.9, ls="--", zorder=5)
    ax.text(0.05, 0.80,
            f"pre growth {d['mean_qtrly_loggrowth_pre']:.2f} log/q",
            transform=ax.transAxes, fontsize=7.5, color=MUTED)
    ax.text(5.6, np.log(agg["level_mw"].iloc[7]) - 1.05,
            f"post growth {d['mean_qtrly_loggrowth_post']:.2f} log/q",
            fontsize=7.5, color=MUTED)
    ax.set_xticks(x, qlabels, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("log total tracked IT power (MW)")
    ax.set_title("(b) Log capacity is near-linear; growth decelerates",
                 loc="left", fontsize=9)

    # (c) placebo break-date grid, S1 shift in mean log additions
    ax = axes[1, 0]
    cand = ab["grid"]
    s1v = [ab["stats"][c]["S1_logadd_shift"] for c in cand]
    colors = [VERMILLION if c == BREAK_Q else GRAY for c in cand]
    ax.bar(np.arange(len(cand)), s1v, width=0.55, color=colors,
           edgecolor="white", linewidth=0.6)
    for i, (c, v) in enumerate(zip(cand, s1v)):
        if c == BREAK_Q:
            ax.text(i, v + 0.04, f"{v:.2f}  rank 3/6", ha="center", fontsize=7,
                    color=INK)
        elif c == RECORD_S1_ARGMAX:
            ax.text(i, v - 0.05, f"argmax\n{v:.2f}", ha="center", va="top",
                    fontsize=7, color=INK)
    ax.set_ylim(0, 1.88)
    ax.set_xticks(np.arange(len(cand)), cand, fontsize=7.5)
    ax.set_ylabel("post$-$pre shift in mean log additions")
    ax.set_title("(c) Placebo break dates: 2025Q1 is never the argmax",
                 loc="left", fontsize=9)

    # (d) top-10 site contributions to post-2025Q1 additions
    ax = axes[1, 1]
    top = dec.head(10).iloc[::-1]
    cols = [ORANGE if n else BLUE for n in top["neocloud"]]
    ax.barh(np.arange(len(top)), top["mw_post"], height=0.62, color=cols,
            edgecolor="white", linewidth=0.6)
    ax.set_yticks(np.arange(len(top)), top["site"], fontsize=7)
    for i, (mw, sh) in enumerate(zip(top["mw_post"], top["share_post"])):
        ax.text(mw + 12, i, f"{sh * 100:.0f}%", va="center", fontsize=6.5,
                color=MUTED)
    ax.set_xlim(0, 1040)
    ax.set_xlabel("MW added 2025Q1–2026Q2")
    ax.set_title("(d) Top sites post-2025Q1 (orange = neocloud, 23.7% total)",
                 loc="left", fontsize=9)

    out = Path(__file__).resolve().parent
    fig.savefig(out / "fig1.pdf", bbox_inches="tight")
    fig.savefig(out / "fig1.png", dpi=200, bbox_inches="tight")
    print(f"wrote {out / 'fig1.pdf'} and {out / 'fig1.png'}", file=sys.stderr)


if __name__ == "__main__":
    main()
