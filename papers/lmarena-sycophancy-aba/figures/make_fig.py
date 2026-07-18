"""Regenerate fig1 (PDF + PNG) for the lmarena-sycophancy-aba mini-paper.

Reads the frozen analysis outputs of the run of record
(``/Users/haukehillebrandt/dev/study-runs/lmarena-sycophancy-aba``, analysis
2026-07, seed 0) — pass a different location with ``--results-root``. The
underlying LMArena parquets (HF ``lmarena-ai/arena-human-preference-140k``) are
NOT committed to this repository; only small derived files written by the
analysis pass are read here:

- ``daily_style_by_model.csv``  per-model daily conv_metadata style means
- ``bt_segments.csv``           segment-specific Bradley-Terry 4o strength
- ``results.json`` / ``results_step.json`` / ``results_diag.json``  headline stats

The headline estimates are asserted against the numbers of record before a
figure is written, so a drifted results vintage fails loudly instead of
silently redrawing different numbers.

Deterministic: pure file reads, no RNG, and ``SOURCE_DATE_EPOCH`` is pinned so
the PDF/PNG bytes are reproducible.

Run (from the repo root):

    uv run python papers/lmarena-sycophancy-aba/figures/make_fig.py \
        --results-root /Users/haukehillebrandt/dev/study-runs/lmarena-sycophancy-aba
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
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Okabe-Ito, colorblind-safe; hues assigned in fixed order by entity.
BLUE = "#0072B2"        # chatgpt-4o-latest (treated live alias)
ORANGE = "#E69F00"      # claude-3-7-sonnet-20250219 (pinned placebo)
VERMILLION = "#D55E00"  # cut dates / highlighted estimate
GRAY = "#666666"        # pooled other pinned controls (context series)
INK = "#222222"

TARGET = "chatgpt-4o-latest-20250326"
CLAUDE = "claude-3-7-sonnet-20250219"

DEPLOY = pd.Timestamp("2025-04-25")
ROLLBACK = pd.Timestamp("2025-04-29")

# Numbers of record (papers/lmarena-sycophancy-aba/main.tex, analysis 2026-07).
RECORD = {
    "deploy_header_dip": -2.405639063710352,     # results_diag dip_perm header
    "deploy_header_dip_p": 0.01282051282051282,
    "rollback_bold_step": 12.110175689096847,    # results_step rollback bold
    "rollback_bold_step_p": 0.0136986301369863,
    "bt_deploy_jump": 0.020862281674395744,      # results bt_local_true
    "bt_deploy_se": 0.14405211582793156,
    "bt_rollback_step": 0.5373279945547271,      # results_step bt_step_rollback
    "bt_rollback_se": 0.13445323672801654,
    "claude37_bold_step": -5.01,                 # results_diag claude37_step_apr29
    "claude37_bt_step": -0.539,
}
ATOL = 5e-6


def check(results_root: Path, skip: bool) -> None:
    res = json.loads((results_root / "results.json").read_text())
    step = json.loads((results_root / "results_step.json").read_text())
    diag = json.loads((results_root / "results_diag.json").read_text())
    got = {
        "deploy_header_dip": diag["dip_perm_deploy_apr26"]["header_per_1k"]["true_dip"],
        "deploy_header_dip_p": diag["dip_perm_deploy_apr26"]["header_per_1k"]["p_two_sided"],
        "rollback_bold_step": step["step_perm_4o"]["rollback_apr29"]["bold_per_1k"]["true_step"],
        "rollback_bold_step_p": step["step_perm_4o"]["rollback_apr29"]["bold_per_1k"][
            "p_two_sided"
        ],
        "bt_deploy_jump": res["bt_local_true"]["deploy_jump"],
        "bt_deploy_se": res["bt_local_true"]["deploy_se"],
        "bt_rollback_step": step["bt_step_rollback"]["step_logodds"],
        "bt_rollback_se": step["bt_step_rollback"]["se"],
        "claude37_bold_step": diag["claude37_step_apr29"]["bold_per_1k"],
        "claude37_bt_step": diag["claude37_step_apr29"]["bt_step_logodds"],
    }
    ok = True
    for k, v in RECORD.items():
        match = bool(np.isclose(got[k], v, atol=ATOL))
        ok = ok and match
        print(f"[{k}] got={got[k]:+.6f} record={v:+.6f} {'OK' if match else 'MISMATCH'}")
    if not ok and not skip:
        sys.exit(
            "headline estimates do not match the numbers of record — wrong "
            "results vintage? (--skip-checks to override)"
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--results-root",
        type=Path,
        default=Path("/Users/haukehillebrandt/dev/study-runs/lmarena-sycophancy-aba"),
    )
    ap.add_argument("--skip-checks", action="store_true")
    args = ap.parse_args()

    check(args.results_root, args.skip_checks)

    daily = pd.read_csv(args.results_root / "daily_style_by_model.csv", parse_dates=["date"])
    segs = pd.read_csv(args.results_root / "bt_segments.csv").set_index("segment")

    daily = daily[daily["date"] <= pd.Timestamp("2025-06-30")]
    d4o = daily[daily["model"] == TARGET].sort_values("date")
    dcl = daily[daily["model"] == CLAUDE].sort_values("date")
    others = daily[~daily["model"].isin([TARGET, CLAUDE])].copy()
    others["w"] = others["n"] * others["bold_per_1k"]
    pooled = (
        others.groupby("date")[["w", "n"]].sum().assign(bold=lambda x: x["w"] / x["n"])
    ).sort_values("date")

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
        1, 2, figsize=(6.6, 2.9), constrained_layout=True, width_ratios=[1.45, 1.0]
    )

    # --- (a) daily bold density: treated alias vs pinned placebos -----------
    ax1.axvspan(DEPLOY, ROLLBACK, color="#f0e6d2", zorder=0)
    for d in (DEPLOY, ROLLBACK):
        ax1.axvline(d, color=VERMILLION, lw=0.9, ls="--" if d == DEPLOY else "-", zorder=1)
    ax1.plot(d4o["date"], d4o["bold_per_1k"], color=BLUE, lw=1.2, zorder=3)
    ax1.plot(dcl["date"], dcl["bold_per_1k"], color=ORANGE, lw=1.2, zorder=3)
    ax1.plot(pooled.index, pooled["bold"], color=GRAY, lw=0.9, alpha=0.85, zorder=2)
    # direct labels (identity never color-alone)
    ax1.text(pd.Timestamp("2025-05-26"), 27.6, "chatgpt-4o-latest", color=BLUE, fontsize=7.5)
    ax1.text(
        pd.Timestamp("2025-05-26"), 4.7, "claude-3-7 (pinned)", color="#a86f00", fontsize=7.5
    )
    ax1.text(
        pd.Timestamp("2025-05-26"), 13.1, "8 other pinned", color=GRAY, fontsize=7.5
    )
    ax1.text(
        DEPLOY - pd.Timedelta(days=1), 33.4, "deploy\nApr 25", color=VERMILLION,
        fontsize=7, ha="right", va="top",
    )
    ax1.text(
        ROLLBACK + pd.Timedelta(days=1), 33.4, "rollback +\narena change\nApr 29",
        color=VERMILLION, fontsize=7, ha="left", va="top",
    )
    ax1.set_ylabel("Bold markers per 1k tokens")
    ax1.set_ylim(0, 34)
    ax1.set_xlim(pd.Timestamp("2025-04-16"), pd.Timestamp("2025-07-01"))
    ax1.xaxis.set_major_locator(mdates.MonthLocator())
    ax1.xaxis.set_minor_locator(mdates.WeekdayLocator(byweekday=0))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax1.set_title("(a) Markdown style, daily means", fontsize=9, loc="left")

    # --- (b) BT strength by ABA segment -------------------------------------
    labels = ["A1\nApr 17-24", "B\nApr 25-28", "A2\nApr 29-May 6"]
    xs = np.arange(3)
    for x, seg in zip(xs, ["A1", "B", "A2"]):
        b = float(segs.loc[seg, "bt_logodds"])
        se = float(segs.loc[seg, "se"])
        color = VERMILLION if seg == "A2" else BLUE
        ax2.plot([x, x], [b - 1.96 * se, b + 1.96 * se], color=color, lw=1.2)
        ax2.plot(
            [x], [b], marker="o", ms=6,
            mfc=color if seg != "A2" else "white", mec=color, mew=1.2, color=color,
        )
    ax2.axhline(float(segs.loc["A1", "bt_logodds"]), color=GRAY, lw=0.7, ls=":")
    ax2.annotate(
        "deploy jump\n+0.02 (SE 0.14)", xy=(0.5, 0.44), fontsize=7, color=INK,
        ha="center",
    )
    ax2.annotate(
        "+0.54 step:\ncontaminated\n(pinned Claude\n$-$0.54 same day)",
        xy=(2.0, 0.62), xytext=(1.62, 0.05), fontsize=7, color=VERMILLION,
        arrowprops=dict(arrowstyle="-", color=VERMILLION, lw=0.7),
    )
    ax2.set_xticks(xs)
    ax2.set_xticklabels(labels, fontsize=7)
    ax2.set_ylabel("4o Bradley-Terry strength (log-odds)")
    ax2.set_ylim(-0.05, 1.15)
    ax2.set_title("(b) Opponent-adjusted strength", fontsize=9, loc="left")

    out = Path(__file__).resolve().parent
    fig.savefig(out / "fig1.pdf")
    fig.savefig(out / "fig1.png", dpi=200)
    print(f"wrote {out / 'fig1.pdf'} and fig1.png")


if __name__ == "__main__":
    main()
