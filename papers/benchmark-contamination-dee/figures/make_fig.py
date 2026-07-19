"""Regenerate fig1 (PDF + PNG) for the benchmark-contamination-dee mini-paper.

Reads the frozen analysis outputs of the run of record
(``/Users/haukehillebrandt/dev/study-runs/benchmark-contamination-dee``,
analysis pass 2026-07, seed 0 for the wild-cluster bootstrap only) -- pass a
different location with ``--results-root``. The Epoch AI source tables are NOT
committed to this repository; only two files written by the analysis pass are
read here:

- ``panel_cells.csv``  896 model x benchmark cells (x, r, group, held_out, ...)
- ``results.json``     every estimate: primary, WCB/RI, robustness, placebos

The headline estimates are asserted against the numbers of record before a
figure is written, so a drifted results vintage fails loudly instead of
silently redrawing different numbers.

Deterministic: pure file reads, no RNG, and ``SOURCE_DATE_EPOCH`` is pinned so
the PDF/PNG bytes are reproducible.

Run (from the repo root):

    uv run python papers/benchmark-contamination-dee/figures/make_fig.py \
        --results-root /Users/haukehillebrandt/dev/study-runs/benchmark-contamination-dee
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

# Okabe-Ito, colorblind-safe; color follows the entity (benchmark group / the
# tau estimand) in every panel. Validated (dataviz six checks, light surface):
# CVD dE 91.9 protan, all PASS.
BLUE = "#0072B2"        # public benchmarks / tau estimates
VERMILLION = "#D55E00"  # held-out benchmarks
INK = "#222222"
MUTED = "#666666"       # annotation text / reference lines only, never a series

# Numbers of record (papers/benchmark-contamination-dee/main.tex, analysis
# 2026-07). Asserted against results.json before drawing.
RECORD_TAU, RECORD_SE, RECORD_WCB_P = 0.1267, 0.0786, 0.145
RECORD_GAMMA, RECORD_GAMMA_SE = -0.1929, 0.0526
RECORD_RI_P1, RECORD_RI_P2, RECORD_RI_N = 0.213, 0.384, 3876
RECORD_N, RECORD_G, RECORD_PRE, RECORD_AT0, RECORD_POST = 896, 19, 227, 1, 668
RECORD_JACKKNIFE = (0.085, 0.171)
RECORD_ROBUST = {  # spec key -> (tau, se)
    "logit_score": (0.087, 0.083),
    "irt_difficulty_controls": (0.044, 0.122),
    "twfe_model_benchmark": (0.026, 0.101),
    "ability_leave_one_out": (0.388, 0.149),
    "ability_heldout_only": (0.354, 0.147),
    "drop_saturated_p95": (0.132, 0.082),
    "release_year_fe": (0.169, 0.081),
}
RECORD_PLACEBOS = {  # cutoff -> (tau, se)
    "-1.0": (0.800, 0.078),
    "-0.5": (0.267, 0.122),
    "0.5": (0.046, 0.070),
    "1.0": (-0.311, 0.074),
}
ATOL = 5e-4


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--results-root",
        type=Path,
        default=Path("/Users/haukehillebrandt/dev/study-runs/benchmark-contamination-dee"),
    )
    ap.add_argument("--skip-checks", action="store_true")
    args = ap.parse_args()

    d = pd.read_csv(args.results_root / "panel_cells.csv")
    res = json.loads((args.results_root / "results.json").read_text())

    # ---- assert the headline estimates against the numbers of record -------
    if not args.skip_checks:
        pri = res["primary_level_contrast"]
        pp = pri["post_x_public"]
        assert abs(pp["coef"] - RECORD_TAU) < ATOL, pp
        assert abs(pp["se"] - RECORD_SE) < ATOL, pp
        assert abs(pp["p_wild_cluster_boot"] - RECORD_WCB_P) < ATOL, pp
        gg = pri["post_heldout_baseline"]
        assert abs(gg["coef"] - RECORD_GAMMA) < ATOL, gg
        assert abs(gg["se"] - RECORD_GAMMA_SE) < ATOL, gg
        ri = res["randomization_inference"]
        assert ri["n_assignments"] == RECORD_RI_N, ri
        assert abs(ri["p_one_sided_ge"] - RECORD_RI_P1) < ATOL, ri
        assert abs(ri["p_two_sided_abs"] - RECORD_RI_P2) < ATOL, ri
        assert pri["n"] == RECORD_N and pri["n_benchmarks"] == RECORD_G, pri
        assert int((d["x"] < 0).sum()) == RECORD_PRE
        assert int((d["x"] == 0).sum()) == RECORD_AT0
        assert int((d["x"] > 0).sum()) == RECORD_POST
        for k, (tv, sv) in RECORD_ROBUST.items():
            c = res["robustness"][k]["post_x_public"]
            assert abs(c["coef"] - tv) < ATOL, (k, c)
            assert abs(c["se"] - sv) < ATOL, (k, c)
        for k, (tv, sv) in RECORD_PLACEBOS.items():
            c = res["placebo_cutoffs"][k]["post_x_public"]
            assert abs(c["coef"] - tv) < ATOL, (k, c)
            assert abs(c["se"] - sv) < ATOL, (k, c)
        lo, hi = res["jackknife"]["range"]
        assert abs(lo - RECORD_JACKKNIFE[0]) < ATOL and abs(hi - RECORD_JACKKNIFE[1]) < ATOL
        assert res["jackknife"]["all_same_sign_as_primary"] is True
        print("numbers-of-record checks: all passed", file=sys.stderr)

    # ---- draw --------------------------------------------------------------
    plt.rcParams.update({
        "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5,
        "axes.edgecolor": MUTED, "axes.linewidth": 0.7,
        "xtick.color": INK, "ytick.color": INK, "text.color": INK,
        "axes.labelcolor": INK, "figure.facecolor": "white",
    })
    fig, (ax, axf) = plt.subplots(
        1, 2, figsize=(9.2, 3.7),
        gridspec_kw={"width_ratios": [1.1, 1.0], "wspace": 0.30})

    # (a) binned event-time means of r, public vs held-out
    d = d[d["r"].notna()].copy()
    d["bin"] = np.floor(d["x"] / 0.5) * 0.5 + 0.25
    for grp, color, label in ((1, BLUE, "public"), (0, VERMILLION, "held-out")):
        g = (d[d["group"] == grp].groupby("bin")["r"]
             .agg(["mean", "sem", "size"]).reset_index())
        g = g[g["size"] >= 3]
        ax.errorbar(g["bin"], g["mean"], yerr=g["sem"], color=color, lw=2,
                    marker="o", ms=5, capsize=0, elinewidth=1.1, zorder=3,
                    mec="white", mew=0.8, label=label)
        ax.annotate(label, (g["bin"].iloc[-1] + 0.12, g["mean"].iloc[-1]),
                    color=INK, fontsize=7.5, fontweight="bold", va="center")
    ax.axvline(0, color=INK, lw=0.9, ls="--", zorder=1)
    ax.axhline(0, color="#dddddd", lw=0.8, zorder=0)
    ax.text(0.07, 0.55, "benchmark release", color=MUTED, fontsize=7,
            rotation=90, va="top")
    ax.set_xlabel("model release $-$ benchmark release (years)")
    ax.set_ylabel("ECI-residualized z-score $r$")
    ax.set_title("(a) Event-time profile by benchmark group", loc="left",
                 fontsize=9)
    ax.set_xlim(-2.1, 3.6)
    ax.legend(frameon=False, fontsize=7.5, loc="lower left")
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#dddddd", lw=0.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    # (b) forest: tau across specifications + placebo cutoffs
    crit = stats.t.ppf(0.975, RECORD_G - 1)
    pp = res["primary_level_contrast"]["post_x_public"]
    rob = res["robustness"]
    rows = [
        ("primary (ECI residual)", pp["coef"], pp["se"],
         pp.get("p_wild_cluster_boot"), True),
        ("logit score", *_cs(rob["logit_score"]), True),
        ("IRT-difficulty controls", *_cs(rob["irt_difficulty_controls"]), True),
        ("model + benchmark FE", *_cs(rob["twfe_model_benchmark"]), True),
        ("ability: leave-one-out", *_cs(rob["ability_leave_one_out"]), True),
        ("ability: held-out only", *_cs(rob["ability_heldout_only"]), True),
        ("drop saturated (>0.95)", *_cs(rob["drop_saturated_p95"]), True),
        ("release-year FE", *_cs(rob["release_year_fe"]), True),
    ]
    for c in ("-0.5", "0.5", "1.0"):  # -1.0 uninterpretable (11 pre cells)
        e = res["placebo_cutoffs"][c]["post_x_public"]
        rows.append((f"placebo cutoff {float(c):+.1f} yr", e["coef"], e["se"],
                     None, False))
    ys = np.arange(len(rows))[::-1]
    axf.axvline(0, color=MUTED, lw=0.8, zorder=1)
    for y, (name, coef, se, pwcb, is_spec) in zip(ys, rows):
        lo, hi = coef - crit * se, coef + crit * se
        axf.plot([lo, hi], [y, y], color=BLUE, lw=1.8,
                 solid_capstyle="round", zorder=2)
        mfc = BLUE if is_spec else "white"
        axf.plot(coef, y, "o", ms=6, mfc=mfc, mec=BLUE, mew=1.2, zorder=3)
        txt = f"{coef:+.2f}"
        if pwcb is not None:
            txt += f"  p={pwcb:.2f}"
        axf.annotate(txt, (hi + 0.06, y), color=MUTED, fontsize=7,
                     va="center")
    axf.set_yticks(ys)
    axf.set_yticklabels([r[0] for r in rows], fontsize=7.5)
    axf.axhline(2.5, color="#dddddd", lw=0.8)
    axf.set_xlabel(r"post $\times$ public premium $\tau$ (z), CR1 95% CI")
    axf.set_title("(b) Specifications (filled) and placebo cutoffs (open)",
                  loc="left", fontsize=9)
    axf.set_xlim(-0.75, 1.35)
    axf.set_axisbelow(True)
    axf.grid(axis="x", color="#dddddd", lw=0.5)
    for s in ("top", "right"):
        axf.spines[s].set_visible(False)

    out = Path(__file__).resolve().parent
    fig.savefig(out / "fig1.pdf", bbox_inches="tight")
    fig.savefig(out / "fig1.png", dpi=200, bbox_inches="tight")
    print(f"wrote {out / 'fig1.pdf'} and {out / 'fig1.png'}", file=sys.stderr)


def _cs(spec: dict) -> tuple[float, float, float | None]:
    c = spec["post_x_public"]
    return c["coef"], c["se"], c.get("p_wild_cluster_boot")


if __name__ == "__main__":
    main()
