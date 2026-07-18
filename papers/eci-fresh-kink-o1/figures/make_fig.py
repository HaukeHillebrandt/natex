"""Regenerate fig1 (PDF + PNG) for the eci-fresh-kink-o1 mini-paper.

Reads the frozen analysis outputs of the run of record
(``/Users/haukehillebrandt/dev/study-runs/eci-fresh-kink-o1``, natex v0.2.0)
— pass a different location with ``--results-root``. The underlying Epoch
eci_scores.csv vintage is NOT committed to this repository; only the derived
panel and the JSON/CSV estimate files written by the analysis pass are read:

- ``input_all.csv``   all 211 aggregated models: days since o1, eci, 1/sigma^2 weight
- ``grid.csv``        the 144-cell specification grid (headline cells asserted)
- ``results.json``    placebo-cutoff grids at bw 540 triangular

The headline estimates are asserted against the numbers of record before a
figure is written, so a drifted results vintage fails loudly instead of
silently redrawing different numbers.  The panel-(a) fit lines are recomputed
here by local-linear (triangular-kernel) WLS per side and asserted against the
slope_left/slope_right of record, which also re-validates the estimator
convention (left: days < 0; right: days >= 0; kernel w = 1 - |d|/bw; CI
weights multiply the kernel).

Deterministic: pure file reads and closed-form WLS, no RNG, and
``SOURCE_DATE_EPOCH`` is pinned so the PDF/PNG bytes are reproducible.

Run (from the repo root):

    uv run python papers/eci-fresh-kink-o1/figures/make_fig.py \
        --results-root /Users/haukehillebrandt/dev/study-runs/eci-fresh-kink-o1
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
BLUE = "#0072B2"        # unweighted estimates / fits
ORANGE = "#E69F00"      # CI-weighted (1/sigma^2) estimates / fits
VERMILLION = "#D55E00"  # o1 cutoff reference line (not a series)
GRAY = "#888888"        # model scatter
DGRAY = "#666666"
INK = "#222222"

BW = 540.0  # headline bandwidth (days), triangular kernel, donut 0

# Numbers of record (papers/eci-fresh-kink-o1/main.tex, analysis 2026-07).
REC_UNW_EST, REC_UNW_SE = 0.006530685694841776, 0.013078528382273465
REC_CIW_EST, REC_CIW_SE = -0.019027317789967226, 0.007881417206031182
REC_SLOPES_UNW = (0.0611, 0.0677)   # pre, post (pts/day, 4 dp of record)
REC_SLOPES_CIW = (0.0522, 0.0331)
REC_PLACEBO_REJECT = {"unweighted": 4, "ci_weighted": 3}  # of 8, alpha=.05
ATOL = 5e-6


def side_slope(d: np.ndarray, y: np.ndarray, w: np.ndarray) -> tuple[float, float]:
    """Weighted least-squares line y = a + b d; returns (a, b)."""
    x = np.column_stack([np.ones_like(d), d])
    xtw = x.T * w
    beta = np.linalg.solve(xtw @ x, xtw @ y)
    return float(beta[0]), float(beta[1])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--results-root",
        type=Path,
        default=Path("/Users/haukehillebrandt/dev/study-runs/eci-fresh-kink-o1"),
    )
    ap.add_argument("--skip-checks", action="store_true")
    args = ap.parse_args()

    panel = pd.read_csv(args.results_root / "input_all.csv")
    grid = pd.read_csv(args.results_root / "grid.csv")
    with open(args.results_root / "results.json") as fh:
        res = json.load(fh)

    def headline(weighting: str) -> pd.Series:
        m = grid[
            (grid["series"] == "all_models")
            & (grid["weighting"] == weighting)
            & (grid["bandwidth"] == BW)
            & (grid["kernel"] == "triangular")
            & (grid["donut"] == 0.0)
        ]
        assert len(m) == 1
        return m.iloc[0]

    unw, ciw = headline("unweighted"), headline("ci_weighted")

    # --- recompute the panel-(a) fit lines and re-validate the convention ---
    d = panel["days"].to_numpy(float)
    y = panel["eci"].to_numpy(float)
    wt = panel["weight"].to_numpy(float)
    fits: dict[str, dict[str, tuple[float, float]]] = {}
    for name, extra in [("unweighted", np.ones_like(d)), ("ci_weighted", wt)]:
        fits[name] = {}
        for side, mask in [("L", (d >= -BW) & (d < 0)), ("R", (d >= 0) & (d <= BW))]:
            kern = (1.0 - np.abs(d[mask]) / BW) * extra[mask]
            fits[name][side] = side_slope(d[mask], y[mask], kern)

    placebos = {w: res["placebos"]["all_models"][w]["cells"] for w in REC_PLACEBO_REJECT}

    def pval(cell: dict) -> float:
        return float(cell["p_value"] if "p_value" in cell else cell["p"])

    n_rej = {w: sum(1 for c in placebos[w] if pval(c) < 0.05) for w in placebos}
    ok = (
        np.isclose(unw["estimate"], REC_UNW_EST, atol=ATOL)
        and np.isclose(unw["se"], REC_UNW_SE, atol=ATOL)
        and np.isclose(ciw["estimate"], REC_CIW_EST, atol=ATOL)
        and np.isclose(ciw["se"], REC_CIW_SE, atol=ATOL)
        and np.allclose(
            [fits["unweighted"]["L"][1], fits["unweighted"]["R"][1]],
            REC_SLOPES_UNW, atol=5e-5,
        )
        and np.allclose(
            [fits["ci_weighted"]["L"][1], fits["ci_weighted"]["R"][1]],
            REC_SLOPES_CIW, atol=5e-5,
        )
        and n_rej == REC_PLACEBO_REJECT
        and all(len(placebos[w]) == 8 for w in placebos)
    )
    print(
        f"[unw] {unw['estimate']:+.6f} (se {unw['se']:.6f}) "
        f"slopes {fits['unweighted']['L'][1]:.4f}/{fits['unweighted']['R'][1]:.4f} | "
        f"[ciw] {ciw['estimate']:+.6f} (se {ciw['se']:.6f}) "
        f"slopes {fits['ci_weighted']['L'][1]:.4f}/{fits['ci_weighted']['R'][1]:.4f} | "
        f"placebos reject unw {n_rej['unweighted']}/8, ciw {n_rej['ci_weighted']}/8 "
        f"{'OK' if ok else 'MISMATCH'}"
    )
    if not ok and not args.skip_checks:
        sys.exit(
            "estimates do not match the numbers of record — wrong results "
            "vintage? (--skip-checks to override)"
        )

    plt.rcParams.update(
        {
            "font.size": 8.5,
            "axes.edgecolor": DGRAY,
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

    # --- (a) all-models ECI vs days since o1, with per-side local fits ------
    ax1.plot(
        d, y, ls="none", marker="o", ms=2.4, mfc="none", mec=GRAY, mew=0.5,
        label="model (n=211)",
    )
    for name, color, label in [
        ("unweighted", BLUE, "unweighted fit"),
        ("ci_weighted", ORANGE, "CI-weighted fit"),
    ]:
        for side, xs in [("L", np.array([-BW, 0.0])), ("R", np.array([0.0, BW]))]:
            a, b = fits[name][side]
            ax1.plot(
                xs, a + b * xs, color=color, lw=1.6,
                label=label if side == "L" else None,
            )
    ax1.axvline(0.0, color=VERMILLION, lw=0.9, zorder=0)
    ax1.text(14, 158, "o1", color=VERMILLION, fontsize=7.5)
    ax1.set_xlim(-600, 680)
    ax1.set_xlabel("Days since o1 (2024-09-12)")
    ax1.set_ylabel("ECI score")
    ax1.legend(
        loc="upper left", fontsize=7.0, handlelength=1.4,
        frameon=True, facecolor="white", edgecolor="none", framealpha=1.0,
    )
    ax1.set_title("(a) ECI, local-linear fits (bw 540, tri)", fontsize=9, loc="left")

    # --- (b) kink at o1 vs placebo cutoffs, both weightings -----------------
    series = [
        ("unweighted", BLUE, -12.0, float(unw["estimate"]), float(unw["se"]),
         float(unw["p"]) < 0.05),
        ("ci_weighted", ORANGE, +12.0, float(ciw["estimate"]), float(ciw["se"]),
         float(ciw["p"]) < 0.05),
    ]
    for name, color, off, est0, se0, sig0 in series:
        cells = [
            {"cutoff": 0.0, "estimate": est0, "se": se0, "sig": sig0}
        ] + [
            {
                "cutoff": float(c["cutoff"]),
                "estimate": float(c["estimate"]),
                "se": float(c["se"]),
                "sig": pval(c) < 0.05,
            }
            for c in placebos[name]
        ]
        for c in cells:
            x = c["cutoff"] + off
            lo, hi = c["estimate"] - 1.96 * c["se"], c["estimate"] + 1.96 * c["se"]
            ax2.plot([x, x], [lo, hi], color=color, lw=1.1)
            ax2.plot(
                [x], [c["estimate"]], marker="o", ms=4.2,
                mfc=color if c["sig"] else "white", mec=color, mew=1.1, color=color,
                label=(
                    ("unweighted" if name == "unweighted" else "CI-weighted")
                    if c["cutoff"] == 0.0 else None
                ),
            )
    ax2.axhline(0.0, color=INK, lw=0.7)
    ax2.axvline(0.0, color=VERMILLION, lw=0.9, zorder=0)
    ax2.text(10, 0.088, "o1", color=VERMILLION, fontsize=7.5)
    ax2.set_xlabel("Cutoff (days relative to o1)")
    ax2.set_ylabel("Kink (ECI pts/day)")
    ax2.set_xlim(-430, 430)
    ax2.set_ylim(-0.078, 0.105)
    ax2.legend(loc="lower left", fontsize=7.0, handlelength=1.0)
    ax2.set_title("(b) Kink at o1 vs. placebo cutoffs", fontsize=9, loc="left")
    ax2.text(
        0.98, 0.02,
        f"filled = nominal $p<0.05$ (HC1)\n"
        f"placebos reject: {n_rej['unweighted']}/8 unw., "
        f"{n_rej['ci_weighted']}/8 CI-w.\nbw 540 d, triangular, donut 0",
        transform=ax2.transAxes, fontsize=6.6, color=DGRAY, va="bottom", ha="right",
    )

    out = Path(__file__).resolve().parent
    fig.savefig(out / "fig1.pdf")
    fig.savefig(out / "fig1.png", dpi=200)
    print(f"wrote {out / 'fig1.pdf'} and fig1.png")


if __name__ == "__main__":
    main()
