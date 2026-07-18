"""Regenerate the four main figures of the epoch-kinks paper (PDF + PNG).

Reads the PUBLIC Epoch AI CSVs (CC-BY 4.0, https://epoch.ai/data) from a local
extraction directory passed as ``--data-root`` — the data are NOT committed to
this repository. Input construction mirrors the analysis pass of record
(``kink-runs/run_kink_pass.py``, 2026-07-16); every headline estimate is
recomputed live with natex v0.2.0 and asserted against the numbers of record in
``docs/case_studies/epoch-kinks.md`` before a figure is written, so a drifted
data vintage fails loudly instead of silently redrawing different numbers.

Deterministic: the estimators are pure weighted least squares and
``SOURCE_DATE_EPOCH`` is pinned so the PDF/PNG bytes are reproducible.

Run (from the repo root):

    uv run python paper/figures/make_figures.py \
        --data-root /path/to/epoch-data/extracted
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("SOURCE_DATE_EPOCH", "0")  # reproducible figure bytes

import numpy as np
import pandas as pd

from natex import difference_in_kinks, regression_kink
from natex.report.figures import OKABE_ITO, kink_fit_plot

O1 = pd.Timestamp("2024-09-12")  # o1-preview release
EXPORT2 = pd.Timestamp("2023-10-17")  # second US export-control round

# Numbers of record (docs/case_studies/epoch-kinks.md). abs tolerance covers
# floating-point noise only — a changed data vintage must fail.
RECORD = {
    "gpqa": (0.002583, 0.000779),
    "metr": (0.006011, 0.002819),
    "eci": (-0.007254, 0.008947),
    "china": (-0.001540, 0.000467),
}
ATOL = 5e-6


def check(name: str, tau: float, se: float, skip: bool) -> None:
    want_tau, want_se = RECORD[name]
    ok = np.isclose(tau, want_tau, atol=ATOL) and np.isclose(se, want_se, atol=ATOL)
    print(f"[{name}] tau={tau:+.6f} se={se:.6f} (record {want_tau:+.6f}/{want_se:.6f})"
          f" {'OK' if ok else 'MISMATCH'}")
    if not ok and not skip:
        sys.exit(f"{name}: estimate does not match the numbers of record — "
                 "wrong data vintage? (--skip-checks to override)")


def styled_kink_figure(running, outcome, *, bandwidth, estimate, out_stem, cutoff_label,
                       xlabel, ylabel, title):
    """natex kink_fit_plot with paper axis labels/title re-applied.

    kink_fit_plot closes its figure after saving; intercept the close, relabel,
    and re-save to the same paths. Presentational only — no numbers change.
    """
    import matplotlib.pyplot as plt

    captured = []
    original_close = plt.close
    plt.close = lambda *a, **k: captured.extend(a)
    try:
        paths = kink_fit_plot(running, outcome, 0.0, bandwidth, out_stem,
                              estimate=estimate, cutoff_label=cutoff_label)
    finally:
        plt.close = original_close
    fig = captured[0]
    ax = fig.axes[0]
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    fig.savefig(paths.png, dpi=150, bbox_inches="tight")
    fig.savefig(paths.pdf, bbox_inches="tight")
    plt.close(fig)
    return paths


def fig_gpqa(root: Path, out: Path, skip: bool) -> None:
    df = pd.read_csv(root / "benchmark_data" / "gpqa_diamond.csv")
    df = df[df["Release date"].notna() & df["mean_score"].notna()].copy()
    df["days"] = (pd.to_datetime(df["Release date"]) - O1).dt.days.astype(float)
    df = df.sort_values("days")
    p = df["mean_score"].clip(1e-6, 1 - 1e-6)
    y = np.log(p / (1 - p)).to_numpy()
    x = df["days"].to_numpy()
    est = regression_kink(y, x, policy_kink=1.0, bandwidth=540)
    check("gpqa", est.tau, est.se, skip)
    styled_kink_figure(
        x, y, bandwidth=540, estimate=est, out_stem=out / "fig_gpqa",
        cutoff_label="o1-preview (2024-09-12)",
        xlabel="days since o1-preview (2024-09-12)",
        ylabel="logit(mean GPQA-Diamond score)",
        title="GPQA-Diamond: sharp RKD in calendar time (bw 540 d, triangular)")


def fig_metr(root: Path, out: Path, skip: bool) -> None:
    df = pd.read_csv(root / "benchmark_data" / "metr_time_horizons_external.csv")
    df = df[df["Release date"].notna()].copy()
    df["days"] = (pd.to_datetime(df["Release date"]) - O1).dt.days.astype(float)
    y = np.log2(df["Time horizon"]).to_numpy()
    x = df["days"].to_numpy()
    est = regression_kink(y, x, policy_kink=1.0, bandwidth=720)
    check("metr", est.tau, est.se, skip)
    styled_kink_figure(
        x, y, bandwidth=720, estimate=est, out_stem=out / "fig_metr",
        cutoff_label="o1-preview (2024-09-12)",
        xlabel="days since o1-preview (2024-09-12)",
        ylabel="log$_2$(50% time horizon, minutes)",
        title="METR 50% time horizon: sharp RKD in calendar time (bw 720 d, triangular)")


def fig_eci(root: Path, out: Path, skip: bool) -> None:
    df = pd.read_csv(root / "benchmark_data" / "epoch_capabilities_index.csv")
    df = df[df["Release date"].notna() & df["ECI Score"].notna()].copy()
    df["days"] = (pd.to_datetime(df["Release date"]) - O1).dt.days.astype(float)
    df = df.sort_values(["days", "ECI Score"])
    y = df["ECI Score"].to_numpy()
    x = df["days"].to_numpy()
    est = regression_kink(y, x, policy_kink=1.0, bandwidth=540)
    check("eci", est.tau, est.se, skip)
    styled_kink_figure(
        x, y, bandwidth=540, estimate=est, out_stem=out / "fig_eci",
        cutoff_label="o1-preview (2024-09-12)",
        xlabel="days since o1-preview (2024-09-12)",
        ylabel="Epoch Capabilities Index score",
        title="ECI (all models): falsification RKD — null (bw 540 d, triangular)")


def _wls_line(x: np.ndarray, y: np.ndarray, bandwidth: float) -> tuple[float, float]:
    """Triangular-kernel weighted local-linear fit; returns (intercept, slope).

    Identical to the natex per-cell fit (pure WLS, triangular weights), used
    here only to DRAW the four group-side lines of the DiK figure.
    """
    w = np.sqrt(1.0 - np.abs(x) / bandwidth)
    design = np.column_stack([np.ones_like(x), x])
    beta, *_ = np.linalg.lstsq(design * w[:, None], y * w, rcond=None)
    return float(beta[0]), float(beta[1])


def fig_china(root: Path, out: Path, skip: bool) -> None:
    import matplotlib.pyplot as plt

    d0 = pd.read_csv(root / "ai_chip_owners" / "cumulative_by_designer.csv")
    d0 = d0[d0["Incomplete"] != True].copy()  # noqa: E712 — drop flagged quarter
    d0["qend"] = pd.to_datetime(d0["End date"])

    def series(owners: list[str]) -> tuple[np.ndarray, np.ndarray]:
        s = (d0[d0.Owner.isin(owners)]
             .groupby("qend")["Compute estimate in H100e (median)"].sum())
        s = s[s > 0]
        days = (s.index - EXPORT2).days.astype(float).to_numpy()
        return days, np.log(s.to_numpy())

    x_cn, y_cn = series(["China"])
    x_hy, y_hy = series(["Amazon", "Microsoft", "Google", "Meta"])

    stacked = pd.DataFrame({
        "days": np.concatenate([x_cn, x_hy]),
        "china": np.concatenate([np.ones_like(x_cn), np.zeros_like(x_hy)]),
        "ln": np.concatenate([y_cn, y_hy]),
    })
    est = difference_in_kinks(stacked["ln"].to_numpy(), stacked["days"].to_numpy(),
                              stacked["china"].to_numpy() == 1,
                              policy_kink_change=1.0, bandwidth=548)
    check("china", est.tau, est.se, skip)

    bw = 548.0
    pad = 1.15 * bw
    blue, verm = OKABE_ITO[0], OKABE_ITO[3]
    rc = {"font.family": "DejaVu Sans", "axes.spines.top": False,
          "axes.spines.right": False, "savefig.bbox": "tight"}
    with plt.rc_context(rc):
        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        for x, y, color, marker, name in ((x_hy, y_hy, blue, "o", "Hyperscalers"),
                                          (x_cn, y_cn, verm, "s", "China (legal)")):
            inside = np.abs(x) <= bw
            context = ~inside & (np.abs(x) <= pad)
            ax.scatter(x[context], y[context], s=16, marker=marker, color="0.8", zorder=1)
            ax.scatter(x[inside], y[inside], s=22, marker=marker, color=color,
                       alpha=0.85, zorder=3, label=name)
            grids = []
            for side in ("left", "right"):
                mask = inside & ((x < 0) if side == "left" else (x >= 0))
                a, b = _wls_line(x[mask], y[mask], bw)
                grid = np.linspace(-bw, 0, 64) if side == "left" else np.linspace(0, bw, 64)
                ax.plot(grid, a + b * grid, color=color, linewidth=1.8, zorder=2)
                grids.append((a, b))
            a_left, b_left = grids[0]
            right = np.linspace(0, bw, 64)
            ax.plot(right, a_left + b_left * right, color=color, linewidth=1.2,
                    linestyle="--", alpha=0.8, zorder=2)
        ax.plot([], [], color="0.4", linewidth=1.2, linestyle="--",
                label="pre-trend continued")
        ax.axvline(0.0, color="black", linestyle="--", linewidth=1.2,
                   label="export controls (2023-10-17)")
        ax.text(0.98, 0.02, f"DiK = {est.tau:+.5f} ln/day\nse = {est.se:.5f}",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=8,
                color="0.25")
        ax.legend(frameon=False, fontsize=8, loc="upper left")
        ax.set_xlim(-pad, pad)
        ax.set_xlabel("days since 2023-10-17 export-control round")
        ax.set_ylabel("ln cumulative H100e stock")
        ax.set_title("China vs hyperscalers: group difference-in-kinks (bw 548 d, triangular)",
                     fontsize=10)
        fig.savefig(out / "fig_china.png", dpi=150)
        fig.savefig(out / "fig_china.pdf")
        plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", type=Path,
                    default=Path("/Users/haukehillebrandt/dev/epoch-data/extracted"),
                    help="local extraction of the public Epoch AI datasets "
                         "(https://epoch.ai/data); not committed to the repo")
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent,
                    help="output directory for fig_*.pdf / fig_*.png")
    ap.add_argument("--skip-checks", action="store_true",
                    help="draw figures even if estimates mismatch the numbers of record")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for fn in (fig_gpqa, fig_metr, fig_eci, fig_china):
        fn(args.data_root, args.out, args.skip_checks)
    print(f"figures written to {args.out}")


if __name__ == "__main__":
    main()
