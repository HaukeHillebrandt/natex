#!/usr/bin/env python3
"""Draw figures/placebo_size.pdf|png from figures/placebo_size.csv.

The CSV holds the numbers of record (one row per series): the share of shifted
placebo cutoffs at which the nominal HC1 5% test rejected, the number of placebo
positions, and the placebo-calibrated p of the declared cutoff.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
FIG = HERE / "figures"

OKABE_ITO = {"blue": "#0072B2", "orange": "#E69F00", "grey": "#999999", "vermillion": "#D55E00"}


def main() -> None:
    df = pd.read_csv(FIG / "placebo_size.csv").sort_values("rejection_rate")
    fig, ax = plt.subplots(figsize=(10, 4.8))
    y = range(len(df))
    colors = [
        OKABE_ITO["blue"] if p <= 0.05 else OKABE_ITO["grey"] for p in df["calibrated_p"]
    ]
    ax.barh(list(y), df["rejection_rate"] * 100, color=colors, height=0.62)
    ax.axvline(5, color=OKABE_ITO["vermillion"], linestyle="--", linewidth=1.2)
    ax.text(5.6, len(df) - 0.45, "nominal size 5%", color=OKABE_ITO["vermillion"],
            fontsize=9, va="center")
    for yi, (rate, n, p) in enumerate(zip(df["rejection_rate"], df["n_placebos"],
                                          df["calibrated_p"])):
        ax.text(rate * 100 + 1.2, yi, f"N = {n}, calibrated p = {p:.2f}", va="center",
                fontsize=8.5)
    ax.set_yticks(list(y))
    ax.set_yticklabels(df["series"], fontsize=9)
    ax.set_xlim(0, 140)
    ax.set_xlabel("Share of shifted placebo cutoffs at which the nominal HC1 test rejects at 5% (%)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("Empirical size of the nominal kink test on eight AI time series",
                 fontsize=11, loc="left")
    fig.text(0.01, 0.01,
             "Blue: declared cutoff is the most extreme of its placebo grid (calibrated p at the"
             " floor 1/(N+1)). Grey: not.", fontsize=8, color="#444444")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(FIG / "placebo_size.pdf")
    fig.savefig(FIG / "placebo_size.png", dpi=150)
    print("wrote", FIG / "placebo_size.pdf", "and .png")


if __name__ == "__main__":
    main()
