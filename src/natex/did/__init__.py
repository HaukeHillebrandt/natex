"""SuDDDS: subset discovery of difference-in-differences (thesis ch. 6 lineage)."""

from natex.did.background import DiDBackground, fit_did_background
from natex.did.panel import CategoricalPanel, build_panel, quantile_bins
from natex.did.statistics import (
    WindowStats,
    bernoulli_window_llr_masks,
    double_beta_llr_masks,
    double_beta_q,
    single_delta_llr,
    single_delta_stats,
    window_stats,
    working_residuals,
)

__all__ = [
    "CategoricalPanel",
    "DiDBackground",
    "WindowStats",
    "bernoulli_window_llr_masks",
    "build_panel",
    "double_beta_llr_masks",
    "double_beta_q",
    "fit_did_background",
    "quantile_bins",
    "single_delta_llr",
    "single_delta_stats",
    "window_stats",
    "working_residuals",
]
