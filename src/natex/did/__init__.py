"""SuDDDS: subset discovery of difference-in-differences (thesis ch. 6 lineage)."""

from natex.did.background import DiDBackground, fit_did_background
from natex.did.controls import ControlResult, dd_control, gess_control, synthetic_control
from natex.did.mdss import SingleDeltaPriority, SubsetState, mdss_optimize
from natex.did.metrics import subset_precision_recall
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
from natex.did.suddds import (
    DiDDiscovery,
    SuDDDSResult,
    default_windows,
    optimize_t0,
    suddds_scan,
)

__all__ = [
    "CategoricalPanel",
    "ControlResult",
    "DiDBackground",
    "DiDDiscovery",
    "SingleDeltaPriority",
    "SubsetState",
    "SuDDDSResult",
    "WindowStats",
    "bernoulli_window_llr_masks",
    "build_panel",
    "dd_control",
    "default_windows",
    "double_beta_llr_masks",
    "double_beta_q",
    "fit_did_background",
    "gess_control",
    "mdss_optimize",
    "optimize_t0",
    "quantile_bins",
    "single_delta_llr",
    "single_delta_stats",
    "subset_precision_recall",
    "suddds_scan",
    "synthetic_control",
    "window_stats",
    "working_residuals",
]
