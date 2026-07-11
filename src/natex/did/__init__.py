"""SuDDDS: subset discovery of difference-in-differences (thesis ch. 6 lineage)."""

from natex.did.background import DiDBackground, fit_did_background
from natex.did.panel import CategoricalPanel, build_panel, quantile_bins

__all__ = [
    "CategoricalPanel",
    "DiDBackground",
    "build_panel",
    "fit_did_background",
    "quantile_bins",
]
