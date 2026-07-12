"""Reporting layer: results bundle, figures, paper templates (spec section 7).

This package RENDERS numbers already computed by discover/validate/estimate —
it adds no inference code beyond the presentational :func:`ivw_pooled`
combiner (documented as indicative only).
"""

from natex.report.bundle import BUNDLE_SCHEMA, PooledEffect, ResultsBundle, ivw_pooled
from natex.report.paper import BANNER, PaperResult, render_paper

__all__ = [
    "BANNER",
    "BUNDLE_SCHEMA",
    "PaperResult",
    "PooledEffect",
    "ResultsBundle",
    "ivw_pooled",
    "render_paper",
]
