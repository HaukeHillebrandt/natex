"""Reporting layer: results bundle, figures, paper templates (spec section 7).

This package RENDERS numbers already computed by discover/validate/estimate —
it adds no inference code beyond the presentational :func:`ivw_pooled`
combiner (documented as indicative only).
"""

from natex.report.bundle import BUNDLE_SCHEMA, PooledEffect, ResultsBundle, ivw_pooled

__all__ = ["BUNDLE_SCHEMA", "PooledEffect", "ResultsBundle", "ivw_pooled"]
