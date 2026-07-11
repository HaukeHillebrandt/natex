"""Instrument and synthetic-control donor discovery (phase 5).

Instrument SELECTION reads only (T, pool, controls) — never the outcome
(the phase's discovery-honesty analog; audit item 10 requires the actual
post-selection first-stage F, never assumed relevance). SC donor scoring
and weighting read only PRE-period outcomes — the post period is the
estimation target (documented method property, mutation-tested).
"""

from natex.iv.donors import (
    DonorScore,
    DonorSelectionResult,
    select_donors,
    select_donors_from_dataset,
    unit_time_matrix,
)
from natex.iv.pipeline import InstrumentDiscovery, discover_instruments
from natex.iv.search import InstrumentSearchResult, select_instruments

__all__ = [
    "DonorScore",
    "DonorSelectionResult",
    "InstrumentDiscovery",
    "InstrumentSearchResult",
    "discover_instruments",
    "select_donors",
    "select_donors_from_dataset",
    "select_instruments",
    "unit_time_matrix",
]
