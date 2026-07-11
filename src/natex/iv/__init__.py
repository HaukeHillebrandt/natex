"""Instrument and synthetic-control donor discovery (phase 5).

Instrument SELECTION reads only (T, pool, controls) — never the outcome
(the phase's discovery-honesty analog; audit item 10 requires the actual
post-selection first-stage F, never assumed relevance).
"""

from natex.iv.pipeline import InstrumentDiscovery, discover_instruments
from natex.iv.search import InstrumentSearchResult, select_instruments

__all__ = [
    "InstrumentDiscovery",
    "InstrumentSearchResult",
    "discover_instruments",
    "select_instruments",
]
