"""Shared helpers for backtests that need the real local datasets (env NATEX_DATA)."""

import os

import pytest

from natex.data.registry import REGISTRY, load_dataset
from natex.data.spec import Dataset


def load_or_skip(name: str, **kwargs) -> Dataset:
    """Load a registered dataset, or skip with fetch instructions if unavailable."""
    if not os.environ.get("NATEX_DATA") and "root" not in kwargs:
        pytest.skip(f"NATEX_DATA not set; dataset {name!r}: {REGISTRY[name].source}")
    try:
        return load_dataset(name, **kwargs)
    except FileNotFoundError as exc:
        pytest.skip(str(exc))


# Session scope: the fixture only hands back the (stateless) helper above, so
# module-scoped fixtures (e.g. the 44k-row LSO scan) may depend on it.
@pytest.fixture(name="load_or_skip", scope="session")
def _load_or_skip_fixture():
    return load_or_skip
