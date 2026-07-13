"""Deterministic dataset profiler: Stage 0 of the analyst pass, no LLM."""

from __future__ import annotations

import dataclasses
import json
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

_TIME_NAME = re.compile(r"date|time|year|month|quarter", re.IGNORECASE)


@dataclass
class ColumnProfile:
    name: str
    dtype: str
    n_unique: int
    missing_frac: float
    is_numeric: bool
    is_binary: bool
    is_time_like: bool
    # Structural-missingness evidence (issue #6): positional row index of the
    # first non-missing cell (None if the column is entirely missing), and the
    # fraction of the column's missing cells that lie in the all-missing row
    # prefix before it (None — never 0.0 — when the column has no missing cell).
    first_valid_index: int | None = None
    prefix_missing_frac: float | None = None
    # Numeric, fully observed, and non-decreasing in row order: such a column
    # can express a row boundary as a declarative ``>=`` filter (issue #6).
    is_monotone: bool = False


@dataclass
class IntakeProfile:
    n_rows: int
    columns: list[ColumnProfile]
    panel_candidates: list[tuple[str, str]] = field(default_factory=list)
    forcing_candidates: list[str] = field(default_factory=list)
    treatment_candidates: list[str] = field(default_factory=list)
    # For every candidate boundary row b (a column's first_valid_index > 0):
    # each monotone column's value AT row b, so a data-blind backend can turn
    # the positional boundary into a value filter (issue #6). JSON round trips
    # turn the int keys into strings; consumers must accept both.
    boundary_values: dict[int, dict[str, float]] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(dataclasses.asdict(self), default=str, indent=1)


def _is_time_like(s: pd.Series, name: str) -> bool:
    if _TIME_NAME.search(name):
        return True
    if pd.api.types.is_datetime64_any_dtype(s):
        return True
    if pd.api.types.is_numeric_dtype(s):
        v = s.dropna()
        if len(v) and np.allclose(v, v.round()) and 1800 <= v.min() and v.max() <= 2100:
            return True
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
        try:
            pd.to_datetime(s.dropna().head(20), errors="raise", format="mixed")
            return True
        except (ValueError, TypeError):
            return False
    return False


def profile(df: pd.DataFrame) -> IntakeProfile:
    cols: list[ColumnProfile] = []
    for name in df.columns:
        s = df[name]
        numeric = pd.api.types.is_numeric_dtype(s)
        uniq = int(s.nunique(dropna=True))
        vals = set(pd.unique(s.dropna()).tolist()) if uniq <= 2 else set()
        notna = s.notna().to_numpy()
        n_missing = int((~notna).sum())
        # Positional first-valid row: every row before it is missing, so the
        # prefix holds exactly ``first_valid`` of the column's missing cells.
        first_valid = int(np.argmax(notna)) if notna.any() else None
        prefix_frac = first_valid / n_missing if first_valid is not None and n_missing else None
        cols.append(
            ColumnProfile(
                name=str(name),
                dtype=str(s.dtype),
                n_unique=uniq,
                missing_frac=float(s.isna().mean()),
                is_numeric=bool(numeric),
                is_binary=bool(numeric and uniq <= 2 and vals <= {0, 1, 0.0, 1.0}),
                is_time_like=_is_time_like(s, str(name)),
                first_valid_index=first_valid,
                prefix_missing_frac=prefix_frac,
                is_monotone=bool(
                    numeric and len(s) and n_missing == 0 and s.is_monotonic_increasing
                ),
            )
        )
    n = len(df)
    monotone_cols = [c.name for c in cols if c.is_monotone]
    boundary_values: dict[int, dict[str, float]] = {
        b: {m: float(df[m].iloc[b]) for m in monotone_cols}
        for b in sorted({c.first_valid_index for c in cols if c.first_valid_index})
    }
    panel: list[tuple[str, str]] = []
    time_cols = [c.name for c in cols if c.is_time_like]
    id_cols = [c.name for c in cols if not c.is_numeric and 1 < c.n_unique < n]
    for u in id_cols:
        for t in time_cols:
            if u != t and df.groupby([u, t]).ngroups >= 0.95 * n:
                panel.append((u, t))
    forcing = [c.name for c in cols if c.is_numeric and not c.is_binary and c.n_unique >= 20]
    treatment = [c.name for c in cols if c.is_binary]
    return IntakeProfile(
        n_rows=n, columns=cols, panel_candidates=panel,
        forcing_candidates=forcing, treatment_candidates=treatment,
        boundary_values=boundary_values,
    )
