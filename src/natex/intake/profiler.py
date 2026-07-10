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


@dataclass
class IntakeProfile:
    n_rows: int
    columns: list[ColumnProfile]
    panel_candidates: list[tuple[str, str]] = field(default_factory=list)
    forcing_candidates: list[str] = field(default_factory=list)
    treatment_candidates: list[str] = field(default_factory=list)

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
        cols.append(
            ColumnProfile(
                name=str(name),
                dtype=str(s.dtype),
                n_unique=uniq,
                missing_frac=float(s.isna().mean()),
                is_numeric=bool(numeric),
                is_binary=bool(numeric and uniq <= 2 and vals <= {0, 1, 0.0, 1.0}),
                is_time_like=_is_time_like(s, str(name)),
            )
        )
    n = len(df)
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
    )
