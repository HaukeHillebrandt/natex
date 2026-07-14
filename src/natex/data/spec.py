"""Dataset abstraction: column-role mapping plus numeric views used by the scan."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from pydantic import BaseModel


class DatasetSpec(BaseModel):
    treatment: str
    outcome: str | None = None
    forcing: list[str]  # may be [] (DiD-only datasets have no forcing variable)
    covariates: list[str]
    time: str | None = None
    unit: str | None = None  # cross-sectional unit id column (e.g. "state")


class Dataset:
    def __init__(self, df: pd.DataFrame, spec: DatasetSpec):
        # Every declared role is validated eagerly — outcome included (issue
        # #26: a missing outcome column must fail here, not as a raw KeyError
        # from ``ds.y`` after an expensive discovery scan). Only the column's
        # EXISTENCE is checked: NaN outcome values stay tolerated (LSO policy).
        roles = [spec.treatment, *([spec.outcome] if spec.outcome is not None else []),
                 *spec.forcing, *spec.covariates]
        missing = [c for c in roles if c not in df.columns]
        if missing:
            raise ValueError(f"columns not in dataframe: {missing}")
        bad = [c for c in spec.forcing if not pd.api.types.is_numeric_dtype(df[c])]
        if bad:
            raise ValueError(f"forcing columns must be numeric: {bad}")
        if not set(spec.forcing) <= set(spec.covariates):
            raise ValueError("forcing columns must be a subset of covariates")
        if spec.time is not None:
            if spec.time not in df.columns:
                raise ValueError(f"time column not in dataframe: {spec.time}")
            if not pd.api.types.is_numeric_dtype(df[spec.time]):
                raise ValueError(f"time column must be numeric: {spec.time}")
        if spec.unit is not None and spec.unit not in df.columns:
            raise ValueError(f"unit column not in dataframe: {spec.unit}")
        # Drop rows with missing values in scan-relevant columns only (never the
        # outcome: discovery uses only (x, z, T) and must tolerate NaN in y).
        # +/-inf counts as missing too (issue #20): dropna leaves inf through,
        # after which Z_std is all-NaN (mean/std over inf), build_geometry
        # crashes far from the cause, and inf in the treatment silently flips
        # treatment_is_binary. Same silent-drop remedy as the NaN policy.
        extra = [c for c in (spec.time, spec.unit) if c is not None]
        scan_cols = list(dict.fromkeys([spec.treatment, *spec.forcing, *spec.covariates, *extra]))
        # Row-loss bookkeeping (issue #1): the deletion policy is by design,
        # but it must never be silent. Every row with a missing/non-finite scan
        # value IS dropped, so each column's bad count among the INPUT rows is
        # exactly the loss attributable to it. Lossless columns are absent —
        # ints only, no fabricated zeros (NaN-never-0.0 lineage).
        self.n_rows_input = len(df)
        # The raw pre-deletion frame (issue #30): per-candidate rebuilds in
        # ``natex.discover`` must re-run listwise deletion from the ORIGINAL
        # rows once a foreign outcome leaves the covariates — rebuilding from
        # the already-truncated ``self.df`` makes those rows unrecoverable and
        # silently defeats this very bookkeeping. A reference, not a copy:
        # ``dropna`` below already returns a new frame.
        self.df_input = df
        dropped: dict[str, int] = {}
        for c in scan_cols:
            if pd.api.types.is_numeric_dtype(df[c]):
                bad = ~np.isfinite(df[c].to_numpy(dtype=float, na_value=np.nan))
            else:
                bad = df[c].isna().to_numpy()
            if bad.any():
                dropped[c] = int(bad.sum())
        self.nan_dropped_by_column = dropped
        clean = df.dropna(subset=scan_cols)
        num_cols = [c for c in scan_cols if pd.api.types.is_numeric_dtype(clean[c])]
        if num_cols:
            clean = clean.loc[np.isfinite(clean[num_cols].to_numpy(dtype=float)).all(axis=1)]
        self.df = clean.reset_index(drop=True)
        self.spec = spec

    @classmethod
    def from_csv(
        cls,
        path: str | Path,
        treatment: str,
        outcome: str | None = None,
        forcing: list[str] | None = None,
        covariates: str | list[str] = "auto",
        time: str | None = None,
        unit: str | None = None,
    ) -> "Dataset":
        df = pd.read_csv(path)
        reserved = {treatment} | ({outcome} if outcome else set())
        if covariates == "auto":
            covariates = [c for c in df.columns if c not in reserved]
        if forcing is None:
            forcing = [c for c in covariates if pd.api.types.is_numeric_dtype(df[c])]
        spec = DatasetSpec(
            treatment=treatment,
            outcome=outcome,
            forcing=forcing,
            covariates=list(covariates),
            time=time,
            unit=unit,
        )
        return cls(df, spec)

    @property
    def n(self) -> int:
        return len(self.df)

    @property
    def n_rows_used(self) -> int:
        """Rows the scan actually sees: ``n_rows_input`` minus listwise deletion."""
        return len(self.df)

    def top_row_loss(self, m: int = 3) -> dict[str, int]:
        """The ``m`` largest per-column row losses, descending (ties by name)."""
        items = sorted(self.nan_dropped_by_column.items(), key=lambda kv: (-kv[1], kv[0]))
        return dict(items[:m])

    @property
    def T(self) -> np.ndarray:
        return self.df[self.spec.treatment].to_numpy(dtype=float)

    @property
    def y(self) -> np.ndarray | None:
        if self.spec.outcome is None:
            return None
        return self.df[self.spec.outcome].to_numpy(dtype=float)

    @property
    def Z(self) -> np.ndarray:
        return self.df[self.spec.forcing].to_numpy(dtype=float)

    @property
    def Z_std(self) -> np.ndarray:
        z = self.Z
        sd = z.std(axis=0, ddof=0)
        sd[sd == 0] = 1.0
        return (z - z.mean(axis=0)) / sd

    def standardize(self, z: np.ndarray) -> np.ndarray:
        """Map raw forcing-space points to Z_std coordinates.

        Uses the SAME per-column moments as ``Z_std`` (mean and sd with
        ddof=0, zero-sd columns pass through unscaled), so
        ``standardize(self.Z)`` is bitwise equal to ``self.Z_std``.
        """
        z = np.asarray(z, dtype=float)
        Z = self.Z
        if z.ndim != 2 or z.shape[1] != Z.shape[1]:
            raise ValueError(f"expected shape (m, {Z.shape[1]}) in forcing space, got {z.shape}")
        sd = Z.std(axis=0, ddof=0)
        sd[sd == 0] = 1.0
        return (z - Z.mean(axis=0)) / sd

    @property
    def X(self) -> np.ndarray:
        return pd.get_dummies(self.df[self.spec.covariates], dtype=float).to_numpy(dtype=float)

    @property
    def treatment_is_binary(self) -> bool:
        vals = np.unique(self.T[~np.isnan(self.T)])
        return vals.size <= 2 and set(vals.tolist()) <= {0.0, 1.0}
