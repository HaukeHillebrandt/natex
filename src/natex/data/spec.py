"""Dataset abstraction: column-role mapping plus numeric views used by the scan."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from pydantic import BaseModel


class DatasetSpec(BaseModel):
    treatment: str
    outcome: str | None = None
    forcing: list[str]
    covariates: list[str]
    time: str | None = None


class Dataset:
    def __init__(self, df: pd.DataFrame, spec: DatasetSpec):
        missing = [c for c in [spec.treatment, *spec.forcing, *spec.covariates] if c not in df.columns]
        if missing:
            raise ValueError(f"columns not in dataframe: {missing}")
        bad = [c for c in spec.forcing if not pd.api.types.is_numeric_dtype(df[c])]
        if bad:
            raise ValueError(f"forcing columns must be numeric: {bad}")
        if not set(spec.forcing) <= set(spec.covariates):
            raise ValueError("forcing columns must be a subset of covariates")
        # Drop rows with missing values in scan-relevant columns only (never the
        # outcome: discovery uses only (x, z, T) and must tolerate NaN in y).
        scan_cols = list(dict.fromkeys([spec.treatment, *spec.forcing, *spec.covariates]))
        self.df = df.dropna(subset=scan_cols).reset_index(drop=True)
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
    ) -> "Dataset":
        df = pd.read_csv(path)
        reserved = {treatment} | ({outcome} if outcome else set())
        if covariates == "auto":
            covariates = [c for c in df.columns if c not in reserved]
        if forcing is None:
            forcing = [c for c in covariates if pd.api.types.is_numeric_dtype(df[c])]
        spec = DatasetSpec(
            treatment=treatment, outcome=outcome, forcing=forcing, covariates=list(covariates), time=time
        )
        return cls(df, spec)

    @property
    def n(self) -> int:
        return len(self.df)

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

    @property
    def X(self) -> np.ndarray:
        return pd.get_dummies(self.df[self.spec.covariates], dtype=float).to_numpy(dtype=float)

    @property
    def treatment_is_binary(self) -> bool:
        vals = np.unique(self.T[~np.isnan(self.T)])
        return vals.size <= 2 and set(vals.tolist()) <= {0.0, 1.0}
