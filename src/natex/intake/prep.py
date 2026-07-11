"""Declarative data-preparation plans: Stage 0 of the analyst pass (spec 6a).

The LLM only ever PROPOSES a :class:`PrepPlan` as pure data; this executor is
the only code that touches the DataFrame — the LLM never emits code. Plans
are validated against the actual frame (unknown columns/roles/ops are
rejected BY NAME) and applied deterministically: the subsample seed lives IN
the plan (``study()`` draws it from the pipeline Generator, then stores it),
so a serialized plan replays bitwise without the caller's Generator.

NaN policy (house rule): NaN is never coerced to 0.0 — filter comparisons on
NaN are False (row dropped, value untouched), ordinal encoding of a NaN cell
is NaN, and discretize leaves NaN cells NaN.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, field_validator

from natex.did.panel import quantile_bins

ROLES = ("treatment", "outcome", "forcing", "covariate", "time", "unit", "ignore")
OPS = ("==", "!=", ">", ">=", "<", "<=", "in", "notna")

_SINGLETON_ROLES = ("treatment", "outcome", "time", "unit")


class PrepFilter(BaseModel):
    """One row filter: keep rows where ``col op value`` holds."""

    col: str
    op: Literal["==", "!=", ">", ">=", "<", "<=", "in", "notna"]
    value: object | None = None  # list required for "in"; ignored for "notna"


class Subsample(BaseModel):
    """Row subsample with a plan-carried seed (replayable without a Generator)."""

    n: int  # validated > 0 in a field_validator (NOT gt=0 —
    seed: int = 0  # constraint keys break strict LLM schemas)

    @field_validator("n")
    @classmethod
    def _n_positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"subsample.n must be >= 1, got {v}")
        return v


class PrepPlan(BaseModel):
    """Declarative, validated, deterministic data-prep plan."""

    version: int = 1
    column_roles: dict[str, str] = Field(default_factory=dict)  # col -> ROLES member
    encodings: dict[str, Literal["onehot", "ordinal"]] = Field(default_factory=dict)
    discretize: dict[str, int] = Field(default_factory=dict)  # col -> n_bins (>= 2)
    drop_cols: list[str] = Field(default_factory=list)
    subsample: Subsample | None = None
    filters: list[PrepFilter] = Field(default_factory=list)

    def validate_against(self, df: pd.DataFrame) -> None:
        """Raise ValueError naming every offending entry; no-op if the plan is valid."""
        known = set(df.columns)
        issues: list[str] = []

        def check_col(col: str, where: str) -> bool:
            if col not in known:
                issues.append(f"unknown column '{col}' in {where}")
                return False
            return True

        for col, role in self.column_roles.items():
            check_col(col, "column_roles")
            if role not in ROLES:
                issues.append(f"unknown role '{role}' for column '{col}' (allowed: {ROLES})")
        for role in _SINGLETON_ROLES:
            cols = [c for c, r in self.column_roles.items() if r == role]
            if len(cols) > 1:
                issues.append(f"role '{role}' assigned to multiple columns: {cols}")

        for col in self.encodings:
            check_col(col, "encodings")
            if col in self.discretize:
                issues.append(f"column '{col}' is both encoded and discretized")

        for col, bins in self.discretize.items():
            ok = check_col(col, "discretize")
            if bins < 2:
                issues.append(f"discretize['{col}']={bins}: n_bins must be >= 2")
            if ok and not pd.api.types.is_numeric_dtype(df[col]):
                issues.append(f"discretize column '{col}' is not numeric (dtype={df[col].dtype})")

        for col in self.drop_cols:
            check_col(col, "drop_cols")
            if col in self.column_roles or col in self.encodings or col in self.discretize:
                issues.append(
                    f"column '{col}' is dropped but also role-assigned/encoded/discretized"
                )

        for f in self.filters:
            check_col(f.col, "filters")
            if f.op == "in" and not isinstance(f.value, (list, tuple)):
                issues.append(
                    f"filter {f.col} in: value must be a list/tuple, "
                    f"got {type(f.value).__name__}"
                )
            if f.op != "notna" and f.value is None:
                issues.append(f"filter {f.col} {f.op}: value must not be None")

        if self.subsample is not None and self.subsample.n < 1:
            issues.append(f"subsample.n must be >= 1, got {self.subsample.n}")

        if issues:
            raise ValueError("invalid PrepPlan: " + "; ".join(issues))

    def apply(self, df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
        """Execute the plan: filters -> drop -> encodings -> discretize -> subsample.

        Deterministic (same plan + same df => bitwise-identical output) and
        never mutates the input frame. Returns ``(df2, log)`` with one
        human-readable log line per step.
        """
        self.validate_against(df)
        out = df.copy()
        log: list[str] = []

        for f in self.filters:
            before = len(out)
            s = out[f.col]
            if f.op == "notna":
                mask = s.notna()
                desc = f"filter {f.col} notna"
            elif f.op == "in":
                values = list(f.value)  # type: ignore[arg-type]  # validated list/tuple
                mask = s.isin(values)
                desc = f"filter {f.col} in {values}"
            else:
                cmp = {"==": s.eq, "!=": s.ne, ">": s.gt, ">=": s.ge, "<": s.lt, "<=": s.le}
                mask = cmp[f.op](f.value) & s.notna()  # NaN comparisons are False
                desc = f"filter {f.col} {f.op} {f.value}"
            out = out.loc[mask]
            log.append(f"{desc}: {before} -> {len(out)} rows")
        if self.filters and len(out) == 0:
            log.append("WARNING: 0 rows remain")

        if self.drop_cols:
            out = out.drop(columns=list(self.drop_cols))
            log.append(f"drop columns: {list(self.drop_cols)}")

        for col, enc in self.encodings.items():
            if enc == "ordinal":
                codes, uniques = pd.factorize(out[col], sort=True)
                vals = codes.astype(float)
                vals[codes == -1] = np.nan  # NaN stays NaN, never code 0
                out[col] = vals
                log.append(f"ordinal {col}: {len(uniques)} categories")
            else:  # onehot
                cols_before = set(out.columns)
                out = pd.get_dummies(out, columns=[col], dtype=float)
                new = [c for c in out.columns if c not in cols_before]
                log.append(f"onehot {col} -> {new}")

        for col, bins in self.discretize.items():
            x = out[col].to_numpy(dtype=float, na_value=np.nan)
            coded = np.full(x.shape, np.nan)
            finite = np.isfinite(x)
            effective = 0
            if finite.any():
                codes, edges = quantile_bins(x[finite], bins)
                coded[finite] = codes.astype(float)
                effective = len(edges) - 1
            out[col] = coded
            log.append(f"discretize {col}: {bins} bins requested, {effective} effective")

        if self.subsample is not None:
            before = len(out)
            take = min(self.subsample.n, before)
            if before:
                # Plan-carried seed: default_rng(seed), NOT the pipeline
                # Generator, so serialized plans replay bitwise on their own.
                rng = np.random.default_rng(self.subsample.seed)
                idx = np.sort(rng.choice(before, size=take, replace=False))
                out = out.iloc[idx]
            log.append(
                f"subsample n={self.subsample.n} seed={self.subsample.seed}: "
                f"{before} -> {len(out)} rows"
            )

        return out, log
