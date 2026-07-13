"""Categorical panel data layer for the DiD (SuDDDS) scan.

Records are coded onto m categorical dimensions (value codes 0..k_j-1 per
dimension j). Subsets are conjunctions over dimensions of unions over values:
``s = {i : code[i, j] in V_j for every dim j}``, represented as per-dimension
boolean masks over value codes; the all-True state is s = D.

The panel carries the outcome ``y`` only as luggage for the estimation stage —
THE SCAN NEVER READS IT.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from natex.data.spec import Dataset


def quantile_bins(x: np.ndarray, bins: int) -> tuple[np.ndarray, np.ndarray]:
    """Code a numeric column into ``bins`` quantile bins.

    Returns ``(codes, edges)`` where ``codes`` are int64 in ``0..b-1`` and
    ``edges`` are the (strictly increasing) bin edges after collapsing
    duplicate quantiles — ties can yield fewer effective bins, which is fine.
    Bins are left-closed: code j means ``edges[j] <= x < edges[j+1]`` (the top
    bin is closed on both sides). ``x`` must be finite (Dataset already dropped
    NaN scan rows).
    """
    x = np.asarray(x, dtype=float)
    if not np.all(np.isfinite(x)):
        raise ValueError("quantile_bins requires finite x (drop NaN/inf rows first)")
    if bins < 1:
        raise ValueError(f"bins must be >= 1, got {bins}")
    edges = np.unique(np.quantile(x, np.linspace(0.0, 1.0, bins + 1)))
    # Interior edges only: min/max never split; empty interior -> single bin 0.
    codes = np.searchsorted(edges[1:-1], x, side="right").astype(np.int64)
    return codes, edges


def _lex_profile_id(codes: np.ndarray, dim_sizes: tuple[int, ...]) -> np.ndarray:
    """Lexicographic profile id over dimension codes (last dimension fastest)."""
    n = codes.shape[0]
    if len(dim_sizes) == 0:
        return np.zeros(n, dtype=np.int64)
    return np.ravel_multi_index(tuple(codes[:, j] for j in range(codes.shape[1])), dim_sizes).astype(
        np.int64
    )


@dataclass
class CategoricalPanel:
    """Coded panel: n records over m categorical dimensions plus (t, theta, y, unit)."""

    codes: np.ndarray  # (n, m) int64 value codes, 0..k_j-1 per dimension
    dim_names: list[str]  # length m
    dim_values: list[np.ndarray]  # per-dim decoded original values (index = code)
    t: np.ndarray  # (n,) float time
    theta: np.ndarray  # (n,) float treatment
    y: np.ndarray | None  # (n,) outcome or None; the SCAN NEVER READS IT
    unit: np.ndarray  # (n,) int64 unit codes, dense 0..n_units-1
    unit_values: np.ndarray  # decoded unit labels (index = code)
    # raw time = t + t_origin (issue #27): nonzero only when an integer time
    # column needed origin normalization to survive float64 conversion; every
    # reported t0/window then lives in the shifted coordinates.
    t_origin: float = 0.0
    _profile_id: np.ndarray | None = field(default=None, init=False, repr=False)

    @property
    def n(self) -> int:
        return self.codes.shape[0]

    @property
    def m(self) -> int:
        return self.codes.shape[1]

    @property
    def dim_sizes(self) -> tuple[int, ...]:
        return tuple(len(v) for v in self.dim_values)

    @property
    def profile_id(self) -> np.ndarray:
        """(n,) int64 lexicographic code over all dimensions (cached)."""
        if self._profile_id is None:
            self._profile_id = _lex_profile_id(self.codes, self.dim_sizes)
        return self._profile_id

    def subset_mask(self, included: list[np.ndarray]) -> np.ndarray:
        """(n,) bool from per-dim value masks (conjunction of unions).

        ``included[j]`` is a boolean mask of length k_j over dimension j's value
        codes; a record is in the subset iff every dimension's code is included.
        """
        if len(included) != self.m:
            raise ValueError(f"expected {self.m} per-dimension masks, got {len(included)}")
        mask = np.ones(self.n, dtype=bool)
        for j, inc in enumerate(included):
            inc = np.asarray(inc, dtype=bool)
            if inc.shape != (self.dim_sizes[j],):
                raise ValueError(
                    f"mask for dim {j} ({self.dim_names[j]}) has shape {inc.shape}, "
                    f"expected ({self.dim_sizes[j]},)"
                )
            mask &= inc[self.codes[:, j]]
        return mask


def build_panel(
    dataset: Dataset, dims: list[str] | None = None, bins: int = 4
) -> CategoricalPanel:
    """Code a Dataset (with ``spec.time`` set) into a CategoricalPanel.

    ``dims`` defaults to all covariates minus the time and unit columns.
    Non-numeric dims and numeric dims with <= ``bins`` distinct values are coded
    by their sorted unique values (so a binary 0/1 column keeps its two values);
    other numeric dims get ``quantile_bins(bins)`` with bin-midpoint decoded
    values. ``unit`` defaults to ``spec.unit``; when absent it falls back to the
    profile id (dependence-preserving nulls then treat each profile as one unit).
    """
    spec = dataset.spec
    if spec.time is None:
        raise ValueError("build_panel requires dataset.spec.time to be set")
    if dims is None:
        excluded = {spec.time, spec.unit}
        dims = [c for c in spec.covariates if c not in excluded]
    missing = [c for c in dims if c not in dataset.df.columns]
    if missing:
        raise ValueError(f"panel dims not in dataframe: {missing}")

    codes_cols: list[np.ndarray] = []
    dim_values: list[np.ndarray] = []
    for name in dims:
        col = dataset.df[name]
        if pd.api.types.is_numeric_dtype(col) and col.nunique() > bins:
            x = col.to_numpy(dtype=float)
            cj, edges = quantile_bins(x, bins)
            values = edges.copy() if edges.size < 2 else (edges[:-1] + edges[1:]) / 2.0
        else:
            cj, uniques = pd.factorize(col, sort=True)
            cj = cj.astype(np.int64)
            values = np.asarray(uniques)
        codes_cols.append(cj)
        dim_values.append(values)
    n = dataset.n
    codes = (
        np.column_stack(codes_cols).astype(np.int64) if codes_cols else np.zeros((n, 0), np.int64)
    )

    if spec.unit is not None:
        uc, uvals = pd.factorize(dataset.df[spec.unit], sort=True)
        unit = uc.astype(np.int64)
        unit_values = np.asarray(uvals)
    else:
        # No unit column: treat each covariate profile as one unit (documented
        # assumption for the dependence-preserving panel nulls).
        pid = _lex_profile_id(codes, tuple(len(v) for v in dim_values))
        unit_values, unit = np.unique(pid, return_inverse=True)
        unit = unit.astype(np.int64)

    # Issue #27: float64 ulp exceeds the time gaps at large integer
    # magnitudes (e.g. ns-since-epoch: ulp at 1e18 is 128), silently merging
    # distinct time points. Integer columns are shifted by their minimum in
    # exact native arithmetic BEFORE conversion (offset kept in t_origin);
    # anything still collapsing fails loudly, never silently.
    t_col = dataset.df[spec.time]
    n_raw_times = int(t_col.nunique())
    t = t_col.to_numpy(dtype=float)
    t_origin = 0.0
    if np.unique(t).size < n_raw_times and pd.api.types.is_integer_dtype(t_col):
        t_raw = t_col.to_numpy()
        origin = t_raw.min()
        t = (t_raw - origin).astype(float)
        t_origin = float(origin)
    if np.unique(t).size < n_raw_times:
        raise ValueError(
            f"time column {spec.time!r} loses distinct values under float64 "
            f"conversion ({n_raw_times} raw vs {np.unique(t).size} converted); "
            "rescale to coarser units (e.g. ns -> s or days)"
        )

    return CategoricalPanel(
        codes=codes,
        dim_names=list(dims),
        dim_values=dim_values,
        t=t,
        theta=dataset.T,
        y=dataset.y,
        unit=unit,
        unit_values=unit_values,
        t_origin=t_origin,
    )
