"""Intercept-continuity placebo tests (audit item 3).

The printed side-mean placebo mechanically rejects valid designs (the running
variable itself has different side means by construction). We instead test the
INTERCEPT jump at the boundary after allowing side-specific linear trends in
the signed distance, with HC1 errors and Holm correction.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import stats

from natex.data.spec import Dataset
from natex.rdd.lord3 import Discovery


def hc1_ols(Xmat: np.ndarray, yvec: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n, p = Xmat.shape
    XtX_inv = np.linalg.pinv(Xmat.T @ Xmat)
    beta = XtX_inv @ (Xmat.T @ yvec)
    e = yvec - Xmat @ beta
    meat = Xmat.T @ (Xmat * (e**2)[:, None])
    cov = XtX_inv @ meat @ XtX_inv * (n / max(n - p, 1))
    return beta, np.sqrt(np.maximum(np.diag(cov), 0.0))


def signed_distance(dataset: Dataset, d: Discovery) -> np.ndarray:
    Z = dataset.Z_std
    return (Z[d.members] - Z[d.center_index]) @ d.normal


@dataclass
class PlaceboReport:
    p_values: dict
    p_holm: dict
    passed: bool
    m: int = 0  # tests that entered the Holm family (finite p-values)
    skipped: dict = field(default_factory=dict)  # column/level -> reason (issue #3)


def _holm(p_values: dict[str, float]) -> dict[str, float]:
    """Holm step-down adjusted p-values; NaN entries excluded and preserved.

    A NaN p (a covariate whose placebo regression fits perfectly, se = 0) must
    never fabricate a finite adjusted p, contaminate the running max, or
    inflate the multiplier m for the finite p-values (finding F-A1, audit
    item 3). Same convention as ``natex.did.effects._holm`` and
    ``natex.validate.panel._holm``.
    """
    out: dict[str, float] = {name: float("nan") for name in p_values}
    usable = [(name, p) for name, p in p_values.items() if not np.isnan(p)]
    m = len(usable)
    running = 0.0
    for rank, (name, p) in enumerate(sorted(usable, key=lambda item: item[1])):
        running = max(running, (m - rank) * p)
        out[name] = min(running, 1.0)
    return out


def placebo_tests(dataset: Dataset, d: Discovery, alpha: float = 0.05) -> PlaceboReport:
    import pandas as pd

    skipped: dict[str, str] = {}
    usable_cols: list[str] = []
    for c in dataset.spec.covariates:
        col = dataset.df[c]
        if not pd.api.types.is_numeric_dtype(col) and col.nunique() == len(col):
            # covariates="auto" sweeps in string date/ID columns; a row-unique
            # column can never be a placebo, and its one-hot levels would each
            # enter the Holm family as a powerless support-<=1 test, silently
            # diluting genuine placebo failures (issue #3).
            skipped[c] = "row-unique non-numeric column (ID/date); excluded before one-hot"
            continue
        usable_cols.append(c)
    Xdf = pd.get_dummies(dataset.df[usable_cols], dtype=float)
    non_forcing = [c for c in Xdf.columns if c not in dataset.spec.forcing]
    s = signed_distance(dataset, d)
    side = d.group1.astype(float)
    design = np.c_[np.ones(s.size), side, s, s * side]
    p_values: dict[str, float] = {}
    for c in non_forcing:
        yv = Xdf[c].to_numpy(dtype=float)[d.members]
        if np.var(yv) == 0:
            skipped[c] = "constant within the neighborhood"
            continue
        vals = np.unique(yv)
        if vals.size == 2 and set(vals.tolist()) <= {0.0, 1.0}:
            s1 = float(yv.sum())
            if min(s1, yv.size - s1) < 2:
                # A minority-count-<2 indicator's intercept jump is pure noise;
                # testing it only dilutes the Holm family (generalizes the
                # var == 0 guard, which is the minority-count-0 case).
                skipped[c] = "degenerate 0/1 level (minority count < 2 in the neighborhood)"
                continue
        beta, se = hc1_ols(design, yv)
        t = beta[1] / se[1] if se[1] > 0 else np.nan
        dof = max(s.size - 4, 1)
        p_values[c] = float(2 * stats.t.sf(abs(t), dof)) if np.isfinite(t) else float("nan")
    p_holm = _holm(p_values)
    m = sum(1 for p in p_values.values() if not np.isnan(p))
    if p_values:
        usable = [v for v in p_holm.values() if not np.isnan(v)]
        # NaN entries stay visible in the report but never decide the battery;
        # if EVERY p is NaN there is nothing usable -> fail loudly (F-A1).
        passed = bool(usable) and all(v > alpha for v in usable)
    else:
        passed = True  # no testable covariate: vacuously passed (documented)
    return PlaceboReport(p_values=p_values, p_holm=p_holm, passed=passed, m=m, skipped=skipped)
