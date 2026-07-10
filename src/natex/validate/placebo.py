"""Intercept-continuity placebo tests (audit item 3).

The printed side-mean placebo mechanically rejects valid designs (the running
variable itself has different side means by construction). We instead test the
INTERCEPT jump at the boundary after allowing side-specific linear trends in
the signed distance, with HC1 errors and Holm correction.
"""

from __future__ import annotations

from dataclasses import dataclass

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


def placebo_tests(dataset: Dataset, d: Discovery, alpha: float = 0.05) -> PlaceboReport:
    import pandas as pd

    Xdf = pd.get_dummies(dataset.df[dataset.spec.covariates], dtype=float)
    non_forcing = [c for c in Xdf.columns if c not in dataset.spec.forcing]
    s = signed_distance(dataset, d)
    side = d.group1.astype(float)
    design = np.c_[np.ones(s.size), side, s, s * side]
    p_values: dict[str, float] = {}
    for c in non_forcing:
        yv = Xdf[c].to_numpy(dtype=float)[d.members]
        if np.var(yv) == 0:
            continue
        beta, se = hc1_ols(design, yv)
        t = beta[1] / se[1] if se[1] > 0 else np.nan
        dof = max(s.size - 4, 1)
        p_values[c] = float(2 * stats.t.sf(abs(t), dof)) if np.isfinite(t) else float("nan")
    names = sorted(p_values, key=p_values.get)
    m = len(names)
    p_holm: dict[str, float] = {}
    running = 0.0
    for rank, name in enumerate(names):
        adj = min((m - rank) * p_values[name], 1.0)
        running = max(running, adj)
        p_holm[name] = running
    passed = all(v > alpha for v in p_holm.values()) if p_holm else True
    return PlaceboReport(p_values=p_values, p_holm=p_holm, passed=passed)
