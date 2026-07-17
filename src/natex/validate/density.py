"""Signed-distance density falsification test (McCrary-style, binned Poisson).

:func:`density_test` is valid only for the FROZEN discovered geometry; it does
not account for the search having selected normal and cutoff (audit item 6).
Use with honest splitting. :func:`binned_poisson_jump` exposes the identical
statistic for DECLARED thresholds (bunching), where no search took place.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

from natex.data.spec import Dataset
from natex.rdd.lord3 import Discovery
from natex.validate.placebo import signed_distance


@dataclass
class DensityReport:
    p_value: float
    theta: float


def binned_poisson_jump(s: np.ndarray, n_bins: int = 20) -> DensityReport:
    """Binned-Poisson intercept-jump test on signed distances ``s`` (cutoff at 0).

    Extracted from ``density_test`` (phase survey, task 4 — pure refactor, NO
    new math) so declared-threshold bunching reuses the identical statistic;
    ``density_test`` now delegates: ``density_test(ds, d, n_bins) ==
    binned_poisson_jump(signed_distance(ds, d), n_bins)``. Non-finite ``s``
    are dropped; ``s`` with fewer than 2 distinct finite values yields
    ``DensityReport(nan, nan)`` — NaN, never 0.
    """
    s = np.asarray(s, dtype=float).ravel()
    s = s[np.isfinite(s)]
    if np.unique(s).size < 2:
        return DensityReport(p_value=float("nan"), theta=float("nan"))
    edges = np.linspace(s.min(), s.max() + 1e-12, n_bins + 1)
    counts, _ = np.histogram(s, bins=edges)
    mids = 0.5 * (edges[:-1] + edges[1:])
    side = (mids >= 0).astype(float)
    X = np.c_[np.ones(n_bins), mids, side, mids * side]
    # Poisson GLM via IRLS
    beta = np.zeros(4)
    for _ in range(100):
        mu = np.exp(np.clip(X @ beta, -30, 30))
        W = mu
        z = X @ beta + (counts - mu) / np.maximum(mu, 1e-12)
        WX = X * W[:, None]
        beta_new = np.linalg.pinv(X.T @ WX) @ (WX.T @ z)
        if np.max(np.abs(beta_new - beta)) < 1e-10:
            beta = beta_new
            break
        beta = beta_new
    mu = np.exp(np.clip(X @ beta, -30, 30))
    cov = np.linalg.pinv(X.T @ (X * mu[:, None]))
    se = float(np.sqrt(max(cov[2, 2], 0.0)))
    theta = float(beta[2])
    p = float(2 * stats.norm.sf(abs(theta / se))) if se > 0 else float("nan")
    return DensityReport(p_value=p, theta=theta)


def density_test(dataset: Dataset, d: Discovery, n_bins: int = 20) -> DensityReport:
    return binned_poisson_jump(signed_distance(dataset, d), n_bins)
