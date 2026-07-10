"""Signed-distance density falsification test (McCrary-style, binned Poisson).

Valid only for the FROZEN discovered geometry; does not account for the search
having selected normal and cutoff (audit item 6). Use with honest splitting.
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


def density_test(dataset: Dataset, d: Discovery, n_bins: int = 20) -> DensityReport:
    s = signed_distance(dataset, d)
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
