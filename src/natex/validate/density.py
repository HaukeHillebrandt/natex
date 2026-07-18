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
    se: float  # Wald SE of theta (issue #41); NaN whenever p_value is NaN


def binned_poisson_jump(
    s: np.ndarray,
    n_bins: int = 20,
    *,
    window: float | None = None,
    max_iter: int = 100,
) -> DensityReport:
    """Binned-Poisson intercept-jump test on signed distances ``s`` (cutoff at 0).

    Extracted from ``density_test`` (phase survey, task 4) so declared-
    threshold bunching reuses the identical statistic; ``density_test``
    delegates: ``density_test(ds, d, n_bins) ==
    binned_poisson_jump(signed_distance(ds, d), n_bins)``. Non-finite ``s``
    are dropped; ``window`` (issue #42) then restricts the fit to
    ``|s| <= window`` so the GLM tests the LOCAL density jump instead of
    binning the full data range. ``s`` with fewer than 2 distinct finite
    values AFTER those subsets yields ``DensityReport(nan, nan, nan)`` —
    NaN, never 0. The Wald SE of theta is reported alongside p and theta
    (issue #41): it stays finite when p underflows to 0.0 and is NaN
    whenever p is NaN.

    The IRLS starts at ``beta = [log(mean(counts)), 0, 0, 0]`` (issue #42:
    the zero start needed ~mean-count iterations and silently exited the cap
    on ~100-count bins, reporting a seed-noisy fabricated p; the mean start
    converges in ~4). A fit that still exhausts ``max_iter`` without
    converging reports NaN — never the statistic of an unconverged beta.
    """
    if window is not None and not window > 0:
        raise ValueError(f"window must be > 0, got {window!r}")
    s = np.asarray(s, dtype=float).ravel()
    s = s[np.isfinite(s)]
    if window is not None:
        s = s[np.abs(s) <= window]
    if np.unique(s).size < 2:
        return DensityReport(p_value=float("nan"), theta=float("nan"), se=float("nan"))
    edges = np.linspace(s.min(), s.max() + 1e-12, n_bins + 1)
    counts, _ = np.histogram(s, bins=edges)
    mids = 0.5 * (edges[:-1] + edges[1:])
    side = (mids >= 0).astype(float)
    X = np.c_[np.ones(n_bins), mids, side, mids * side]
    # Poisson GLM via IRLS
    beta = np.array([np.log(max(counts.mean(), 1e-12)), 0.0, 0.0, 0.0])
    converged = False
    for _ in range(max_iter):
        mu = np.exp(np.clip(X @ beta, -30, 30))
        W = mu
        z = X @ beta + (counts - mu) / np.maximum(mu, 1e-12)
        WX = X * W[:, None]
        beta_new = np.linalg.pinv(X.T @ WX) @ (WX.T @ z)
        if np.max(np.abs(beta_new - beta)) < 1e-10:
            beta = beta_new
            converged = True
            break
        beta = beta_new
    if not converged:
        return DensityReport(p_value=float("nan"), theta=float("nan"), se=float("nan"))
    mu = np.exp(np.clip(X @ beta, -30, 30))
    cov = np.linalg.pinv(X.T @ (X * mu[:, None]))
    se = float(np.sqrt(max(cov[2, 2], 0.0)))
    theta = float(beta[2])
    p = float(2 * stats.norm.sf(abs(theta / se))) if se > 0 else float("nan")
    return DensityReport(p_value=p, theta=theta, se=se if se > 0 else float("nan"))


def density_test(dataset: Dataset, d: Discovery, n_bins: int = 20) -> DensityReport:
    return binned_poisson_jump(signed_distance(dataset, d), n_bins)
