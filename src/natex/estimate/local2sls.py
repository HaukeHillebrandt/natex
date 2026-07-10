"""Local effect estimation at a discovered discontinuity.

Primary estimator (audit item 4): ordinary 2SLS with the FROZEN discovered side
indicator as the excluded instrument, side-specific linear controls in the
signed distance, HC1 sandwich errors, and always-on first-stage diagnostics
(audit item 10). The papers' 'group instrument' (Eq 5.14) is deliberately not
implemented: it is inconsistent as printed.

Orientation convention: the hyperplane normal's sign (and hence which side is
``group1``) is arbitrary upstream, so the instrument is canonically oriented
toward the higher-treatment side before estimation. This leaves tau, its SE,
and the CI exactly invariant (a reparametrization of the non-treatment
columns) and makes the reported first-stage jump nonnegative; relevance
(audit item 10) is magnitude-based (F = t^2) either way.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

from natex.data.spec import Dataset
from natex.rdd.lord3 import Discovery
from natex.validate.placebo import hc1_ols, signed_distance


@dataclass
class EffectEstimate:
    tau: float
    se: float
    ci: tuple[float, float]
    method: str
    first_stage_jump: float
    first_stage_t: float
    weak_instrument: bool


def _oriented_side(Tm: np.ndarray, group1: np.ndarray) -> np.ndarray:
    """Return the side indicator oriented toward the higher-treatment side."""
    g = group1.astype(bool)
    if not g.any() or g.all():
        return g
    return ~g if Tm[g].mean() < Tm[~g].mean() else g


def _first_stage(Tm: np.ndarray, controls: np.ndarray, g: np.ndarray):
    Xfs = np.c_[controls, g]
    beta, se = hc1_ols(Xfs, Tm)
    jump, jump_se = beta[-1], se[-1]
    t = jump / jump_se if jump_se > 0 else float("nan")
    return float(jump), float(t)


def _package(tau, se, method, fs_jump, fs_t):
    z = stats.norm.ppf(0.975)
    return EffectEstimate(
        tau=float(tau),
        se=float(se),
        ci=(float(tau - z * se), float(tau + z * se)),
        method=method,
        first_stage_jump=fs_jump,
        first_stage_t=fs_t,
        weak_instrument=bool(fs_t**2 < 10.0) if np.isfinite(fs_t) else True,
    )


def local_2sls(dataset: Dataset, d: Discovery) -> EffectEstimate:
    if dataset.y is None:
        raise ValueError("dataset has no outcome column")
    m = d.members
    ym, Tm = dataset.y[m], dataset.T[m]
    s = signed_distance(dataset, d)
    g = _oriented_side(Tm, d.group1).astype(float)
    controls = np.c_[np.ones(m.size), s, s * g]
    Zmat = np.c_[controls, g]
    Xmat = np.c_[controls, Tm]
    n, p = Xmat.shape
    ZtX_inv = np.linalg.pinv(Zmat.T @ Xmat)
    beta = ZtX_inv @ (Zmat.T @ ym)
    e = ym - Xmat @ beta
    meat = Zmat.T @ (Zmat * (e**2)[:, None])
    cov = ZtX_inv @ meat @ ZtX_inv.T * (n / max(n - p, 1))
    fs_jump, fs_t = _first_stage(Tm, controls, g)
    return _package(beta[-1], np.sqrt(max(cov[-1, -1], 0.0)), "2sls", fs_jump, fs_t)


def wald_estimate(dataset: Dataset, d: Discovery) -> EffectEstimate:
    if dataset.y is None:
        raise ValueError("dataset has no outcome column")
    m = d.members
    ym, Tm = dataset.y[m], dataset.T[m]
    g = _oriented_side(Tm, d.group1)
    dy = ym[g].mean() - ym[~g].mean()
    dt = Tm[g].mean() - Tm[~g].mean()
    tau = dy / dt if dt != 0 else float("nan")
    n1, n0 = int(g.sum()), int((~g).sum())
    vy = ym[g].var(ddof=1) / n1 + ym[~g].var(ddof=1) / n0
    vt = Tm[g].var(ddof=1) / n1 + Tm[~g].var(ddof=1) / n0
    se = float("nan") if dt == 0 else float(np.sqrt(max(vy / dt**2 + (dy**2 / dt**4) * vt, 0.0)))
    fs_se = float(np.sqrt(vt))
    fs_t = dt / fs_se if fs_se > 0 else float("nan")
    return _package(tau, se, "wald", float(dt), float(fs_t))
