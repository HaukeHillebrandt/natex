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
    n_used: int = 0  # members with finite y actually used


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


def _package(tau, se, method, fs_jump, fs_t, n_used):
    z = stats.norm.ppf(0.975)
    return EffectEstimate(
        tau=float(tau),
        se=float(se),
        ci=(float(tau - z * se), float(tau + z * se)),
        method=method,
        first_stage_jump=fs_jump,
        first_stage_t=fs_t,
        weak_instrument=bool(fs_t**2 < 10.0) if np.isfinite(fs_t) else True,
        n_used=n_used,
    )


def _nan_estimate(method, fs_jump, fs_t, n_used):
    """Underdetermined estimate: NaN effect (never 0.0), instrument flagged weak."""
    nan = float("nan")
    return EffectEstimate(
        tau=nan,
        se=nan,
        ci=(nan, nan),
        method=method,
        first_stage_jump=fs_jump,
        first_stage_t=fs_t,
        weak_instrument=True,
        n_used=n_used,
    )


def _wald_first_stage(Tm: np.ndarray, g: np.ndarray):
    n1, n0 = int(g.sum()), int((~g).sum())
    dt = Tm[g].mean() - Tm[~g].mean()
    vt = Tm[g].var(ddof=1) / n1 + Tm[~g].var(ddof=1) / n0
    fs_se = float(np.sqrt(vt))
    fs_t = dt / fs_se if fs_se > 0 else float("nan")
    return float(dt), float(fs_t), float(vt)


def _fallback_first_stage(T_all: np.ndarray, s_all: np.ndarray, group1: np.ndarray, method: str):
    """First-stage diagnostics from the finite-T member rows (else NaN)."""
    tkeep = np.isfinite(T_all)
    Tt, gt1 = T_all[tkeep], group1[tkeep]
    if Tt.size == 0:
        return float("nan"), float("nan")
    g = _oriented_side(Tt, gt1)
    if not g.any() or g.all():
        return float("nan"), float("nan")
    if method == "wald":
        dt, fs_t, _ = _wald_first_stage(Tt, g)
        return dt, fs_t
    st, gf = s_all[tkeep], g.astype(float)
    return _first_stage(Tt, np.c_[np.ones(Tt.size), st, st * gf], gf)


def local_2sls(dataset: Dataset, d: Discovery) -> EffectEstimate:
    if dataset.y is None:
        raise ValueError("dataset has no outcome column")
    m = d.members
    y_all, T_all = dataset.y[m], dataset.T[m]
    s_all = signed_distance(dataset, d)
    g1_all = d.group1.astype(bool)
    keep = np.isfinite(y_all)
    n_used = int(keep.sum())
    ym, Tm, s = y_all[keep], T_all[keep], s_all[keep]
    if n_used < 8:
        return _nan_estimate("2sls", *_fallback_first_stage(T_all, s_all, g1_all, "2sls"), n_used)
    g = _oriented_side(Tm, g1_all[keep])
    if not g.any() or g.all():
        return _nan_estimate("2sls", *_fallback_first_stage(T_all, s_all, g1_all, "2sls"), n_used)
    gf = g.astype(float)
    controls = np.c_[np.ones(n_used), s, s * gf]
    Zmat = np.c_[controls, gf]
    Xmat = np.c_[controls, Tm]
    n, p = Xmat.shape
    ZtX_inv = np.linalg.pinv(Zmat.T @ Xmat)
    beta = ZtX_inv @ (Zmat.T @ ym)
    e = ym - Xmat @ beta
    meat = Zmat.T @ (Zmat * (e**2)[:, None])
    cov = ZtX_inv @ meat @ ZtX_inv.T * (n / max(n - p, 1))
    fs_jump, fs_t = _first_stage(Tm, controls, gf)
    return _package(beta[-1], np.sqrt(max(cov[-1, -1], 0.0)), "2sls", fs_jump, fs_t, n_used)


def wald_estimate(dataset: Dataset, d: Discovery) -> EffectEstimate:
    if dataset.y is None:
        raise ValueError("dataset has no outcome column")
    m = d.members
    y_all, T_all = dataset.y[m], dataset.T[m]
    s_all = signed_distance(dataset, d)
    g1_all = d.group1.astype(bool)
    keep = np.isfinite(y_all)
    n_used = int(keep.sum())
    ym, Tm = y_all[keep], T_all[keep]
    if n_used < 8:
        return _nan_estimate("wald", *_fallback_first_stage(T_all, s_all, g1_all, "wald"), n_used)
    g = _oriented_side(Tm, g1_all[keep])
    if not g.any() or g.all():
        return _nan_estimate("wald", *_fallback_first_stage(T_all, s_all, g1_all, "wald"), n_used)
    dt, fs_t, vt = _wald_first_stage(Tm, g)
    if dt == 0:
        return _nan_estimate("wald", dt, fs_t, n_used)
    n1, n0 = int(g.sum()), int((~g).sum())
    dy = ym[g].mean() - ym[~g].mean()
    tau = dy / dt
    vy = ym[g].var(ddof=1) / n1 + ym[~g].var(ddof=1) / n0
    se = float(np.sqrt(max(vy / dt**2 + (dy**2 / dt**4) * vt, 0.0)))
    return _package(tau, se, "wald", dt, fs_t, n_used)
