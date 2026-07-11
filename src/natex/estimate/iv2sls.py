"""General k-instrument 2SLS with HC1 sandwich errors and Hansen J.

Audit item 4: 2SLS is the ONLY estimator family — the papers' printed
group-instrument form (Eq 5.14, W = T - mu) is inconsistent and is never
implemented. Audit item 10: first-stage relevance is never assumed — the HC1
Wald F of the instrument block (and the instruments' partial R^2 after
controls) is always computed and a ``weak_instrument`` flag raised when
F < 10 (a heuristic convention, not a Stock-Yogo critical value).

Exclusion is untestable. The Hansen J statistic tests only the
OVERIDENTIFYING restrictions given at least one valid instrument; it can
never certify exclusion itself. When the model is just-identified (k == 1)
``j_stat``/``j_p`` are None — never a fabricated value — and ``j_df`` is 0.

NaN policy (spec section 5 item 8): every underdetermined or degenerate path
returns NaN estimates, never 0.0.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import stats

_WEAK_F_THRESHOLD = 10.0


@dataclass
class IVEstimate:
    tau: float
    se: float
    ci: tuple[float, float]
    method: str  # "2sls"
    first_stage_F: float  # HC1 Wald F of the instrument block in the first stage
    partial_r2: float  # first-stage partial R^2 of instruments after controls
    weak_instrument: bool  # first_stage_F < 10.0 (NaN F -> True)
    j_stat: float | None  # Hansen J; None when just-identified (k == 1)
    j_p: float | None
    j_df: int  # k - 1 (0 when k == 1)
    n_used: int  # rows with finite (y, T, instruments, controls)
    ar_ci: tuple[float, float] | None = None  # filled by task 2
    ar_kind: str | None = None  # filled by task 2
    extras: dict = field(default_factory=dict)


def _as_2d(a: np.ndarray) -> np.ndarray:
    arr = np.asarray(a, dtype=float)
    return arr[:, None] if arr.ndim == 1 else arr


def _nan_estimate(
    k: int,
    n_used: int,
    extras: dict,
    first_stage_f: float = float("nan"),
    partial_r2: float = float("nan"),
) -> IVEstimate:
    """Underdetermined/degenerate estimate: NaN effect (never 0.0), flagged weak."""
    nan = float("nan")
    return IVEstimate(
        tau=nan,
        se=nan,
        ci=(nan, nan),
        method="2sls",
        first_stage_F=first_stage_f,
        partial_r2=partial_r2,
        weak_instrument=True,
        j_stat=None,
        j_p=None,
        j_df=max(k - 1, 0),
        n_used=n_used,
        extras=extras,
    )


def _residualize(a: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """Residual of each column of ``a`` after least-squares projection on ``basis``."""
    coef, *_ = np.linalg.lstsq(basis, a, rcond=None)
    return a - basis @ coef


def _first_stage_diagnostics(
    t_vec: np.ndarray, cfull: np.ndarray, z_mat: np.ndarray
) -> tuple[float, float]:
    """HC1 Wald F that the k instrument coefficients are jointly zero in
    T ~ [1, controls, instruments] (chi^2/k form), plus the instruments'
    partial R^2 after the controls."""
    n = t_vec.size
    k = z_mat.shape[1]
    w_mat = np.c_[cfull, z_mat]
    p = w_mat.shape[1]
    wtw_inv = np.linalg.pinv(w_mat.T @ w_mat)
    gamma = wtw_inv @ (w_mat.T @ t_vec)
    u = t_vec - w_mat @ gamma
    meat = w_mat.T @ (w_mat * (u**2)[:, None])
    cov = wtw_inv @ meat @ wtw_inv * (n / max(n - p, 1))
    g = gamma[-k:]
    wald = float(g @ np.linalg.pinv(cov[-k:, -k:]) @ g)
    f_stat = wald / k
    t_res = _residualize(t_vec, cfull)
    z_res = _residualize(z_mat, cfull)
    sst = float(t_res @ t_res)
    if sst <= 0:
        return f_stat, float("nan")
    resid = t_res - z_res @ np.linalg.lstsq(z_res, t_res, rcond=None)[0]
    return f_stat, float(1.0 - (resid @ resid) / sst)


def _hansen_j(e: np.ndarray, z_res: np.ndarray, k: int) -> tuple[float | None, float | None, int]:
    """Hansen J with df = k - 1 (one endogenous regressor); None when k == 1."""
    if k < 2:
        return None, None, 0
    m = z_res.T @ e
    s_mat = z_res.T @ (z_res * (e**2)[:, None])
    j = float(m @ np.linalg.pinv(s_mat) @ m)
    return j, float(stats.chi2.sf(j, k - 1)), k - 1


def iv_2sls(
    y: np.ndarray,
    T: np.ndarray,
    instruments: np.ndarray,
    controls: np.ndarray | None = None,
    alpha: float = 0.05,
) -> IVEstimate:
    """HC1-sandwich 2SLS of ``y`` on ``T`` with ``instruments`` (n, k), k >= 1.

    An intercept is added internally; ``controls`` (n, q) enter both the
    structural equation and the instrument set. Rows with any non-finite
    value in (y, T, instruments, controls) are dropped and counted in
    ``extras["n_dropped"]``.
    """
    y = np.asarray(y, dtype=float).ravel()
    t_all = np.asarray(T, dtype=float).ravel()
    z_all = _as_2d(instruments)
    c_all = _as_2d(controls) if controls is not None else np.empty((y.size, 0))
    k = z_all.shape[1]
    extras: dict = {}
    if k == 0:
        extras["n_dropped"] = 0
        extras["reason"] = "empty instrument list"
        return _nan_estimate(k, 0, extras)
    finite = (
        np.isfinite(y)
        & np.isfinite(t_all)
        & np.isfinite(z_all).all(axis=1)
        & np.isfinite(c_all).all(axis=1)
    )
    n_used = int(finite.sum())
    extras["n_dropped"] = int(y.size - n_used)
    p = c_all.shape[1] + 2  # intercept + controls + T
    if n_used < p + 3:
        extras["reason"] = "underdetermined: fewer than p + 3 finite rows"
        return _nan_estimate(k, n_used, extras)
    ym, tm = y[finite], t_all[finite]
    cfull = np.c_[np.ones(n_used), c_all[finite]]
    z_mat = z_all[finite]
    zfull = np.c_[cfull, z_mat]
    x_mat = np.c_[cfull, tm]
    if np.linalg.matrix_rank(zfull) < zfull.shape[1] or np.linalg.matrix_rank(x_mat) < p:
        extras["rank_deficient"] = True
    x_hat = zfull @ np.linalg.lstsq(zfull, x_mat, rcond=None)[0]
    if np.linalg.matrix_rank(x_hat) < p:
        # tau's column is in the null space of the projected design: unidentified.
        extras["rank_deficient"] = True
        return _nan_estimate(k, n_used, extras)
    a_inv = np.linalg.pinv(x_hat.T @ x_mat)
    beta = a_inv @ (x_hat.T @ ym)
    e = ym - x_mat @ beta
    meat = x_hat.T @ (x_hat * (e**2)[:, None])
    cov = a_inv @ meat @ a_inv.T * (n_used / max(n_used - p, 1))
    tau = float(beta[-1])
    se = float(np.sqrt(max(cov[-1, -1], 0.0)))
    zcrit = float(stats.norm.ppf(1.0 - alpha / 2.0))
    f_stat, partial_r2 = _first_stage_diagnostics(tm, cfull, z_mat)
    j_stat, j_p, j_df = _hansen_j(e, _residualize(z_mat, cfull), k)
    return IVEstimate(
        tau=tau,
        se=se,
        ci=(tau - zcrit * se, tau + zcrit * se),
        method="2sls",
        first_stage_F=f_stat,
        partial_r2=partial_r2,
        weak_instrument=bool(f_stat < _WEAK_F_THRESHOLD) if np.isfinite(f_stat) else True,
        j_stat=j_stat,
        j_p=j_p,
        j_df=j_df,
        n_used=n_used,
        extras=extras,
    )
