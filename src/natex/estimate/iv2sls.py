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
Identification is counted by the EFFECTIVE rank of the instruments after
partialling on [1, controls] (issue #11): duplicated or control-collinear
instruments are collapsed to an orthonormal basis (tau unchanged) with
``extras["rank_deficient"]``/``extras["k_effective"]`` set, so a rank-1
instrument block can never buy a structurally guaranteed J pass, an
F deflated by k/k_eff, or a misparametrized AR set.

Weak-IV-robust inference (audit section 3 adopted): ``ar_confidence_set``
inverts the Anderson–Rubin test in closed form. With controls-plus-intercept
partialled out of (y, T, Z) and P the projection onto the partialled
instruments, the level-alpha set is {tau : r'Ar <= 0} for r = y~ - tau T~,
A = P - c_k (I - P), c_k = k F_crit(k, n-q-k; 1-alpha)/(n-q-k) — a quadratic
in tau whose four honest outcomes ("interval", "empty", "disjoint",
"unbounded") are reported as found, never coerced to a finite interval.
k = 1 is exactly Fieller; the set is bounded iff the homoskedastic
first-stage F exceeds F_crit.

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
    j_stat: float | None  # Hansen J; None when just-identified (effective k == 1)
    j_p: float | None
    j_df: int  # k_eff - 1 (0 when just-identified; k_eff < k flagged in extras)
    n_used: int  # rows with finite (y, T, instruments, controls)
    ar_ci: tuple[float, float] | None = None  # AR/Fieller interval when kind == "interval"
    ar_kind: str | None = None  # "interval" | "empty" | "disjoint" | "unbounded"
    extras: dict = field(default_factory=dict)


@dataclass
class ARSet:
    kind: str  # "interval" | "empty" | "disjoint" | "unbounded"
    interval: tuple[float, float] | None  # for "interval"; None otherwise
    rays: tuple[tuple[float, float], tuple[float, float]] | None  # "disjoint": (-inf,r1],[r2,inf)
    ar_at_2sls: float  # AR statistic evaluated at the 2SLS point estimate
    f_crit: float


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


def _effective_instruments(z_mat: np.ndarray, z_res: np.ndarray) -> tuple[int, np.ndarray]:
    """Effective rank of the control-residualized instruments plus an
    orthonormal basis of their column space (issue #11).

    Duplicated instruments — or instruments collinear with [1, controls] —
    collapse the moment space: keeping the nominal k would deflate the
    first-stage Wald F by k/k_eff, fabricate a structurally guaranteed
    Hansen J pass (df >= 1 on a just-identified model), and misparametrize
    the AR set's f_crit and dof. Rank is measured AFTER partialling so
    control-collinear instruments also collapse, with the matrix_rank-style
    tolerance anchored to the PRE-partialling spectral norm of ``z_mat``
    (residualization is a contraction, so a residual that is pure float
    noise relative to the original columns counts as rank 0). The basis
    spans exactly col(z_res), leaving the 2SLS projection — and hence tau —
    unchanged.
    """
    if z_res.shape[1] == 0:
        return 0, z_res
    u, s, _ = np.linalg.svd(z_res, full_matrices=False)
    scale = max(float(np.linalg.norm(z_mat, ord=2)), float(s.max(initial=0.0)))
    tol = scale * max(z_res.shape) * np.finfo(float).eps
    r = int(np.count_nonzero(s > tol))
    return r, u[:, :r]


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


def _ar_set_partialled(
    y_t: np.ndarray, t_t: np.ndarray, z_t: np.ndarray, dof: int, alpha: float
) -> ARSet:
    """Closed-form AR set from already-partialled (y~, T~, Z~) with dof = n - q - k.

    g(tau) = r'Ar, A = P - c_k(I-P), r = y~ - tau T~; using P idempotent this
    is the quadratic a tau^2 + b tau + c with a = T~'A T~, b = -2 y~'A T~,
    c = y~'A y~, each moment expanded as (1+c_k) m'Pm - c_k m'm.
    """
    k = z_t.shape[1]
    f_crit = float(stats.f.ppf(1.0 - alpha, k, dof))
    c_k = k * f_crit / dof
    py = z_t @ np.linalg.lstsq(z_t, y_t, rcond=None)[0]
    pt = z_t @ np.linalg.lstsq(z_t, t_t, rcond=None)[0]
    ypy, tpt, ypt = float(y_t @ py), float(t_t @ pt), float(y_t @ pt)
    yy, tt, yt = float(y_t @ y_t), float(t_t @ t_t), float(y_t @ t_t)
    a = (1.0 + c_k) * tpt - c_k * tt
    b = -2.0 * ((1.0 + c_k) * ypt - c_k * yt)
    c = (1.0 + c_k) * ypy - c_k * yy

    # AR statistic at the (partialled) 2SLS point estimate.
    ar_at = float("nan")
    if tpt > 0:
        tau_hat = ypt / tpt
        r = y_t - tau_hat * t_t
        r_p_r = float(r @ (py - tau_hat * pt))
        denom = (float(r @ r) - r_p_r) / dof
        if denom > 0:
            ar_at = (r_p_r / k) / denom
        elif r_p_r > 0:
            ar_at = float("inf")

    scale = abs((1.0 + c_k) * tpt) + abs(c_k * tt)
    if abs(a) <= 1e-12 * scale:  # degenerate quadratic: linear fallback
        b_scale = abs((1.0 + c_k) * ypt) + abs(c_k * yt)
        if abs(b) <= 2e-12 * b_scale:  # constant: whole line or nothing
            kind = "unbounded" if c <= 0 else "empty"
        else:
            kind = "unbounded"  # a half-line {tau : b tau + c <= 0}
        return ARSet(kind, None, None, ar_at, f_crit)
    disc = b * b - 4.0 * a * c
    if disc > 0:
        sq = float(np.sqrt(disc))
        r1, r2 = sorted(((-b - sq) / (2.0 * a), (-b + sq) / (2.0 * a)))
        if a > 0:
            return ARSet("interval", (r1, r2), None, ar_at, f_crit)
        rays = ((float("-inf"), r1), (r2, float("inf")))
        return ARSet("disjoint", None, rays, ar_at, f_crit)
    if a > 0:  # rejected at every tau; possible only when k >= 2
        return ARSet("empty", None, None, ar_at, f_crit)
    return ARSet("unbounded", None, None, ar_at, f_crit)


def ar_confidence_set(
    y: np.ndarray,
    T: np.ndarray,
    instruments: np.ndarray,
    controls: np.ndarray | None = None,
    alpha: float = 0.05,
) -> ARSet:
    """Closed-form Anderson–Rubin/Fieller confidence set for the effect of T.

    Intercept and ``controls`` are partialled out of (y, T, instruments);
    the level-alpha set inverts AR(tau) <= F_crit(k, n-q-k) with k the
    EFFECTIVE rank of the partialled instruments (issue #11): duplicated or
    control-collinear instruments collapse to their orthonormal basis, and
    instruments with no variation left after partialling raise. The four
    possible set kinds are reported honestly — a weak first stage yields
    "unbounded" or "disjoint" (never a fabricated finite interval); "empty"
    (model rejected at every tau) is reachable only when k >= 2. k = 1 is
    exactly the Fieller construction.
    """
    y = np.asarray(y, dtype=float).ravel()
    t_all = np.asarray(T, dtype=float).ravel()
    z_all = _as_2d(instruments)
    c_all = _as_2d(controls) if controls is not None else np.empty((y.size, 0))
    k = z_all.shape[1]
    if k == 0:
        raise ValueError("empty instrument list")
    finite = (
        np.isfinite(y)
        & np.isfinite(t_all)
        & np.isfinite(z_all).all(axis=1)
        & np.isfinite(c_all).all(axis=1)
    )
    n = int(finite.sum())
    q = 1 + c_all.shape[1]
    if n <= q:
        raise ValueError("underdetermined: need n - q - k_eff >= 1 finite rows")
    cfull = np.c_[np.ones(n), c_all[finite]]
    y_t = _residualize(y[finite], cfull)
    t_t = _residualize(t_all[finite], cfull)
    z_t = _residualize(z_all[finite], cfull)
    k_eff, z_basis = _effective_instruments(z_all[finite], z_t)
    if k_eff == 0:
        raise ValueError("instruments collinear with controls (no variation after partialling)")
    dof = n - q - k_eff
    if dof < 1:
        raise ValueError("underdetermined: need n - q - k_eff >= 1 finite rows")
    return _ar_set_partialled(y_t, t_t, z_t if k_eff == k else z_basis, dof, alpha)


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
    z_res = _residualize(z_mat, cfull)
    k_eff, z_basis = _effective_instruments(z_mat, z_res)
    if k_eff == 0:
        extras["rank_deficient"] = True
        extras["k_effective"] = 0
        extras["reason"] = "instruments collinear with controls"
        return _nan_estimate(k_eff, n_used, extras)
    if k_eff < k:
        # Honest flagged degradation: swap in an orthonormal basis of the
        # partialled instrument space (identical 2SLS projection, so tau is
        # unchanged) and use k_eff wherever the nominal k appeared.
        extras["rank_deficient"] = True
        extras["k_effective"] = k_eff
        z_mat = z_basis
        z_res = z_basis
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
    j_stat, j_p, j_df = _hansen_j(e, z_res, k_eff)
    ar_ci: tuple[float, float] | None = None
    ar_kind: str | None = None
    dof = n_used - cfull.shape[1] - k_eff
    if dof >= 1:
        ar = _ar_set_partialled(
            _residualize(ym, cfull), _residualize(tm, cfull), z_res, dof, alpha
        )
        ar_kind = ar.kind
        ar_ci = ar.interval
        extras["ar_at_2sls"] = ar.ar_at_2sls
        if ar.kind == "disjoint":
            extras["ar_rays"] = ar.rays
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
        ar_ci=ar_ci,
        ar_kind=ar_kind,
        extras=extras,
    )
