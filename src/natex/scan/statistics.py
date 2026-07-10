"""LLR observation-model kernels for the neighborhood scan.

Normal model: audit-verified compact form
    LLR = B1*B0 / (2*(B1+B0)) * (m1 - m0)^2,
with B_g the group precision mass and m_g the precision-weighted group mean.
"""

from __future__ import annotations

import numpy as np
from scipy.special import expit


def normal_llr_all_splits(r: np.ndarray, w: np.ndarray, G: np.ndarray) -> np.ndarray:
    """LLR of every candidate split (columns of boolean G; True = group 1).

    Splits with an empty side score exactly 0.0. Degeneracy is detected from
    group *counts*, not precision masses: computing B0 = B - B1 in floating
    point leaves ~1e-16 residue for full columns, which would otherwise leak
    a spurious nonzero LLR.
    """
    wr = w * r
    Gt = G.T.astype(float)
    B1 = Gt @ w
    C1 = Gt @ wr
    B = float(w.sum())
    C = float(wr.sum())
    B0 = B - B1
    C0 = C - C1
    with np.errstate(divide="ignore", invalid="ignore"):
        m1 = C1 / B1
        m0 = C0 / B0
        llr = 0.5 * (m1 - m0) ** 2 * (B1 * B0) / (B1 + B0)
    n1 = G.sum(axis=0)
    degenerate = (n1 == 0) | (n1 == G.shape[0]) | (B1 <= 0.0) | (B0 <= 0.0)
    return np.where(degenerate, 0.0, llr)


def normal_glr_common_variance(r: np.ndarray, G: np.ndarray) -> np.ndarray:
    """Unknown-variance GLR per split: (k/2) * log(RSS0 / RSS1).

    A perfect two-mean fit (RSS1 == 0) is the sigma -> 0 boundary supremum of
    the alternative; per the audit convention (pure-group splits scored via
    boundary likelihood suprema, never NA/inf) RSS1 is floored at the smallest
    positive normal float, so such splits score finitely and sharper
    separations (larger RSS0) still rank strictly higher.
    """
    k = r.size
    rss0 = float(np.sum((r - r.mean()) ** 2))
    Gf = G.astype(float)
    n1 = Gf.sum(axis=0)
    n0 = k - n1
    with np.errstate(divide="ignore", invalid="ignore"):
        m1 = (Gf.T @ r) / n1
        m0 = ((1 - Gf).T @ r) / n0
        rss1 = (Gf.T @ r**2) - n1 * m1**2 + ((1 - Gf).T @ r**2) - n0 * m0**2
        rss1 = np.maximum(rss1, np.finfo(float).tiny)
        out = 0.5 * k * np.log(rss0 / rss1)
    out = np.where((n1 == 0) | (n0 == 0) | (rss0 <= 0.0), 0.0, out)
    return np.maximum(out, 0.0)


def offset_log_lik(theta: float, t: np.ndarray, eta: np.ndarray) -> float:
    """Bernoulli log-likelihood at offset theta: sum[t*z - log(1+e^z)], z = eta + theta.

    For theta = +/-inf with the corresponding pure group (all 1 / all 0) this
    returns 0.0 — the boundary supremum of the likelihood (audit correction:
    pure-group splits are scored, never NA).
    """
    if np.isposinf(theta):
        return 0.0 if np.all(t == 1) else -np.inf
    if np.isneginf(theta):
        return 0.0 if np.all(t == 0) else -np.inf
    z = eta + theta
    return float(np.sum(t * z - np.logaddexp(0.0, z)))


def fit_log_odds_offset(t: np.ndarray, eta: np.ndarray) -> float:
    """MLE of theta in P(T=1) = logistic(eta + theta) via bracketed Newton.

    The log-likelihood is concave in theta; Newton steps on the score
    sum(t - pi) are kept inside a shrinking bracket (bisection fallback).
    Pure groups return +/-inf (boundary supremum).
    """
    s = float(t.sum())
    if s == 0:
        return -np.inf
    if s == t.size:
        return np.inf
    lo, hi = -30.0, 30.0
    theta = 0.0
    for _ in range(200):
        pi = expit(eta + theta)
        score = float(np.sum(t - pi))
        if abs(score) < 1e-11:
            break
        if score > 0:
            lo = theta
        else:
            hi = theta
        info = float(np.sum(pi * (1.0 - pi)))
        step = score / max(info, 1e-12)
        cand = theta + step
        theta = cand if lo < cand < hi else 0.5 * (lo + hi)
    return theta


def _sup_log_lik(t: np.ndarray, eta: np.ndarray) -> float:
    return offset_log_lik(fit_log_odds_offset(t, eta), t, eta)


def bernoulli_llr_all_splits_reference(t: np.ndarray, eta: np.ndarray, G: np.ndarray) -> np.ndarray:
    """Phase-1 per-split Python loop, kept as the parity reference.

    For each column g of boolean G: sup_{theta0,theta1} l - sup_theta l, i.e.
    the alternative gives each side its own offset. Always >= 0; degenerate
    (one-sided) splits score exactly 0.0.
    """
    ll0 = _sup_log_lik(t, eta)
    m = G.shape[1]
    out = np.zeros(m)
    for j in range(m):
        g = G[:, j]
        if g.all() or (~g).all():
            out[j] = 0.0
            continue
        ll1 = _sup_log_lik(t[g], eta[g]) + _sup_log_lik(t[~g], eta[~g])
        out[j] = max(ll1 - ll0, 0.0)
    return out


def fit_log_odds_offsets(t: np.ndarray, eta: np.ndarray, M: np.ndarray) -> np.ndarray:
    """Vectorized per-column MLE of theta in P(T=1) = logistic(eta + theta).

    M is a boolean (k, m) membership mask, one column per group. Returns
    theta (m,), with +/-inf for pure columns (all masked t equal 1 / 0;
    empty columns count as all-zeros) — the boundary suprema of audit item
    21. Mixed columns run the same bracketed Newton as the scalar
    ``fit_log_odds_offset`` (audit item 22), all columns at once: brackets
    start at (-30, 30), Newton steps outside the bracket fall back to
    bisection, convergence at |score| < 1e-11, max 200 iterations.
    Converged columns are retired from the iteration.
    """
    t = np.asarray(t, dtype=float)
    Mf = M.astype(float)
    cnt = Mf.sum(axis=0)
    s = Mf.T @ t
    lower = s == 0.0
    upper = (s == cnt) & ~lower
    out = np.where(lower, -np.inf, np.where(upper, np.inf, 0.0))
    act = np.flatnonzero(~(lower | upper))
    if act.size == 0:
        return out
    Msub = Mf[:, act]
    s_sub = s[act]
    theta = np.zeros(act.size)
    lo = np.full(act.size, -30.0)
    hi = np.full(act.size, 30.0)
    for _ in range(200):
        p = expit(eta[:, None] + theta[None, :])
        score = s_sub - (p * Msub).sum(axis=0)
        done = np.abs(score) < 1e-11
        if done.any():
            out[act[done]] = theta[done]
            keep = ~done
            if not keep.any():
                return out
            act, s_sub, Msub = act[keep], s_sub[keep], Msub[:, keep]
            theta, lo, hi = theta[keep], lo[keep], hi[keep]
            score, p = score[keep], p[:, keep]
        info = (p * (1.0 - p) * Msub).sum(axis=0)
        pos = score > 0.0
        lo = np.where(pos, theta, lo)
        hi = np.where(pos, hi, theta)
        cand = theta + score / np.maximum(info, 1e-12)
        theta = np.where((cand > lo) & (cand < hi), cand, 0.5 * (lo + hi))
    out[act] = theta
    return out


def masked_offset_log_lik(
    theta: np.ndarray, t: np.ndarray, eta: np.ndarray, M: np.ndarray
) -> np.ndarray:
    """Per-column masked Bernoulli log-likelihood at offsets theta.

    Column j: sum_i M_ij * (t_i * z_ij - log(1 + e^{z_ij})), z_ij = eta_i +
    theta_j. At theta_j = +/-inf the value is the boundary supremum 0.0 for
    the matching pure column (masked t all ones / all zeros; both vacuously
    true for an empty column), -inf otherwise (audit item 21: pure groups
    scored via boundary suprema, never NA).
    """
    t = np.asarray(t, dtype=float)
    theta = np.asarray(theta, dtype=float)
    Mf = M.astype(float)
    cnt = Mf.sum(axis=0)
    s = Mf.T @ t
    finite = np.isfinite(theta)
    z = eta[:, None] + np.where(finite, theta, 0.0)[None, :]
    ll = ((t[:, None] * z - np.logaddexp(0.0, z)) * Mf).sum(axis=0)
    at_posinf = np.where(s == cnt, 0.0, -np.inf)
    at_neginf = np.where(s == 0.0, 0.0, -np.inf)
    return np.where(finite, ll, np.where(theta > 0, at_posinf, at_neginf))


def bernoulli_llr_all_splits(t: np.ndarray, eta: np.ndarray, G: np.ndarray) -> np.ndarray:
    """LLR of every candidate split for the Bernoulli offset model (vectorized).

    Same contract as ``bernoulli_llr_all_splits_reference`` (allclose to
    atol 1e-8, including pure-group boundary-suprema columns): each column g
    scores sup over per-side offsets minus the single shared-offset supremum,
    always >= 0, with degenerate (one-sided) columns exactly 0.0. Newton is
    vectorized across all split-sides via ``fit_log_odds_offsets``.
    """
    t = np.asarray(t, dtype=float)
    ll0 = _sup_log_lik(t, eta)
    Gc = ~G
    ll1 = masked_offset_log_lik(
        fit_log_odds_offsets(t, eta, G), t, eta, G
    ) + masked_offset_log_lik(fit_log_odds_offsets(t, eta, Gc), t, eta, Gc)
    n1 = G.sum(axis=0)
    degenerate = (n1 == 0) | (n1 == G.shape[0])
    return np.where(degenerate, 0.0, np.maximum(ll1 - ll0, 0.0))
