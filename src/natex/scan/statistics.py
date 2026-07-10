"""LLR observation-model kernels for the neighborhood scan.

Normal model: audit-verified compact form
    LLR = B1*B0 / (2*(B1+B0)) * (m1 - m0)^2,
with B_g the group precision mass and m_g the precision-weighted group mean.
"""

from __future__ import annotations

import numpy as np


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
