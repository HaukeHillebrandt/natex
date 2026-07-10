"""Neighborhood construction and candidate hyperplane partitions.

Tie convention (audit item 23): signed distance >= 0 => group 1; the center
point lies on every candidate plane and always belongs to group 1.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def knn_indices(z_std: np.ndarray, k: int) -> np.ndarray:
    if k < 2:
        raise ValueError("k must be >= 2")
    tree = cKDTree(z_std)
    _, idx = tree.query(z_std, k=k)
    idx = np.atleast_2d(idx)
    # cKDTree returns self as the first neighbor for exact matches, but ties can
    # reorder; enforce self-first deterministically.
    n = z_std.shape[0]
    for i in range(n):
        if idx[i, 0] != i and i in idx[i]:
            j = int(np.where(idx[i] == i)[0][0])
            idx[i, 0], idx[i, j] = idx[i, j], idx[i, 0]
    return idx


def candidate_partitions(cz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    dots = cz @ cz.T
    G = dots >= 0.0  # column j: split by normal (x_j - center)
    k = G.shape[0]
    seen: set[bytes] = set()
    keep: list[int] = []
    for j in range(G.shape[1]):
        col = G[:, j]
        s = int(col.sum())
        if s == 0 or s == k:
            continue  # degenerate (includes the center's own zero normal)
        # DEE complement dedup (audit: legacy `~col` check was dead code because
        # tied points -- the center at least -- are group 1 under BOTH
        # orientations). The antipodal normal -n assigns group 1 to {dots <= 0},
        # so that is the complement key.
        b1, b2 = col.tobytes(), (dots[:, j] <= 0.0).tobytes()
        if b1 in seen or b2 in seen:
            continue
        seen.add(b1)
        seen.add(b2)
        keep.append(j)
    return G[:, keep], np.asarray(keep, dtype=int)


def local_residual_variance(r: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Per-point residual variance over each point's OWN k-neighborhood.

    Audit item 20: the legacy implementation indexed reverse neighbors by
    mistake; this function is row-wise over idx (own neighborhoods) by
    construction. Floor is data-scaled (audit item 24), never absolute.
    """
    local = np.var(r[idx], axis=1, ddof=1)
    floor = 1e-3 * float(np.var(r, ddof=1))
    return np.maximum(local, floor)
