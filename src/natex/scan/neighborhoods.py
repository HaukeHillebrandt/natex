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
        # Dedup on actual membership masks only (issue #8). Under the tie
        # convention every candidate mask contains the center, so exact
        # complements are impossible among candidates; antipodal normals
        # yield genuinely DISTINCT partitions (tied rows stay group 1 under
        # both orientations) and both must be scored. Duplicate normals
        # (identical masks) still collapse.
        b1 = col.tobytes()
        if b1 in seen:
            continue
        seen.add(b1)
        keep.append(j)
    return G[:, keep], np.asarray(keep, dtype=int)


def local_residual_variance(r: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Per-point residual variance over each point's OWN k-neighborhood.

    Audit item 20: the legacy implementation indexed reverse neighbors by
    mistake; this function is row-wise over idx (own neighborhoods) by
    construction. Floor is data-scaled (audit item 24), never absolute.

    A zero (or non-finite) global residual variance means the Normal treatment
    background fits the treatment exactly; the floor would be 0, the precision
    weights inf, and every LLR NaN -- NaN then wins argmax and poisons the
    randomization p-value (NaN >= NaN is False -> p = 1/(Q+1)). Fail loudly
    instead (issue #9); discover() isolates this as status="failed".
    """
    gv = float(np.var(r, ddof=1))
    if not np.isfinite(gv) or gv <= 0.0:
        raise ValueError(
            "Normal treatment background fits the treatment exactly -- zero "
            "residual variance (constant or exactly-linear treatment); the "
            "Normal scan model is degenerate and the LLR is undefined"
        )
    local = np.var(r[idx], axis=1, ddof=1)
    return np.maximum(local, 1e-3 * gv)
