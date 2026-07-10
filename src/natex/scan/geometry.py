"""Scan geometry cache (audit §3, adopted improvements).

The scan geometry — kNN neighborhoods plus deduped candidate partitions —
depends only on ``Z_std``, which is identical across all Q null replicas of the
randomization test. Building it once and sharing it removes the dominant
avoidable cost of ``randomization_test`` without changing a single bit of the
output (replica draw order is untouched).

``shrink(k2)`` exploits the Kmax-NN prefix property: cKDTree returns neighbors
sorted by distance, so the first ``k2`` columns of a ``k``-NN index array ARE
the exact ``k2``-NN lists (self stays in column 0). One build at Kmax serves
every smaller k.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from natex.scan import neighborhoods


@dataclass
class ScanGeometry:
    """kNN index table plus a lazy per-center partition cache.

    Memory note: fully populated, the partition cache holds ~n * k^2 booleans
    (each center caches a (k, <=k-1) mask G plus its keep vector) — e.g.
    44k centers x 50 x 49 ~ 108 MB. Acceptable for the target scales; callers
    that only touch a center subset pay only for the centers they visit.
    """

    k: int
    idx: np.ndarray  # (n, k) own-neighborhood indices, self first
    _partitions: dict[int, tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)

    def partitions_for(self, i: int, Z_std: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Deduped candidate partitions (G, keep) for center i, cached lazily."""
        cached = self._partitions.get(i)
        if cached is None:
            cz = Z_std[self.idx[i]] - Z_std[i]
            cached = neighborhoods.candidate_partitions(cz)
            self._partitions[i] = cached
        return cached

    def shrink(self, k2: int) -> ScanGeometry:
        """Exact k2-NN geometry via the Kmax prefix; fresh empty partition cache."""
        if k2 > self.k:
            raise ValueError(f"cannot shrink to k2={k2} > k={self.k}")
        return ScanGeometry(k=k2, idx=self.idx[:, :k2])


def build_geometry(Z_std: np.ndarray, k: int) -> ScanGeometry:
    """kNN index table for Z_std with an empty (lazily filled) partition cache."""
    return ScanGeometry(k=k, idx=neighborhoods.knn_indices(Z_std, k=k))
