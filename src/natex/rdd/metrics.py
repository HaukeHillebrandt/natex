"""Alignment metrics between discovered splits and known ground truth."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from natex.rdd.lord3 import Discovery, LoRD3Result


def _entropy(p: float) -> float:
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return float(-p * np.log2(p) - (1 - p) * np.log2(1 - p))


def normalized_information_gain(true_D: np.ndarray, members: np.ndarray, group1: np.ndarray) -> float:
    t = true_D[members].astype(bool)
    h = _entropy(t.mean())
    if h == 0.0:
        return 0.0
    cond = 0.0
    for side in (group1, ~group1):
        if side.sum() == 0:
            continue
        cond += side.mean() * _entropy(t[side].mean())
    return max((h - cond) / h, 0.0)


@dataclass
class DiscoveryCluster:
    representative: Discovery  # highest-LLR member
    center_z: np.ndarray  # raw-z of representative center (copy)
    size: int
    max_llr: float


def cluster_discoveries(
    result: LoRD3Result,
    Z_raw: np.ndarray,
    tol: float | np.ndarray,
    top: int | None = None,
) -> list[DiscoveryCluster]:
    """Greedily cluster discovery centers for multi-cutoff assertions.

    Walks ``result.discoveries`` in descending LLR (the list is already sorted)
    and assigns each discovery to the first existing cluster whose representative
    center is within ``tol`` on EVERY raw-z dimension (Chebyshev distance after
    per-dimension scaling), otherwise opens a new cluster. ``Z_raw`` is the
    dataset's raw (unstandardized) forcing matrix so tolerances are interpretable
    in the data's own units; a scalar ``tol`` broadcasts across dimensions.
    ``top`` restricts clustering to the first ``top`` discoveries (default: all).
    The returned list is ordered by ``max_llr`` descending.
    """
    Z = np.asarray(Z_raw, dtype=float)
    if Z.ndim == 1:
        Z = Z[:, None]
    tol_arr = np.broadcast_to(np.asarray(tol, dtype=float), (Z.shape[1],))
    if np.any(tol_arr < 0) or not np.all(np.isfinite(tol_arr)):
        raise ValueError("tol must be finite and non-negative")
    discoveries = result.discoveries if top is None else result.discoveries[:top]
    clusters: list[DiscoveryCluster] = []
    for d in discoveries:
        center = Z[d.center_index]
        for c in clusters:
            if np.all(np.abs(center - c.center_z) <= tol_arr):
                c.size += 1
                break
        else:
            clusters.append(
                DiscoveryCluster(
                    representative=d,
                    center_z=center.copy(),
                    size=1,
                    max_llr=float(d.llr),
                )
            )
    clusters.sort(key=lambda c: c.max_llr, reverse=True)
    return clusters
