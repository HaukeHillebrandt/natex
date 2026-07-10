"""Coarse-to-fine scan (spec 6d): seeded center subsample, top-m localization,
full-resolution rescan near candidates.

Spec 6b contract — never silently truncate: the result always reports which
centers were scanned (``fine_centers``, ``frac_centers_scanned``) and with what
parameters (``params``), so callers (CLI, results bundle) can state coverage.

Deterministic given the caller's Generator: the only stochastic step is the
coarse center subsample; localization and both scans are pure functions of it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy.spatial import cKDTree

from natex.scan.geometry import ScanGeometry, build_geometry

if TYPE_CHECKING:
    from natex.data.spec import Dataset
    from natex.rdd.lord3 import LoRD3Result


@dataclass
class CoarseToFineResult:
    result: LoRD3Result  # fine-stage discoveries (full-resolution, subset of centers)
    coarse_result: LoRD3Result
    fine_centers: np.ndarray  # dataset indices scanned at full resolution
    frac_centers_scanned: float  # len(unique(coarse union fine centers)) / n
    params: dict  # n_coarse, top_m, radius_mult, k, model, degree, seed note


def coarse_to_fine_scan(
    dataset: Dataset,
    k: int = 50,
    n_coarse: int = 2000,
    top_m: int = 20,
    radius_mult: float = 2.0,
    model: str = "auto",
    degree: int = 1,
    rng: np.random.Generator | None = None,
    geometry: ScanGeometry | None = None,
) -> CoarseToFineResult:
    # Runtime import breaks the natex.scan <-> natex.rdd.lord3 import cycle
    # (lord3 needs scan.geometry; the scan package re-exports this module).
    from natex.rdd.lord3 import lord3_scan

    if rng is None:
        raise ValueError("pass an explicit numpy Generator (reproducibility contract)")
    Z = dataset.Z_std
    n = dataset.n
    # Build (or accept) the full geometry once: the kNN query is cheap even at
    # 44k rows; the savings live in per-center partition/LLR work.
    if geometry is None:
        geometry = build_geometry(Z, k)
    elif geometry.k != k:
        raise ValueError(f"geometry.k={geometry.k} disagrees with k={k}")

    # Coarse stage: seeded subsample of centers.
    coarse_centers = rng.choice(n, size=min(n_coarse, n), replace=False)
    coarse_result = lord3_scan(
        dataset, k=k, model=model, degree=degree, centers=coarse_centers, geometry=geometry
    )

    if coarse_centers.shape[0] == n:
        # Degenerate small-n case: the coarse stage already visited every
        # center, so full resolution means all of them — no localization.
        fine_centers = np.arange(n)
    else:
        # Localization: around each top-m coarse discovery, take every point
        # within radius_mult * r_k(center), where r_k is the distance from the
        # center to the k-th (farthest) member of its own kNN list.
        tree = cKDTree(Z)
        fine_sets: list[np.ndarray] = []
        for d in coarse_result.top(top_m):
            c = d.center_index
            r_k = float(np.linalg.norm(Z[geometry.idx[c, -1]] - Z[c]))
            ball = tree.query_ball_point(Z[c], r=radius_mult * r_k)
            fine_sets.append(np.asarray(ball, dtype=int))
        fine_centers = (
            np.unique(np.concatenate(fine_sets)) if fine_sets else np.empty(0, dtype=int)
        )

    # Fine stage: full-resolution rescan over the union of localized sets.
    result = lord3_scan(
        dataset, k=k, model=model, degree=degree, centers=fine_centers, geometry=geometry
    )

    scanned = np.union1d(coarse_centers, fine_centers)
    return CoarseToFineResult(
        result=result,
        coarse_result=coarse_result,
        fine_centers=fine_centers,
        frac_centers_scanned=float(scanned.shape[0]) / float(n),
        params={
            "n_coarse": n_coarse,
            "top_m": top_m,
            "radius_mult": radius_mult,
            "k": k,
            "model": model,
            "degree": degree,
            "seed_note": "coarse centers drawn from the caller's Generator; "
            "same seed and data give an identical scan",
        },
    )
