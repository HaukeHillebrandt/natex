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
    from collections.abc import Callable

    from natex.data.spec import Dataset
    from natex.rdd.lord3 import LoRD3Result


def _coarse_fine_stages(
    dataset: Dataset,
    coarse_centers: np.ndarray,
    k: int,
    top_m: int,
    radius_mult: float,
    model: str,
    degree: int,
    geometry: ScanGeometry,
    tree: cKDTree | None = None,
) -> tuple[LoRD3Result, LoRD3Result, np.ndarray]:
    """Coarse scan over FROZEN centers -> top-m localization -> fine rescan.

    The center subsample is the pipeline's only treatment-independent
    randomness; which fine centers get rescanned depends on the dataset's
    treatment through the coarse discoveries, so null-replica datasets must
    rerun all three stages themselves (issue #21) — hence this helper takes
    ``coarse_centers`` as a frozen array rather than drawing it.
    """
    # Runtime import breaks the natex.scan <-> natex.rdd.lord3 import cycle
    # (lord3 needs scan.geometry; the scan package re-exports this module).
    from natex.rdd.lord3 import lord3_scan

    Z = dataset.Z_std
    n = dataset.n
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
        if tree is None:
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
    return coarse_result, result, fine_centers


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
    if rng is None:
        raise ValueError("pass an explicit numpy Generator (reproducibility contract)")
    n = dataset.n
    # Build (or accept) the full geometry once: the kNN query is cheap even at
    # 44k rows; the savings live in per-center partition/LLR work.
    if geometry is None:
        geometry = build_geometry(dataset.Z_std, k)
    elif geometry.k != k:
        raise ValueError(f"geometry.k={geometry.k} disagrees with k={k}")

    # Coarse stage: seeded subsample of centers.
    coarse_centers = rng.choice(n, size=min(n_coarse, n), replace=False)
    coarse_result, result, fine_centers = _coarse_fine_stages(
        dataset, coarse_centers, k, top_m, radius_mult, model, degree, geometry
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


def coarse_to_fine_search(
    dataset: Dataset,
    coarse_centers: np.ndarray,
    k: int = 50,
    top_m: int = 20,
    radius_mult: float = 2.0,
    model: str = "auto",
    degree: int = 1,
    geometry: ScanGeometry | None = None,
) -> Callable[[Dataset], LoRD3Result]:
    """Procedure-matched replica search for the fitted-null calibration (issue #21).

    Returns ``search(ds_star) -> LoRD3Result`` for
    :func:`natex.validate.randomization.randomization_test`: each null replica
    reruns the WHOLE treatment-dependent coarse-to-fine pipeline on its own
    treatment draw — coarse scan over the frozen, treatment-independent center
    subsample ``coarse_centers`` (conditioned on, exactly as in the observed
    run), that replica's own top-``top_m`` localization, then the
    full-resolution rescan — so replica maxima follow the same search
    procedure as the observed statistic. Rescanning replicas over ALL centers
    instead would give stochastically larger null maxima (a full-scan max
    dominates any subset max) and inflate the p-value.

    ``geometry`` and the kd-tree depend only on ``Z_std``, which replicas
    share with ``dataset`` (only the treatment column is redrawn), so both are
    built once here and reused by every replica; ``model`` should be the
    resolved kind from the observed scan result.
    """
    coarse_centers = np.asarray(coarse_centers, dtype=int)
    if geometry is None:
        geometry = build_geometry(dataset.Z_std, k)
    elif geometry.k != k:
        raise ValueError(f"geometry.k={geometry.k} disagrees with k={k}")
    tree = cKDTree(dataset.Z_std)

    def search(ds_star: Dataset) -> LoRD3Result:
        _, result, _ = _coarse_fine_stages(
            ds_star, coarse_centers, k, top_m, radius_mult, model, degree, geometry, tree
        )
        return result

    return search
