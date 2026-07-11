"""DEE step 1: repair overlapping LoRD3 discoveries into disjoint quasi-experiments.

Corrected Algorithm 1 (Jakubowski et al., JMLR 2023) per docs/math_audit_final.md:

- repo risk 1: every input is explicit (dataset, discoveries, m_prime, k_prime,
  t_side) -- no module-level ``train_data`` or other global state.
- repo risk 2: the FIRST candidate faces the same per-side support threshold as
  every later one (the paper's repo installed it unchecked).
- repo risk 3: ``projected_center`` is the direct orthogonal projection of the
  member centroid onto the discovery's frozen hyperplane,
  ``centroid - ((centroid - Z_std[center]) @ n) * n`` -- deterministic, no
  random Gram-Schmidt basis, no RNG anywhere in this module.
- audit 23 tie convention: signed distance >= 0 => group 1 (same as the scan);
  Voronoi ownership ties go to the earlier-accepted center, deterministically.
- audit 7: ``experiment_radius`` reports the per-experiment radius rho feeding
  the corrected Theorem-1 diagnostic (E[q^2] <= 2*rho^2, sqrt(2) factor; see
  the method card). natex never uses the paper's uncorrected printed bound.

All geometry lives in Z_std space (the scan's space). Repair never reads y.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from natex.data.spec import Dataset
from natex.rdd.lord3 import Discovery, LoRD3Result


@dataclass
class QuasiExperiment:
    """Disjoint local experiment; duck-type-compatible with ``rdd.lord3.Discovery``.

    ``center_index, members, group1, normal`` carry the exact names and types
    the downstream stages expect, so ``local_2sls``, ``placebo_tests``,
    ``density_test`` and ``signed_distance`` work on it unchanged.
    """

    center_index: int  # dataset row index of the accepted center
    members: np.ndarray  # int row indices: k'-NN ball  ∩ Voronoi cell (center included)
    group1: np.ndarray  # bool over members; signed distance >= 0 side (audit 23)
    normal: np.ndarray  # unit normal in Z_std space, frozen from the Discovery
    llr: float  # the discovery LLR that ranked this center
    centroid: np.ndarray  # Z_std[members].mean(axis=0)
    projected_center: np.ndarray  # centroid projected onto the frozen hyperplane


@dataclass
class VKNNResult:
    experiments: list[QuasiExperiment]  # in acceptance order (LLR rank order)
    accepted: np.ndarray  # int ranks into the candidate list that were accepted
    rejected: np.ndarray  # int ranks rejected by the support test
    k_prime: int  # effective k' (clamped to n)
    t_side: int


def _ordered_candidates(discoveries: Sequence[Discovery], m_prime: int) -> list[Discovery]:
    """Top-m_prime candidates, defensively re-sorted if llr is not non-increasing."""
    cands = list(discoveries)
    llrs = [d.llr for d in cands]
    if any(llrs[i] < llrs[i + 1] for i in range(len(llrs) - 1)):
        cands.sort(key=lambda d: d.llr, reverse=True)  # stable: equal llr keeps input order
    return cands[: max(int(m_prime), 0)]


def _member_sets(
    Z: np.ndarray, trial: list[Discovery], balls: dict[int, np.ndarray]
) -> list[np.ndarray]:
    """k'-NN ball ∩ Voronoi cell for every center in ``trial`` (acceptance order).

    Squared distances use one formula per center so exact geometric ties are
    bitwise ties; ``np.argmin`` then awards the row to the EARLIER-accepted
    center (deterministic tie-break).
    """
    d2 = np.stack([((Z - Z[int(d.center_index)]) ** 2).sum(axis=1) for d in trial])
    owner = np.argmin(d2, axis=0)
    out: list[np.ndarray] = []
    for j, d in enumerate(trial):
        ball = balls[int(d.center_index)]
        out.append(np.sort(ball[owner[ball] == j]))
    return out


def _both_sides_supported(
    Z: np.ndarray, d: Discovery, members: np.ndarray, t_side: int
) -> bool:
    s = (Z[members] - Z[int(d.center_index)]) @ np.asarray(d.normal, dtype=float)
    g1 = s >= 0.0
    return int(g1.sum()) >= t_side and int((~g1).sum()) >= t_side


def voronoi_knn_repair(
    dataset: Dataset,
    discoveries: Sequence[Discovery],
    m_prime: int,
    k_prime: int = 200,
    t_side: int = 30,
) -> VKNNResult:
    """Forward-stepwise repair of the top-m_prime discoveries into disjoint experiments.

    Tentatively add each LLR-ranked candidate, reassign every accepted center's
    index set to its k'-NN ball intersected with its Voronoi cell (over accepted
    centers only, distances in Z_std), and accept iff EVERY accepted center --
    including the new one and including the very first -- retains >= t_side
    members on both sides of its frozen hyperplane; otherwise reject the
    candidate and restore the previous index sets. RNG-free by construction.
    """
    Z = dataset.Z_std
    n = Z.shape[0]
    kq = min(int(k_prime), n)
    cands = _ordered_candidates(discoveries, m_prime)

    balls: dict[int, np.ndarray] = {}
    if cands:
        tree = cKDTree(Z)  # built once; queries cached per center
        for d in cands:
            c = int(d.center_index)
            if c not in balls:
                _, idx = tree.query(Z[c], k=kq)
                balls[c] = np.atleast_1d(np.asarray(idx, dtype=int))

    accepted: list[Discovery] = []
    members: list[np.ndarray] = []
    accepted_ranks: list[int] = []
    rejected_ranks: list[int] = []
    for rank, cand in enumerate(cands):
        trial = accepted + [cand]
        trial_members = _member_sets(Z, trial, balls)
        if all(
            _both_sides_supported(Z, d, mem, t_side)
            for d, mem in zip(trial, trial_members, strict=True)
        ):
            accepted, members = trial, trial_members
            accepted_ranks.append(rank)
        else:
            rejected_ranks.append(rank)  # restore: keep the previous accepted sets

    experiments: list[QuasiExperiment] = []
    for d, mem in zip(accepted, members, strict=True):
        c = int(d.center_index)
        nhat = np.asarray(d.normal, dtype=float)
        s = (Z[mem] - Z[c]) @ nhat
        centroid = Z[mem].mean(axis=0)
        projected = centroid - ((centroid - Z[c]) @ nhat) * nhat
        experiments.append(
            QuasiExperiment(
                center_index=c,
                members=mem,
                group1=s >= 0.0,
                normal=nhat,
                llr=float(d.llr),
                centroid=centroid,
                projected_center=projected,
            )
        )
    return VKNNResult(
        experiments=experiments,
        accepted=np.asarray(accepted_ranks, dtype=int),
        rejected=np.asarray(rejected_ranks, dtype=int),
        k_prime=kq,
        t_side=int(t_side),
    )


def select_m_prime(
    scan_result: LoRD3Result, null_max_llrs: np.ndarray, level: float = 0.95
) -> int:
    """M' = number of per-center max LLRs STRICTLY above the null-max quantile.

    ``null_max_llrs`` comes from the phase-1 fitted-null Monte Carlo
    (``validate.randomization.RandomizationReport.null_max_llrs``); this is the
    paper's rural-roads M' rule. Zero is a legal answer (empty candidate list).
    """
    llrs = np.asarray([d.llr for d in scan_result.discoveries], dtype=float)
    if llrs.size == 0:
        return 0
    null = np.asarray(null_max_llrs, dtype=float)
    if null.size == 0:
        raise ValueError("null_max_llrs is empty")
    return int(np.sum(llrs > float(np.quantile(null, level))))


def experiment_radius(dataset: Dataset, e: QuasiExperiment) -> float:
    """rho = max member distance from the projected center, in Z_std (audit 7).

    Diagnostic input to the corrected Theorem-1 bound E[q^2] <= 2*rho^2 (with
    the sqrt(2) factor, valid along fixed-shape scaling only -- method card).
    NaN (never 0.0) for an empty experiment.
    """
    if e.members.size == 0:
        return float("nan")
    diff = dataset.Z_std[e.members] - e.projected_center
    return float(np.max(np.linalg.norm(diff, axis=1)))
