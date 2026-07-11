"""Repaired Multidimensional Subset Scan (MDSS) — thesis Algorithm 8.

Algorithm 8 as printed is deletion-only (it slices ``s_tau ∩ {x_j = v}``, so
a value dropped once can never return) and leaves the priorities of empty
slices undefined. The audit-13 repair implemented here: per dimension j,
build the *relaxed* state (dimension j unconstrained, every other dimension
as-is), generate candidate value subsets of dimension j on the relaxed
records, and **retain the incumbent explicitly** — a candidate replaces the
current constraint only when its LLR strictly exceeds the current state's.
The LLR is therefore weakly increasing by construction.

Candidate generation per dimension (audit item 16): when the dimension's
cardinality ``k_j`` is small (``k_j <= exhaustive_max_values``), all
``2^k_j - 1`` nonempty value subsets are enumerated and scored in one
vectorized evaluator call — exact for the printed statistic, and the natex
default. At larger cardinality one of three heuristic priority orderings is
used, and the LLR is evaluated at every prefix of the ordering:

* ``"greedy"`` — start from the best single value, iteratively add the value
  maximizing the combined ``q1 - q2`` of the union (the thesis's greedy
  remedy for the Simpson's-paradox failure of naive ``q1 - q2`` ordering);
* ``"wcc"`` — weighted convex combinations: draw ``n_rho`` values of
  ``rho ~ U(0, 1)`` from ``rng`` and order values by
  ``rho * q1 + (1 - rho) * (-q2)`` for each draw;
* ``"single_delta"`` — order values by ``gamma_k = C_tilde_k / B_tilde_k``
  both descending **and** ascending (both signs of Delta, audit item 15).

Undefined (empty-slice) priorities sort last — they still appear in the
trailing prefixes, so no value subset is silently unreachable. Discovery
never reads the outcome ``y``; failed evaluator columns (NaN) are never
treated as 0.0 — they are simply never selected.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from natex.did.panel import CategoricalPanel
from natex.did.statistics import WindowStats, single_delta_stats

Evaluator = Callable[[np.ndarray], np.ndarray]
"""(n, S) boolean mask matrix -> (S,) LLR values (closure over a fixed (T0, W) + model)."""

_PRIORITIES = ("greedy", "wcc", "single_delta")
_MAX_EXACT_VALUES = 20  # hard cap: never enumerate more than 2^20 - 1 candidate columns


@dataclass
class SubsetState:
    """Conjunction-of-unions subset state: per-dimension bool masks over value codes.

    ``included[j]`` has length ``k_j``; a record is in the subset iff every
    dimension's value code is included. The all-True state is ``s = D``.
    """

    included: list[np.ndarray]

    def mask(self, panel: CategoricalPanel) -> np.ndarray:
        """(n,) record membership mask."""
        return panel.subset_mask(self.included)

    def values(self, panel: CategoricalPanel) -> dict[str, list]:
        """Decoded included values per CONSTRAINED dimension (all-True dims omitted)."""
        out: dict[str, list] = {}
        for j, inc in enumerate(self.included):
            inc = np.asarray(inc, dtype=bool)
            if not bool(inc.all()):
                out[panel.dim_names[j]] = np.asarray(panel.dim_values[j])[inc].tolist()
        return out


@dataclass
class SingleDeltaPriority:
    """Per-profile corrected single-Delta stats for priority ordering (audit 15).

    ``C_tilde`` / ``B_tilde`` are the profiled sufficient statistics from
    :func:`natex.did.statistics.single_delta_stats`, indexed by profile id.
    ``c_rec`` / ``b_rec`` distribute each profile's statistic evenly over its
    records, so a mask sum ``M.T @ c_rec`` equals the sum of ``C_tilde`` over
    the profiles inside the mask — exact whenever masks contain whole
    profiles, which every conjunction-of-unions subset over the SAME panel
    dimensions does (profiles are constant on each dimension).
    """

    profile_id: np.ndarray  # (n,) record -> profile id
    C_tilde: np.ndarray  # (P,) per-profile
    B_tilde: np.ndarray  # (P,) per-profile
    c_rec: np.ndarray = field(init=False, repr=False)  # (n,) record-distributed C_tilde
    b_rec: np.ndarray = field(init=False, repr=False)  # (n,) record-distributed B_tilde

    def __post_init__(self) -> None:
        pid = np.asarray(self.profile_id)
        counts = np.bincount(pid, minlength=self.C_tilde.shape[0]).astype(float)
        per = counts[pid]  # >= 1 for every present profile
        self.c_rec = self.C_tilde[pid] / per
        self.b_rec = self.B_tilde[pid] / per

    @classmethod
    def from_window_stats(
        cls, ws: WindowStats, profile_id: np.ndarray, n_profiles: int | None = None
    ) -> SingleDeltaPriority:
        C_tilde, B_tilde = single_delta_stats(ws, profile_id, n_profiles=n_profiles)
        return cls(profile_id=np.asarray(profile_id), C_tilde=C_tilde, B_tilde=B_tilde)


def _nonempty_value_subsets(k: int) -> np.ndarray:
    """(2^k - 1, k) bool — every nonempty subset of a dimension's value codes."""
    bits = np.arange(1, 2**k, dtype=np.int64)
    return ((bits[:, None] >> np.arange(k, dtype=np.int64)[None, :]) & 1).astype(bool)


def _prefix_masks(order: np.ndarray) -> np.ndarray:
    """(k, k) bool — row p includes the first p+1 values of ``order``."""
    k = order.size
    V = np.zeros((k, k), dtype=bool)
    for p, v in enumerate(order):
        V[p:, v] = True
    return V


def _value_side_sums(
    ws: WindowStats, codes_j: np.ndarray, k: int, relaxed: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per-value (C1, B1, C0, B0) over the slices ``relaxed & (x_j = v)``."""
    Vslice = (codes_j[:, None] == np.arange(k)[None, :]) & relaxed[:, None]
    Vt = Vslice.T.astype(float)
    c1 = Vt @ (ws.c * ws.g1)
    b1 = Vt @ (ws.b * ws.g1)
    g0 = ws.g0
    c0 = Vt @ (ws.c * g0)
    b0 = Vt @ (ws.b * g0)
    return c1, b1, c0, b0


def _greedy_order(
    c1v: np.ndarray, b1v: np.ndarray, c0v: np.ndarray, b0v: np.ndarray
) -> np.ndarray:
    """Thesis greedy remedy: iteratively add the value maximizing the union's q1 - q2.

    Empty-side unions have undefined priority (0/0 -> NaN) and sort last;
    ties break on the lowest value code (deterministic).
    """
    k = c1v.shape[0]
    order = np.empty(k, dtype=np.int64)
    used = np.zeros(k, dtype=bool)
    c1 = b1 = c0 = b0 = 0.0
    for step in range(k):
        cand = np.flatnonzero(~used)
        with np.errstate(divide="ignore", invalid="ignore"):
            pr = (c1 + c1v[cand]) / (b1 + b1v[cand]) - (c0 + c0v[cand]) / (b0 + b0v[cand])
        pr = np.where(np.isnan(pr), -np.inf, pr)
        nxt = int(cand[int(np.argmax(pr))])
        order[step] = nxt
        used[nxt] = True
        c1 += float(c1v[nxt])
        b1 += float(b1v[nxt])
        c0 += float(c0v[nxt])
        b0 += float(b0v[nxt])
    return order


def _wcc_prefixes(
    c1v: np.ndarray,
    b1v: np.ndarray,
    c0v: np.ndarray,
    b0v: np.ndarray,
    rng: np.random.Generator,
    n_rho: int,
) -> np.ndarray:
    """Prefix families for ``n_rho`` draws of rho ~ U(0,1): rho*q1 + (1-rho)*(-q2)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        q1 = c1v / b1v
        neg_q2 = -(c0v / b0v)
    rhos = rng.uniform(0.0, 1.0, size=n_rho)
    stacks = []
    for rho in rhos:
        pr = rho * q1 + (1.0 - rho) * neg_q2
        pr = np.where(np.isnan(pr), -np.inf, pr)
        stacks.append(_prefix_masks(np.argsort(-pr, kind="stable")))
    return np.vstack(stacks)


def _single_delta_prefixes(
    prio: SingleDeltaPriority, codes_j: np.ndarray, k: int, relaxed: np.ndarray
) -> np.ndarray:
    """Prefix families for gamma = C_tilde/B_tilde, descending AND ascending (audit 15)."""
    Vslice = (codes_j[:, None] == np.arange(k)[None, :]) & relaxed[:, None]
    Vt = Vslice.T.astype(float)
    Cv = Vt @ prio.c_rec
    Bv = Vt @ prio.b_rec
    with np.errstate(divide="ignore", invalid="ignore"):
        gamma = np.where(Bv > 0.0, Cv / Bv, np.nan)
    # NaN sorts last under np.argsort in both directions (-NaN is still NaN).
    desc = np.argsort(-gamma, kind="stable")
    asc = np.argsort(gamma, kind="stable")
    return np.vstack([_prefix_masks(desc), _prefix_masks(asc)])


def mdss_optimize(
    panel: CategoricalPanel,
    evaluator: Evaluator,
    priority: str,
    priority_stats: object,
    rng: np.random.Generator,
    init: SubsetState | None = None,
    n_rho: int = 10,
    exhaustive_max_values: int = 12,
    max_sweeps: int = 25,
    tol: float = 1e-12,
    *,
    trace: list[float] | None = None,
) -> tuple[SubsetState, float]:
    """Repaired Algorithm 8: conditionally optimize the subset for a fixed (T0, W).

    Each sweep shuffles the dimension order via ``rng`` (Alg 8 line 3). Per
    dimension j (audit-13 repair): relax dimension j, generate candidate
    value subsets on the relaxed records — exhaustively when
    ``k_j <= exhaustive_max_values`` (audit 16), else prefixes of the chosen
    priority ordering — and accept the best candidate only when its LLR
    exceeds the current state's by more than ``tol``; otherwise the current
    constraint for dimension j is retained. Converged when a full sweep
    accepts no update (or after ``max_sweeps``). Returns ``(state, llr)``;
    the LLR trace is weakly increasing by construction (appended to
    ``trace``, when given, after the initial evaluation and after every
    dimension step).

    ``priority_stats`` must be a :class:`WindowStats` for ``"greedy"`` /
    ``"wcc"`` and a :class:`SingleDeltaPriority` for ``"single_delta"``;
    ``init=None`` starts from ``s = D``. ``init`` is copied, never mutated.
    """
    if not isinstance(rng, np.random.Generator):
        raise TypeError(f"rng must be a numpy Generator, got {type(rng).__name__}")
    if priority not in _PRIORITIES:
        raise ValueError(f"priority must be one of {_PRIORITIES}, got {priority!r}")
    if priority == "single_delta":
        if not isinstance(priority_stats, SingleDeltaPriority):
            raise TypeError(
                "priority='single_delta' requires SingleDeltaPriority stats, got "
                f"{type(priority_stats).__name__}"
            )
    elif not isinstance(priority_stats, WindowStats):
        raise TypeError(
            f"priority={priority!r} requires WindowStats, got {type(priority_stats).__name__}"
        )
    if priority == "wcc" and n_rho < 1:
        raise ValueError(f"n_rho must be >= 1, got {n_rho}")
    if max_sweeps < 1:
        raise ValueError(f"max_sweeps must be >= 1, got {max_sweeps}")

    m = panel.m
    sizes = panel.dim_sizes
    if init is None:
        included = [np.ones(k, dtype=bool) for k in sizes]
    else:
        if len(init.included) != m:
            raise ValueError(
                f"init has {len(init.included)} dimension masks, panel has {m} dims"
            )
        included = []
        for j, inc in enumerate(init.included):
            inc = np.asarray(inc, dtype=bool)
            if inc.shape != (sizes[j],):
                raise ValueError(
                    f"init mask for dim {j} ({panel.dim_names[j]}) has shape "
                    f"{inc.shape}, expected ({sizes[j]},)"
                )
            included.append(inc.copy())

    cur_llr = float(np.asarray(evaluator(panel.subset_mask(included)[:, None]), dtype=float)[0])
    if trace is not None:
        trace.append(cur_llr)

    for _sweep in range(max_sweeps):
        improved = False
        for j in rng.permutation(m):
            j = int(j)
            k = sizes[j]
            codes_j = panel.codes[:, j]
            # Relaxed state: dimension j unconstrained, all other dims as-is.
            relaxed = np.ones(panel.n, dtype=bool)
            for jj in range(m):
                if jj != j:
                    relaxed &= included[jj][panel.codes[:, jj]]
            if k <= exhaustive_max_values:
                if k > _MAX_EXACT_VALUES:
                    raise ValueError(
                        f"refusing exhaustive enumeration over 2^{k} - 1 subsets "
                        f"(dim {panel.dim_names[j]}); lower exhaustive_max_values"
                    )
                V = _nonempty_value_subsets(k)
            elif priority == "greedy":
                sums = _value_side_sums(priority_stats, codes_j, k, relaxed)
                V = _prefix_masks(_greedy_order(*sums))
            elif priority == "wcc":
                sums = _value_side_sums(priority_stats, codes_j, k, relaxed)
                V = _wcc_prefixes(*sums, rng=rng, n_rho=n_rho)
            else:  # single_delta
                V = _single_delta_prefixes(priority_stats, codes_j, k, relaxed)

            M = relaxed[:, None] & V[:, codes_j].T
            llrs = np.asarray(evaluator(M), dtype=float)
            llrs = np.where(np.isnan(llrs), -np.inf, llrs)  # failed columns never win
            best = int(np.argmax(llrs))
            # Audit-13 incumbent retention: accept only a strict improvement.
            if llrs[best] > cur_llr + tol:
                included[j] = V[best].copy()
                cur_llr = float(llrs[best])
                improved = True
            if trace is not None:
                trace.append(cur_llr)
        if not improved:
            break

    return SubsetState(included=included), float(cur_llr)
