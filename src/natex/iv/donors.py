"""Synthetic-control donor selection with pre-trend scoring (phase 5).

Abadie, Diamond & Hainmueller (2010) donor selection on a unit-by-time
outcome matrix: candidates are units (other than the treated) with finite
outcomes at EVERY pre-``t0`` time where the treated unit is observed (the
phase-3 balanced-donor rule; dropped candidates are counted in
``extras["n_dropped_incomplete"]``). Candidates are scored against the
treated pre-trajectory (RMSE and Pearson correlation over the common pre
times), the top ``n_donors`` form the pool, and simplex weights
(``natex.estimate.simplex.fit_simplex_weights``, the shared deterministic
SLSQP fitter) build the counterfactual and post-period ATT.

Pre-only honesty: scoring, ranking, and weight fitting read ONLY pre-``t0``
outcome columns — the post period is the estimation target. This inherent
SC use of pre-period outcomes is a documented method property, enforced by
mutation tests (changing post outcomes never moves scores or weights).

Failure paths (never 0.0): no complete candidate or treated unobserved
pre-``t0`` -> ``att_post = NaN``, ``pre_rmspe = +inf``, all-NaN
counterfactual, reason in ``extras["failure"]``; no defined post period ->
``att_post = NaN``, ``post_rmspe = +inf``.

DOCUMENTED DEVIATION (same as phase 3): outcome-only pre-fit; Abadie et
al.'s covariate V-weights are not implemented.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from natex.data.spec import Dataset
from natex.estimate.simplex import fit_simplex_weights, weighted_counterfactual

_SCORING = ("rmse", "corr")


@dataclass
class DonorScore:
    """Pre-period fit of one complete donor candidate to the treated unit."""

    unit: object
    pre_rmse: float  # vs treated pre-trajectory (common finite pre times)
    pre_corr: float  # Pearson over the same times; NaN if < 3 points
    rank: int  # 1 = best by the active scoring method


@dataclass
class DonorSelectionResult:
    """Donor pool, simplex weights, counterfactual and post-period ATT."""

    treated_unit: object
    t0: float
    donors: list[object]  # selected pool, score order
    scores: list[DonorScore]  # ALL complete candidates, ranked
    weights: np.ndarray  # simplex weights over `donors`
    y0_hat: np.ndarray  # (n_t,) counterfactual for the treated unit, NaN where undefined
    times: np.ndarray  # (n_t,) sorted unique times
    pre_rmspe: float  # +inf if no defined pre time (never 0 on failure)
    post_rmspe: float
    att_post: float  # mean post-period (y_treated - y0_hat); NaN on failure
    effect_by_time: np.ndarray  # (n_t,) gap, NaN where undefined
    extras: dict = field(default_factory=dict)


def unit_time_matrix(
    df: pd.DataFrame, unit: str, time: str, outcome: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(Y, units, times): unit-by-time mean-aggregated outcome matrix, NaN for empty cells.

    Duplicate ``(unit, time)`` rows mean-aggregate (NaN outcomes skipped);
    a cell with no finite outcome is NaN — never 0. ``units`` and ``times``
    are sorted unique values; ``times`` is cast to float.
    """
    for col in (unit, time, outcome):
        if col not in df.columns:
            raise ValueError(f"column not in dataframe: {col!r}")
    wide = df.groupby([unit, time], sort=True)[outcome].mean().unstack(time)
    return (
        wide.to_numpy(dtype=float),
        wide.index.to_numpy(),
        wide.columns.to_numpy(dtype=float),
    )


def select_donors(
    Y: np.ndarray,
    units: np.ndarray,
    times: np.ndarray,
    treated_unit: object,
    t0: float,
    n_donors: int | None = None,  # None -> all complete candidates
    scoring: str = "rmse",  # "rmse" | "corr"
) -> DonorSelectionResult:
    """Score, select and weight donors for ``treated_unit`` at cutoff ``t0``.

    See the module docstring for the balanced-donor rule, pre-only honesty
    and failure-path policy.
    """
    if scoring not in _SCORING:
        raise ValueError(f"scoring must be one of {_SCORING}, got {scoring!r}")
    if n_donors is not None and n_donors < 1:
        raise ValueError(f"n_donors must be >= 1 or None, got {n_donors}")
    Y = np.asarray(Y, dtype=float)
    units = np.asarray(units)
    times = np.asarray(times, dtype=float)
    if Y.shape != (units.size, times.size):
        raise ValueError(
            f"Y must be (n_units, n_times) = ({units.size}, {times.size}), got {Y.shape}"
        )
    hit = np.flatnonzero(units == treated_unit)
    if hit.size != 1:
        raise ValueError(
            f"treated_unit {treated_unit!r} matches {hit.size} rows of units (need exactly 1)"
        )
    i_treated = int(hit[0])

    order = np.argsort(times, kind="stable")
    times = times[order]
    Y = Y[:, order]
    n_t = times.size
    t0 = float(t0)

    y_treated = Y[i_treated]
    pre = times < t0
    post = ~pre
    common = pre & np.isfinite(y_treated)  # treated-observed pre times

    cand_idx = np.array([i for i in range(units.size) if i != i_treated], dtype=int)
    n_candidates = int(cand_idx.size)

    def _fail(reason: str, n_dropped: int = 0) -> DonorSelectionResult:
        return DonorSelectionResult(
            treated_unit=treated_unit,
            t0=t0,
            donors=[],
            scores=[],
            weights=np.zeros(0),
            y0_hat=np.full(n_t, np.nan),
            times=times,
            pre_rmspe=float("inf"),  # never 0 on failure
            post_rmspe=float("inf"),
            att_post=float("nan"),
            effect_by_time=np.full(n_t, np.nan),
            extras={
                "failure": reason,
                "n_candidates": n_candidates,
                "n_dropped_incomplete": n_dropped,
                "n_common_pre_times": int(common.sum()),
                "converged": False,
            },
        )

    if not common.any():
        return _fail("treated unit has no finite outcome before t0 (no common pre time)")
    if n_candidates == 0:
        return _fail("no candidate units besides the treated unit")

    # Balanced-donor rule: complete candidates have finite outcomes at EVERY
    # common pre time (phase-3 rule; see did/controls.synthetic_control).
    complete = np.all(np.isfinite(Y[cand_idx][:, common]), axis=1)
    n_dropped = int(n_candidates - complete.sum())
    cand_idx = cand_idx[complete]
    if cand_idx.size == 0:
        return _fail("no candidate has finite outcomes at every treated pre time", n_dropped)

    # ---- pre-period scoring (reads ONLY the common pre columns) ----
    y_pre = y_treated[common]  # (n_common,)
    donors_pre = Y[cand_idx][:, common]  # (n_cand, n_common), all finite
    resid = donors_pre - y_pre[None, :]
    rmse = np.sqrt(np.mean(resid**2, axis=1))
    n_common = int(common.sum())
    if n_common >= 3:
        yc = y_pre - y_pre.mean()
        dc = donors_pre - donors_pre.mean(axis=1, keepdims=True)
        denom = np.sqrt((dc**2).sum(axis=1) * float(yc @ yc))
        corr = np.where(denom > 0.0, (dc @ yc) / np.where(denom > 0.0, denom, 1.0), np.nan)
    else:
        corr = np.full(cand_idx.size, np.nan)  # undefined, never fabricated

    key = rmse if scoring == "rmse" else -corr
    rank_order = np.argsort(key, kind="stable")  # NaN keys sort last
    scores = [
        DonorScore(
            unit=units[cand_idx[i]],
            pre_rmse=float(rmse[i]),
            pre_corr=float(corr[i]),
            rank=r + 1,
        )
        for r, i in enumerate(rank_order)
    ]

    k = cand_idx.size if n_donors is None else min(int(n_donors), int(cand_idx.size))
    donor_rows = cand_idx[rank_order[:k]]
    donors = [units[j] for j in donor_rows]

    # ---- simplex weights on pre columns only; counterfactual on all times ----
    fit = fit_simplex_weights(y_pre, Y[donor_rows][:, common].T)
    y0_hat = weighted_counterfactual(Y[donor_rows], fit.weights)  # (n_t,)
    gap = y_treated - y0_hat

    ok_pre = pre & np.isfinite(gap)
    pre_rmspe = float(np.sqrt(np.mean(gap[ok_pre] ** 2))) if ok_pre.any() else float("inf")
    ok_post = post & np.isfinite(gap)
    post_rmspe = float(np.sqrt(np.mean(gap[ok_post] ** 2))) if ok_post.any() else float("inf")
    att_post = float(np.mean(gap[ok_post])) if ok_post.any() else float("nan")

    return DonorSelectionResult(
        treated_unit=treated_unit,
        t0=t0,
        donors=donors,
        scores=scores,
        weights=np.asarray(fit.weights, dtype=float),
        y0_hat=y0_hat,
        times=times,
        pre_rmspe=pre_rmspe,
        post_rmspe=post_rmspe,
        att_post=att_post,
        effect_by_time=gap,
        extras={
            "n_candidates": n_candidates,
            "n_dropped_incomplete": n_dropped,
            "n_common_pre_times": n_common,
            "n_post_times_defined": int(ok_post.sum()),
            "converged": fit.converged,
            "fit_sse": fit.sse,
            "scoring": scoring,
            "note": "outcome-only pre-fit; Abadie covariate V-weights not implemented",
        },
    )


def select_donors_from_dataset(
    dataset: Dataset,
    treated_unit: object,
    t0: float,
    n_donors: int | None = None,
    scoring: str = "rmse",
) -> DonorSelectionResult:
    """Adapter: requires spec.unit, spec.time, spec.outcome; delegates to select_donors."""
    spec = dataset.spec
    missing = [
        f"spec.{name}"
        for name, value in (("unit", spec.unit), ("time", spec.time), ("outcome", spec.outcome))
        if value is None
    ]
    if missing:
        raise ValueError(f"donor selection requires unit, time and outcome roles; missing: {missing}")
    Y, units, times = unit_time_matrix(dataset.df, spec.unit, spec.time, spec.outcome)
    return select_donors(
        Y, units, times, treated_unit, t0, n_donors=n_donors, scoring=scoring
    )
