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

Inference: :func:`sc_placebo_test` is Abadie et al.'s in-space placebo
test — every complete donor refit as pseudo-treated under the identical
selection rule (treated unit removed from every placebo panel), post/pre
RMSPE ratios, +1-rank p-value. Deterministic, no rng anywhere.

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
_MIN_USABLE_PLACEBOS = 5  # fewer usable placebos -> p = NaN (phase-3 policy, never a fake 1.0)


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


# ---------------------------------------------------------------------------
# Abadie in-space RMSPE-ratio placebo inference (audit item 1/5 lineage)
# ---------------------------------------------------------------------------


def _rmspe_ratio(pre_rmspe: float, post_rmspe: float) -> float:
    """Post/pre RMSPE ratio; NaN on every failure path, never a fake 0.0.

    ``pre = +inf`` / ``post = +inf`` mark failed fits or an undefined
    period (the :func:`select_donors` failure policy), so the ratio is NaN
    — ``post / inf`` would fabricate 0.0. A genuinely perfect pre fit
    (``pre == 0``, ``post > 0``) is +inf; 0/0 is NaN.
    """
    if not (np.isfinite(pre_rmspe) and np.isfinite(post_rmspe)):
        return float("nan")
    if pre_rmspe == 0.0:
        return float("inf") if post_rmspe > 0.0 else float("nan")
    return float(post_rmspe / pre_rmspe)


@dataclass
class SCPlaceboReport:
    """In-space placebo test: +1-rank p-value on post/pre RMSPE ratios."""

    p_value: float  # (1 + #{placebo ratio >= treated}) / (n_used + 1); NaN if n_used < 5
    ratio_treated: float  # post_rmspe / pre_rmspe of the treated run
    ratios: np.ndarray  # (n_used,) usable placebo ratios (may include +inf), sorted descending
    placebo_units: list[object]  # aligned with `ratios`
    n_skipped: int  # undefined (NaN) ratios — failed fits or 0/0 — plus poor-fit exclusions
    extras: dict = field(default_factory=dict)


def sc_placebo_test(
    Y: np.ndarray,
    units: np.ndarray,
    times: np.ndarray,
    treated_unit: object,
    t0: float,
    n_donors: int | None = None,
    scoring: str = "rmse",
    exclude_poor_fit: float | None = None,
) -> SCPlaceboReport:
    """Abadie, Diamond & Hainmueller (2010) in-space placebo inference.

    Every complete donor candidate of the treated run is refit as
    pseudo-treated under the IDENTICAL selection rule (same ``n_donors``
    and ``scoring``); the treated unit's row is removed from every placebo
    panel, so it can never enter a placebo donor pool (its real post-period
    effect must not contaminate placebo counterfactuals). Each fit yields
    ratio = post-RMSPE / pre-RMSPE — sign-agnostic, so the test is
    two-sided by construction (audit item 5) — and the p-value is the
    +1-rank Monte Carlo form (audit item 1 lineage):

        p = (1 + #{placebo ratio >= treated ratio}) / (n_used + 1)

    Placebos with an UNDEFINED (NaN) ratio — a failed fit, or 0/0 from an
    exactly-zero pre-RMSPE with no post gap — are skipped and counted in
    ``n_skipped`` (never a fake 0.0 ratio). A zero pre-RMSPE with real post
    divergence gives ratio = +inf: a defined, maximally extreme value that
    is KEPT (counted in ``extras["n_inf_ratio"]``) — it sorts first, ties
    with an infinite treated ratio, and keeps the +1-rank p conservative.
    With ``exclude_poor_fit = m``, placebos whose pre-RMSPE exceeds
    ``m x`` the treated unit's are also dropped (Abadie's poor-pre-fit
    exclusion) and listed in ``extras["poor_fit_units"]``. Fewer than
    ``_MIN_USABLE_PLACEBOS`` (5) usable placebos — or an undefined treated
    ratio — gives ``p = NaN``, never a fake 1.0 (phase-3 policy).
    Deterministic: no rng anywhere.
    """
    if exclude_poor_fit is not None and not exclude_poor_fit > 0:
        raise ValueError(f"exclude_poor_fit must be > 0 or None, got {exclude_poor_fit}")
    treated_res = select_donors(
        Y, units, times, treated_unit, t0, n_donors=n_donors, scoring=scoring
    )
    ratio_treated = _rmspe_ratio(treated_res.pre_rmspe, treated_res.post_rmspe)

    units_arr = np.asarray(units)
    i_treated = int(np.flatnonzero(units_arr == treated_unit)[0])  # validated above
    Y_placebo = np.delete(np.asarray(Y, dtype=float), i_treated, axis=0)
    units_placebo = np.delete(units_arr, i_treated)

    candidates = [s.unit for s in treated_res.scores]  # every complete donor, score order
    pools: dict = {}
    pre_rmspes: dict = {}
    kept: list[tuple[object, float]] = []
    poor_fit_units: list[object] = []
    n_failed = 0
    n_zero_pre = 0
    n_inf_ratio = 0
    for u in candidates:
        res = select_donors(
            Y_placebo, units_placebo, times, u, t0, n_donors=n_donors, scoring=scoring
        )
        pools[u] = list(res.donors)
        pre_rmspes[u] = res.pre_rmspe
        ratio = _rmspe_ratio(res.pre_rmspe, res.post_rmspe)
        if np.isnan(ratio):
            # NaN is the only UNDEFINED case: pre_rmspe == 0.0 marks 0/0 (or
            # an undefined post period), anything else a failed fit. +inf — a
            # perfect pre fit with real post divergence — is a defined,
            # maximally extreme ratio and is kept (issue #15): dropping it
            # shrank the +1-rank numerator, anti-conservative.
            if res.pre_rmspe == 0.0:
                n_zero_pre += 1
            else:
                n_failed += 1
            continue
        if exclude_poor_fit is not None and res.pre_rmspe > exclude_poor_fit * treated_res.pre_rmspe:
            poor_fit_units.append(u)
            continue
        if np.isinf(ratio):
            n_inf_ratio += 1
        kept.append((u, ratio))

    raw = np.asarray([r for _, r in kept], dtype=float)
    order = np.argsort(-raw, kind="stable")
    ratios = raw[order]
    placebo_units = [kept[i][0] for i in order]
    n_used = int(ratios.size)
    n_skipped = n_failed + n_zero_pre + len(poor_fit_units)

    if n_used < _MIN_USABLE_PLACEBOS or np.isnan(ratio_treated):
        p_value = float("nan")
    else:
        p_value = float((1 + int(np.sum(ratios >= ratio_treated))) / (n_used + 1))

    extras = {
        "n_placebo_candidates": len(candidates),
        "n_used": n_used,
        "n_failed": n_failed,
        "n_zero_pre_rmspe": n_zero_pre,  # NaN ratios with pre_rmspe == 0 (0/0 or undefined post)
        "n_inf_ratio": n_inf_ratio,  # kept +inf ratios (perfect pre fit, divergent post)
        "n_poor_fit": len(poor_fit_units),
        "poor_fit_units": poor_fit_units,
        "placebo_pools": pools,
        "placebo_pre_rmspe": pre_rmspes,
        "treated_pre_rmspe": treated_res.pre_rmspe,
        "treated_post_rmspe": treated_res.post_rmspe,
        "exclude_poor_fit": exclude_poor_fit,
        "n_donors": n_donors,
        "scoring": scoring,
    }
    if "failure" in treated_res.extras:
        extras["treated_failure"] = treated_res.extras["failure"]
    return SCPlaceboReport(
        p_value=p_value,
        ratio_treated=ratio_treated,
        ratios=ratios,
        placebo_units=placebo_units,
        n_skipped=n_skipped,
        extras=extras,
    )
