"""Diagnostic battery for known-cutoff kink designs (IZA DP 18313).

Implements the paper's validation battery: bandwidth and donut sensitivity
grids plus placebo kinks on a cutoff grid with empirical size (Figure A3),
per-period event-study kink contrasts relative to a base period (Figure A2,
Panel D), predetermined covariates as placebo outcomes, and the binned
pre/post density-difference kink test (Figure A4 and Table A3).

Every check reuses the right-minus-left reduced-form contrast from
``natex.kink.estimate`` with a unit denominator, so orientation and row
handling are identical to the headline estimator. All results are
falsification evidence, never assumption certification, and failed
computations report ``NaN`` with a reason, never ``0.0``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from natex.kink.estimate import (
    KinkEstimate,
    _canonical_zero,
    _post_indicator,
    _validate_common,
    difference_in_kinks,
    regression_kink,
)


@dataclass
class PlaceboKink:
    """Reduced-form kink contrast at one placebo cutoff."""

    cutoff: float
    estimate: float
    se: float
    p_value: float
    n_used: int
    reason: str | None


@dataclass
class PlaceboKinkGrid:
    """Placebo-cutoff battery with its empirical rejection share."""

    placebos: list[PlaceboKink]
    alpha: float
    n_evaluated: int
    n_significant: int
    empirical_size: float


@dataclass
class CovariateKink:
    """Kink contrast with a predetermined covariate as the outcome."""

    name: str
    estimate: float
    se: float
    p_value: float
    n_used: int
    reason: str | None


@dataclass
class EventStudyKink:
    """One period's kink contrast relative to the base period."""

    period: object
    estimate: float
    se: float
    ci: tuple[float, float]
    p_value: float
    n_used: int
    reason: str | None


@dataclass
class KinkEventStudy:
    """Per-period difference-in-kinks contrasts relative to ``base_period``."""

    base_period: object
    kinks: list[EventStudyKink]


@dataclass
class DensityKinkDifference:
    """Kink in the binned post-minus-pre running-variable density difference."""

    estimate: float
    se: float
    p_value: float
    n_bins: int
    bin_width: float
    n_pre: int
    n_post: int
    degree: int
    bin_centers: np.ndarray
    density_difference: np.ndarray
    reason: str | None


def _p_value(estimate: float, se: float, critical_df: int | None) -> float:
    if not (np.isfinite(estimate) and np.isfinite(se)):
        return float("nan")
    if se == 0.0:
        return float("nan") if estimate == 0.0 else 0.0
    statistic = abs(estimate / se)
    if critical_df is not None:
        return float(2.0 * stats.t.sf(statistic, critical_df))
    return float(2.0 * stats.norm.sf(statistic))


def _contrast_row(estimate: KinkEstimate) -> tuple[float, float, float, int, str | None]:
    value = float(estimate.reduced_form)
    se = float(estimate.reduced_form_se)
    slopes = estimate.extras.get("outcome_slopes")
    if se == 0.0 and slopes:
        # An exact fit leaves roundoff dust in the contrast; canonicalize it
        # exactly as the estimator does for an exact-zero first stage.
        value = _canonical_zero(value, slopes)
    p_value = _p_value(value, se, estimate.extras.get("critical_df"))
    return (value, se, p_value, estimate.n_used, estimate.extras.get("reason"))


def _reduced_form(
    y,
    running,
    *,
    post,
    cutoff: float,
    bandwidth: float,
    degree: int,
    kernel: str,
    donut: float,
    covariates,
    clusters,
    alpha: float,
) -> KinkEstimate:
    """Outcome kink contrast with a unit denominator (sharp reduced form)."""
    if post is None:
        return regression_kink(
            y,
            running,
            policy_kink=1.0,
            cutoff=cutoff,
            bandwidth=bandwidth,
            degree=degree,
            kernel=kernel,
            donut=donut,
            covariates=covariates,
            clusters=clusters,
            alpha=alpha,
        )
    return difference_in_kinks(
        y,
        running,
        post,
        policy_kink_change=1.0,
        cutoff=cutoff,
        bandwidth=bandwidth,
        degree=degree,
        kernel=kernel,
        donut=donut,
        covariates=covariates,
        clusters=clusters,
        alpha=alpha,
    )


def sensitivity_grid(
    y,
    running,
    *,
    bandwidths,
    donuts=(0.0,),
    post=None,
    treatment=None,
    policy_kink: float | None = None,
    policy_kink_change: float | None = None,
    cutoff: float = 0.0,
    degree: int = 1,
    kernel: str = "triangular",
    covariates=None,
    clusters=None,
    alpha: float = 0.05,
) -> list[KinkEstimate]:
    """Re-estimate the design over a bandwidth-by-donut grid (paper Fig. A3 A-B).

    Returns the full ``KinkEstimate`` for every combination, in
    ``bandwidths``-major order; each estimate records its own ``bandwidth``
    and ``donut`` in ``extras``.
    """
    bandwidth_grid = [float(h) for h in np.atleast_1d(np.asarray(bandwidths, dtype=float))]
    donut_grid = [float(d) for d in np.atleast_1d(np.asarray(donuts, dtype=float))]
    if not bandwidth_grid or not donut_grid:
        raise ValueError("bandwidths and donuts must be non-empty")
    invalid = [
        (h, d)
        for h in bandwidth_grid
        for d in donut_grid
        if not (np.isfinite(h) and np.isfinite(d) and 0.0 <= d < h)
    ]
    if invalid:
        raise ValueError(f"invalid bandwidth/donut combinations: {invalid}")
    if post is None and policy_kink_change is not None:
        raise ValueError("policy_kink_change applies only to a difference-in-kinks design")
    if post is not None and policy_kink is not None:
        raise ValueError("policy_kink applies only to a cross-sectional RKD")

    results: list[KinkEstimate] = []
    for h in bandwidth_grid:
        for d in donut_grid:
            if post is None:
                results.append(
                    regression_kink(
                        y,
                        running,
                        treatment=treatment,
                        policy_kink=policy_kink,
                        cutoff=cutoff,
                        bandwidth=h,
                        degree=degree,
                        kernel=kernel,
                        donut=d,
                        covariates=covariates,
                        clusters=clusters,
                        alpha=alpha,
                    )
                )
            else:
                results.append(
                    difference_in_kinks(
                        y,
                        running,
                        post,
                        treatment=treatment,
                        policy_kink_change=policy_kink_change,
                        cutoff=cutoff,
                        bandwidth=h,
                        degree=degree,
                        kernel=kernel,
                        donut=d,
                        covariates=covariates,
                        clusters=clusters,
                        alpha=alpha,
                    )
                )
    return results


def placebo_kinks(
    y,
    running,
    cutoffs,
    *,
    bandwidth: float,
    post=None,
    degree: int = 1,
    kernel: str = "triangular",
    donut: float = 0.0,
    covariates=None,
    clusters=None,
    alpha: float = 0.05,
) -> PlaceboKinkGrid:
    """Reduced-form kink contrasts at shifted placebo cutoffs (paper Fig. A3 C).

    Evaluates the outcome's right-minus-left slope contrast (post-minus-pre
    when ``post`` is given) at each supplied cutoff with a unit denominator.
    ``empirical_size`` is the share of evaluable contrasts with
    ``p_value < alpha``; supply cutoffs away from the true kink so that the
    share estimates false-rejection size. Unevaluable cutoffs are reported
    as ``NaN`` with a reason and excluded from the share.
    """
    cutoff_grid = [float(c) for c in np.atleast_1d(np.asarray(cutoffs, dtype=float))]
    if not cutoff_grid:
        raise ValueError("cutoffs must be non-empty")
    placebos: list[PlaceboKink] = []
    n_evaluated = 0
    n_significant = 0
    for placebo_cutoff in cutoff_grid:
        estimate = _reduced_form(
            y,
            running,
            post=post,
            cutoff=placebo_cutoff,
            bandwidth=bandwidth,
            degree=degree,
            kernel=kernel,
            donut=donut,
            covariates=covariates,
            clusters=clusters,
            alpha=alpha,
        )
        value, se, p_value, n_used, reason = _contrast_row(estimate)
        if np.isfinite(p_value):
            n_evaluated += 1
            if p_value < alpha:
                n_significant += 1
        placebos.append(
            PlaceboKink(
                cutoff=placebo_cutoff,
                estimate=value,
                se=se,
                p_value=p_value,
                n_used=n_used,
                reason=reason,
            )
        )
    empirical_size = n_significant / n_evaluated if n_evaluated else float("nan")
    return PlaceboKinkGrid(
        placebos=placebos,
        alpha=alpha,
        n_evaluated=n_evaluated,
        n_significant=n_significant,
        empirical_size=float(empirical_size),
    )


def covariate_kinks(
    covariates,
    running,
    *,
    bandwidth: float,
    post=None,
    cutoff: float = 0.0,
    degree: int = 1,
    kernel: str = "triangular",
    donut: float = 0.0,
    clusters=None,
    alpha: float = 0.05,
) -> list[CovariateKink]:
    """Predetermined covariates as placebo outcomes (paper Fig. A4 B-D, Table A3).

    Each named covariate is run through the same reduced-form kink contrast
    used for the outcome. A significant covariate kink (change) signals
    selection or composition at the cutoff. A kink that is stable over time
    cancels in the DiK contrast, exactly as for the outcome.
    """
    if not isinstance(covariates, Mapping) or not covariates:
        raise ValueError("covariates must be a non-empty mapping of name -> values")
    rows: list[CovariateKink] = []
    for name, values in covariates.items():
        estimate = _reduced_form(
            np.asarray(values, dtype=float),
            running,
            post=post,
            cutoff=cutoff,
            bandwidth=bandwidth,
            degree=degree,
            kernel=kernel,
            donut=donut,
            covariates=None,
            clusters=clusters,
            alpha=alpha,
        )
        value, se, p_value, n_used, reason = _contrast_row(estimate)
        rows.append(
            CovariateKink(
                name=str(name),
                estimate=value,
                se=se,
                p_value=p_value,
                n_used=n_used,
                reason=reason,
            )
        )
    return rows


def event_study_kinks(
    y,
    running,
    period,
    *,
    base_period,
    bandwidth: float,
    cutoff: float = 0.0,
    degree: int = 1,
    kernel: str = "triangular",
    donut: float = 0.0,
    covariates=None,
    clusters=None,
    alpha: float = 0.05,
) -> KinkEventStudy:
    """Per-period kink contrasts relative to a base period (paper Fig. A2 D).

    For every period other than ``base_period``, runs a sharp two-period
    difference-in-kinks (that period versus the base) with a unit
    denominator. Estimates for pre-reform periods near zero are the paper's
    pseudo-test of parallel kink trends; they cannot prove the assumption.
    """
    y_all = np.asarray(y, dtype=float)
    running_all = np.asarray(running, dtype=float)
    period_all = np.asarray(period)
    if period_all.ndim != 1 or period_all.size != y_all.size:
        raise ValueError(f"period must be one-dimensional with {y_all.size} rows")
    covariates_all = None if covariates is None else np.asarray(covariates, dtype=float)
    if covariates_all is not None and covariates_all.ndim == 1:
        covariates_all = covariates_all[:, None]
    clusters_all = None if clusters is None else np.asarray(clusters)

    present = ~pd.isna(period_all)
    labels = np.sort(pd.unique(period_all[present]))
    if not any(label == base_period for label in labels):
        raise ValueError(f"base_period {base_period!r} does not appear in period")

    kinks: list[EventStudyKink] = []
    for label in labels:
        if label == base_period:
            continue
        mask = present & ((period_all == label) | (period_all == base_period))
        estimate = difference_in_kinks(
            y_all[mask],
            running_all[mask],
            period_all[mask] == label,
            policy_kink_change=1.0,
            cutoff=cutoff,
            bandwidth=bandwidth,
            degree=degree,
            kernel=kernel,
            donut=donut,
            covariates=None if covariates_all is None else covariates_all[mask],
            clusters=None if clusters_all is None else clusters_all[mask],
            alpha=alpha,
        )
        value, se, p_value, n_used, reason = _contrast_row(estimate)
        critical_value = estimate.extras["critical_value"]
        ci = (value - critical_value * se, value + critical_value * se)
        kinks.append(
            EventStudyKink(
                period=label,
                estimate=value,
                se=se,
                ci=ci,
                p_value=p_value,
                n_used=n_used,
                reason=reason,
            )
        )
    return KinkEventStudy(base_period=base_period, kinks=kinks)


def density_kink_difference(
    running,
    post,
    *,
    bandwidth: float,
    cutoff: float = 0.0,
    n_bins: int = 80,
    degree: int = 2,
    kernel: str = "triangular",
    alpha: float = 0.05,
) -> DensityKinkDifference:
    """Kink in the binned pre/post density difference (paper Fig. A4 A, Table A3).

    Bins the running variable within the bandwidth window on both sides of
    the cutoff, forms the post-minus-pre difference of per-period density
    estimates, and fits the side-specific local polynomial to that
    difference. The estimate is the change in the first-order term at the
    cutoff. The paper's Table A3 specification (``n_bins=80``,
    ``degree=13``) is available by override, but the default is ``degree=2``
    because the degree-13 bin regression grossly over-rejects under the null
    in calibration (HC1 with 14 parameters per 40-bin side). This is a
    manipulation falsification check only — bin-level HC1 inference treats
    estimated frequencies as data.
    """
    _validate_common(cutoff, bandwidth, degree, kernel, 0.0, alpha)
    if isinstance(n_bins, bool) or not isinstance(n_bins, (int, np.integer)) or n_bins < 4:
        raise ValueError("n_bins must be an even integer >= 4")
    if n_bins % 2:
        raise ValueError("n_bins must be even so the cutoff is a bin edge")
    if n_bins // 2 <= degree + 2:
        raise ValueError(
            f"n_bins={n_bins} leaves no residual degrees of freedom per side for degree {degree}"
        )
    running_all = np.asarray(running, dtype=float)
    if running_all.ndim != 1:
        raise ValueError("running must be one-dimensional")
    post_all, post_ok = _post_indicator(post, running_all.size)
    distance = running_all - cutoff
    window = post_ok & np.isfinite(running_all) & (np.abs(distance) <= bandwidth)

    edges = np.linspace(-bandwidth, bandwidth, n_bins + 1)
    bin_width = float(edges[1] - edges[0])
    bin_centers = 0.5 * (edges[:-1] + edges[1:])
    counts: dict[bool, int] = {}
    densities: dict[bool, np.ndarray] = {}
    for flag in (False, True):
        rows = window & (post_all == flag)
        n_rows = int(rows.sum())
        counts[flag] = n_rows
        if n_rows:
            histogram, _ = np.histogram(distance[rows], bins=edges)
            densities[flag] = histogram / (n_rows * bin_width)
        else:
            densities[flag] = np.full(n_bins, np.nan)
    density_difference = densities[True] - densities[False]

    fit = regression_kink(
        density_difference,
        bin_centers,
        policy_kink=1.0,
        cutoff=0.0,
        bandwidth=bandwidth,
        degree=degree,
        kernel=kernel,
        alpha=alpha,
    )
    estimate, se, p_value, _, reason = _contrast_row(fit)
    empty = [name for name, flag in (("pre", False), ("post", True)) if counts[flag] == 0]
    if empty:
        reason = f"no {' or '.join(empty)} rows inside the bandwidth window"
    return DensityKinkDifference(
        estimate=estimate,
        se=se,
        p_value=p_value,
        n_bins=int(n_bins),
        bin_width=bin_width,
        n_pre=counts[False],
        n_post=counts[True],
        degree=int(degree),
        bin_centers=bin_centers,
        density_difference=density_difference,
        reason=reason,
    )
