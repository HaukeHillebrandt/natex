"""Known-cutoff regression-kink and difference-in-kinks estimation.

All slope contrasts use the explicit ``right minus left`` convention.  The
ratio is invariant to reversing that convention, but the reported reduced
form and first stage are not, so the orientation is part of the public API.

The DiK point estimator follows Böckerman, Jysmä, and Kanninen (2025), Eqs.
9--10.  Inference is a natex extension: stacked kernel WLS with HC1 or CR1
sandwich covariance, the outcome/first-stage cross covariance in fuzzy
ratios, and a Fieller confidence set for weak denominators.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

_KERNELS = {"triangular", "uniform", "epanechnikov"}
_WEAK_F_THRESHOLD = 10.0
_RELATIVE_ROUNDOFF_TOL = 1_000.0 * np.finfo(float).eps


@dataclass
class KinkEstimate:
    """Local-polynomial kink-ratio estimate and design diagnostics."""

    tau: float
    se: float
    ci: tuple[float, float]
    method: str  # sharp_rkd | fuzzy_rkd | sharp_dik | fuzzy_dik
    reduced_form: float
    reduced_form_se: float
    first_stage: float
    first_stage_se: float
    first_stage_F: float
    weak_first_stage: bool
    n_used: int
    n_by_cell: dict[str, int]
    fieller_ci: tuple[float, float] | None = None
    fieller_kind: str | None = None  # interval | empty | disjoint | unbounded
    extras: dict = field(default_factory=dict)


@dataclass
class _Prepared:
    y: np.ndarray
    treatment: np.ndarray | None
    x: np.ndarray
    weights: np.ndarray
    design: np.ndarray
    contrast: np.ndarray
    cell: np.ndarray
    cell_labels: tuple[str, ...]
    clusters: np.ndarray | None
    n_input: int
    n_dropped_nonfinite: int
    n_outside_bandwidth: int
    n_donut_excluded: int
    n_zero_weight_excluded: int
    n_by_cell: dict[str, int]
    n_covariates: int
    n_covariates_dropped_constant: int


@dataclass
class _Fit:
    beta: np.ndarray
    residual: np.ndarray
    bread: np.ndarray
    cov: np.ndarray
    rank: int
    response_scale: float


def _as_vector(name: str, values, n: int | None = None) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if n is not None and arr.size != n:
        raise ValueError(f"{name} has {arr.size} rows; expected {n}")
    return arr


def _as_covariates(values, n: int) -> np.ndarray:
    if values is None:
        return np.empty((n, 0), dtype=float)
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim != 2 or arr.shape[0] != n:
        raise ValueError(f"covariates must have shape ({n}, q), got {arr.shape}")
    return arr


def _validate_common(
    cutoff: float,
    bandwidth: float,
    degree: int,
    kernel: str,
    donut: float,
    alpha: float,
) -> None:
    if not np.isfinite(cutoff):
        raise ValueError("cutoff must be finite")
    if not np.isfinite(bandwidth) or bandwidth <= 0.0:
        raise ValueError("bandwidth must be finite and positive")
    if isinstance(degree, bool) or not isinstance(degree, (int, np.integer)) or degree < 1:
        raise ValueError("degree must be an integer >= 1")
    if kernel not in _KERNELS:
        raise ValueError(f"kernel must be one of {sorted(_KERNELS)}, got {kernel!r}")
    if not np.isfinite(donut) or donut < 0.0 or donut >= bandwidth:
        raise ValueError("donut must be finite and satisfy 0 <= donut < bandwidth")
    if not np.isfinite(alpha) or not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1)")


def _validate_first_stage(treatment, policy_kink: float | None, name: str) -> None:
    if (treatment is None) == (policy_kink is None):
        raise ValueError(
            f"supply exactly one of treatment (fuzzy) or {name} (sharp)"
        )
    if policy_kink is not None and (not np.isfinite(policy_kink) or policy_kink == 0.0):
        raise ValueError(f"{name} must be finite and nonzero")


def _kernel_weights(u: np.ndarray, kernel: str) -> np.ndarray:
    a = np.abs(u)
    if kernel == "triangular":
        return 1.0 - a
    if kernel == "epanechnikov":
        return 0.75 * (1.0 - a**2)
    return np.ones_like(u)


def _post_indicator(post, n: int) -> tuple[np.ndarray, np.ndarray]:
    raw = np.asarray(post)
    if raw.ndim != 1 or raw.size != n:
        raise ValueError(f"post must be one-dimensional with {n} rows")
    present = ~pd.isna(raw)
    if pd.api.types.is_numeric_dtype(raw.dtype):
        present &= np.isfinite(raw.astype(float))
    values = set(pd.unique(raw[present]).tolist())
    if not values <= {0, 1, False, True}:
        raise ValueError("post must contain only boolean or 0/1 values")
    indicator = np.zeros(n, dtype=bool)
    indicator[present] = raw[present].astype(bool)
    if set(indicator[present].tolist()) != {False, True}:
        raise ValueError("post must contain both pre and post rows")
    return indicator, present


def _cluster_present(clusters, n: int) -> tuple[np.ndarray | None, np.ndarray]:
    if clusters is None:
        return None, np.ones(n, dtype=bool)
    raw = np.asarray(clusters)
    if raw.ndim != 1 or raw.size != n:
        raise ValueError(f"clusters must be one-dimensional with {n} rows")
    present = ~pd.isna(raw)
    if pd.api.types.is_numeric_dtype(raw.dtype):
        present &= np.isfinite(raw.astype(float))
    return raw, present


def _prepare(
    y,
    running,
    *,
    treatment,
    post,
    cutoff: float,
    bandwidth: float,
    degree: int,
    kernel: str,
    donut: float,
    covariates,
    clusters,
) -> _Prepared:
    y_all = _as_vector("y", y)
    n = y_all.size
    running_all = _as_vector("running", running, n)
    treatment_all = (
        None if treatment is None else _as_vector("treatment", treatment, n)
    )
    covariates_all = _as_covariates(covariates, n)
    cluster_all, cluster_ok = _cluster_present(clusters, n)
    if post is None:
        post_all = None
        post_ok = np.ones(n, dtype=bool)
        cell_labels = ("left", "right")
    else:
        post_all, post_ok = _post_indicator(post, n)
        cell_labels = ("pre_left", "pre_right", "post_left", "post_right")

    finite = np.isfinite(y_all) & np.isfinite(running_all) & post_ok & cluster_ok
    if treatment_all is not None:
        finite &= np.isfinite(treatment_all)
    if covariates_all.shape[1]:
        finite &= np.isfinite(covariates_all).all(axis=1)
    n_dropped_nonfinite = int(n - finite.sum())

    distance = running_all - cutoff
    within_bandwidth = np.abs(distance) <= bandwidth
    outside_bandwidth = int(np.sum(finite & ~within_bandwidth))
    donut_excluded = int(
        np.sum(finite & within_bandwidth & (np.abs(distance) < donut))
    )
    in_window = within_bandwidth & (np.abs(distance) >= donut)
    selected = finite & in_window
    u_all = distance / bandwidth
    positive_weight = np.zeros(n, dtype=bool)
    positive_weight[selected] = _kernel_weights(u_all[selected], kernel) > 0.0
    zero_weight_excluded = int(np.sum(selected & ~positive_weight))
    selected &= positive_weight

    y_fit = y_all[selected]
    treatment_fit = None if treatment_all is None else treatment_all[selected]
    u = u_all[selected]
    right = u >= 0.0
    if post_all is None:
        cell = right.astype(np.int64)
    else:
        cell = post_all[selected].astype(np.int64) * 2 + right.astype(np.int64)
    weights = _kernel_weights(u, kernel)
    n_by_cell = {
        label: int(np.sum(cell == j)) for j, label in enumerate(cell_labels)
    }

    block_size = degree + 1
    n_blocks = len(cell_labels)
    block_design = np.zeros((u.size, n_blocks * block_size), dtype=float)
    powers = np.column_stack([u**j for j in range(block_size)]) if u.size else np.empty(
        (0, block_size)
    )
    for j in range(n_blocks):
        rows = cell == j
        block_design[rows, j * block_size : (j + 1) * block_size] = powers[rows]

    cov_fit = covariates_all[selected]
    dropped_constant = 0
    if cov_fit.shape[1] and cov_fit.shape[0]:
        centered_covariates = cov_fit.copy()
        for j in range(len(cell_labels)):
            rows = cell == j
            if np.any(rows):
                centered_covariates[rows] -= cov_fit[rows][0]
        column_scale = np.max(np.abs(centered_covariates), axis=0)
        nonzero = column_scale > 0.0
        normalized = np.zeros_like(centered_covariates)
        normalized[:, nonzero] = (
            centered_covariates[:, nonzero] / column_scale[nonzero]
        )
        mean = normalized.mean(axis=0)
        scale = normalized.std(axis=0, ddof=0)
        threshold = np.finfo(float).eps * max(cov_fit.shape[0], 1)
        keep = nonzero & (scale > threshold)
        dropped_constant = int((~keep).sum())
        cov_fit = (normalized[:, keep] - mean[keep]) / scale[keep]
    elif cov_fit.shape[1]:
        dropped_constant = cov_fit.shape[1]
        cov_fit = np.empty((0, 0), dtype=float)
    design = np.c_[block_design, cov_fit]

    contrast = np.zeros(design.shape[1], dtype=float)
    slope_scale = 1.0 / bandwidth
    if post_all is None:
        contrast[1] = -slope_scale
        contrast[block_size + 1] = slope_scale
    else:
        contrast[1] = slope_scale
        contrast[block_size + 1] = -slope_scale
        contrast[2 * block_size + 1] = -slope_scale
        contrast[3 * block_size + 1] = slope_scale

    cluster_fit = None
    if cluster_all is not None:
        cluster_fit, _ = pd.factorize(cluster_all[selected], sort=False)

    return _Prepared(
        y=y_fit,
        treatment=treatment_fit,
        x=u * bandwidth,
        weights=weights,
        design=design,
        contrast=contrast,
        cell=cell,
        cell_labels=cell_labels,
        clusters=cluster_fit,
        n_input=n,
        n_dropped_nonfinite=n_dropped_nonfinite,
        n_outside_bandwidth=outside_bandwidth,
        n_donut_excluded=donut_excluded,
        n_zero_weight_excluded=zero_weight_excluded,
        n_by_cell=n_by_cell,
        n_covariates=cov_fit.shape[1],
        n_covariates_dropped_constant=dropped_constant,
    )


def _score_meat(
    design: np.ndarray,
    weights: np.ndarray,
    residual_a: np.ndarray,
    residual_b: np.ndarray,
    clusters: np.ndarray | None,
) -> tuple[np.ndarray, float, int | None]:
    score_a = design * (weights * residual_a)[:, None]
    score_b = design * (weights * residual_b)[:, None]
    n, p = design.shape
    if clusters is None:
        factor = n / max(n - p, 1)
        return score_a.T @ score_b, factor, None
    n_clusters = int(clusters.max(initial=-1) + 1)
    if n_clusters < 2:
        raise ValueError("cluster-robust inference requires at least two clusters")
    sums_a = np.zeros((n_clusters, p), dtype=float)
    sums_b = np.zeros((n_clusters, p), dtype=float)
    np.add.at(sums_a, clusters, score_a)
    np.add.at(sums_b, clusters, score_b)
    factor = (n_clusters / (n_clusters - 1)) * ((n - 1) / max(n - p, 1))
    return sums_a.T @ sums_b, factor, n_clusters


def _fit(prepared: _Prepared, values: np.ndarray) -> _Fit:
    design, weights = prepared.design, prepared.weights
    centered_values = values.copy()
    cell_locations = np.zeros(len(prepared.cell_labels), dtype=float)
    for j in range(len(prepared.cell_labels)):
        rows = prepared.cell == j
        cell_locations[j] = values[rows][0]
        centered_values[rows] -= cell_locations[j]
    response_scale = float(np.max(np.abs(centered_values), initial=0.0))
    normalized_values = (
        centered_values / response_scale
        if response_scale > 0.0
        else centered_values
    )
    sqrt_weights = np.sqrt(weights)
    weighted_design = design * sqrt_weights[:, None]
    centered_beta, _, rank, _ = np.linalg.lstsq(
        weighted_design, sqrt_weights * normalized_values, rcond=None
    )
    rank = int(rank)
    p = design.shape[1]
    if rank < p:
        raise ValueError(f"rank-deficient local-polynomial design ({rank} < {p})")
    beta = centered_beta * response_scale
    block_size = (p - prepared.n_covariates) // len(prepared.cell_labels)
    for j, location in enumerate(cell_locations):
        beta[j * block_size] += location
    xtwx = design.T @ (design * weights[:, None])
    bread = np.linalg.inv(xtwx)
    residual = normalized_values - design @ centered_beta
    scaled_sse = float(np.sum(weights * residual**2))
    scaled_value_ss = float(np.sum(weights * normalized_values**2))
    if scaled_sse <= _RELATIVE_ROUNDOFF_TOL**2 * scaled_value_ss:
        residual = np.zeros_like(residual)
    meat, factor, _ = _score_meat(
        design, weights, residual, residual, prepared.clusters
    )
    cov = bread @ meat @ bread * factor
    return _Fit(
        beta=beta,
        residual=residual,
        bread=bread,
        cov=cov,
        rank=rank,
        response_scale=response_scale,
    )


def _cross_cov(prepared: _Prepared, fit_a: _Fit, fit_b: _Fit) -> np.ndarray:
    meat, factor, _ = _score_meat(
        prepared.design,
        prepared.weights,
        fit_a.residual,
        fit_b.residual,
        prepared.clusters,
    )
    return fit_a.bread @ meat @ fit_b.bread * factor


def _contrast(fit: _Fit, vector: np.ndarray) -> tuple[float, float, float]:
    value = float(vector @ fit.beta)
    vector_scale = float(np.max(np.abs(vector), initial=0.0))
    if vector_scale == 0.0:
        return value, 0.0, 0.0
    normalized_vector = vector / vector_scale
    normalized_variance = float(normalized_vector @ fit.cov @ normalized_vector)
    normalized_variance = max(normalized_variance, 0.0)
    normalized_se = float(np.sqrt(normalized_variance) * vector_scale)
    se = float(normalized_se * fit.response_scale)
    return value, se, normalized_se


def _contrast_correlation(
    fit_a: _Fit,
    fit_b: _Fit,
    cross_cov: np.ndarray,
    vector: np.ndarray,
) -> float:
    vector_scale = float(np.max(np.abs(vector), initial=0.0))
    if vector_scale == 0.0:
        return 0.0
    normalized_vector = vector / vector_scale
    var_a = max(float(normalized_vector @ fit_a.cov @ normalized_vector), 0.0)
    var_b = max(float(normalized_vector @ fit_b.cov @ normalized_vector), 0.0)
    if var_a == 0.0 or var_b == 0.0:
        return 0.0
    covariance = float(normalized_vector @ cross_cov @ normalized_vector)
    correlation = covariance / (np.sqrt(var_a) * np.sqrt(var_b))
    return float(np.clip(correlation, -1.0, 1.0))


def _canonical_zero(value: float, slopes: dict[str, float]) -> float:
    scale = max((abs(v) for v in slopes.values()), default=0.0)
    if value == 0.0 or (scale > 0.0 and abs(value) <= _RELATIVE_ROUNDOFF_TOL * scale):
        return 0.0
    return value


def _cell_slopes(
    fit: _Fit, prepared: _Prepared, degree: int, bandwidth: float
) -> tuple[dict[str, float], dict[str, float]]:
    block_size = degree + 1
    slopes: dict[str, float] = {}
    ses: dict[str, float] = {}
    for j, label in enumerate(prepared.cell_labels):
        vector = np.zeros(prepared.design.shape[1], dtype=float)
        vector[j * block_size + 1] = 1.0 / bandwidth
        value, se, _ = _contrast(fit, vector)
        slopes[label] = value
        ses[label] = se
    return slopes, ses


def _fieller(
    numerator: float,
    denominator: float,
    se_numerator: float,
    se_denominator: float,
    correlation: float,
    critical_value: float,
) -> tuple[str, tuple[float, float] | None, dict]:
    critical_squared = critical_value**2
    numerator_scale = max(abs(numerator), se_numerator)
    denominator_scale = max(abs(denominator), se_denominator)
    roundoff = _RELATIVE_ROUNDOFF_TOL

    if numerator_scale == 0.0 and denominator_scale == 0.0:
        return "unbounded", None, {}
    if numerator_scale == 0.0:
        normalized_denominator = denominator / denominator_scale
        normalized_var_denominator = (se_denominator / denominator_scale) ** 2
        a_left = normalized_denominator**2
        a_right = critical_squared * normalized_var_denominator
        a = a_left - a_right
        a_tol = roundoff * (abs(a_left) + abs(a_right))
        if a <= a_tol:
            return "unbounded", None, {}
        return "interval", (0.0, 0.0), {}
    if denominator_scale == 0.0:
        normalized_numerator = numerator / numerator_scale
        normalized_var_numerator = (se_numerator / numerator_scale) ** 2
        c_left = normalized_numerator**2
        c_right = critical_squared * normalized_var_numerator
        c = c_left - c_right
        c_tol = roundoff * (abs(c_left) + abs(c_right))
        return ("unbounded", None, {}) if c <= c_tol else ("empty", None, {})

    normalized_numerator = numerator / numerator_scale
    normalized_denominator = denominator / denominator_scale
    normalized_se_numerator = se_numerator / numerator_scale
    normalized_se_denominator = se_denominator / denominator_scale
    normalized_var_numerator = normalized_se_numerator**2
    normalized_var_denominator = normalized_se_denominator**2
    normalized_covariance = (
        correlation * normalized_se_numerator * normalized_se_denominator
    )
    ratio_scale = numerator_scale / denominator_scale

    a_left = normalized_denominator**2
    a_right = critical_squared * normalized_var_denominator
    a = a_left - a_right
    a_tol = roundoff * (abs(a_left) + abs(a_right))
    b_left = -2.0 * normalized_numerator * normalized_denominator
    b_right = 2.0 * critical_squared * normalized_covariance
    b = b_left + b_right
    b_tol = roundoff * (abs(b_left) + abs(b_right))
    c_left = normalized_numerator**2
    c_right = critical_squared * normalized_var_numerator
    c = c_left - c_right
    c_tol = roundoff * (abs(c_left) + abs(c_right))

    if abs(a) <= a_tol:
        if abs(b) <= b_tol:
            return ("unbounded", None, {}) if c <= c_tol else ("empty", None, {})
        endpoint = float((-c / b) * ratio_scale)
        ray = ((float("-inf"), endpoint),) if b > 0 else ((endpoint, float("inf")),)
        return "unbounded", None, {"fieller_rays": ray}
    discriminant = b * b - 4.0 * a * c
    disc_scale = abs(b * b) + abs(4.0 * a * c)
    if abs(discriminant) <= roundoff * disc_scale:
        discriminant = 0.0
    if discriminant < 0.0:
        return ("empty", None, {}) if a > 0.0 else ("unbounded", None, {})
    if discriminant == 0.0:
        root = float((-b / (2.0 * a)) * ratio_scale)
        if a > 0.0:
            return "interval", (root, root), {}
        return "unbounded", None, {}
    root = float(np.sqrt(discriminant))
    q = -0.5 * (b + np.copysign(root, b))
    roots = (q / a, c / q) if q != 0.0 else (
        (-b - root) / (2.0 * a),
        (-b + root) / (2.0 * a),
    )
    r1, r2 = sorted(float(value * ratio_scale) for value in roots)
    if a > 0.0:
        return "interval", (r1, r2), {}
    rays = ((float("-inf"), r1), (r2, float("inf")))
    return "disjoint", None, {"fieller_rays": rays}


def _nan_estimate(
    method: str,
    prepared: _Prepared,
    reason: str,
    base_extras: dict,
) -> KinkEstimate:
    nan = float("nan")
    extras = {**base_extras, "reason": reason}
    return KinkEstimate(
        tau=nan,
        se=nan,
        ci=(nan, nan),
        method=method,
        reduced_form=nan,
        reduced_form_se=nan,
        first_stage=nan,
        first_stage_se=nan,
        first_stage_F=nan,
        weak_first_stage=True,
        n_used=prepared.y.size,
        n_by_cell=prepared.n_by_cell,
        extras=extras,
    )


def _estimate(
    y,
    running,
    *,
    treatment,
    policy_kink: float | None,
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
    sharp = policy_kink is not None
    design_name = "dik" if post is not None else "rkd"
    method = f"{'sharp' if sharp else 'fuzzy'}_{design_name}"
    prepared = _prepare(
        y,
        running,
        treatment=treatment,
        post=post,
        cutoff=cutoff,
        bandwidth=bandwidth,
        degree=degree,
        kernel=kernel,
        donut=donut,
        covariates=covariates,
        clusters=clusters,
    )
    n, p = prepared.design.shape
    n_clusters = None
    if prepared.clusters is not None:
        n_clusters = int(prepared.clusters.max(initial=-1) + 1)
    critical_df = (
        n_clusters - 1
        if prepared.clusters is not None and n_clusters is not None and n_clusters >= 2
        else None
    )
    critical_value = float(
        stats.t.isf(alpha / 2.0, critical_df)
        if critical_df is not None
        else stats.norm.isf(alpha / 2.0)
    )
    extras = {
        "cutoff": cutoff,
        "bandwidth": bandwidth,
        "degree": degree,
        "kernel": kernel,
        "donut": donut,
        "contrast": "right_minus_left",
        "inference": "CR1" if prepared.clusters is not None else "HC1",
        "n_clusters": n_clusters,
        "critical_df": critical_df,
        "critical_value": critical_value,
        "n_input": prepared.n_input,
        "n_dropped_nonfinite": prepared.n_dropped_nonfinite,
        "n_outside_bandwidth": prepared.n_outside_bandwidth,
        "n_donut_excluded": prepared.n_donut_excluded,
        "n_zero_weight_excluded": prepared.n_zero_weight_excluded,
        "n_covariates": prepared.n_covariates,
        "n_covariates_dropped_constant": prepared.n_covariates_dropped_constant,
    }
    under = [label for label, count in prepared.n_by_cell.items() if count < degree + 1]
    if under:
        return _nan_estimate(
            method,
            prepared,
            f"underdetermined cells for degree {degree}: {under}",
            extras,
        )
    saturated = [
        label for label, count in prepared.n_by_cell.items() if count == degree + 1
    ]
    if saturated:
        return _nan_estimate(
            method,
            prepared,
            f"insufficient residual degrees of freedom in cells: {saturated}",
            extras,
        )
    if n <= p:
        return _nan_estimate(
            method,
            prepared,
            f"underdetermined local-polynomial design: {n} rows for {p} parameters",
            extras,
        )
    if prepared.clusters is not None and (n_clusters is None or n_clusters < 2):
        return _nan_estimate(
            method, prepared, "cluster-robust inference needs at least two clusters", extras
        )
    if prepared.clusters is not None:
        clusters_by_cell = {
            label: int(np.unique(prepared.clusters[prepared.cell == j]).size)
            for j, label in enumerate(prepared.cell_labels)
        }
        extras["n_clusters_by_cell"] = clusters_by_cell
        unsupported = [
            label for label, count in clusters_by_cell.items() if count < 2
        ]
        if unsupported:
            return _nan_estimate(
                method,
                prepared,
                f"cluster-robust inference needs at least two clusters in every cell: {unsupported}",
                extras,
            )

    try:
        outcome_fit = _fit(prepared, prepared.y)
        outcome_slopes, outcome_slope_se = _cell_slopes(
            outcome_fit, prepared, degree, bandwidth
        )
        reduced_form, reduced_form_se, _ = _contrast(
            outcome_fit, prepared.contrast
        )
        extras["outcome_slopes"] = outcome_slopes
        extras["outcome_slope_se"] = outcome_slope_se
        extras["rank"] = outcome_fit.rank

        if sharp:
            first_stage = float(policy_kink)
            first_stage_se = 0.0
            first_stage_f = float("inf")
            weak = False
            covariance = 0.0
            correlation = 0.0
            treatment_fit = None
        else:
            assert prepared.treatment is not None
            treatment_fit = _fit(prepared, prepared.treatment)
            first_stage_slopes, first_stage_slope_se = _cell_slopes(
                treatment_fit, prepared, degree, bandwidth
            )
            first_stage, first_stage_se, _ = _contrast(
                treatment_fit, prepared.contrast
            )
            first_stage = _canonical_zero(first_stage, first_stage_slopes)
            cross = _cross_cov(prepared, outcome_fit, treatment_fit)
            correlation = _contrast_correlation(
                outcome_fit, treatment_fit, cross, prepared.contrast
            )
            with np.errstate(over="ignore", under="ignore", invalid="ignore"):
                covariance = float(
                    np.multiply(
                        correlation,
                        np.multiply(reduced_form_se, first_stage_se),
                    )
                )
            if first_stage_se == 0.0:
                first_stage_f = float("inf") if first_stage != 0.0 else float("nan")
            else:
                with np.errstate(over="ignore"):
                    first_stage_f = float(
                        np.square(np.divide(first_stage, first_stage_se))
                    )
            weak = not bool(first_stage_f >= _WEAK_F_THRESHOLD)
            extras["first_stage_slopes"] = first_stage_slopes
            extras["first_stage_slope_se"] = first_stage_slope_se
            if post is not None:
                extras["first_stage_kinks"] = {
                    "pre": first_stage_slopes["pre_right"]
                    - first_stage_slopes["pre_left"],
                    "post": first_stage_slopes["post_right"]
                    - first_stage_slopes["post_left"],
                }
        if post is not None:
            extras["outcome_kinks"] = {
                "pre": outcome_slopes["pre_right"] - outcome_slopes["pre_left"],
                "post": outcome_slopes["post_right"] - outcome_slopes["post_left"],
            }
    except (np.linalg.LinAlgError, ValueError) as exc:
        return _nan_estimate(method, prepared, str(exc), extras)

    fieller_kind = None
    fieller_ci = None
    if first_stage == 0.0 or not np.isfinite(first_stage):
        tau = se = float("nan")
        ci = (float("nan"), float("nan"))
        extras["reason"] = "the first-stage slope contrast is zero or non-finite"
    else:
        with np.errstate(over="ignore", invalid="ignore"):
            tau = float(np.divide(reduced_form, first_stage))
        combined_se = reduced_form_se
        if not sharp and np.isfinite(tau):
            assert prepared.treatment is not None
            try:
                combined_fit = _fit(
                    prepared, prepared.y - tau * prepared.treatment
                )
                _, combined_se, _ = _contrast(combined_fit, prepared.contrast)
            except (np.linalg.LinAlgError, ValueError) as exc:
                return _nan_estimate(method, prepared, str(exc), extras)
        if not np.isfinite(tau):
            tau = se = float("nan")
            ci = (float("nan"), float("nan"))
            extras["reason"] = "the slope ratio is non-finite in floating-point units"
        else:
            with np.errstate(over="ignore"):
                se = float(np.divide(combined_se, abs(first_stage)))
            ci = (tau - critical_value * se, tau + critical_value * se)
    if not sharp:
        fieller_kind, fieller_ci, fieller_extras = _fieller(
            reduced_form,
            first_stage,
            reduced_form_se,
            first_stage_se,
            correlation,
            critical_value,
        )
        extras.update(fieller_extras)
    extras["outcome_first_stage_covariance"] = covariance
    extras["outcome_first_stage_correlation"] = correlation
    extras["smoothing_bias_caveat"] = (
        "conventional local-polynomial interval; inspect bandwidth and polynomial sensitivity"
    )
    return KinkEstimate(
        tau=float(tau),
        se=float(se),
        ci=(float(ci[0]), float(ci[1])),
        method=method,
        reduced_form=float(reduced_form),
        reduced_form_se=float(reduced_form_se),
        first_stage=float(first_stage),
        first_stage_se=float(first_stage_se),
        first_stage_F=float(first_stage_f),
        weak_first_stage=weak,
        n_used=n,
        n_by_cell=prepared.n_by_cell,
        fieller_ci=fieller_ci,
        fieller_kind=fieller_kind,
        extras=extras,
    )


def regression_kink(
    y,
    running,
    *,
    treatment=None,
    policy_kink: float | None = None,
    cutoff: float = 0.0,
    bandwidth: float,
    degree: int = 1,
    kernel: str = "triangular",
    donut: float = 0.0,
    covariates=None,
    clusters=None,
    alpha: float = 0.05,
) -> KinkEstimate:
    """Estimate a sharp or fuzzy regression kink at a known cutoff.

    Supply exactly one of ``policy_kink`` (known right-minus-left policy
    slope change; sharp RKD) or ``treatment`` (observed stochastic policy
    variable; fuzzy RKD).  ``bandwidth`` is required because natex does not
    claim an automatic derivative-optimal selector.
    """
    _validate_common(cutoff, bandwidth, degree, kernel, donut, alpha)
    _validate_first_stage(treatment, policy_kink, "policy_kink")
    return _estimate(
        y,
        running,
        treatment=treatment,
        policy_kink=policy_kink,
        post=None,
        cutoff=cutoff,
        bandwidth=bandwidth,
        degree=degree,
        kernel=kernel,
        donut=donut,
        covariates=covariates,
        clusters=clusters,
        alpha=alpha,
    )


def difference_in_kinks(
    y,
    running,
    post,
    *,
    treatment=None,
    policy_kink_change: float | None = None,
    cutoff: float = 0.0,
    bandwidth: float,
    degree: int = 1,
    kernel: str = "triangular",
    donut: float = 0.0,
    covariates=None,
    clusters=None,
    alpha: float = 0.05,
) -> KinkEstimate:
    """Estimate a sharp or fuzzy difference-in-kinks design.

    The numerator is ``(right-left slope kink)_post - (... )_pre``.
    Supply the corresponding known ``policy_kink_change`` for a sharp DiK,
    or the observed policy variable as ``treatment`` for fuzzy DiK.

    A fuzzy DiK has the paper's positive-weight causal interpretation only
    when latent policy-schedule composition is stable (or validly reweighted)
    across periods at the cutoff and individual kink changes share one sign.
    These identifying assumptions are not testable by this function.
    """
    _validate_common(cutoff, bandwidth, degree, kernel, donut, alpha)
    _validate_first_stage(treatment, policy_kink_change, "policy_kink_change")
    return _estimate(
        y,
        running,
        treatment=treatment,
        policy_kink=policy_kink_change,
        post=post,
        cutoff=cutoff,
        bandwidth=bandwidth,
        degree=degree,
        kernel=kernel,
        donut=donut,
        covariates=covariates,
        clusters=clusters,
        alpha=alpha,
    )
