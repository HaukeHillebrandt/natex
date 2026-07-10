"""Fitted-null Monte Carlo calibration of the max-LLR scan statistic.

This is a parametric bootstrap against the fitted null model, NOT an exact
randomization test (audit item 1): the observed data are not exchangeable with
replicas drawn from a model fitted to them. p-values use the +1 rank rule.
Bernoulli replicas are direct Bernoulli(p_hat) draws (audit item 2).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from natex.data.spec import Dataset
from natex.rdd.lord3 import LoRD3Result, fit_treatment_model, lord3_scan
from natex.scan.geometry import ScanGeometry, build_geometry
from natex.scan.neighborhoods import local_residual_variance


@dataclass
class RandomizationReport:
    p_value: float
    observed_max_llr: float
    null_max_llrs: np.ndarray
    q: int


def _draw_null_treatment(kind, p_or_mu, sigma2, rng):
    if kind == "bernoulli":
        return rng.binomial(1, p_or_mu).astype(float)
    return p_or_mu + np.sqrt(sigma2) * rng.standard_normal(p_or_mu.size)


def randomization_test(
    dataset: Dataset,
    scan_result: LoRD3Result,
    Q: int = 99,
    rng: np.random.Generator | None = None,
    scan_kwargs: dict | None = None,
    geometry: ScanGeometry | None = None,
    centers: np.ndarray | None = None,
) -> RandomizationReport:
    if rng is None:
        raise ValueError("pass an explicit numpy Generator")
    scan_kwargs = dict(scan_kwargs or {})
    scan_kwargs.setdefault("k", scan_result.k)
    kind = scan_result.model
    X, T, Z = dataset.X, dataset.T, dataset.Z_std
    # Geometry depends only on Z_std, which is identical across all replicas:
    # build once, reuse everywhere. Replica draw order is unchanged, so
    # p-values are bit-identical with or without the cache.
    if geometry is None:
        geometry = build_geometry(Z, scan_kwargs["k"])
    predict, _ = fit_treatment_model(X, T, kind, scan_kwargs.get("degree", 1))
    fitted = predict(X)
    sigma2 = None
    if kind == "normal":
        sigma2 = local_residual_variance(T - fitted, geometry.idx)
    else:
        fitted = np.clip(fitted, 1e-6, 1 - 1e-6)

    observed = scan_result.discoveries[0].llr
    null_max = np.empty(Q)
    for q_i in range(Q):
        t_star = _draw_null_treatment(kind, fitted, sigma2, rng)
        df_star = dataset.df.copy()
        df_star[dataset.spec.treatment] = t_star
        ds_star = Dataset(df_star, dataset.spec)
        res_star = lord3_scan(
            ds_star, model=kind, rng=rng, geometry=geometry, centers=centers, **scan_kwargs
        )
        null_max[q_i] = res_star.discoveries[0].llr if res_star.discoveries else 0.0

    p = (1.0 + float(np.sum(null_max >= observed))) / (Q + 1.0)
    return RandomizationReport(p_value=p, observed_max_llr=observed, null_max_llrs=null_max, q=Q)
