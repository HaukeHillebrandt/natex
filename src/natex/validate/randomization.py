"""Fitted-null Monte Carlo calibration of the max-LLR scan statistic.

This is a parametric bootstrap against the fitted null model, NOT an exact
randomization test (audit item 1): the observed data are not exchangeable with
replicas drawn from a model fitted to them. p-values use the +1 rank rule.
Bernoulli replicas are direct Bernoulli(p_hat) draws (audit item 2).
"""

from __future__ import annotations

from collections.abc import Callable
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
    search: Callable[[Dataset], LoRD3Result] | None = None,
) -> RandomizationReport:
    """Calibrate ``scan_result``'s max LLR against Q fitted-null replicas.

    ``search`` (issue #21): when the observed statistic came from a
    treatment-adaptive search procedure (e.g. the coarse-to-fine scan, whose
    fine-stage center subset is localized around the observed treatment's own
    coarse discoveries), pass a callable mapping a replica dataset to its
    :class:`LoRD3Result` under the SAME procedure — see
    :func:`natex.scan.coarse.coarse_to_fine_search`. The default ``None``
    rescans every replica at full resolution over ``centers``, which is
    procedure-matched only for a full (or fixed-center) observed scan; using
    it for a coarse-to-fine observed statistic gives stochastically larger
    replica maxima and an inflated p-value.

    ``scan_kwargs["model"]`` is accepted for callers that share scan
    configuration with the observed scan, but it must equal
    ``scan_result.model``. The replica model is always taken from the observed
    result so that it cannot be supplied twice to :func:`lord3_scan`.
    """
    if rng is None:
        raise ValueError("pass an explicit numpy Generator")
    # Issue #25: mirror the panel contract (validate/panel.py) verbatim —
    # Q=0 fabricated a vacuous p=1.0 from zero draws, Q=-1 crashed in
    # np.empty, and an empty scan result raised a raw IndexError.
    if Q < 1:
        raise ValueError(f"Q must be >= 1, got {Q}")
    if not scan_result.discoveries:
        raise ValueError("scan_result has no discoveries: nothing to calibrate")
    observed = scan_result.discoveries[0].llr
    if not np.isfinite(observed):
        # NaN >= NaN is False, so a non-finite statistic would silently score
        # the minimum attainable p = 1/(Q+1) (issue #9) — reject before doing
        # Q replica scans' worth of work.
        raise ValueError(
            f"non-finite observed max LLR ({observed}): the scan statistic is "
            "degenerate and cannot be ranked"
        )
    scan_kwargs = dict(scan_kwargs or {})
    kind = scan_result.model
    configured_model = scan_kwargs.pop("model", kind)
    if configured_model != kind:
        raise ValueError(
            f"scan_kwargs model ({configured_model!r}) does not match "
            f"scan_result.model ({kind!r})"
        )
    scan_kwargs.setdefault("k", scan_result.k)
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

    null_max = np.empty(Q)
    for q_i in range(Q):
        t_star = _draw_null_treatment(kind, fitted, sigma2, rng)
        df_star = dataset.df.copy()
        df_star[dataset.spec.treatment] = t_star
        ds_star = Dataset(df_star, dataset.spec)
        if search is None:
            res_star = lord3_scan(
                ds_star, model=kind, rng=rng, geometry=geometry, centers=centers, **scan_kwargs
            )
        else:
            res_star = search(ds_star)
        null_max[q_i] = res_star.discoveries[0].llr if res_star.discoveries else 0.0

    if not (np.isfinite(observed) and np.isfinite(null_max).all()):
        # Defense in depth (issue #9): NaN >= NaN is False, so a non-finite
        # statistic would silently score the minimum attainable p = 1/(Q+1).
        raise ValueError(
            f"non-finite max LLR in randomization test (observed={observed}, "
            f"non-finite replicas={int(np.sum(~np.isfinite(null_max)))} of {Q}); "
            "the scan statistic is degenerate and cannot be ranked"
        )
    p = (1.0 + float(np.sum(null_max >= observed))) / (Q + 1.0)
    return RandomizationReport(p_value=p, observed_max_llr=observed, null_max_llrs=null_max, q=Q)
