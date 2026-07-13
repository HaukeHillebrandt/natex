"""DEE orchestrator: end-to-end debiasing of an observational CATE estimator.

Pipeline (each stage consumes the PREVIOUS stage's object -- the VKNN index
sets are computed once and passed by object, never recomputed downstream,
fixing the paper repo's silent step-4 row-mismatch truncation, repo risk 4):

    voronoi_knn_repair -> [balance_filter] -> experiment_effects (frozen-side
    2SLS, HC1) -> drop non-finite tau/se into diagnostics ->
    experiment_crossfit_cate (audit 9) -> bias_obs = obs - tau ->
    stage-1 noise (chi-square measurement model, dee/noise.py) ->
    HeteroskedasticGP bias + direct surfaces -> model weights -> query
    predictions and the audit-8 mixture posterior.

Conventions that bind here (docs/math_audit_final.md + phase-4 plan):

- **Sign convention (pinned, Codex #30):** ``bias_u = cate_obs(center_u) -
  tau_hat_u``; ``cate_debiased(x) = cate_raw(x) - bias_gp_mean(x)``.
- All GP/estimator geometry lives in Z_std space; ``query`` arrives in RAW
  forcing units and is standardized once via ``Dataset.standardize`` (bitwise
  consistent with ``Z_std``).
- Model A = "debias" (obs - bias-GP; its posterior is the bias posterior
  shifted by ``cate_raw`` -- the observational estimator's own uncertainty is
  not modeled, matching the paper; documented). Model B = "direct"
  (CATE-extrapolation GP). ``ModelWeights.w_debias`` weighs model A.
- One caller-supplied ``numpy.random.Generator`` drives every stochastic call
  (crossfit folds + estimator seeds, noise/GP fit restarts, stacking folds);
  identical seed => identical output.
- NaN policy: experiments with non-finite 2SLS tau/se are excluded from both
  GPs and listed in ``diagnostics["dropped"]``; usable experiments whose
  cross-fitted obs CATE is non-finite are additionally excluded from the BIAS
  side only (issue #23; dropped reason "non-finite cross-fitted obs CATE",
  count in ``diagnostics["n_experiments_used_bias"]``). Fewer than 3 usable
  experiments => every GP-derived field is None/NaN (never 0.0) with
  ``diagnostics["reason"]``; fewer than 3 bias-usable experiments => the bias
  side (``gp_bias``, ``w_debias``, ``mixture``, ``cate_debiased``) degenerates
  the same way while ``gp_direct``/``cate_direct`` are still produced. Weak
  instruments are kept, not dropped (audit 10): their large SE^2 downweights
  them in the heteroskedastic GP.
- A NaN stacking weight falls back to ``loo_weights`` (documented contract in
  ``dee/bma.py``), with the stacking detail preserved under
  ``detail["stacking_fallback"]``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from natex.data.spec import Dataset
from natex.dee.bma import (
    MixturePosterior,
    ModelWeights,
    buffered_stacking_weights,
    loo_weights,
    mixture_posterior,
    mll_weights,
)
from natex.dee.gp import GPPosterior, HeteroskedasticGP, _require_rng
from natex.dee.noise import smooth_noise
from natex.dee.observational import (
    ObservationalEstimator,
    default_factory,
    experiment_crossfit_cate,
)
from natex.dee.vknn import (
    VKNNResult,
    balance_filter,
    experiment_effects,
    experiment_radius,
    voronoi_knn_repair,
)
from natex.estimate.local2sls import EffectEstimate
from natex.rdd.lord3 import Discovery, LoRD3Result

_WEIGHTINGS = ("stacking", "loo", "mll")


@dataclass
class DEEResult:
    """Everything ``dee_debias`` produced; all per-experiment arrays are aligned
    with ``vknn.experiments`` (acceptance order, post balance filter)."""

    vknn: VKNNResult
    effects: list[EffectEstimate]  # aligned with vknn.experiments
    used: np.ndarray  # bool over experiments: finite tau & se, post-filter
    obs_at_centers: np.ndarray  # cross-fitted obs-CATE at projected centers (audit 9)
    bias_obs: np.ndarray  # obs_at_centers - tau_hat (pinned sign convention)
    noise_var: np.ndarray  # smoothed (stage-1) noise variances actually used
    gp_bias: HeteroskedasticGP | None  # fit on used & finite(obs_at_centers) rows (issue #23)
    gp_direct: HeteroskedasticGP | None  # fit on all used rows
    weights: ModelWeights
    query: np.ndarray  # raw-unit query points as given
    cate_raw: np.ndarray  # full-fit obs estimator at query
    cate_debiased: np.ndarray  # cate_raw - gp_bias posterior mean at query
    cate_direct: np.ndarray  # gp_direct posterior mean at query
    mixture: MixturePosterior | None  # over query points
    diagnostics: dict[str, Any]


def _model_weights(
    weighting: str,
    centers_used: np.ndarray,
    tau_used: np.ndarray,
    obs_used: np.ndarray,
    nv_used: np.ndarray,
    gp_bias: HeteroskedasticGP,
    gp_direct: HeteroskedasticGP,
    n_folds: int,
    rng: np.random.Generator,
) -> ModelWeights:
    if weighting == "mll":
        return mll_weights(gp_bias, gp_direct)
    if weighting == "loo":
        return loo_weights(gp_bias, gp_direct)
    weights = buffered_stacking_weights(
        centers_used, tau_used, obs_used, nv_used, rng, n_folds=n_folds
    )
    if not np.isfinite(weights.w_debias):
        fallback = loo_weights(gp_bias, gp_direct)
        fallback.detail["stacking_fallback"] = weights.detail
        return fallback
    return weights


def dee_debias(
    dataset: Dataset,
    query: np.ndarray,
    discoveries: LoRD3Result | Sequence[Discovery],
    *,
    m_prime: int,
    k_prime: int = 200,
    t_side: int = 30,
    factory: Callable[[], ObservationalEstimator] | None = None,
    weighting: str = "stacking",
    smooth_noise_stage1: bool = True,
    balance_alpha: float | None = 0.05,
    n_folds: int = 5,
    rng: np.random.Generator | None = None,
) -> DEEResult:
    """Run the full DEE debiasing pipeline; see the module docstring.

    Parameters
    ----------
    query : (m, d) query points in RAW forcing units (standardized internally).
    discoveries : a ``LoRD3Result`` or any sequence of ``Discovery`` objects.
    m_prime : number of top-LLR candidates offered to the VKNN repair
        (``select_m_prime`` supplies it from the randomization-test null).
    factory : zero-arg ``ObservationalEstimator`` factory; ``None`` selects the
        core-deps T-learner via ``default_factory(rng)``.
    weighting : "stacking" (default; buffered predictive stacking) | "loo" | "mll".
    smooth_noise_stage1 : when True (default), SE^2 pass through the chi-square
        measurement-model GP (``smooth_noise``, df = n_used - 4); when False,
        raw HC1 SE^2 are used directly.
    balance_alpha : placebo-battery level for the balance filter; ``None``
        disables the filter entirely.
    n_folds : folds for BOTH the leave-experiment-out cross-fit (audit 9) and
        the buffered stacking.
    rng : REQUIRED explicit numpy Generator (reproducibility contract).
    """
    rng = _require_rng(rng)
    if weighting not in _WEIGHTINGS:
        raise ValueError(f"weighting must be one of {list(_WEIGHTINGS)}, got {weighting!r}")
    if dataset.y is None:
        raise ValueError("dee_debias needs a dataset with an outcome column")
    query = np.asarray(query, dtype=float)
    query_std = dataset.standardize(query)  # validates (m, d); ValueError on mismatch
    if factory is None:
        factory = default_factory(rng)

    # --- discovery-side stages (never read y) --------------------------------
    if isinstance(discoveries, LoRD3Result):
        discs: Sequence[Discovery] = discoveries.discoveries
    else:
        discs = list(discoveries)
    vknn = voronoi_knn_repair(dataset, discs, m_prime, k_prime=k_prime, t_side=t_side)
    if balance_alpha is not None:
        vknn = balance_filter(dataset, vknn, alpha=balance_alpha)
    u = len(vknn.experiments)

    # --- local effects + usability mask ---------------------------------------
    effects = experiment_effects(dataset, vknn, method="2sls")
    tau = np.array([e.tau for e in effects], dtype=float)
    se = np.array([e.se for e in effects], dtype=float)
    used = np.isfinite(tau) & np.isfinite(se)
    dropped = [
        {
            "experiment": int(i),
            "center_index": int(vknn.experiments[i].center_index),
            "reason": "non-finite 2SLS tau/se",
        }
        for i in np.flatnonzero(~used)
    ]

    # --- cross-fitted bias observations (audit 9) -----------------------------
    obs_at_centers = experiment_crossfit_cate(dataset, vknn, factory, rng, n_folds=n_folds)
    bias_obs = obs_at_centers - tau  # pinned sign: bias = obs - tau
    # Issue #23: the bias surface also needs a finite cross-fitted obs CATE at
    # the center; tau/se-usable experiments failing that are dropped from the
    # BIAS side only (the healthy direct model keeps them).
    used_bias = used & np.isfinite(obs_at_centers)
    dropped.extend(
        {
            "experiment": int(i),
            "center_index": int(vknn.experiments[i].center_index),
            "reason": "non-finite cross-fitted obs CATE",
        }
        for i in np.flatnonzero(used & ~used_bias)
    )

    # --- stage-1 noise ---------------------------------------------------------
    if u:
        centers = np.stack(
            [np.asarray(e.projected_center, dtype=float) for e in vknn.experiments]
        )
    else:
        centers = np.empty((0, dataset.Z_std.shape[1]))
    se2 = se**2
    noise_var = np.full(u, np.nan)
    if used.any():
        if smooth_noise_stage1:
            df = np.array([float(e.n_used - 4) for e in effects])
            noise_var[used] = smooth_noise(centers[used], se2[used], df[used], rng=rng)
        else:
            noise_var[used] = se2[used]

    n_used = int(used.sum())
    n_used_bias = int(used_bias.sum())
    diagnostics: dict[str, Any] = {
        "m_prime": int(m_prime),
        "dropped": dropped,
        "radii": np.array([experiment_radius(dataset, e) for e in vknn.experiments]),
        "n_experiments": u,
        "n_experiments_used": n_used,
        "n_experiments_used_bias": n_used_bias,
        "buffer": None,
        "fold_sizes": None,
    }
    m = query_std.shape[0]

    if n_used < 3:
        # Degenerate: no surface to fit. Every GP-derived field is None/NaN --
        # never 0.0 -- and cate_raw is skipped too (nothing to debias against).
        diagnostics["reason"] = (
            f"only {n_used} usable experiments (< 3); GP-derived outputs are NaN"
        )
        return DEEResult(
            vknn=vknn,
            effects=effects,
            used=used,
            obs_at_centers=obs_at_centers,
            bias_obs=bias_obs,
            noise_var=noise_var,
            gp_bias=None,
            gp_direct=None,
            weights=ModelWeights(
                w_debias=float("nan"),
                strategy=weighting,
                detail={"reason": diagnostics["reason"]},
            ),
            query=query,
            cate_raw=np.full(m, np.nan),
            cate_debiased=np.full(m, np.nan),
            cate_direct=np.full(m, np.nan),
            mixture=None,
            diagnostics=diagnostics,
        )

    if n_used_bias < 3:
        # Bias-side degenerate (issue #23): the debias model has no surface to
        # fit -- gp_bias/weights/mixture/cate_debiased are None/NaN, never
        # 0.0 -- but the direct model is healthy, so fit and keep it.
        diagnostics["reason"] = (
            f"only {n_used_bias} experiments with a finite cross-fitted obs CATE (< 3); "
            "bias-side outputs are NaN, direct-side outputs are kept"
        )
        gp_direct = HeteroskedasticGP.fit(centers[used], tau[used], noise_var[used], rng=rng)
        full_model = factory().fit(dataset.Z_std, dataset.T, dataset.y)
        cate_raw = np.asarray(full_model.predict_cate(query_std), dtype=float)
        return DEEResult(
            vknn=vknn,
            effects=effects,
            used=used,
            obs_at_centers=obs_at_centers,
            bias_obs=bias_obs,
            noise_var=noise_var,
            gp_bias=None,
            gp_direct=gp_direct,
            weights=ModelWeights(
                w_debias=float("nan"),
                strategy=weighting,
                detail={"reason": diagnostics["reason"]},
            ),
            query=query,
            cate_raw=cate_raw,
            cate_debiased=np.full(m, np.nan),
            cate_direct=np.asarray(gp_direct.posterior(query_std).mean, dtype=float),
            mixture=None,
            diagnostics=diagnostics,
        )

    # --- bias / direct surfaces ------------------------------------------------
    # gp_bias fits the bias-usable rows EXPLICITLY (issue #23) so its
    # fit_report matches diagnostics; gp_direct keeps every usable experiment.
    gp_bias = HeteroskedasticGP.fit(
        centers[used_bias], bias_obs[used_bias], noise_var[used_bias], rng=rng
    )
    gp_direct = HeteroskedasticGP.fit(centers[used], tau[used], noise_var[used], rng=rng)

    # --- model weights -----------------------------------------------------------
    # Stacking scores model A's held-out predictive, which needs a finite obs
    # CATE -- so the weighting sees the bias-usable rows (identical to the
    # previous internal drop, but with consistent detail/fold indices).
    weights = _model_weights(
        weighting,
        centers[used_bias],
        tau[used_bias],
        obs_at_centers[used_bias],
        noise_var[used_bias],
        gp_bias,
        gp_direct,
        n_folds,
        rng,
    )
    if weights.strategy == "stacking":
        diagnostics["buffer"] = weights.detail.get("buffer")
        diagnostics["fold_sizes"] = [
            {"fold": f["fold"], "n_train": f["n_train"], "n_test": len(f["test_idx"])}
            for f in weights.detail.get("folds", [])
        ]

    # --- query predictions -------------------------------------------------------
    full_model = factory().fit(dataset.Z_std, dataset.T, dataset.y)
    cate_raw = np.asarray(full_model.predict_cate(query_std), dtype=float)
    post_bias = gp_bias.posterior(query_std)
    post_direct = gp_direct.posterior(query_std)
    cate_debiased = cate_raw - post_bias.mean
    # Model A posterior: the bias posterior shifted by cate_raw (the estimator's
    # own uncertainty is not modeled -- paper behavior, documented above).
    post_a = GPPosterior(mean=cate_debiased.copy(), cov=post_bias.cov)
    mixture = mixture_posterior(post_a, post_direct, weights.w_debias)

    return DEEResult(
        vknn=vknn,
        effects=effects,
        used=used,
        obs_at_centers=obs_at_centers,
        bias_obs=bias_obs,
        noise_var=noise_var,
        gp_bias=gp_bias,
        gp_direct=gp_direct,
        weights=weights,
        query=query,
        cate_raw=cate_raw,
        cate_debiased=cate_debiased,
        cate_direct=np.asarray(post_direct.mean, dtype=float),
        mixture=mixture,
        diagnostics=diagnostics,
    )
