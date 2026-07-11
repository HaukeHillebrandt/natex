"""Model weights and mixture posterior for the two DEE debiasing models.

Model A ("debias") predicts tau(x) = cate_obs(x) - bias_gp(x) with the pinned
sign convention bias_u = cate_obs_u - tau_hat_u (math audit, Codex #30).
Model B ("direct") is the CATE-extrapolation GP fit on tau_hat itself.

Weighting strategies:

- :func:`mll_weights` / :func:`loo_weights` -- softmax of the two log scores.
  These are the paper's baselines, kept for comparison only; superseded by
  buffered stacking as the default (audit section 3 adopted).
- :func:`buffered_stacking_weights` -- the default. Hyperparameters of BOTH
  GPs are refit within each spatially buffered fold (train centers within
  ``buffer`` of any held-out center are excluded), and the mixture weight is
  chosen on a 101-point grid over [0, 1] maximizing the summed held-out log
  mixture density. Deterministic 1-D optimization -- NOT softmax-PLP and NOT
  the paper's unseeded 1-MC "random dist" strategy (audit section 3; repo
  risk 8).

Mixture posterior (audit 8):

- covariance ``w*Sa + (1-w)*Sb + w*(1-w)*(mu_a-mu_b)(mu_a-mu_b)^T`` -- the
  exact two-component mixture second moment, not a naive convex combination.
- sampling draws ONE Bernoulli(w) model label per posterior draw, then the
  whole draw from that component. Independent per-point labels average the
  bimodality away across query points and produce too-narrow aggregate
  intervals -- the defect audit 8 flags.

House rules: explicit ``numpy.random.Generator`` everywhere stochastic;
identical seed => identical output; NaN never 0.0 on failure (a NaN score or
too few centers yields NaN weights, and callers fall back to
:func:`loo_weights` -- documented); no bare except.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from natex.dee.gp import (
    GPPosterior,
    HeteroskedasticGP,
    _as_2d,
    _require_rng,
    _sq_dists,
)

_LOG_2PI = math.log(2.0 * math.pi)


def _softmax_pair(score_debias: float, score_direct: float) -> float:
    """exp(s_a) / (exp(s_a) + exp(s_b)) computed stably; NaN if either is NaN."""
    if not (np.isfinite(score_debias) and np.isfinite(score_direct)):
        return float("nan")
    # 1 / (1 + exp(s_b - s_a)); exp saturates safely at +/- inf
    with np.errstate(over="ignore"):
        return float(1.0 / (1.0 + np.exp(score_direct - score_debias)))


@dataclass
class ModelWeights:
    """Mixture weight on model A = debias; ``w_direct = 1 - w_debias``."""

    w_debias: float
    strategy: str  # "mll" | "loo" | "stacking"
    detail: dict[str, Any]


def mll_weights(gp_bias: HeteroskedasticGP, gp_direct: HeteroskedasticGP) -> ModelWeights:
    """Softmax of the two exact MLLs (paper baseline, kept for comparison)."""
    s_a = float(gp_bias.log_marginal_likelihood())
    s_b = float(gp_direct.log_marginal_likelihood())
    return ModelWeights(
        w_debias=_softmax_pair(s_a, s_b),
        strategy="mll",
        detail={"score_debias": s_a, "score_direct": s_b},
    )


def loo_weights(gp_bias: HeteroskedasticGP, gp_direct: HeteroskedasticGP) -> ModelWeights:
    """Softmax of the two closed-form LOO log predictives (paper baseline)."""
    s_a = float(gp_bias.loo_log_predictive())
    s_b = float(gp_direct.loo_log_predictive())
    return ModelWeights(
        w_debias=_softmax_pair(s_a, s_b),
        strategy="loo",
        detail={"score_debias": s_a, "score_direct": s_b},
    )


def buffered_folds(
    Xc: np.ndarray, n_folds: int, buffer: float, rng: np.random.Generator
) -> list[tuple[np.ndarray, np.ndarray]]:
    """rng-shuffled disjoint folds with a spatial train/test buffer.

    Every center is held out exactly once. Train indices for a fold exclude
    the held-out centers AND any center strictly within ``buffer`` (Euclidean
    distance in the space of ``Xc``, i.e. Z_std upstream) of ANY held-out
    center in that fold. Deterministic given ``rng``.
    """
    rng = _require_rng(rng)
    X = _as_2d(Xc)
    u = X.shape[0]
    n_folds = int(n_folds)
    if not 2 <= n_folds <= u:
        raise ValueError(f"n_folds must be in [2, {u}], got {n_folds}")
    buffer = float(buffer)
    if not buffer >= 0.0:  # also rejects NaN
        raise ValueError(f"buffer must be a finite value >= 0, got {buffer}")
    if not np.all(np.isfinite(X)):
        raise ValueError("Xc must be finite; filter non-finite centers upstream")

    dist = np.sqrt(_sq_dists(X, X))
    groups = np.array_split(rng.permutation(u), n_folds)
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    for g in groups:
        test = np.sort(g)
        keep = np.ones(u, dtype=bool)
        keep[test] = False
        keep &= ~(dist[:, test] < buffer).any(axis=1)
        folds.append((np.flatnonzero(keep), test))
    return folds


def _normal_logpdf(x: np.ndarray, mean: np.ndarray, var: np.ndarray) -> np.ndarray:
    """Elementwise Normal log density; NaN (never 0.0) where var <= 0 or non-finite."""
    x = np.asarray(x, dtype=float)
    var = np.asarray(var, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = -0.5 * (_LOG_2PI + np.log(var) + (x - mean) ** 2 / var)
    return np.where(np.isfinite(out) & (var > 0.0), out, np.nan)


def _median_nn_distance(X: np.ndarray) -> float:
    """Median nearest-neighbor distance among the centers (the "auto" buffer)."""
    dist = np.sqrt(_sq_dists(X, X))
    np.fill_diagonal(dist, np.inf)
    return float(np.median(dist.min(axis=1)))


def buffered_stacking_weights(
    centers: np.ndarray,
    tau_hat: np.ndarray,
    obs_at_centers: np.ndarray,
    noise_var: np.ndarray,
    rng: np.random.Generator,
    n_folds: int = 5,
    buffer: float | str = "auto",
    n_restarts: int = 2,
) -> ModelWeights:
    """Buffered predictive stacking of the debias and direct models (default).

    Within each buffered fold both GPs' hyperparameters are refit on the
    buffered training subset (rng restarts). The held-out predictive for
    center i is Normal(obs_at_centers[i] - bias_gp(center_i),
    bias_var + noise_var[i]) under model A and Normal(direct_gp mean,
    var + noise_var[i]) under model B, both scored at target ``tau_hat[i]``;
    ``w`` maximizes the summed held-out log mixture density over a 101-point
    grid on [0, 1] (audit section 3: deterministic 1-D optimization, no
    softmax-PLP). Folds whose buffered training set is too small to fit a GP
    contribute nothing and are recorded in ``detail["skipped_folds"]``.

    Fewer than 3 usable centers => ``w_debias`` is NaN (never 0.0) and the
    caller falls back to :func:`loo_weights` (documented contract). If
    ``u < n_folds``, ``n_folds`` is reduced to ``max(2, u)`` with a diagnostic
    in ``detail``.
    """
    rng = _require_rng(rng)
    X = _as_2d(centers)
    tau = np.asarray(tau_hat, dtype=float).ravel()
    obs = np.asarray(obs_at_centers, dtype=float).ravel()
    nv = np.asarray(noise_var, dtype=float).ravel()
    n = X.shape[0]
    if not (tau.shape[0] == obs.shape[0] == nv.shape[0] == n):
        raise ValueError(
            f"length mismatch: centers {n}, tau_hat {tau.shape[0]}, "
            f"obs_at_centers {obs.shape[0]}, noise_var {nv.shape[0]}"
        )

    finite = (
        np.all(np.isfinite(X), axis=1)
        & np.isfinite(tau)
        & np.isfinite(obs)
        & np.isfinite(nv)
    )
    detail: dict[str, Any] = {"n_dropped": int((~finite).sum()), "n_used": int(finite.sum())}
    u = int(finite.sum())
    if u < 3:
        detail["reason"] = "fewer than 3 usable centers; fall back to loo_weights"
        return ModelWeights(w_debias=float("nan"), strategy="stacking", detail=detail)

    Xf, tauf, obsf, nvf = X[finite], tau[finite], obs[finite], nv[finite]
    orig_idx = np.flatnonzero(finite)

    n_folds = int(n_folds)
    if n_folds < 2:
        raise ValueError(f"n_folds must be >= 2, got {n_folds}")
    if u < n_folds:
        detail["n_folds_requested"] = n_folds
        n_folds = max(2, u)
    detail["n_folds"] = n_folds

    buf = _median_nn_distance(Xf) if isinstance(buffer, str) and buffer == "auto" else float(buffer)
    if not buf >= 0.0:  # also rejects NaN
        raise ValueError(f"buffer must be 'auto' or a finite value >= 0, got {buffer!r}")
    detail["buffer"] = buf

    log_a = np.full(u, np.nan)  # held-out log density under model A (debias)
    log_b = np.full(u, np.nan)  # ... under model B (direct)
    fold_records: list[dict[str, Any]] = []
    skipped: list[int] = []
    for k, (tr, te) in enumerate(buffered_folds(Xf, n_folds, buf, rng)):
        if tr.size < 2:  # a GP needs >= 2 training rows (dee/gp.py degeneracy rule)
            skipped.append(k)
            fold_records.append(
                {"fold": k, "n_train": int(tr.size), "test_idx": orig_idx[te].tolist()}
            )
            continue
        gp_bias = HeteroskedasticGP.fit(
            Xf[tr], obsf[tr] - tauf[tr], nvf[tr], rng=rng, n_restarts=n_restarts
        )
        gp_direct = HeteroskedasticGP.fit(
            Xf[tr], tauf[tr], nvf[tr], rng=rng, n_restarts=n_restarts
        )
        post_bias = gp_bias.posterior(Xf[te])
        post_dir = gp_direct.posterior(Xf[te])
        var_bias = np.maximum(np.diag(post_bias.cov), 0.0) + nvf[te]
        var_dir = np.maximum(np.diag(post_dir.cov), 0.0) + nvf[te]
        log_a[te] = _normal_logpdf(tauf[te], obsf[te] - post_bias.mean, var_bias)
        log_b[te] = _normal_logpdf(tauf[te], post_dir.mean, var_dir)
        fold_records.append(
            {
                "fold": k,
                "n_train": int(tr.size),
                "test_idx": orig_idx[te].tolist(),
                "log_score_debias": float(np.nansum(log_a[te])),
                "log_score_direct": float(np.nansum(log_b[te])),
            }
        )
    detail["folds"] = fold_records
    detail["skipped_folds"] = skipped

    scored = np.isfinite(log_a) & np.isfinite(log_b)
    detail["n_scored"] = int(scored.sum())
    if not scored.any():
        detail["reason"] = "no held-out center received a finite score from both models"
        return ModelWeights(w_debias=float("nan"), strategy="stacking", detail=detail)

    la, lb = log_a[scored], log_b[scored]
    grid = np.linspace(0.0, 1.0, 101)
    with np.errstate(divide="ignore"):  # log(0) = -inf at the grid endpoints
        totals = np.array(
            [
                float(np.sum(np.logaddexp(np.log(w) + la, np.log1p(-w) + lb)))
                for w in grid
            ]
        )
    best = int(np.argmax(totals))  # ties -> smallest w (deterministic)
    detail["grid_best_log_score"] = float(totals[best])
    return ModelWeights(w_debias=float(grid[best]), strategy="stacking", detail=detail)


@dataclass
class MixturePosterior:
    """Two-component mixture of GP posteriors with the audit-8 covariance."""

    mean: np.ndarray  # (m,) w mu_a + (1-w) mu_b
    cov: np.ndarray  # (m, m) w Sa + (1-w) Sb + w(1-w)(mu_a-mu_b)(mu_a-mu_b)^T
    w: float
    post_a: GPPosterior = field(repr=False)
    post_b: GPPosterior = field(repr=False)

    def sample(self, rng: np.random.Generator | None = None, size: int = 1) -> np.ndarray:
        """(size, m) draws: ONE Bernoulli(w) model label per draw (audit 8).

        The whole draw then comes from that component, so cross-point
        bimodality survives aggregation. NaN ``w`` yields all-NaN draws.
        """
        rng = _require_rng(rng)
        size = int(size)
        m = self.mean.shape[0]
        if not np.isfinite(self.w):
            rng.random(size)  # keep the rng stream advancing uniformly
            return np.full((size, m), np.nan)
        labels = rng.random(size) < self.w  # True -> model A
        out = np.empty((size, m))
        out[labels] = self.post_a.sample(rng, size=int(labels.sum()))
        out[~labels] = self.post_b.sample(rng, size=int(size - labels.sum()))
        return out


def mixture_posterior(post_a: GPPosterior, post_b: GPPosterior, w: float) -> MixturePosterior:
    """Analytic two-component mixture of ``post_a`` (debias) and ``post_b`` (direct).

    ``w`` is the weight on model A. NaN ``w`` (degenerate weighting upstream)
    propagates NaN mean/cov -- never 0.0. Finite ``w`` outside [0, 1] raises.
    """
    w = float(w)
    if np.isfinite(w) and not 0.0 <= w <= 1.0:
        raise ValueError(f"w must be in [0, 1] (or NaN), got {w}")
    mu_a = np.asarray(post_a.mean, dtype=float).ravel()
    mu_b = np.asarray(post_b.mean, dtype=float).ravel()
    if mu_a.shape != mu_b.shape:
        raise ValueError(f"posterior sizes differ: {mu_a.shape} vs {mu_b.shape}")
    m = mu_a.shape[0]
    if not np.isfinite(w):
        return MixturePosterior(
            mean=np.full(m, np.nan),
            cov=np.full((m, m), np.nan),
            w=w,
            post_a=post_a,
            post_b=post_b,
        )
    diff = mu_a - mu_b
    cov = w * post_a.cov + (1.0 - w) * post_b.cov + w * (1.0 - w) * np.outer(diff, diff)
    return MixturePosterior(
        mean=w * mu_a + (1.0 - w) * mu_b,
        cov=cov,
        w=w,
        post_a=post_a,
        post_b=post_b,
    )
