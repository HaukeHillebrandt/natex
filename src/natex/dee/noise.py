"""Hierarchical stage-1 noise smoothing for DEE local-effect variances.

Audit section 3 (adopted): Normal likelihood + chi-square measurement model on
SE^2 -- not both a t likelihood and a latent variance. With
``SE^2 * df / sigma^2 ~ chi^2_df``, ``log SE^2`` equals ``log sigma^2`` plus a
KNOWN bias ``psi(df/2) - log(df/2)`` with KNOWN variance ``psi_1(df/2)``
(scipy digamma / trigamma). We debias ``log SE^2``, fit the exact
heteroskedastic GP (``dee/gp.py``) with that measurement variance as known
per-point noise, and back-transform the latent posterior by the log-normal
mean ``exp(m + v/2)``, floored at ``floor_frac * median(finite SE^2)`` --
data-scaled, never an absolute constant (audit 24 lineage).

Conventions that bind here (docs/math_audit_final.md + house rules):

- the SE^2 are estimated (HC1 upstream), never treated as known independent
  GP noise (audit 9): they enter only through this measurement model.
- NaN never 0.0: non-finite ``se2`` propagates NaN at that index only; a
  degenerate fit (fewer than 2 usable rows) returns NaN, never a filler.
- one ``numpy.random.Generator`` through every stochastic call (GP restarts);
  identical seed => identical output.
- ``df=None`` fallback (documented heuristic): no debiasing constant is known,
  so ``v = log se2`` and the measurement variance is the pooled sample
  variance of the finite ``v``.
- ``df`` values below 1 are clipped to 1 with a ``UserWarning`` diagnostic.
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy.special import digamma, polygamma

from natex.dee.gp import HeteroskedasticGP, _as_2d, _require_rng


def log_se2_bias(df: np.ndarray) -> np.ndarray:
    """``E[log SE^2] - log sigma^2 = psi(df/2) - log(df/2)`` under chi^2_df."""
    df = np.asarray(df, dtype=float)
    return digamma(df / 2.0) - np.log(df / 2.0)


def log_se2_measurement_var(df: np.ndarray) -> np.ndarray:
    """``Var[log SE^2] = psi_1(df/2)`` (trigamma) under chi^2_df."""
    df = np.asarray(df, dtype=float)
    return polygamma(1, df / 2.0)


def smooth_noise(
    X: np.ndarray,
    se2: np.ndarray,
    df: np.ndarray | None,
    rng: np.random.Generator,
    floor_frac: float = 1e-3,
) -> np.ndarray:
    """Smooth per-experiment SE^2 via the chi-square measurement-model GP.

    Parameters
    ----------
    X : (u, d) experiment centers (Z_std space).
    se2 : (u,) observed squared standard errors (HC1, estimated upstream).
    df : (u,) per-experiment residual dof (``n_used - 4`` for the frozen-side
        2SLS); ``None`` selects the pooled-variance fallback (docstring above).
    rng : explicit numpy Generator (GP restart draws).
    floor_frac : floor = ``floor_frac * median(finite se2)`` -- data-scaled.

    Returns
    -------
    (u,) smoothed noise variances. NaN exactly where ``se2`` is non-finite
    (callers drop those experiments); zero/negative finite ``se2`` rows are
    excluded from training but predicted from the GP and floored.
    """
    rng = _require_rng(rng)
    X = _as_2d(X)
    se2 = np.asarray(se2, dtype=float).ravel()
    u = se2.shape[0]
    if X.shape[0] != u:
        raise ValueError(f"X has {X.shape[0]} rows but se2 has {u}")
    if not float(floor_frac) > 0.0:
        raise ValueError("floor_frac must be > 0")

    finite_se2 = np.isfinite(se2)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_se2 = np.log(se2)  # -inf at 0, NaN at negative/non-finite: train-masked below

    if df is None:
        v = log_se2
        vf = v[np.isfinite(v)]
        pooled = float(np.var(vf, ddof=1)) if vf.size >= 2 else float("nan")
        meas_var = np.full(u, pooled)
    else:
        df_arr = np.broadcast_to(np.asarray(df, dtype=float), (u,)).astype(float)
        below = np.isfinite(df_arr) & (df_arr < 1.0)
        if below.any():
            warnings.warn(
                f"smooth_noise: {int(below.sum())} df value(s) < 1 clipped to 1",
                stacklevel=2,
            )
            df_arr = np.maximum(df_arr, 1.0)  # NaN df stays NaN (row train-masked below)
        v = log_se2 - log_se2_bias(df_arr)
        meas_var = log_se2_measurement_var(df_arr)

    floor = float(floor_frac) * float(np.median(se2[finite_se2])) if finite_se2.any() else np.nan

    train = np.isfinite(v) & np.isfinite(meas_var) & np.all(np.isfinite(X), axis=1)
    if int(train.sum()) < 2:
        # degenerate: nothing to smooth from -- NaN, never 0.0
        return np.full(u, np.nan)

    gp = HeteroskedasticGP.fit(X[train], v[train], meas_var[train], rng=rng)
    post = gp.posterior(X)  # non-finite X rows propagate NaN at those indices only
    post_var = np.maximum(np.diag(post.cov), 0.0)  # clip roundoff negatives
    out = np.exp(post.mean + 0.5 * post_var)  # log-normal mean back-transform
    out = np.maximum(out, floor)  # NaN propagates through maximum
    out[~finite_se2] = np.nan
    return out
