"""Exact heteroskedastic-noise RBF GP on core deps (numpy/scipy only).

DEE fits its bias and direct-CATE surfaces on tens of experiment centers, so
the exact O(n^3) GP is trivial (design spec section 10: small-N defaults; the
optional GPyTorch backend behind ``natex[gp]`` exists only for scale).

Conventions that bind here (docs/math_audit_final.md + house rules):

- per-point noise variances enter as KNOWN heteroskedastic noise; upstream,
  ``dee/noise.py`` produces them from the chi-square measurement model, so
  they are documented as estimated-then-smoothed, never raw classical SE^2.
- one ``numpy.random.Generator`` through every stochastic call (restart draws,
  prior/posterior sampling); identical seed => identical output.
- NaN never 0.0: fewer than 2 finite training rows => NaN mean/cov/MLL/LOO.
- distances are computed on inputs as given (callers pass Z_std-space points).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize

_LOG_2PI = math.log(2.0 * math.pi)
_MAX_JITTER = 1e-4


def _require_rng(rng: np.random.Generator | None) -> np.random.Generator:
    if rng is None:
        raise ValueError("pass an explicit numpy Generator")
    if not isinstance(rng, np.random.Generator):
        raise TypeError(f"rng must be a numpy Generator, got {type(rng).__name__}")
    return rng


def _as_2d(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    if X.ndim != 2:
        raise ValueError(f"X must be (n, d), got shape {X.shape}")
    return X


def _sq_dists(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Pairwise squared Euclidean distances, clipped at 0 against roundoff."""
    d2 = (
        (A**2).sum(axis=1)[:, None]
        + (B**2).sum(axis=1)[None, :]
        - 2.0 * (A @ B.T)
    )
    return np.maximum(d2, 0.0)


def rbf_kernel(
    A: np.ndarray, B: np.ndarray, lengthscale: float, outputscale: float
) -> np.ndarray:
    """outputscale * exp(-||a - b||^2 / (2 * lengthscale^2)), shape (len(A), len(B))."""
    A, B = _as_2d(A), _as_2d(B)
    if lengthscale <= 0.0 or outputscale < 0.0:
        raise ValueError("lengthscale must be > 0 and outputscale >= 0")
    return float(outputscale) * np.exp(-_sq_dists(A, B) / (2.0 * float(lengthscale) ** 2))


def _chol_with_jitter(K: np.ndarray, jitter: float = 0.0) -> tuple[np.ndarray, bool]:
    """Lower Cholesky of K (+ escalating jitter*I). Escalates jitter x10 up to 1e-4.

    Returns ``(L, lower)`` usable with ``scipy.linalg.cho_solve``. Raises
    ``numpy.linalg.LinAlgError`` if the matrix is not PD even at max jitter.
    """
    n = K.shape[0]
    eye = np.eye(n)
    j = float(jitter)
    while True:
        try:
            c, low = cho_factor(K + j * eye if j > 0.0 else K, lower=True)
            return c, low
        except np.linalg.LinAlgError:
            j = 1e-8 if j == 0.0 else j * 10.0
            if j > _MAX_JITTER:
                raise np.linalg.LinAlgError(
                    f"matrix not positive definite even with jitter {_MAX_JITTER}"
                ) from None


def sample_gp_prior(
    X: np.ndarray,
    lengthscale: float,
    outputscale: float,
    rng: np.random.Generator | None = None,
    size: int = 1,
    jitter: float = 1e-8,
) -> np.ndarray:
    """(size, n) zero-mean prior draws via Cholesky with jitter escalation."""
    rng = _require_rng(rng)
    X = _as_2d(X)
    K = rbf_kernel(X, X, lengthscale, outputscale)
    L, _ = _chol_with_jitter(K, jitter=jitter)
    z = rng.standard_normal((int(size), X.shape[0]))
    return z @ np.tril(L).T


@dataclass
class GPPosterior:
    """Latent-function posterior at the query points (noise NOT added)."""

    mean: np.ndarray  # (m,)
    cov: np.ndarray  # (m, m)

    def sample(self, rng: np.random.Generator | None = None, size: int = 1) -> np.ndarray:
        """(size, m) posterior draws; all-NaN draws for a degenerate posterior."""
        rng = _require_rng(rng)
        m = self.mean.shape[0]
        if not (np.all(np.isfinite(self.mean)) and np.all(np.isfinite(self.cov))):
            rng.standard_normal((int(size), m))  # keep the rng stream advancing uniformly
            return np.full((int(size), m), np.nan)
        L, _ = _chol_with_jitter(self.cov, jitter=1e-12)
        z = rng.standard_normal((int(size), m))
        return self.mean[None, :] + z @ np.tril(L).T


def _neg_mll_and_grad(
    theta: np.ndarray, d2: np.ndarray, y: np.ndarray, noise_var: np.ndarray
) -> tuple[float, np.ndarray]:
    """Negative exact MLL and gradient in theta = (log l, log s2, mean_const)."""
    log_l, log_s2, mean = theta
    n = y.shape[0]
    ell2 = math.exp(2.0 * log_l)
    s2 = math.exp(log_s2)
    R = np.exp(-d2 / (2.0 * ell2))
    Kt = s2 * R + np.diag(noise_var)
    try:
        L, low = _chol_with_jitter(Kt)
    except np.linalg.LinAlgError:
        return 1e25, np.zeros(3)
    r = y - mean
    alpha = cho_solve((L, low), r)
    logdet = 2.0 * np.sum(np.log(np.diag(L)))
    mll = -0.5 * float(r @ alpha) - 0.5 * logdet - 0.5 * n * _LOG_2PI
    Kinv = cho_solve((L, low), np.eye(n))
    W = np.outer(alpha, alpha) - Kinv  # dMLL/dtheta_k = 0.5 tr(W dK/dtheta_k)
    dK_dlog_l = s2 * R * (d2 / ell2)
    dK_dlog_s2 = s2 * R
    g = np.array(
        [
            0.5 * float(np.sum(W * dK_dlog_l)),
            0.5 * float(np.sum(W * dK_dlog_s2)),
            float(np.sum(alpha)),
        ]
    )
    if not np.isfinite(mll):
        return 1e25, np.zeros(3)
    return -mll, -g


@dataclass
class HeteroskedasticGP:
    """Exact RBF GP with constant mean and KNOWN per-point noise variances.

    Construct directly with fixed hyperparameters, or via :meth:`fit` (exact
    MLL maximized by L-BFGS-B over (log lengthscale, log outputscale,
    mean_const) with rng-seeded restarts). The Cholesky of
    ``K + diag(noise_var)`` and ``alpha = K~^-1 (y - mean_const)`` are cached
    at construction.
    """

    lengthscale: float
    outputscale: float
    mean_const: float
    X: np.ndarray  # (n, d) training inputs actually used (finite rows)
    y: np.ndarray  # (n,)
    noise_var: np.ndarray  # (n,) known noise variances
    fit_report: dict[str, Any] | None = field(default=None, repr=False)
    _chol: np.ndarray | None = field(default=None, init=False, repr=False)
    _alpha: np.ndarray | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.X = _as_2d(self.X) if np.asarray(self.X).size else np.empty((0, 1))
        self.y = np.asarray(self.y, dtype=float).ravel()
        self.noise_var = np.broadcast_to(
            np.asarray(self.noise_var, dtype=float), self.y.shape
        ).copy()
        if self.X.shape[0] != self.y.shape[0]:
            raise ValueError(
                f"X has {self.X.shape[0]} rows but y has {self.y.shape[0]}"
            )
        if np.any(self.noise_var[np.isfinite(self.noise_var)] < 0.0):
            raise ValueError("noise_var must be nonnegative")
        if self._degenerate():
            return
        Kt = rbf_kernel(self.X, self.X, self.lengthscale, self.outputscale) + np.diag(
            self.noise_var
        )
        L, _ = _chol_with_jitter(Kt)
        self._chol = np.tril(L)
        self._alpha = cho_solve((self._chol, True), self.y - self.mean_const)

    def _degenerate(self) -> bool:
        return (
            self.X.shape[0] < 2
            or not np.isfinite(self.lengthscale)
            or not np.isfinite(self.outputscale)
            or not np.isfinite(self.mean_const)
            or not (np.all(np.isfinite(self.y)) and np.all(np.isfinite(self.noise_var)))
        )

    @classmethod
    def fit(
        cls,
        X: np.ndarray,
        y: np.ndarray,
        noise_var: np.ndarray,
        rng: np.random.Generator | None = None,
        n_restarts: int = 4,
        lengthscale_bounds: tuple[float, float] = (1e-2, 1e2),
        outputscale_bounds: tuple[float, float] = (1e-4, 1e4),
    ) -> HeteroskedasticGP:
        """Maximize the exact MLL; deterministic given ``rng``.

        Non-finite (X, y, noise_var) rows are dropped and counted in
        ``fit_report['n_dropped']``. With fewer than 2 finite rows the model
        is degenerate: posterior/MLL/LOO are NaN, never 0.0.
        """
        rng = _require_rng(rng)
        X = _as_2d(X)
        y = np.asarray(y, dtype=float).ravel()
        noise_var = np.broadcast_to(np.asarray(noise_var, dtype=float), y.shape)
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"X has {X.shape[0]} rows but y has {y.shape[0]}")
        finite = np.isfinite(y) & np.isfinite(noise_var) & np.all(np.isfinite(X), axis=1)
        n_dropped = int((~finite).sum())
        Xf, yf, nvf = X[finite], y[finite], noise_var[finite].astype(float)
        report: dict[str, Any] = {
            "n_dropped": n_dropped,
            "n_used": int(finite.sum()),
            "starts": [],
            "best_mll": float("nan"),
        }
        if Xf.shape[0] < 2:
            gp = cls(
                lengthscale=float("nan"),
                outputscale=float("nan"),
                mean_const=float("nan"),
                X=Xf,
                y=yf,
                noise_var=nvf,
                fit_report=report,
            )
            return gp

        d2 = _sq_dists(Xf, Xf)
        lo_l, hi_l = (math.log(b) for b in lengthscale_bounds)
        lo_s, hi_s = (math.log(b) for b in outputscale_bounds)
        y_mean, y_var = float(yf.mean()), float(yf.var())
        y_sd = math.sqrt(y_var) if y_var > 0.0 else 1.0

        # start 1: deterministic heuristic (median distance, data variance).
        off = d2[np.triu_indices_from(d2, k=1)]
        med = math.sqrt(float(np.median(off[off > 0.0]))) if np.any(off > 0.0) else 1.0
        starts = [
            np.array(
                [
                    np.clip(math.log(med), lo_l, hi_l),
                    np.clip(math.log(max(y_var, outputscale_bounds[0])), lo_s, hi_s),
                    y_mean,
                ]
            )
        ]
        # rng-seeded restarts: log-uniform in bounds, mean jittered around y_mean.
        for _ in range(int(n_restarts)):
            starts.append(
                np.array(
                    [
                        rng.uniform(lo_l, hi_l),
                        rng.uniform(lo_s, hi_s),
                        y_mean + y_sd * rng.standard_normal(),
                    ]
                )
            )

        best_theta, best_mll = None, -np.inf
        for theta0 in starts:
            init_neg, _ = _neg_mll_and_grad(theta0, d2, yf, nvf)
            res = minimize(
                _neg_mll_and_grad,
                theta0,
                args=(d2, yf, nvf),
                method="L-BFGS-B",
                jac=True,
                bounds=[(lo_l, hi_l), (lo_s, hi_s), (None, None)],
                options={"maxiter": 500, "ftol": 1e-12, "gtol": 1e-8},
            )
            final_mll = -float(res.fun)
            report["starts"].append(
                {
                    "init_theta": theta0.copy(),
                    "init_mll": -float(init_neg),
                    "final_theta": np.asarray(res.x, dtype=float),
                    "final_mll": final_mll,
                }
            )
            if final_mll > best_mll:
                best_mll, best_theta = final_mll, np.asarray(res.x, dtype=float)
        report["best_mll"] = best_mll
        assert best_theta is not None
        return cls(
            lengthscale=math.exp(best_theta[0]),
            outputscale=math.exp(best_theta[1]),
            mean_const=float(best_theta[2]),
            X=Xf,
            y=yf,
            noise_var=nvf,
            fit_report=report,
        )

    def log_marginal_likelihood(self) -> float:
        """Exact MLL at the stored hyperparameters; NaN when degenerate."""
        if self._chol is None or self._alpha is None:
            return float("nan")
        r = self.y - self.mean_const
        logdet = 2.0 * float(np.sum(np.log(np.diag(self._chol))))
        return float(
            -0.5 * (r @ self._alpha) - 0.5 * logdet - 0.5 * self.y.shape[0] * _LOG_2PI
        )

    def posterior(self, Xq: np.ndarray) -> GPPosterior:
        """Latent posterior at ``Xq``; NaN mean/cov (never 0.0) when degenerate."""
        Xq = _as_2d(Xq)
        m = Xq.shape[0]
        if self._chol is None or self._alpha is None:
            return GPPosterior(mean=np.full(m, np.nan), cov=np.full((m, m), np.nan))
        Ks = rbf_kernel(self.X, Xq, self.lengthscale, self.outputscale)  # (n, m)
        mean = self.mean_const + Ks.T @ self._alpha
        v = cho_solve((self._chol, True), Ks)  # K~^-1 K*
        cov = rbf_kernel(Xq, Xq, self.lengthscale, self.outputscale) - Ks.T @ v
        cov = 0.5 * (cov + cov.T)  # symmetrize against roundoff
        return GPPosterior(mean=mean, cov=cov)

    def loo_log_predictive(self) -> float:
        """Closed-form leave-one-out log predictive density of the NOISY y_i.

        Standard identities on K~ = K + diag(noise): mu_-i = y_i -
        [K~^-1 r]_i / [K~^-1]_ii and sigma^2_-i = 1 / [K~^-1]_ii
        (Rasmussen & Williams eq. 5.10-5.12); no refits. NaN when degenerate.
        """
        if self._chol is None or self._alpha is None:
            return float("nan")
        n = self.y.shape[0]
        Kinv_diag = np.diag(cho_solve((self._chol, True), np.eye(n)))
        var_loo = 1.0 / Kinv_diag
        resid = self._alpha * var_loo  # y_i - mu_-i
        return float(
            np.sum(-0.5 * np.log(2.0 * np.pi * var_loo) - resid**2 / (2.0 * var_loo))
        )
