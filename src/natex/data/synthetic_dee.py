"""Scaled DEE simulation-1 DGP (Jakubowski et al., JMLR 2023) with an exact
closed-form conditional-bias construction.

Corrected reconstruction of the paper's simulation 1 (the paper's exact
complier-shift calibration is repo code, not printed math — phase-4 plan task 7):

- ``X ~ U[0, 1]^2``; nested corner instruments ``Z_j = 1[x0 >= b_j and x1 >= b_j]``
  for thresholds ``b1 < b2`` (two square-corner RDs, the repo's step 1).
- Complier type ``G ~ Categorical(type_probs)`` i.i.d. with
  0=never-taker, 1=complier-Z1, 2=complier-Z2, 3=always-taker; observed treatment
  ``D = 1[G=3] + 1[G=1] Z1 + 1[G=2] Z2`` (always/never-takers give overlap
  everywhere, hence every ``type_prob`` must be > 0).
- tau(.) and beta(.) are GP surfaces (one ``sample_gp_prior`` draw each, RBF with
  the given lengthscales/outputscale), or constants when ``constant_surfaces``
  is set.
- **Exact conditional-bias identity**: the region ``r(X) in {00, 10, 11}``
  (Z1 Z2 patterns; nesting makes 01 impossible) has closed-form
  ``q1_r = P(G=3 | D=1, r)`` and ``q0_r = P(G=0 | D=0, r)`` from ``type_probs``.
  With ``a_r = 1 / (q1_r + q0_r)`` and
  ``c_i = beta(X_i) * a_r * (1[G_i=3] - 1[G_i=0])``,
  ``E[c | D=1, X] - E[c | D=0, X] = beta(X)`` exactly (G is independent of X and
  1[G=3] vanishes on D=0 rows, 1[G=0] on D=1 rows), so the
  conditional-on-observables contrast is ``tau(X) + beta(X)`` by construction —
  the sampled bias surface IS the observational bias.
- ``y = X @ g + tau(X) * D + c + noise_sd * eps``, ``g ~ N(0, I_2)``.

Documented deviation (GP surfaces): the exact joint prior draw over the stacked
n + grid^2 points is O((n + m)^3) and infeasible at the n >= 1e5 scales the
identity tests use. Instead each surface is ONE exact ``sample_gp_prior`` draw
on a fixed 25x25 anchor lattice over [0, 1]^2, extended to the stacked
train+query points by noiseless kriging (RBF posterior-mean) interpolation.
With anchor spacing 1/25 far below the supported lengthscales the interpolant
is numerically indistinguishable from an exact draw, and the bias identity is
exact conditional on the realized surface values regardless.

rng consumption order (determinism contract): X uniforms -> complier types ->
[tau anchor draw -> beta anchor draw, skipped when ``constant_surfaces`` is
set] -> g -> outcome noise.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.linalg import cho_solve

from natex.data.spec import Dataset, DatasetSpec
from natex.dee.gp import _chol_with_jitter, rbf_kernel, sample_gp_prior

_ANCHOR_GRID = 25  # exact GP draw on a 25x25 lattice, kriged to the data points
_EVAL_CHUNK = 32768  # cap the (chunk, 625) cross-kernel blocks at large n


@dataclass
class DEETruth:
    cate_train: np.ndarray  # (n,) tau(X_i)
    bias_train: np.ndarray  # (n,) beta(X_i)
    cate_query: np.ndarray  # (m,) tau at grid points
    bias_query: np.ndarray  # (m,)
    query: np.ndarray  # (m, 2) raw-unit grid points
    complier_type: np.ndarray  # (n,) int: 0=never, 1=complier-Z1, 2=complier-Z2, 3=always
    thresholds: tuple[float, float]


def _sample_surface(
    points: np.ndarray, lengthscale: float, outputscale: float, rng: np.random.Generator
) -> np.ndarray:
    """One GP-prior surface at ``points``: exact anchor-lattice draw + kriging."""
    g = (np.arange(_ANCHOR_GRID) + 0.5) / _ANCHOR_GRID
    a0, a1 = np.meshgrid(g, g, indexing="ij")
    anchors = np.column_stack([a0.ravel(), a1.ravel()])
    f = sample_gp_prior(anchors, lengthscale, outputscale, rng=rng)[0]
    Kaa = rbf_kernel(anchors, anchors, lengthscale, outputscale)
    chol = _chol_with_jitter(Kaa, jitter=1e-8)
    w = cho_solve(chol, f)
    out = np.empty(points.shape[0], dtype=float)
    for s in range(0, points.shape[0], _EVAL_CHUNK):
        blk = points[s : s + _EVAL_CHUNK]
        out[s : s + blk.shape[0]] = rbf_kernel(blk, anchors, lengthscale, outputscale) @ w
    return out


def _region_confound_scale(probs: np.ndarray) -> np.ndarray:
    """a_r = 1 / (q1_r + q0_r) for regions r = 00, 10, 11 (indices 0, 1, 2)."""
    p0, p1, p2, p3 = probs
    q1 = np.array([1.0, p3 / (p3 + p1), p3 / (p1 + p2 + p3)])
    q0 = np.array([p0 / (p0 + p1 + p2), p0 / (p0 + p2), 1.0])
    return 1.0 / (q1 + q0)


def make_dee_synthetic(
    n: int,
    *,
    cate_lengthscale: float = 0.5,
    bias_lengthscale: float = 0.5,
    outputscale: float = 1.0,
    thresholds: tuple[float, float] = (1.0 / 3.0, 2.0 / 3.0),
    type_probs: tuple[float, float, float, float] = (0.25, 0.25, 0.25, 0.25),
    grid: int = 25,
    noise_sd: float = 0.5,
    constant_surfaces: tuple[float, float] | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[Dataset, DEETruth]:
    """Scaled simulation-1 dataset + ground truth. See the module docstring.

    ``constant_surfaces=(tau0, beta0)`` overrides both GP draws with constants
    (exact tests); it consumes NO surface rng draws.
    """
    if rng is None:
        raise ValueError("pass an explicit numpy Generator (reproducibility contract)")
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    b1, b2 = (float(b) for b in thresholds)
    if not 0.0 < b1 < b2 < 1.0:
        raise ValueError(f"thresholds must satisfy 0 < b1 < b2 < 1, got {thresholds}")
    probs = np.asarray(type_probs, dtype=float)
    if probs.shape != (4,) or np.any(probs <= 0.0) or abs(float(probs.sum()) - 1.0) > 1e-8:
        raise ValueError(
            "type_probs must be 4 strictly positive probabilities summing to 1 "
            f"(overlap in every region needs every complier type), got {type_probs}"
        )
    if grid < 2:
        raise ValueError(f"grid must be >= 2, got {grid}")
    if cate_lengthscale <= 0.0 or bias_lengthscale <= 0.0 or outputscale <= 0.0:
        raise ValueError("lengthscales and outputscale must be > 0")
    if noise_sd < 0.0:
        raise ValueError(f"noise_sd must be >= 0, got {noise_sd}")

    x = rng.uniform(0.0, 1.0, size=(n, 2))
    gtype = rng.choice(4, size=n, p=probs).astype(np.int64)

    z1 = (x[:, 0] >= b1) & (x[:, 1] >= b1)
    z2 = (x[:, 0] >= b2) & (x[:, 1] >= b2)
    region = z1.astype(int) + z2.astype(int)  # 0=00, 1=10, 2=11 (01 impossible)
    D = ((gtype == 3) | ((gtype == 1) & z1) | ((gtype == 2) & z2)).astype(float)

    qs = (np.arange(grid) + 0.5) / grid  # cell-center query lattice in (0, 1)^2
    q0m, q1m = np.meshgrid(qs, qs, indexing="ij")
    query = np.column_stack([q0m.ravel(), q1m.ravel()])
    m = query.shape[0]

    if constant_surfaces is not None:
        tau0, beta0 = (float(v) for v in constant_surfaces)
        cate_train = np.full(n, tau0)
        bias_train = np.full(n, beta0)
        cate_query = np.full(m, tau0)
        bias_query = np.full(m, beta0)
    else:
        stacked = np.vstack([x, query])
        tau_all = _sample_surface(stacked, cate_lengthscale, outputscale, rng)
        beta_all = _sample_surface(stacked, bias_lengthscale, outputscale, rng)
        cate_train, cate_query = tau_all[:n], tau_all[n:]
        bias_train, bias_query = beta_all[:n], beta_all[n:]

    a = _region_confound_scale(probs)[region]
    c = bias_train * a * ((gtype == 3).astype(float) - (gtype == 0).astype(float))

    g = rng.normal(0.0, 1.0, size=2)
    y = x @ g + cate_train * D + c + noise_sd * rng.normal(0.0, 1.0, size=n)

    df = pd.DataFrame({"x0": x[:, 0], "x1": x[:, 1], "D": D, "y": y})
    spec = DatasetSpec(
        treatment="D", outcome="y", forcing=["x0", "x1"], covariates=["x0", "x1"]
    )
    truth = DEETruth(
        cate_train=cate_train,
        bias_train=bias_train,
        cate_query=cate_query,
        bias_query=bias_query,
        query=query,
        complier_type=gtype,
        thresholds=(b1, b2),
    )
    return Dataset(df, spec), truth
