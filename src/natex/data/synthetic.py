"""Synthetic benchmark DGPs from KDD 2018 Eqs 17-22, with the audit corrections:
binary treatment uses an additive LOG-ODDS shift (the printed '+mu' is a typo).

Phase-2 fidelity options (every new kwarg defaults to phase-1 behavior):

- ``confounder="uniform"``: u_i ~ U(0, 1) (Eq 17); phase-1 default stays N(0, 0.5).
- ``boundary="random"``: per-dimension b_j ~ U(0, 1) (Eq 18), redrawn from the seeded
  generator until the empirical region mass lies in
  [min_region_frac, 1 - min_region_frac] — a documented deviation from the paper that
  guards degenerate empty corners.
- ``heteroskedastic=True``: eps_T, eps_p, eps_y ~ N(0, mean_j(x_ij)) per point (Eq 19;
  the second parameter is a variance, so the noise scale is sqrt(mean_j x_ij)).
  eps_p enters the binary-treatment log-odds (Eq 21) only in this mode.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import expit

from natex.data.spec import Dataset, DatasetSpec

_MAX_BOUNDARY_DRAWS = 1000


def draw_confounder(n: int, kind: str, rng: np.random.Generator) -> np.ndarray:
    """Unobserved confounder u: phase-1 'normal' N(0, 0.5) or Eq-17 'uniform' U(0, 1)."""
    if kind == "normal":
        return rng.normal(0.0, 0.5, size=n)
    if kind == "uniform":
        return rng.uniform(0.0, 1.0, size=n)
    raise ValueError(f"unknown confounder kind: {kind!r} (expected 'normal' or 'uniform')")


def _corner_region(
    x: np.ndarray,
    pz: int,
    boundary: float | str,
    min_region_frac: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Axis-aligned corner region D_i = prod_j 1[x_ij >= b_j] over the first pz dims."""
    if isinstance(boundary, str):
        if boundary != "random":
            raise ValueError(f"boundary must be a scalar or 'random', got {boundary!r}")
        lo, hi = min_region_frac, 1.0 - min_region_frac
        for _ in range(_MAX_BOUNDARY_DRAWS):
            b = rng.uniform(0.0, 1.0, size=pz)  # Eq 18
            D = np.all(x[:, :pz] >= b, axis=1)
            if lo <= D.mean() <= hi:
                return D
        raise RuntimeError(
            f"no random boundary gave region mass in [{lo}, {hi}] after "
            f"{_MAX_BOUNDARY_DRAWS} draws; lower pz or min_region_frac"
        )
    return np.all(x[:, :pz] >= float(boundary), axis=1)


def make_synthetic(
    n: int,
    px: int = 2,
    pz: int = 2,
    zeta: float = 3.0,
    tau: float = 2.0,
    kind: str = "binary",
    discont: str = "square",
    rng: np.random.Generator | None = None,
    boundary: float | str = 0.5,
    min_region_frac: float = 0.05,
    heteroskedastic: bool = False,
    confounder: str = "normal",
) -> tuple[Dataset, np.ndarray]:
    if rng is None:
        raise ValueError("pass an explicit numpy Generator (reproducibility contract)")
    if pz > px:
        raise ValueError("pz cannot exceed px")
    if not 0.0 < min_region_frac < 0.5:
        raise ValueError(f"min_region_frac must lie in (0, 0.5), got {min_region_frac}")
    x = rng.uniform(0.0, 1.0, size=(n, px))
    u = draw_confounder(n, confounder, rng)
    if discont != "square":
        raise ValueError("only the 'square' (corner) region is implemented")
    D = _corner_region(x, pz, boundary, min_region_frac, rng)

    # Noise scale: phase-1 homoskedastic 0.5, or Eq 19 sqrt(mean_j x_ij) per point.
    scale: float | np.ndarray = np.sqrt(x.mean(axis=1)) if heteroskedastic else 0.5

    g = rng.normal(0.0, 1.0, size=px)
    if kind == "real":
        T = x @ g + zeta * D + rng.normal(0.0, scale, size=n) + u
    elif kind == "binary":
        logits = x @ g + u + (zeta / 2.0) * (2.0 * D - 1.0)
        if heteroskedastic:
            logits = logits + rng.normal(0.0, scale, size=n)  # eps_p, Eq 21
        p = expit(logits)
        T = rng.binomial(1, p).astype(float)
    else:
        raise ValueError(f"unknown kind: {kind}")

    gy = rng.normal(0.0, 1.0, size=px)
    y = x @ gy + tau * T + rng.normal(0.0, scale, size=n) + u

    cols = {f"x{j}": x[:, j] for j in range(px)}
    df = pd.DataFrame(cols)
    df["T"] = T
    df["y"] = y
    spec = DatasetSpec(
        treatment="T",
        outcome="y",
        forcing=[f"x{j}" for j in range(pz)],
        covariates=[f"x{j}" for j in range(px)],
    )
    return Dataset(df, spec), D


def inject_label_noise(T: np.ndarray, rho: float, rng: np.random.Generator) -> np.ndarray:
    """KDD fuzzification protocol: return T_rho with P(T_rho = T) = rho exactly.

    Each binary label is flipped independently with probability 1 - rho, so
    rho = 1 recovers T and rho = 0.5 destroys all signal. Never mutates the input.
    """
    arr = np.asarray(T)
    if not 0.5 <= rho <= 1.0:
        raise ValueError(f"rho must lie in [0.5, 1], got {rho}")
    if not np.isin(arr, (0, 1)).all():
        raise ValueError("inject_label_noise requires binary T with values in {0, 1}")
    out = arr.copy()
    flip = rng.random(out.shape) < (1.0 - rho)
    out[flip] = 1 - out[flip]
    return out
