"""Synthetic benchmark DGPs from KDD 2018 Eqs 19-21, with the audit corrections:
binary treatment uses an additive LOG-ODDS shift (the printed '+mu' is a typo)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import expit

from natex.data.spec import Dataset, DatasetSpec


def make_synthetic(
    n: int,
    px: int = 2,
    pz: int = 2,
    zeta: float = 3.0,
    tau: float = 2.0,
    kind: str = "binary",
    discont: str = "square",
    rng: np.random.Generator | None = None,
) -> tuple[Dataset, np.ndarray]:
    if rng is None:
        raise ValueError("pass an explicit numpy Generator (reproducibility contract)")
    if pz > px:
        raise ValueError("pz cannot exceed px")
    x = rng.uniform(0.0, 1.0, size=(n, px))
    u = rng.normal(0.0, 0.5, size=n)
    if discont != "square":
        raise ValueError("phase 1 implements the 'square' (corner) region only")
    D = np.all(x[:, :pz] >= 0.5, axis=1)

    g = rng.normal(0.0, 1.0, size=px)
    if kind == "real":
        T = x @ g + zeta * D + rng.normal(0.0, 0.5, size=n) + u
    elif kind == "binary":
        p = expit(x @ g + u + (zeta / 2.0) * (2.0 * D - 1.0))
        T = rng.binomial(1, p).astype(float)
    else:
        raise ValueError(f"unknown kind: {kind}")

    gy = rng.normal(0.0, 1.0, size=px)
    y = x @ gy + tau * T + rng.normal(0.0, 0.5, size=n) + u

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
