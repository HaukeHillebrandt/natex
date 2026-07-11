"""Factor-model synthetic-control DGP with known donor support (phase 5).

Panel factor model (Abadie, Diamond & Hainmueller 2010 setting):

    y_ut = mu_u + lambda_u @ f_t + noise * eps_ut

with unit intercepts ``mu_u ~ N(0, 5^2)``, factor loadings
``lambda_u ~ N(0, I_{n_factors})`` and RANDOM-WALK factors
``f_t = cumsum(N(0, I))`` — persistent trends, the setting where
pre-trend matching is informative. The treated unit's ``(mu, lambda)`` is
an EXACT convex combination — a uniform random simplex (Dirichlet(1,..,1))
draw over ``k_true`` donors chosen without replacement — of those donors'
parameters, so the noiseless treated trajectory is spanned by the donor
pool and the synthetic-control estimand is well posed.

Identifiable donor support (documented design choice): the chosen true
donors are relocated into a tight cluster at the EDGE of the donor cloud
(cluster center ``mu = 12`` = 2.4 donor sd, loadings ``1.5`` = 1.5 sd;
within-cluster deviations are the donors' original deviations shrunk by
0.4 — no extra rng draws). Two failure modes of a fully exchangeable
design force this: (i) a convex combination is typically as CLOSE to
random non-members as to its own vertices (squared-distance ratio ~2), so
top-k pre-RMSE screening cannot recover the support at any noise level;
(ii) an interior treated unit is reconstructable by many mixtures of
far-apart donors, so simplex weights are non-unique and leak off the true
support. An extreme treated unit is the Abadie-Diamond-Hainmueller
California setting, where SC weights are identified (calibration tables
in ``tests/test_donors.py``).

Post-``t0`` the treated unit receives a constant additive ``effect``;
times are ``0, ..., n_pre + n_post - 1`` with ``t0 = n_pre``.

Reproducibility contract (repo convention): one explicit
``numpy.random.Generator`` drives every draw; omitting it raises
ValueError. Draw order is fixed (mu, lambda, f, donor choice, Dirichlet
weights, eps) regardless of parameter values.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class SCSyntheticData:
    """One synthetic SC panel plus its ground truth."""

    df: pd.DataFrame  # long form: unit, time, y (treated unit named "treated")
    units: list[str]
    treated_unit: str
    t0: float
    true_donors: list[str]  # donors carrying the true convex weights
    true_weights: np.ndarray  # aligned with true_donors
    effect: float  # constant additive post-period effect on the treated unit


def make_sc_synthetic(
    n_units: int = 20,
    n_pre: int = 15,
    n_post: int = 10,
    n_factors: int = 2,
    k_true: int = 3,
    effect: float = 10.0,
    noise: float = 0.5,
    rng: np.random.Generator | None = None,
) -> SCSyntheticData:
    """Draw one factor-model SC panel; see the module docstring."""
    if rng is None:
        raise ValueError("pass an explicit numpy Generator (reproducibility contract)")
    if not isinstance(rng, np.random.Generator):
        raise TypeError(f"rng must be a numpy Generator, got {type(rng).__name__}")
    if n_units < 2:
        raise ValueError(f"n_units must be >= 2 (one treated + one donor), got {n_units}")
    if not 1 <= k_true <= n_units - 1:
        raise ValueError(f"need 1 <= k_true <= n_units - 1 donors, got k_true={k_true}, n_units={n_units}")
    if n_pre < 1:
        raise ValueError(f"n_pre must be >= 1, got {n_pre}")
    if n_post < 0:
        raise ValueError(f"n_post must be >= 0, got {n_post}")
    if n_factors < 1:
        raise ValueError(f"n_factors must be >= 1, got {n_factors}")
    if noise < 0:
        raise ValueError(f"noise must be >= 0, got {noise}")

    n_donors = n_units - 1
    n_t = n_pre + n_post
    donor_names = [f"unit{j:02d}" for j in range(1, n_donors + 1)]
    treated_unit = "treated"
    units = donor_names + [treated_unit]
    times = np.arange(n_t, dtype=float)
    t0 = float(n_pre)

    mu = rng.normal(0.0, 5.0, size=n_donors)
    lam = rng.normal(0.0, 1.0, size=(n_donors, n_factors))
    f = np.cumsum(rng.normal(0.0, 1.0, size=(n_factors, n_t)), axis=1)  # random walk
    idx = rng.choice(n_donors, size=k_true, replace=False)
    w = rng.dirichlet(np.ones(k_true))  # uniform random simplex draw
    eps = rng.standard_normal((n_units, n_t))

    # Identifiable donor support: relocate the true donors into a tight
    # cluster at the edge of the donor cloud (see the module docstring).
    # No rng draws consumed.
    shrink = 0.4
    mu[idx] = 12.0 + shrink * (mu[idx] - mu[idx].mean())
    lam[idx] = 1.5 + shrink * (lam[idx] - lam[idx].mean(axis=0))

    mu_treated = float(w @ mu[idx])
    lam_treated = w @ lam[idx]  # (n_factors,)

    base = np.empty((n_units, n_t))
    base[:n_donors] = mu[:, None] + lam @ f
    base[n_donors] = mu_treated + lam_treated @ f
    y = base + noise * eps
    y[n_donors, times >= t0] += effect

    df = pd.DataFrame(
        {
            "unit": np.repeat(np.asarray(units, dtype=object), n_t),
            "time": np.tile(times, n_units),
            "y": y.ravel(),
        }
    )
    return SCSyntheticData(
        df=df,
        units=units,
        treated_unit=treated_unit,
        t0=t0,
        true_donors=[donor_names[i] for i in idx],
        true_weights=np.asarray(w, dtype=float),
        effect=float(effect),
    )
