"""Sparse-first-stage IV synthetic DGP (BCCH exponential design).

Belloni, Chen, Chernozhukov & Hansen (2012, Econometrica) benchmark sparse
instrument selection on an "exponential design" first stage: coefficients
``pi_j`` proportional to ``0.7**(j-1)`` on the first ``s`` of ``p`` candidate
instruments and zero elsewhere, rescaled so the POPULATION concentration
parameter

    mu^2 = n * pi' Sigma pi / sigma_v^2        (sigma_v^2 = 1 here)

hits a chosen target ``mu2`` (BCCH use mu^2 in {30, 180}). Instruments are
jointly normal with Toeplitz correlation ``Sigma_jk = rho_z**|j-k|``; the
first-stage error ``v`` and the structural error ``e`` are bivariate normal
with unit variances and correlation ``endog``, so the OLS plim bias of the
naive y-on-T regression is ``endog / (pi' Sigma pi + 1)`` — tunable
endogeneity.

Exclusion violators are plantable: the LAST ``n_invalid`` pool columns enter
the outcome directly with coefficient ``phi``. They carry no first-stage
signal (``pi = 0`` off support), so they violate exclusion, not relevance;
``s + n_invalid <= p`` is enforced so violators never overlap
``true_support``.

``IVSyntheticData.concentration`` is the REALIZED sample analog
``pi' Z'Z pi / var_hat(v)`` (ddof=1). It fluctuates around the targeted
population ``mu2`` with relative sd ~ ``sqrt(2/n)`` per factor — the
fidelity test checks it lands within 25% of the target, which catches
rescaling bugs (e.g. dropping Sigma from the quadratic form roughly doubles
it at ``rho_z = 0.5``).

Reproducibility contract (repo convention): one explicit
``numpy.random.Generator`` drives every draw; omitting it raises ValueError.
The rng stream is consumed identically regardless of parameter values
(Z, then v, then the independent component of e).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class IVSyntheticData:
    """One synthetic IV draw plus its ground truth."""

    df: pd.DataFrame  # columns z1..zp, T, y
    pool_names: list[str]  # ["z1", ..., "zp"]
    true_support: list[str]  # the s relevant instruments (first s columns)
    invalid_names: list[str]  # exclusion violators (last n_invalid columns)
    tau: float
    pi: np.ndarray  # (p,) true first-stage coefficients
    concentration: float  # realized pi' Z'Z pi / var_hat(v)


def make_iv_synthetic(
    n: int = 500,
    p: int = 50,
    s: int = 5,
    mu2: float = 180.0,
    rho_z: float = 0.5,
    endog: float = 0.6,
    tau: float = 1.0,
    n_invalid: int = 0,
    phi: float = 0.5,
    rng: np.random.Generator | None = None,
) -> IVSyntheticData:
    """Draw one BCCH exponential-design IV dataset; see the module docstring.

    ``mu2 = 0`` is allowed and yields ``pi = 0`` — a pure-noise pool for
    honest "no instrument exists" tests.
    """
    if rng is None:
        raise ValueError("pass an explicit numpy Generator (reproducibility contract)")
    if not isinstance(rng, np.random.Generator):
        raise TypeError(f"rng must be a numpy Generator, got {type(rng).__name__}")
    if n < 2:
        raise ValueError(f"n must be >= 2, got {n}")
    if p < 1:
        raise ValueError(f"p must be >= 1, got {p}")
    if not 1 <= s <= p:
        raise ValueError(f"need 1 <= s <= p, got s={s}, p={p}")
    if n_invalid < 0 or s + n_invalid > p:
        raise ValueError(
            f"need 0 <= n_invalid and s + n_invalid <= p, got s={s}, n_invalid={n_invalid}, p={p}"
        )
    if mu2 < 0:
        raise ValueError(f"mu2 must be >= 0, got {mu2}")
    if not abs(rho_z) < 1:
        raise ValueError(f"|rho_z| must be < 1, got {rho_z}")
    if not abs(endog) <= 1:
        raise ValueError(f"|endog| must be <= 1, got {endog}")

    idx = np.arange(p)
    sigma = rho_z ** np.abs(np.subtract.outer(idx, idx))  # Toeplitz AR(1), PD for |rho_z| < 1
    z = rng.standard_normal((n, p)) @ np.linalg.cholesky(sigma).T

    pi_shape = 0.7 ** np.arange(s)  # BCCH exponential design
    quad = float(pi_shape @ sigma[:s, :s] @ pi_shape)  # pi_shape' Sigma_ss pi_shape > 0
    pi = np.zeros(p)
    pi[:s] = pi_shape * np.sqrt(mu2 / (n * quad))  # population n pi' Sigma pi = mu2 (sigma_v = 1)

    v = rng.standard_normal(n)
    e = endog * v + np.sqrt(1.0 - endog**2) * rng.standard_normal(n)
    t_vec = z @ pi + v
    y = tau * t_vec + e
    if n_invalid:
        y = y + phi * z[:, p - n_invalid :].sum(axis=1)

    fitted = z @ pi
    concentration = float(fitted @ fitted / np.var(v, ddof=1))

    names = [f"z{j}" for j in range(1, p + 1)]
    df = pd.DataFrame(z, columns=names)
    df["T"] = t_vec
    df["y"] = y
    return IVSyntheticData(
        df=df,
        pool_names=names,
        true_support=names[:s],
        invalid_names=names[p - n_invalid :] if n_invalid else [],
        tau=float(tau),
        pi=pi,
        concentration=concentration,
    )
