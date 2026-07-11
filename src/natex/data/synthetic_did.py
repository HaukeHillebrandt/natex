"""Ch. 6 synthetic DiD generator (thesis Eqs 6.22-6.28) with the audit's repairs.

Data-generating process (docs/math_audit_final.md, "pure typos" section):

* **Eq 6.22** — covariates and the unobserved confounder are discrete uniform:
  ``x_ij, u_i ~ DiscreteUniform(1..V)``; ``t_i ~ DiscreteUniform`` over
  ``periods`` integer times ``1..periods``.
* **Eq 6.23** — heteroskedastic noise ``eps ~ N(0, mean_j x_ij)``; the second
  parameter is a VARIANCE, so the noise scale is ``sqrt(mean_j x_ij)``.
* **Eqs 6.24-6.25 (Codex #23, dimensionally invalid as printed)** — the thesis
  draws per-(dimension, value) coefficients ``gamma ~ N(0, I_{d x V})`` yet
  multiplies ``x_i * gamma`` as if gamma were length-d. Repair: coefficients
  are PER VALUE and enter as lookups, never an inner product with the codes:
  ``theta_i = sum_j gamma_theta[j, x_ij] + zeta * 1[x_i in s_I] * 1[t_i >= T0]
  + eps_theta_i + u_i``.
* **Eq 6.26** — ``y_i = sum_j gamma_y[j, x_ij] + tau * theta_i + eps_y_i + u_i``
  (same per-value repair for gamma_y).
* **Eqs 6.27-6.28 (Codex #24/#25)** — ``hetero_group=True`` with the default
  ``hetero_kind="scaled_noise"`` replaces the y noise by the printed
  time-scaled term ``eps_y_i * 1[x_i in s_g] * t_i`` with
  ``s_g = s_I  union  s_c`` and ``s_c`` a random subset of UNTREATED profiles
  (Eq 6.28's set notation repaired to ``s_c  subset of  D \\ s_I``). As the
  audit notes, this creates HETEROSKEDASTICITY, not the correlation the thesis
  prose claims — and because per-record mean-zero noise averages out of every
  per-time control mean, it biases NO control method and cannot reproduce the
  Fig 6.5 method ranking (calibration: 16 seed-configs, all three controls
  within ~0.3 of tau after dose normalization).

* **``hetero_kind="shock"``** — the PROSE-intent correlated variant needed
  for the Fig 6.5 analog: Eq 6.27's ``eps * 1[divergent] * t_i`` with eps
  drawn ONCE PER TIME PERIOD (``N(0, hetero_scale^2)``, shared by every
  divergent record at that time) instead of per record — within-subset
  correlation, exactly what the thesis prose claims. Two documented
  deviations from the printed equation, both forced by identifiability
  (calibration, 8 seeds x several configs): (i) the shocks hit s_c ONLY, not
  ``s_g = s_I union s_c`` — shocks shared with s_I are common to the treated
  subset while every monotone-profile-expansion control catches them only
  DILUTED (a conjunction-of-unions cover cannot isolate s_c, so no method
  can cancel them and no ranking exists). On s_c only, the variant realizes
  the Fig 6.5 failure mode: pooled controls cannot disaggregate the
  divergent subset — DD absorbs ~|s_c|/|control| of every shock, synthetic
  control overfits its few pre-period times with dozens of donors — while
  GESS's pre-MSE search excludes s_c. (ii) s_c is redrawn until AVOIDABLE by
  monotone profile expansion: at least one dimension constrained by both s_I
  and s_c has disjoint value sets — GESS can only build controls from
  profile expansions (thesis limitation), so an s_c intersecting every
  expansion contaminates every method equally and the ranking is again
  unidentifiable. The thesis base config (8-value dimensions) makes such
  collisions rare, which is why its averaged Fig 6.5 shows clean
  order-of-magnitude gaps.

``theta_kind="binary"`` is a documented natex addition (audit item 19 needs a
binary-treatment variant): the Eq 6.25 latent is thresholded at its median, so
the planted jump becomes a jump in P(theta = 1).

The intervention subset s_I picks ``s_values`` random values in each of
``s_dims`` random dimensions (thesis base config: 2 x 2); T0 is drawn from the
middle half of the time range so both sides always hold data.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from natex.data.spec import Dataset, DatasetSpec

_MAX_SC_DRAWS = 100


@dataclass
class DiDTruth:
    """Ground truth of one synthetic draw.

    ``included[j]`` is a length-V boolean mask over the value grid ``1..V``
    (index ``v - 1``) — all-True for unconstrained dimensions, matching the
    conjunction-of-unions subset convention. ``hetero_mask`` is the divergent
    record mask of the Eq 6.27 variant (``None`` unless ``hetero_group=True``):
    s_g = s_I union s_c for ``hetero_kind="scaled_noise"``, s_c alone for
    ``"shock"`` (whose shared per-period draws are in ``hetero_shocks``).
    """

    included: list[np.ndarray]  # per-dim value masks of s_I
    record_mask: np.ndarray  # (n,) treated-subset membership
    t0: float
    zeta: float
    tau: float
    hetero_mask: np.ndarray | None = None  # (n,) divergent records (hetero_group only)
    hetero_shocks: np.ndarray | None = None  # (periods,) shared shocks ("shock" only)


def _random_subset_masks(
    d: int, V: int, s_dims: int, s_values: int, rng: np.random.Generator
) -> list[np.ndarray]:
    """Per-dim value masks: s_values random values in each of s_dims random dims."""
    included = [np.ones(V, dtype=bool) for _ in range(d)]
    for j in rng.choice(d, size=s_dims, replace=False):
        mask = np.zeros(V, dtype=bool)
        mask[rng.choice(V, size=s_values, replace=False)] = True
        included[int(j)] = mask
    return included


def _subset_record_mask(x: np.ndarray, included: list[np.ndarray]) -> np.ndarray:
    """(n,) bool: conjunction over dims of value membership (x values 1..V)."""
    mask = np.ones(x.shape[0], dtype=bool)
    for j, inc in enumerate(included):
        mask &= inc[x[:, j] - 1]
    return mask


def _avoidable(inc_i: list[np.ndarray], inc_c: list[np.ndarray]) -> bool:
    """True iff some dim constrained by BOTH s_I and s_c has disjoint values.

    Any monotone profile expansion of s_I that keeps that dimension's values
    then excludes s_c entirely, so a GESS-style control CAN avoid the
    divergent subset (module docstring, ``hetero_kind="shock"``).
    """
    return any(
        not a.all() and not c.all() and not (a & c).any()
        for a, c in zip(inc_i, inc_c, strict=True)
    )


def make_did_synthetic(
    n: int = 2000,
    d: int = 4,
    V: int = 8,
    periods: int = 10,
    zeta: float = 10.0,
    tau: float = 10.0,
    s_dims: int = 2,
    s_values: int = 2,
    theta_kind: str = "real",
    hetero_group: bool = False,
    hetero_kind: str = "scaled_noise",
    hetero_scale: float = 8.0,
    rng: np.random.Generator | None = None,
) -> tuple[Dataset, DiDTruth]:
    """Draw one ch.6 synthetic panel; returns ``(Dataset, DiDTruth)``.

    The returned Dataset has ``time="t"``, ``unit=None``, ``forcing=[]``,
    ``covariates=[x0..x{d-1}]`` (integer values on the grid ``1..V``),
    treatment ``"theta"`` and outcome ``"y"``. Note the covariates are
    d discrete values each — pass ``bins >= V`` to :func:`natex.did.suddds
    .suddds_scan` so they stay categorical instead of being quantile-merged.
    """
    if rng is None:
        raise ValueError("pass an explicit numpy Generator (reproducibility contract)")
    if not isinstance(rng, np.random.Generator):
        raise TypeError(f"rng must be a numpy Generator, got {type(rng).__name__}")
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    if d < 1:
        raise ValueError(f"d must be >= 1, got {d}")
    if V < 2:
        raise ValueError(f"V must be >= 2, got {V}")
    if periods < 3:
        raise ValueError(f"periods must be >= 3 (pre data, post data, middle T0), got {periods}")
    if not 1 <= s_dims <= d:
        raise ValueError(f"s_dims must lie in 1..d={d}, got {s_dims}")
    if not 1 <= s_values <= V:
        raise ValueError(f"s_values must lie in 1..V={V}, got {s_values}")
    if theta_kind not in ("real", "binary"):
        raise ValueError(f"theta_kind must be 'real' or 'binary', got {theta_kind!r}")
    if hetero_kind not in ("scaled_noise", "shock"):
        raise ValueError(f"hetero_kind must be 'scaled_noise' or 'shock', got {hetero_kind!r}")
    if not hetero_scale > 0.0:
        raise ValueError(f"hetero_scale must be > 0, got {hetero_scale}")

    # Eq 6.22: covariates, confounder and time, all discrete uniform.
    x = rng.integers(1, V + 1, size=(n, d))
    u = rng.integers(1, V + 1, size=n).astype(float)
    t = rng.integers(1, periods + 1, size=n).astype(float)

    # T0 from the middle half of the time range (both sides always hold data).
    times = np.arange(1, periods + 1)
    lo = max(periods // 4, 1)
    hi = max(periods - periods // 4, lo + 1)
    T0 = float(rng.choice(times[lo:hi]))
    post = t >= T0

    # s_I: s_values random values in each of s_dims random dimensions.
    included = _random_subset_masks(d, V, s_dims, s_values, rng)
    record_mask = _subset_record_mask(x, included)

    # Eq 6.23: eps ~ N(0, mean_j x_ij) — the second parameter is a variance.
    scale = np.sqrt(x.mean(axis=1))

    # Eqs 6.24-6.25 (repaired): per-value coefficient lookups.
    gamma_theta = rng.normal(0.0, 1.0, size=(d, V))
    theta = gamma_theta[np.arange(d)[None, :], x - 1].sum(axis=1)
    theta = theta + zeta * (record_mask & post) + rng.normal(0.0, 1.0, size=n) * scale + u
    if theta_kind == "binary":
        # Documented addition: threshold the Eq 6.25 latent at its median so
        # the planted jump becomes a jump in P(theta = 1).
        theta = (theta >= np.median(theta)).astype(float)

    # Eq 6.26 (same per-value repair); Eq 6.27 swaps in time-scaled noise on s_g.
    gamma_y = rng.normal(0.0, 1.0, size=(d, V))
    y_base = gamma_y[np.arange(d)[None, :], x - 1].sum(axis=1)
    eps_y = rng.normal(0.0, 1.0, size=n) * scale
    hetero_mask: np.ndarray | None = None
    hetero_shocks: np.ndarray | None = None
    if hetero_group:
        # Eq 6.28 (repaired set notation): s_c is a random subset of UNTREATED
        # profiles, drawn like s_I and redrawn until it contributes records
        # outside s_I; hetero_kind="shock" additionally requires s_c to be
        # AVOIDABLE by monotone profile expansion (see module docstring).
        for _ in range(_MAX_SC_DRAWS):
            sc_included = _random_subset_masks(d, V, s_dims, s_values, rng)
            sc_mask = _subset_record_mask(x, sc_included) & ~record_mask
            if sc_mask.any() and (hetero_kind != "shock" or _avoidable(included, sc_included)):
                break
        else:
            raise RuntimeError(
                f"no admissible untreated subset s_c found in {_MAX_SC_DRAWS} draws; "
                "lower s_dims/s_values or raise n"
            )
        if hetero_kind == "shock":
            # Prose-intent correlated variant: one shared draw PER TIME PERIOD,
            # time-scaled, on the s_c records only (module docstring).
            hetero_shocks = rng.normal(0.0, hetero_scale, size=periods)
            hetero_mask = sc_mask
            eps_y = eps_y + hetero_shocks[t.astype(np.int64) - 1] * t * sc_mask
        else:
            hetero_mask = record_mask | sc_mask
            # As printed (Codex #24): heteroskedastic time-scaled noise, no
            # noise outside s_g — heteroskedasticity, not the claimed
            # correlation.
            eps_y = eps_y * (hetero_mask * t)
    y = y_base + tau * theta + eps_y + u

    df = pd.DataFrame({f"x{j}": x[:, j] for j in range(d)})
    df["t"] = t
    df["theta"] = theta
    df["y"] = y
    spec = DatasetSpec(
        treatment="theta",
        outcome="y",
        forcing=[],
        covariates=[f"x{j}" for j in range(d)],
        time="t",
        unit=None,
    )
    truth = DiDTruth(
        included=included,
        record_mask=record_mask,
        t0=T0,
        zeta=float(zeta),
        tau=float(tau),
        hetero_mask=hetero_mask,
        hetero_shocks=hetero_shocks,
    )
    return Dataset(df, spec), truth
