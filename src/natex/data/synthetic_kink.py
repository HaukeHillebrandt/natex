"""Seeded synthetic data for regression-kink and difference-in-kinks designs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from natex.data.spec import Dataset, DatasetSpec


@dataclass(frozen=True)
class RKDTruth:
    tau: float
    cutoff: float
    policy_intercept: float
    policy_left_slope: float
    policy_kink: float
    bias_kink: float
    expected_rkd: float
    fuzzy: bool


@dataclass(frozen=True)
class DiKTruth:
    tau: float
    cutoff: float
    policy_kink_pre: float
    policy_kink_post: float
    policy_kink_change: float
    policy_left_slope_pre: float
    policy_left_slope_post: float
    bias_kink_pre: float
    bias_kink_post: float
    expected_dik: float
    expected_rkd_post: float
    fuzzy: bool


def _require_rng(rng: np.random.Generator | None) -> np.random.Generator:
    if rng is None:
        raise ValueError("pass an explicit numpy Generator (reproducibility contract)")
    if not isinstance(rng, np.random.Generator):
        raise TypeError(f"rng must be a numpy Generator, got {type(rng).__name__}")
    return rng


def _require_finite(**values: float) -> None:
    for name, value in values.items():
        try:
            finite = np.isfinite(value)
        except TypeError:
            finite = False
        if not isinstance(finite, (bool, np.bool_)) or not bool(finite):
            raise ValueError(f"{name} must be finite")


def _validate_common(
    n: int,
    treatment_noise: float,
    outcome_noise: float,
) -> None:
    if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n < 20:
        raise ValueError("n must be an integer >= 20")
    if not np.isfinite(treatment_noise) or treatment_noise < 0.0:
        raise ValueError("treatment_noise must be finite and non-negative")
    if not np.isfinite(outcome_noise) or outcome_noise < 0.0:
        raise ValueError("outcome_noise must be finite and non-negative")


def _schedule(
    x: np.ndarray,
    intercept: float,
    left_slope: float | np.ndarray,
    kink: float | np.ndarray,
) -> np.ndarray:
    return intercept + left_slope * x + kink * np.maximum(x, 0.0)


def make_rkd_synthetic(
    n: int = 2000,
    *,
    tau: float = 2.0,
    cutoff: float = 0.0,
    policy_intercept: float = 0.5,
    policy_left_slope: float = 0.3,
    policy_kink: float = 0.8,
    bias_kink: float = 0.0,
    outcome_left_slope: float = 0.4,
    level_jump: float = 0.0,
    fuzzy: bool = False,
    treatment_noise: float = 0.35,
    outcome_noise: float = 0.5,
    rng: np.random.Generator | None = None,
) -> tuple[Dataset, RKDTruth]:
    """Draw one known-cutoff RKD repeated cross-section.

    ``bias_kink`` is a violation oracle: the population RKD ratio is
    ``tau + bias_kink / policy_kink``.  ``level_jump`` is absorbed by the
    side-specific intercepts and does not alter the slope estimand.
    """
    rng = _require_rng(rng)
    _require_finite(
        tau=tau,
        cutoff=cutoff,
        policy_intercept=policy_intercept,
        policy_left_slope=policy_left_slope,
        policy_kink=policy_kink,
        bias_kink=bias_kink,
        outcome_left_slope=outcome_left_slope,
        level_jump=level_jump,
        treatment_noise=treatment_noise,
        outcome_noise=outcome_noise,
    )
    _validate_common(n, treatment_noise, outcome_noise)
    if policy_kink == 0.0:
        raise ValueError("policy_kink must be nonzero")
    x = rng.uniform(-1.0, 1.0, size=n)
    policy = _schedule(x, policy_intercept, policy_left_slope, policy_kink)
    if fuzzy:
        policy = policy + rng.normal(0.0, treatment_noise, size=n)
    y = (
        1.0
        + outcome_left_slope * x
        + bias_kink * np.maximum(x, 0.0)
        + level_jump * (x >= 0.0)
        + tau * policy
        + rng.normal(0.0, outcome_noise, size=n)
    )
    df = pd.DataFrame({"running": cutoff + x, "policy": policy, "y": y})
    spec = DatasetSpec(
        treatment="policy",
        outcome="y",
        forcing=["running"],
        covariates=["running"],
    )
    truth = RKDTruth(
        tau=float(tau),
        cutoff=float(cutoff),
        policy_intercept=float(policy_intercept),
        policy_left_slope=float(policy_left_slope),
        policy_kink=float(policy_kink),
        bias_kink=float(bias_kink),
        expected_rkd=float(tau + bias_kink / policy_kink),
        fuzzy=bool(fuzzy),
    )
    return Dataset(df, spec), truth


def make_dik_synthetic(
    n: int = 4000,
    *,
    tau: float = 2.0,
    cutoff: float = 0.0,
    policy_intercept: float = 0.5,
    policy_left_slope_pre: float = 0.4,
    policy_left_slope_post: float = -0.1,
    policy_kink_pre: float = 0.2,
    policy_kink_post: float = 1.0,
    bias_kink_pre: float = 0.6,
    bias_kink_post: float = 0.6,
    outcome_left_slope_pre: float = -0.2,
    outcome_left_slope_post: float = 0.3,
    level_jump: float = 0.0,
    fuzzy: bool = False,
    treatment_noise: float = 0.35,
    outcome_noise: float = 0.5,
    rng: np.random.Generator | None = None,
) -> tuple[Dataset, DiKTruth]:
    """Draw a balanced two-period DiK repeated cross-section.

    Both policy slopes move by default, so there is no clean control side.
    A stable non-policy kink (``bias_kink_pre == bias_kink_post``) biases each
    cross-sectional RKD but cancels from the DiK ratio.  Different values give
    the exact parallel-kink-trends violation oracle stored in ``expected_dik``.
    """
    rng = _require_rng(rng)
    _require_finite(
        tau=tau,
        cutoff=cutoff,
        policy_intercept=policy_intercept,
        policy_left_slope_pre=policy_left_slope_pre,
        policy_left_slope_post=policy_left_slope_post,
        policy_kink_pre=policy_kink_pre,
        policy_kink_post=policy_kink_post,
        bias_kink_pre=bias_kink_pre,
        bias_kink_post=bias_kink_post,
        outcome_left_slope_pre=outcome_left_slope_pre,
        outcome_left_slope_post=outcome_left_slope_post,
        level_jump=level_jump,
        treatment_noise=treatment_noise,
        outcome_noise=outcome_noise,
    )
    _validate_common(n, treatment_noise, outcome_noise)
    if n % 2:
        raise ValueError("n must be even so pre and post samples are balanced")
    kink_change = policy_kink_post - policy_kink_pre
    if not np.isfinite(kink_change) or kink_change == 0.0:
        raise ValueError("policy kink change must be finite and nonzero")
    n_period = n // 2
    post = np.repeat([False, True], n_period)
    x = rng.uniform(-1.0, 1.0, size=n)
    permutation = rng.permutation(n)
    post = post[permutation]
    x = x[permutation]
    left_slope = np.where(post, policy_left_slope_post, policy_left_slope_pre)
    kink = np.where(post, policy_kink_post, policy_kink_pre)
    policy = _schedule(x, policy_intercept, left_slope, kink)
    if fuzzy:
        policy = policy + rng.normal(0.0, treatment_noise, size=n)
    bias_kink = np.where(post, bias_kink_post, bias_kink_pre)
    outcome_left_slope = np.where(
        post, outcome_left_slope_post, outcome_left_slope_pre
    )
    y = (
        1.0
        + post.astype(float)
        + outcome_left_slope * x
        + bias_kink * np.maximum(x, 0.0)
        + level_jump * (x >= 0.0)
        + tau * policy
        + rng.normal(0.0, outcome_noise, size=n)
    )
    df = pd.DataFrame(
        {
            "running": cutoff + x,
            "post": post.astype(np.int64),
            "policy": policy,
            "y": y,
        }
    )
    spec = DatasetSpec(
        treatment="policy",
        outcome="y",
        forcing=["running"],
        covariates=["running"],
        time="post",
    )
    truth = DiKTruth(
        tau=float(tau),
        cutoff=float(cutoff),
        policy_kink_pre=float(policy_kink_pre),
        policy_kink_post=float(policy_kink_post),
        policy_kink_change=float(kink_change),
        policy_left_slope_pre=float(policy_left_slope_pre),
        policy_left_slope_post=float(policy_left_slope_post),
        bias_kink_pre=float(bias_kink_pre),
        bias_kink_post=float(bias_kink_post),
        expected_dik=float(tau + (bias_kink_post - bias_kink_pre) / kink_change),
        expected_rkd_post=(
            float(tau + bias_kink_post / policy_kink_post)
            if policy_kink_post != 0.0
            else float("nan")
        ),
        fuzzy=bool(fuzzy),
    )
    return Dataset(df, spec), truth
