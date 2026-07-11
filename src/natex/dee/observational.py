"""DEE conditioned-on-observables CATE layer with leave-experiment-out cross-fitting.

Audit 9 (docs/math_audit_final.md): fitting the observational estimator on the
same rows that produced the local-IV effects correlates the two stages, so the
bias observation ``cate_obs(center_u) - tau_hat_u`` is contaminated. Repair:
``experiment_crossfit_cate`` assigns experiments to folds and predicts each
experiment's centroid from a model whose training rows exclude EVERY fold-mate
experiment's members -- in particular its own.

Conventions (phase-4 plan):

- Features are ``dataset.Z_std`` (the scan's space); query points are the
  experiments' ``projected_center`` coordinates, already in Z_std.
- The core-deps default is a sklearn T-learner (two GradientBoostingRegressors,
  CATE = mu1_hat - mu0_hat); it requires binary treatment. Continuous or
  multi-valued treatments need the econml ``CausalForestDML`` adapter behind
  the ``natex[ml]`` extra.
- Determinism: estimator seeds derive from the caller's Generator via
  ``int(rng.integers(2**32))``; fold assignment uses the same Generator.
- NaN policy: an underdetermined arm (< min_treated rows after dropping
  non-finite outcomes) yields NaN predictions, never 0.0. Rows with non-finite
  outcomes are excluded from training (Dataset keeps them for discovery).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

from natex.data.spec import Dataset
from natex.dee.gp import _require_rng
from natex.dee.vknn import VKNNResult


@runtime_checkable
class ObservationalEstimator(Protocol):
    """Anything that fits (X, T, y) and predicts a CATE at query points."""

    def fit(self, X: np.ndarray, T: np.ndarray, y: np.ndarray) -> "ObservationalEstimator": ...

    def predict_cate(self, Xq: np.ndarray) -> np.ndarray: ...


@dataclass
class TLearner:
    """Core-deps T-learner: one GradientBoostingRegressor per treatment arm.

    ``seed`` pins both arms' regressors (callers derive it as
    ``int(rng.integers(2**32))``). If either arm has fewer than ``min_treated``
    rows with finite outcomes, ``predict_cate`` returns NaN (never 0.0).
    """

    seed: int
    n_estimators: int = 200
    max_depth: int = 3
    learning_rate: float = 0.05
    min_treated: int = 20  # per-arm minimum; below -> predict_cate returns NaN

    _models: tuple[GradientBoostingRegressor, GradientBoostingRegressor] | None = field(
        default=None, init=False, repr=False
    )
    _n_features: int | None = field(default=None, init=False, repr=False)

    def fit(self, X: np.ndarray, T: np.ndarray, y: np.ndarray) -> "TLearner":
        X = np.asarray(X, dtype=float)
        T = np.asarray(T, dtype=float)
        y = np.asarray(y, dtype=float)
        if X.ndim != 2:
            raise ValueError(f"X must be (n, d), got shape {X.shape}")
        if T.shape != (X.shape[0],) or y.shape != (X.shape[0],):
            raise ValueError("X, T, y must have matching first dimensions")
        vals = np.unique(T[np.isfinite(T)])
        if not set(vals.tolist()) <= {0.0, 1.0}:
            raise ValueError(
                "TLearner requires binary treatment (values in {0, 1}); for continuous or "
                "multi-valued treatments supply an econml CausalForestDML factory "
                "(natex[ml] extra)"
            )
        self._n_features = X.shape[1]
        keep = np.isfinite(y)  # never train on non-finite outcomes
        X, T, y = X[keep], T[keep], y[keep]
        treated = T == 1.0
        if int(treated.sum()) < self.min_treated or int((~treated).sum()) < self.min_treated:
            self._models = None  # underdetermined arm: NaN downstream, never 0.0
            return self
        s1, s0 = (int(s) for s in np.random.SeedSequence(self.seed).generate_state(2))
        mu1 = self._regressor(s1).fit(X[treated], y[treated])
        mu0 = self._regressor(s0).fit(X[~treated], y[~treated])
        self._models = (mu0, mu1)
        return self

    def _regressor(self, seed: int) -> GradientBoostingRegressor:
        return GradientBoostingRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            random_state=seed,
        )

    def predict_cate(self, Xq: np.ndarray) -> np.ndarray:
        if self._n_features is None:
            raise ValueError("call fit before predict_cate")
        Xq = np.asarray(Xq, dtype=float)
        if Xq.ndim != 2 or Xq.shape[1] != self._n_features:
            raise ValueError(f"Xq must be (m, {self._n_features}), got shape {Xq.shape}")
        if self._models is None:
            return np.full(Xq.shape[0], np.nan)
        mu0, mu1 = self._models
        return np.asarray(mu1.predict(Xq) - mu0.predict(Xq), dtype=float)


def default_factory(rng: np.random.Generator) -> Callable[[], ObservationalEstimator]:
    """Zero-arg factory producing independently seeded TLearners from ``rng``."""
    rng = _require_rng(rng)

    def factory() -> ObservationalEstimator:
        return TLearner(seed=int(rng.integers(2**32)))

    return factory


def _assign_folds(u: int, n_folds: int, rng: np.random.Generator) -> np.ndarray:
    """rng-assign u experiments to min(n_folds, u) balanced non-empty folds."""
    if n_folds < 1:
        raise ValueError(f"n_folds must be >= 1, got {n_folds}")
    f = min(int(n_folds), u)
    fold_of = np.empty(u, dtype=int)
    fold_of[rng.permutation(u)] = np.arange(u) % f
    return fold_of


def experiment_crossfit_cate(
    dataset: Dataset,
    result: VKNNResult,
    factory: Callable[[], ObservationalEstimator],
    rng: np.random.Generator,
    n_folds: int = 5,
) -> np.ndarray:
    """Observational CATE at each experiment's ``projected_center``, cross-fitted.

    Experiments are rng-assigned to ``min(n_folds, u)`` folds; the fold-f model
    is fit on every dataset row EXCEPT the union of fold-f experiments' members
    (audit 9: the bias observation at a center never comes from a model trained
    on that experiment's own rows). Returns a (u,) array in acceptance order;
    NaN from an underdetermined arm propagates, never 0.0.
    """
    rng = _require_rng(rng)
    y = dataset.y
    if y is None:
        raise ValueError("experiment_crossfit_cate needs a dataset with an outcome column")
    experiments = result.experiments
    u = len(experiments)
    out = np.full(u, np.nan)
    if u == 0:
        return out
    Z = dataset.Z_std
    T = dataset.T
    fold_of = _assign_folds(u, n_folds, rng)
    for f in range(int(fold_of.max()) + 1):
        in_fold = np.flatnonzero(fold_of == f)
        train = np.ones(Z.shape[0], dtype=bool)
        for j in in_fold:
            train[np.asarray(experiments[j].members, dtype=int)] = False
        model = factory().fit(Z[train], T[train], y[train])
        centers = np.stack(
            [np.asarray(experiments[j].projected_center, dtype=float) for j in in_fold]
        )
        out[in_fold] = model.predict_cate(centers)
    return out
