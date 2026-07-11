"""econml CausalForestDML adapter for the DEE observational layer (natex[ml]).

Optional-extra counterpart of the core-deps T-learner in
``dee/observational.py``: implements the same ``ObservationalEstimator``
protocol but supports **continuous and multi-valued treatments** as well as
binary ones (the TLearner gap -- it refuses non-binary T with a pointer here).
econml is imported inside :meth:`CausalForestEstimator.fit`, so this module
always imports on core deps; calling ``fit`` without econml raises an
``ImportError`` naming ``pip install "natex-discovery[ml]"``.

Conventions inherited from the observational layer (phase-4 plan, audit 9):

- Features are ``Z_std``; the experiment-level cross-fitting in
  ``experiment_crossfit_cate`` accepts a zero-arg factory of these estimators
  exactly like the TLearner (derive seeds via ``int(rng.integers(2**32))``).
- Determinism: ``seed`` pins econml's ``random_state`` and ``n_jobs``
  defaults to 1 -- parallel tree reductions reorder float sums (~1e-15
  run-to-run drift), so identical seed and data => bitwise identical
  predictions only serially. Pass ``n_jobs=-1`` to trade that for speed.
- econml requires ``n_estimators`` divisible by its ``subforest_size``
  (default 4); econml raises otherwise.
- NaN policy: an underdetermined fit (a binary arm below ``min_treated`` rows
  with finite outcomes, or -- continuous case -- fewer than ``2 * min_treated``
  usable rows or zero treatment variance) yields NaN predictions, never 0.0.
  Rows with non-finite outcomes or treatments are excluded from training.
"""

from __future__ import annotations

from typing import Any

import numpy as np


class CausalForestEstimator:
    """``ObservationalEstimator`` backed by ``econml.dml.CausalForestDML``.

    Parameters
    ----------
    seed : pins econml's ``random_state`` (callers derive it from the pipeline
        Generator as ``int(rng.integers(2**32))``).
    n_estimators : forest size (paper-scale default 1000; tests use fewer).
    min_treated : per-arm minimum for binary T (mirrors ``TLearner``); for
        continuous T at least ``2 * min_treated`` usable rows are required.
    **cf_kwargs : forwarded verbatim to ``CausalForestDML`` (e.g. ``model_y``,
        ``model_t``, ``cv``, ``criterion``).
    """

    def __init__(
        self, seed: int, n_estimators: int = 1000, min_treated: int = 20, **cf_kwargs: Any
    ) -> None:
        self.seed = int(seed)
        self.n_estimators = int(n_estimators)
        self.min_treated = int(min_treated)
        self.cf_kwargs = cf_kwargs
        self._model: Any | None = None
        self._n_features: int | None = None

    def fit(self, X: np.ndarray, T: np.ndarray, y: np.ndarray) -> "CausalForestEstimator":
        try:
            from econml.dml import CausalForestDML
        except ImportError:
            raise ImportError(
                "CausalForestEstimator needs econml; install the optional extra: "
                'pip install "natex-discovery[ml]"'
            ) from None
        X = np.asarray(X, dtype=float)
        T = np.asarray(T, dtype=float)
        y = np.asarray(y, dtype=float)
        if X.ndim != 2:
            raise ValueError(f"X must be (n, d), got shape {X.shape}")
        if T.shape != (X.shape[0],) or y.shape != (X.shape[0],):
            raise ValueError("X, T, y must have matching first dimensions")
        self._n_features = X.shape[1]
        keep = np.isfinite(y) & np.isfinite(T)  # never train on non-finite rows
        X, T, y = X[keep], T[keep], y[keep]
        discrete = set(np.unique(T).tolist()) <= {0.0, 1.0}
        if discrete:
            treated = T == 1.0
            underdetermined = (
                int(treated.sum()) < self.min_treated
                or int((~treated).sum()) < self.min_treated
            )
        else:
            underdetermined = T.shape[0] < 2 * self.min_treated or float(np.var(T)) == 0.0
        if underdetermined:
            self._model = None  # NaN downstream, never 0.0
            return self
        model = CausalForestDML(
            n_estimators=self.n_estimators,
            discrete_treatment=discrete,
            random_state=self.seed,
            **{"n_jobs": 1, **self.cf_kwargs},  # serial by default: bitwise determinism
        )
        model.fit(y, T, X=X)
        self._model = model
        return self

    def predict_cate(self, Xq: np.ndarray) -> np.ndarray:
        if self._n_features is None:
            raise ValueError("call fit before predict_cate")
        Xq = np.asarray(Xq, dtype=float)
        if Xq.ndim != 2 or Xq.shape[1] != self._n_features:
            raise ValueError(f"Xq must be (m, {self._n_features}), got shape {Xq.shape}")
        if self._model is None:
            return np.full(Xq.shape[0], np.nan)
        return np.asarray(self._model.effect(Xq), dtype=float).ravel()
