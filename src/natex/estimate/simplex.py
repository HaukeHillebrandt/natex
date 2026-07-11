"""Shared simplex-weight fitter for synthetic-control counterfactuals.

Extracted verbatim from ``did/controls.py`` (phase 3) so DiD synthetic
controls (``did.controls.synthetic_control``) and IV/SC donor selection
(``iv.donors``) share one deterministic fitter. Two invariants carry over:

* **Scale invariance** (the phase-3 fix): the SSE objective is normalized
  by ``y_target @ y_target`` because SLSQP's internal accuracy threshold is
  absolute — on raw-scale outcomes (prop99: SSE ~ 5e3 at the uniform start)
  it declares success after ~5 iterations without leaving the uniform
  start. Regressions: ``test_did_controls.py::
  test_synthetic_control_scale_invariant_optimization`` and
  ``test_simplex.py::test_scale_invariance``.
* **NaN, never 0**: :func:`weighted_counterfactual` renormalizes present
  donors while the missing weight mass stays within tolerance and returns
  NaN beyond it — a silently zeroed time never appears.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

MISSING_W_TOL = 0.1  # a time is defined while the total weight of MISSING
# donor cells stays <= this; present weights are renormalized. SLSQP
# solutions are not sparse (dozens of O(1e-2) weights), so requiring every
# weighted donor present would void almost every time on a thin panel.


@dataclass
class SimplexFit:
    """Deterministic simplex-constrained least-squares fit."""

    weights: np.ndarray  # (n_donors,) w >= 0, sum w = 1
    sse: float  # de-normalized to target units
    converged: bool


def fit_simplex_weights(y_target: np.ndarray, Y_donors: np.ndarray) -> SimplexFit:
    """Fit ``w >= 0, sum(w) = 1`` minimizing ``||y_target - Y_donors @ w||^2``.

    ``y_target`` is ``(n_common,)`` and ``Y_donors`` is
    ``(n_common, n_donors)``. Solved with SLSQP from the uniform start
    (deterministic, no rng). The SSE is scale-normalized by
    ``y_target @ y_target`` during optimization (the phase-3 fix; see the
    module docstring) and de-normalized back to target units in
    ``SimplexFit.sse``.
    """
    y_fit = np.asarray(y_target, dtype=float)
    y_ctrl = np.asarray(Y_donors, dtype=float)
    n_c = y_ctrl.shape[1]
    w0 = np.full(n_c, 1.0 / n_c)

    scale = float(y_fit @ y_fit)
    if scale <= 0.0:
        scale = 1.0

    def objective(w: np.ndarray) -> float:
        r = y_fit - y_ctrl @ w
        return float(r @ r) / scale

    def gradient(w: np.ndarray) -> np.ndarray:
        return -2.0 * (y_ctrl.T @ (y_fit - y_ctrl @ w)) / scale

    result = minimize(
        objective,
        w0,
        jac=gradient,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * n_c,
        constraints=[
            {
                "type": "eq",
                "fun": lambda w: float(w.sum() - 1.0),
                "jac": lambda w: np.ones_like(w),
            }
        ],
        options={"maxiter": 500, "ftol": 1e-12},
    )
    return SimplexFit(
        weights=np.asarray(result.x, dtype=float),
        sse=float(result.fun) * scale,  # de-normalized back to target units
        converged=bool(result.success),
    )


def weighted_counterfactual(
    contrib: np.ndarray, w: np.ndarray, missing_tol: float = MISSING_W_TOL
) -> np.ndarray:
    """(n_t,) donor-weighted mean per time from ``contrib`` (n_donors, n_t).

    Missing donor cells (NaN) are dropped and the PRESENT weights are
    renormalized while the missing weight mass is <= ``missing_tol``;
    beyond that the time is NaN — never a silent 0. The renormalization
    error is bounded by the missing mass times the donor-mean spread.
    """
    present = np.isfinite(contrib)
    missing_w = (~present).T.astype(float) @ w  # (n_t,)
    present_w = present.T.astype(float) @ w
    num = np.where(present, contrib, 0.0).T @ w
    ok = (missing_w <= missing_tol) & (present_w > 0.0)
    return np.where(ok, num / np.where(present_w > 0.0, present_w, 1.0), np.nan)
