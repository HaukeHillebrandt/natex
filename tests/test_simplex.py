"""Helper-level tests for the shared simplex-weight fitter (phase 5 task 6).

The fitter is extracted verbatim from ``did/controls.py`` (phase 3); the
scale-invariance behavior it encodes is regression-tested end to end in
``test_did_controls.py::test_synthetic_control_scale_invariant_optimization``
and here at the helper level.
"""

import numpy as np

from natex.estimate.simplex import (
    MISSING_W_TOL,
    SimplexFit,
    fit_simplex_weights,
    weighted_counterfactual,
)


def _convex_problem() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Deterministic (y_target, Y_donors, w_true): exact convex combination."""
    rng = np.random.default_rng(7)
    n_common, n_donors = 12, 3
    Y = rng.uniform(1.0, 5.0, size=(n_common, n_donors))
    w_true = np.array([0.2, 0.5, 0.3])
    return Y @ w_true, Y, w_true


def test_exact_convex_combination_recovery():
    y, Y, w_true = _convex_problem()
    fit = fit_simplex_weights(y, Y)
    assert isinstance(fit, SimplexFit)
    assert fit.converged
    assert np.max(np.abs(fit.weights - w_true)) < 1e-3
    assert fit.sse < 1e-10  # de-normalized, in target units


def test_scale_invariance():
    # Phase-3 regression at the helper level: the SSE objective is normalized
    # by y_target @ y_target, so rescaling target and donors together (unit
    # change) leaves the optimization path — and the weights — unchanged.
    y, Y, _ = _convex_problem()
    fit_raw = fit_simplex_weights(y, Y)
    fit_big = fit_simplex_weights(y * 1e4, Y * 1e4)
    assert np.max(np.abs(fit_big.weights - fit_raw.weights)) < 1e-6
    # sse is de-normalized to target units: scales by 1e8 (both ~0 here).
    assert fit_big.sse < 1e-10 * 1e8


def test_simplex_constraints_on_random_problem():
    # Noisy seeded problem with NO exact convex representation: the solution
    # must still satisfy the simplex constraints.
    rng = np.random.default_rng(42)
    Y = rng.normal(size=(20, 8))
    y = rng.normal(size=20)
    fit = fit_simplex_weights(y, Y)
    assert fit.weights.min() >= -1e-12
    assert abs(fit.weights.sum() - 1.0) <= 1e-8
    assert np.isfinite(fit.sse) and fit.sse >= 0.0


def test_weighted_counterfactual_nan_renormalization():
    # Donor 0 carries weight 0.05 <= MISSING_W_TOL, donor 1 carries 0.6.
    w = np.array([0.05, 0.6, 0.35])
    contrib = np.array(
        [
            [1.0, np.nan, 1.0],  # donor 0: missing at t=1 (mass 0.05, under tol)
            [2.0, 2.0, np.nan],  # donor 1: missing at t=2 (mass 0.6, over tol)
            [4.0, 4.0, 4.0],
        ]
    )
    out = weighted_counterfactual(contrib, w)
    # t=0 all present: 0.05*1 + 0.6*2 + 0.35*4 = 2.65
    assert np.isclose(out[0], 2.65)
    # t=1 renormalized over present donors: (0.6*2 + 0.35*4) / 0.95
    assert np.isclose(out[1], (0.6 * 2.0 + 0.35 * 4.0) / 0.95)
    # t=2 missing mass 0.6 > tol: NaN, never 0
    assert np.isnan(out[2])
    assert not np.any(out == 0.0)


def test_weighted_counterfactual_tolerance_parameter():
    w = np.array([0.05, 0.95])
    contrib = np.array([[np.nan], [3.0]])
    assert MISSING_W_TOL == 0.1
    # Default tolerance (0.1) admits the 0.05 missing mass.
    assert np.isclose(weighted_counterfactual(contrib, w)[0], 3.0)
    # A stricter tolerance voids the time — NaN, never 0.
    assert np.isnan(weighted_counterfactual(contrib, w, missing_tol=0.01)[0])
