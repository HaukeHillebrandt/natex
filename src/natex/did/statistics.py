"""Window-restricted LLR kernels for the SuDDDS RDiT scan (thesis ch. 6).

Every kernel restricts to the window ``[T0-W, T0)`` (pre, g0) / ``[T0, T0+W)``
(post, g1): records outside contribute nothing, and all kernels evaluate many
candidate subsets at once (boolean masks matrix ``M`` of shape ``(n, S)``),
mirroring the vectorized phase-2 scan kernels.

Observation models and audit corrections (docs/math_audit_final.md):

* **Double-beta** (Eqs 6.7-6.11) — separate pre/post precision-weighted means.
  Sufficient statistics use sigma^2 weights, ``c_i = r_i / sigma_i^2`` and
  ``b_i = 1 / sigma_i^2`` (the thesis page prints sigma_i; adjudicated in
  audit section 1). Eq 6.10 prose swaps q1/q2 <-> beta_g0/beta_g1 — a typo,
  the equations are right (audit item 17): here ``q1`` is the post-side mean.
* **Single-Delta** (Eqs 6.13-6.16, repaired per audit item 15) — the profile
  means mu_i must be profiled out under BOTH hypotheses and the statistic
  restricted to the window. With ``d_j = +1`` (post) / ``-1`` (pre) this gives
  the corrected sufficient statistics C-tilde/B-tilde and the profile GLR
  ``C^2 / (2 B)`` with MLE ``Delta_hat = C / B``; the statistic is even in
  Delta, so both signs are covered (they enter separately only through the
  priority ordering downstream).
* **Bernoulli window LLR** (audit item 19) — the model class must match the
  treatment's type: for binary theta the DiD contrast is scored with exact
  Bernoulli log-odds offsets (reusing the masked phase-2 machinery), never
  with a Normal approximation. Working residuals exist ONLY to order
  priorities.

Discovery never reads the outcome ``y``; degenerate subsets score exactly
0.0 (detected from in-window COUNTS, never float residue); failures are NaN,
never 0.0.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from natex.scan.statistics import fit_log_odds_offsets, masked_offset_log_lik


@dataclass
class WindowStats:
    """Per-record sufficient statistics for a fixed (T0, W).

    Records outside the window have ``c = b = 0`` and ``g1 = False``, so
    subset sums automatically restrict to the window. Invariant:
    ``g1`` implies ``in_window``; the pre side is ``in_window & ~g1``.
    """

    in_window: np.ndarray  # (n,) bool — t in [T0-W, T0+W)
    g1: np.ndarray  # (n,) bool — post side, t in [T0, T0+W)
    c: np.ndarray  # (n,) w * r, zeroed outside window
    b: np.ndarray  # (n,) w = 1/sigma2, zeroed outside window

    @property
    def g0(self) -> np.ndarray:
        """Pre side, t in [T0-W, T0)."""
        return self.in_window & ~self.g1


def window_stats(
    t: np.ndarray, r: np.ndarray, sigma2: np.ndarray, T0: float, W: float
) -> WindowStats:
    """Build :class:`WindowStats` for the window ``[T0-W, T0) / [T0, T0+W)``.

    ``sigma2`` must be finite and strictly positive (weights are 1/sigma2);
    ``W`` must be > 0. Raises ``ValueError`` otherwise — never silently
    zeroes a failure.
    """
    t = np.asarray(t, dtype=float)
    r = np.asarray(r, dtype=float)
    sigma2 = np.asarray(sigma2, dtype=float)
    if not (t.shape == r.shape == sigma2.shape):
        raise ValueError(
            f"t, r, sigma2 must share one shape, got {t.shape}, {r.shape}, {sigma2.shape}"
        )
    if not np.isfinite(W) or W <= 0.0:
        raise ValueError(f"W must be a finite positive width, got {W}")
    if not np.isfinite(T0):
        raise ValueError(f"T0 must be finite, got {T0}")
    if not np.all(np.isfinite(sigma2) & (sigma2 > 0.0)):
        raise ValueError("sigma2 must be finite and > 0 (data-scaled floor upstream)")
    in_window = (t >= T0 - W) & (t < T0 + W)
    g1 = (t >= T0) & in_window
    w = 1.0 / sigma2
    b = np.where(in_window, w, 0.0)
    c = np.where(in_window, w * r, 0.0)
    return WindowStats(in_window=in_window, g1=g1, c=c, b=b)


def double_beta_llr_masks(ws: WindowStats, M: np.ndarray) -> np.ndarray:
    """Double-beta LLR (Eq 6.9) per subset column of boolean ``M`` (n, S).

    Per column: ``C1^2/(2 B1) + C0^2/(2 B0) - (C1+C0)^2/(2 (B1+B0))`` with
    C1/B1 the post-side and C0/B0 the pre-side subset sums of c and b.
    Degenerate columns (no in-window records on either side) score exactly
    0.0, detected from in-window COUNTS per side, not float residue — the
    same guard as ``normal_llr_all_splits``. Always >= 0.
    """
    M = np.asarray(M, dtype=bool)
    g1 = ws.g1
    g0 = ws.g0
    Mt = M.T.astype(float)
    C1 = Mt @ (ws.c * g1)
    B1 = Mt @ (ws.b * g1)
    C0 = Mt @ (ws.c * g0)
    B0 = Mt @ (ws.b * g0)
    n1 = M.T.astype(np.int64) @ g1.astype(np.int64)
    n0 = M.T.astype(np.int64) @ g0.astype(np.int64)
    with np.errstate(divide="ignore", invalid="ignore"):
        llr = C1**2 / (2.0 * B1) + C0**2 / (2.0 * B0) - (C1 + C0) ** 2 / (2.0 * (B1 + B0))
    degenerate = (n1 == 0) | (n0 == 0) | (B1 <= 0.0) | (B0 <= 0.0)
    return np.where(degenerate, 0.0, np.maximum(llr, 0.0))


def double_beta_q(ws: WindowStats, M: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-column ratios (Eq 6.10): ``q1 = C1/B1`` (post), ``q2 = C0/B0`` (pre).

    These are the precision-weighted residual means and the MLEs of the side
    means (the thesis PROSE pairs them backwards — audit item 17; the
    equations, and this function, are right). Empty sides are NaN, never 0.
    """
    M = np.asarray(M, dtype=bool)
    g1 = ws.g1
    g0 = ws.g0
    Mt = M.T.astype(float)
    C1 = Mt @ (ws.c * g1)
    B1 = Mt @ (ws.b * g1)
    C0 = Mt @ (ws.c * g0)
    B0 = Mt @ (ws.b * g0)
    n1 = M.T.astype(np.int64) @ g1.astype(np.int64)
    n0 = M.T.astype(np.int64) @ g0.astype(np.int64)
    with np.errstate(divide="ignore", invalid="ignore"):
        q1 = np.where((n1 == 0) | (B1 <= 0.0), np.nan, C1 / B1)
        q2 = np.where((n0 == 0) | (B0 <= 0.0), np.nan, C0 / B0)
    return q1, q2


def single_delta_stats(
    ws: WindowStats, profile_id: np.ndarray, n_profiles: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Audit-15 corrected single-Delta sufficient statistics per PROFILE.

    Returns ``(C_tilde, B_tilde)`` indexed by profile id. Over the in-window
    records of profile i, with ``d_j = +1`` (post) / ``-1`` (pre) and
    ``w_j = 1/sigma_j^2``::

        delta_bar_i = sum(w d) / sum(w)
        B_tilde_i   = sum(w) - (sum(w d))^2 / sum(w)   [= sum w (d - delta_bar)^2]
        C_tilde_i   = sum(w (d - delta_bar_i) r)

    which is the exact profile-out of mu_i under BOTH hypotheses (the thesis
    freezes mu_i at the H0 MLE and uses B_i = sum w — audit item 15's
    counterexample). Profiles with < 2 in-window records or one empty side
    cannot identify Delta and get ``B_tilde = C_tilde = 0`` exactly.
    """
    profile_id = np.asarray(profile_id)
    if profile_id.shape != ws.b.shape:
        raise ValueError(f"profile_id shape {profile_id.shape} != records {ws.b.shape}")
    n_prof = int(profile_id.max()) + 1 if profile_id.size else 0
    if n_profiles is not None:
        if n_profiles < n_prof:
            raise ValueError(f"n_profiles={n_profiles} < max(profile_id)+1={n_prof}")
        n_prof = n_profiles
    d = np.where(ws.g1, 1.0, -1.0)  # out-of-window values are inert: b = c = 0 there
    sw = np.bincount(profile_id, weights=ws.b, minlength=n_prof)
    swd = np.bincount(profile_id, weights=ws.b * d, minlength=n_prof)
    sc = np.bincount(profile_id, weights=ws.c, minlength=n_prof)
    sdc = np.bincount(profile_id, weights=ws.c * d, minlength=n_prof)
    n1 = np.bincount(profile_id[ws.g1], minlength=n_prof)
    n0 = np.bincount(profile_id[ws.g0], minlength=n_prof)
    with np.errstate(divide="ignore", invalid="ignore"):
        delta_bar = swd / sw
        B_tilde = sw - swd**2 / sw
        C_tilde = sdc - delta_bar * sc
    degenerate = (n1 == 0) | (n0 == 0) | (n1 + n0 < 2)
    B_tilde = np.where(degenerate, 0.0, B_tilde)
    C_tilde = np.where(degenerate, 0.0, C_tilde)
    return C_tilde, B_tilde


def single_delta_llr(
    C_sel: float | np.ndarray, B_sel: float | np.ndarray
) -> float | np.ndarray:
    """Profiled single-Delta GLR: ``C^2 / (2 B)``; ``Delta_hat = C / B``.

    ``C_sel``/``B_sel`` are sums of C_tilde/B_tilde over the selected
    profiles. 0.0 when ``B <= 0`` (no identifying variation). Even in the
    sign of Delta (audit item 15: both signs enter through the priority
    ordering, the maximized statistic is sign-agnostic). Always >= 0.
    """
    C = np.asarray(C_sel, dtype=float)
    B = np.asarray(B_sel, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(B > 0.0, C * C / (2.0 * B), 0.0)
    if np.ndim(C_sel) == 0 and np.ndim(B_sel) == 0:
        return float(out)
    return out


def bernoulli_window_llr_masks(
    theta: np.ndarray, eta: np.ndarray, ws: WindowStats, M: np.ndarray
) -> np.ndarray:
    """Bernoulli window LLR per subset column (audit item 19, binary theta).

    Column s: H0 fits one common log-odds offset over ``s & window``; H1 fits
    separate pre/post offsets over ``s & g0`` and ``s & g1``. Reuses the
    phase-2 masked machinery (``fit_log_odds_offsets`` /
    ``masked_offset_log_lik``) on the three mask stacks in a single Newton
    batch. Pure sides hit the boundary suprema (0.0 log-likelihood, audit
    item 21 convention); degenerate columns (an empty in-window side) score
    exactly 0.0. Always >= 0 and finite.
    """
    theta = np.asarray(theta, dtype=float)
    eta = np.asarray(eta, dtype=float)
    M = np.asarray(M, dtype=bool)
    M1 = M & ws.g1[:, None]
    M0 = M & ws.g0[:, None]
    Mw = M1 | M0
    stacked = np.concatenate([Mw, M0, M1], axis=1)
    th = fit_log_odds_offsets(theta, eta, stacked)
    ll = masked_offset_log_lik(th, theta, eta, stacked)
    s = M.shape[1]
    ll_win, ll_pre, ll_post = ll[:s], ll[s : 2 * s], ll[2 * s :]
    degenerate = (M1.sum(axis=0) == 0) | (M0.sum(axis=0) == 0)
    return np.where(degenerate, 0.0, np.maximum(ll_pre + ll_post - ll_win, 0.0))


def working_residuals(theta: np.ndarray, p_hat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Working residuals ``(r, sigma2) = (theta - p_hat, p_hat (1 - p_hat))``.

    Used ONLY to order priorities for the Bernoulli model (via
    :func:`window_stats` on r/sigma2); the LLR evaluation itself stays
    exact-Bernoulli (:func:`bernoulli_window_llr_masks`). ``p_hat`` must lie
    strictly inside (0, 1) — the fitted background already clips.
    """
    theta = np.asarray(theta, dtype=float)
    p_hat = np.asarray(p_hat, dtype=float)
    return theta - p_hat, p_hat * (1.0 - p_hat)
