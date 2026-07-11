"""DiD treatment background model: theta ~ f(unit, t) (thesis Eq 6.4 lineage).

Fits the expected treatment level per record from unit fixed effects plus a
polynomial in time, so the RDiT scan works on residuals (normal model) or
log-odds offsets (Bernoulli model). Discovery never reads the outcome ``y`` —
this module only ever touches ``panel.theta``, ``panel.t``, ``panel.unit`` and
``panel.profile_id``.

Design notes (docs/math_audit_final.md):

* **Eq 6.4 rank deficiency** — the printed design (global intercept + full set
  of unit dummies) is rank-deficient. We drop the global intercept and keep
  the full dummy set, solving with :func:`numpy.linalg.lstsq` (minimum-norm
  pseudoinverse solution), so any remaining collinearity (e.g. degree >= 1
  with few time points) is handled without reference-category bookkeeping.
* **Audit item 24** — no absolute variance floor. Per-*profile* residual
  variances are shrunk toward the global residual variance with a
  count-based weight ``lam = n0 / (n0 + n_prof)`` (``n0 = 10``), and the
  final floor is *data-scaled*: ``1e-12 * s2_global``, never an absolute
  constant.
* **Bernoulli path** — ridge-penalized logistic regression (``C=1.0``) on
  standardized features, mirroring ``natex.rdd.lord3.fit_treatment_model``:
  under perfect separation the unpenalized MLE does not exist, the fitted
  background would absorb the discontinuity, and Bernoulli(p-hat) null
  replicas would degenerate into exact copies of theta. Standardization makes
  the penalty independent of feature units.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from natex.did.panel import CategoricalPanel

_P_CLIP = 1e-6
_N0 = 10.0  # count-based shrinkage prior weight (records)
_FLOOR_SCALE = 1e-12  # data-scaled variance floor multiplier (never absolute)


@dataclass
class DiDBackground:
    """Fitted treatment background over a categorical panel."""

    kind: str  # "normal" | "bernoulli"
    fitted: np.ndarray  # f(x, t) (normal) or p_hat (bernoulli), (n,)
    r: np.ndarray | None  # theta - fitted (normal only)
    sigma2: np.ndarray | None  # per-record variance (normal only)
    eta: np.ndarray | None  # logit(clip(p_hat)) (bernoulli only)


def _design_matrix(panel: CategoricalPanel, degree: int, unit_effects: bool) -> np.ndarray:
    """Unit one-hot intercepts (or a single intercept) + standardized-time powers.

    Time is standardized to zero mean / unit sd for conditioning before the
    polynomial expansion. No global intercept alongside the dummies (Eq 6.4
    rank fix); residual rank deficiency is left to the lstsq/pinv solver.
    """
    if degree < 0:
        raise ValueError(f"degree must be >= 0, got {degree}")
    t = np.asarray(panel.t, dtype=float)
    sd = float(t.std())
    ts = (t - float(t.mean())) / (sd if sd > 0 else 1.0)
    cols: list[np.ndarray] = []
    if unit_effects:
        one_hot = np.zeros((panel.n, len(panel.unit_values)))
        one_hot[np.arange(panel.n), panel.unit] = 1.0
        cols.append(one_hot)
    else:
        cols.append(np.ones((panel.n, 1)))
    for p in range(1, degree + 1):
        cols.append(ts[:, None] ** p)
    return np.hstack(cols)


def _profile_sigma2(
    r: np.ndarray, profile_id: np.ndarray, shrink: float | None
) -> np.ndarray:
    """Per-record variance from per-profile residual variance, shrunk (audit 24).

    ``sigma2_prof = (1 - lam) * s2_prof + lam * s2_global`` with
    ``lam = n0 / (n0 + n_prof)`` (``n0 = 10``) unless ``shrink`` pins ``lam``
    directly. Final floor ``max(sigma2, 1e-12 * s2_global)`` — scaled by the
    data, never an absolute constant.
    """
    _, inv = np.unique(profile_id, return_inverse=True)
    counts = np.bincount(inv).astype(float)
    means = np.bincount(inv, weights=r) / counts
    s2_prof = np.bincount(inv, weights=(r - means[inv]) ** 2) / counts
    s2_global = float(np.var(r))
    if shrink is None:
        lam = _N0 / (_N0 + counts)
    else:
        if not 0.0 <= shrink <= 1.0:
            raise ValueError(f"shrink must be in [0, 1], got {shrink}")
        lam = np.full_like(counts, float(shrink))
    sigma2_prof = (1.0 - lam) * s2_prof + lam * s2_global
    sigma2_prof = np.maximum(sigma2_prof, _FLOOR_SCALE * s2_global)
    return sigma2_prof[inv]


def fit_did_background(
    panel: CategoricalPanel,
    model: str = "auto",
    degree: int = 1,
    unit_effects: bool = True,
    shrink: float | None = None,
) -> DiDBackground:
    """Fit the treatment background theta ~ unit effects + polynomial time.

    Parameters
    ----------
    panel:
        Coded panel; only ``theta``, ``t``, ``unit`` and ``profile_id`` are
        read (the scan never reads ``y``).
    model:
        ``"auto"`` selects ``"bernoulli"`` iff theta is binary in {0, 1},
        else ``"normal"``.
    degree:
        Polynomial order R in (standardized) time.
    unit_effects:
        Per-unit one-hot intercepts (thesis 6.4.4 uses state fixed effects);
        ``False`` fits a single global intercept.
    shrink:
        ``None`` (default) uses the count-based shrinkage weight
        ``lam = 10 / (10 + n_prof)``; a float in [0, 1] pins ``lam`` for every
        profile (0 = raw per-profile variance + scaled floor, 1 = global).
    """
    theta = np.asarray(panel.theta, dtype=float)
    if model == "auto":
        vals = np.unique(theta[~np.isnan(theta)])
        binary = vals.size <= 2 and set(vals.tolist()) <= {0.0, 1.0}
        model = "bernoulli" if binary else "normal"
    if model not in ("normal", "bernoulli"):
        raise ValueError(f"model must be 'auto', 'normal' or 'bernoulli', got {model!r}")

    design = _design_matrix(panel, degree, unit_effects)

    if model == "normal":
        beta, *_ = np.linalg.lstsq(design, theta, rcond=None)
        fitted = design @ beta
        r = theta - fitted
        sigma2 = _profile_sigma2(r, panel.profile_id, shrink)
        return DiDBackground(kind="normal", fitted=fitted, r=r, sigma2=sigma2, eta=None)

    scaler = StandardScaler().fit(design)
    est = LogisticRegression(C=1.0, max_iter=1000).fit(
        scaler.transform(design), theta.astype(int)
    )
    p_hat = est.predict_proba(scaler.transform(design))[:, 1]
    eta = logit(np.clip(p_hat, _P_CLIP, 1.0 - _P_CLIP))
    return DiDBackground(kind="bernoulli", fitted=p_hat, r=None, sigma2=None, eta=eta)
