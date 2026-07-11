"""Belloni-style plug-in Lasso instrument selection (BCCH 2012, Econometrica).

Selection reads only ``(T, pool, controls)`` — the function signature cannot
even receive an outcome (spec section 5 discovery-honesty analog). Audit
item 10: first-stage relevance is never assumed — the HC1 Wald F of the
POST-Lasso selected block (and its partial R^2 after controls) is always
computed, with ``weak = F < 10 or NaN``.

Pinned algorithm (phase-5 plan, "Design conventions"):

1. Partial ``[1, controls]`` out of ``d = T`` and every pool column by OLS
   residuals (Frisch–Waugh), then solve
   ``min_pi (1/n)||d - Z pi||^2 + (lam/n) sum_j psi_j |pi_j|``.
2. Plug-in penalty ``lam = 2 c sqrt(n) Phi^{-1}(1 - gamma/(2p))`` with
   ``c = 1.1`` and ``gamma = 0.1 / log(max(n, p))`` (documented choice),
   heteroskedastic loadings ``psi_j = sqrt((1/n) sum_i z_ij^2 eps_i^2)``,
   iterated: ``eps^0`` = residual of ``d`` on the 5 pool columns most
   correlated with it (BCCH-style init; plain centered ``d`` when p < 5),
   refreshed from post-Lasso residuals, stopping when the support is stable
   or after ``max_iter`` iterations.
3. Exact sklearn mapping: with ``u_j = z_j / psi_j`` and ``b_j = psi_j pi_j``
   the objective equals ``2 [(1/2n)||d - U b||^2 + (lam/2n)||b||_1]``, i.e.
   ``sklearn.linear_model.Lasso(alpha = lam/(2n), fit_intercept=False)`` on
   the partialled (hence centered) data; recover ``pi_j = b_j / psi_j``.

Zero-variance pool columns (including columns exactly collinear with the
controls, which have zero variance after partialling) get ``psi_j = inf``
semantics: they are excluded up front with a diagnostic in
``extras["dropped_zero_variance"]`` — never a divide-by-zero. Their ``pi``
entries are 0.0 and their reported loading is 0.0, the literal value of the
loading formula on a zero column; the exclusion itself is structural.

``lam="cv"`` uses ``LassoCV`` with rng-seeded 5-fold shuffling only to CHOOSE
lambda; the iterated-loading fit is then identical to an explicit float
lambda. Documented caveat: a CV lambda voids the plug-in sparsity guarantee —
the plug-in is the default. The rng is consumed only on this path; "plugin"
and float lambdas are RNG-free and bitwise deterministic.

NaN policy (spec section 5 item 8): empty selection is reported honestly —
``selected=[]``, ``first_stage_F=NaN``, ``partial_r2=NaN``, ``weak=True`` —
never a fabricated 0.0.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import stats
from sklearn.linear_model import Lasso, LassoCV
from sklearn.model_selection import KFold

from natex.estimate.iv2sls import _WEAK_F_THRESHOLD, _first_stage_diagnostics, _residualize

_SKLEARN_MAX_ITER = 10_000
_INIT_CORR_COLUMNS = 5  # BCCH-style eps^0: residual of d on the 5 most correlated columns


@dataclass
class InstrumentSearchResult:
    selected: list[str]  # post-Lasso support, pool_names order
    pi_lasso: np.ndarray  # (p,) weighted-Lasso coefs on the ORIGINAL column scale
    pi_post: np.ndarray  # (p,) post-Lasso OLS coefs (0.0 off-support)
    lam: float  # penalty actually used
    loadings: np.ndarray  # (p,) final penalty loadings psi_j
    first_stage_F: float  # HC1 Wald F of selected block (NaN when selected == [])
    partial_r2: float  # NaN when selected == []
    weak: bool  # F < 10.0 or NaN
    n_iter: int  # plug-in iterations performed
    extras: dict = field(default_factory=dict)  # dropped zero-variance cols, support trace, ...


def _empty_result(
    p: int, lam_val: float, loadings: np.ndarray, n_iter: int, extras: dict
) -> InstrumentSearchResult:
    """Honest 'no instrument found': NaN diagnostics (never 0.0), weak=True."""
    nan = float("nan")
    return InstrumentSearchResult(
        selected=[],
        pi_lasso=np.zeros(p),
        pi_post=np.zeros(p),
        lam=lam_val,
        loadings=loadings,
        first_stage_F=nan,
        partial_r2=nan,
        weak=True,
        n_iter=n_iter,
        extras=extras,
    )


def _loadings(z_partial: np.ndarray, eps: np.ndarray) -> np.ndarray:
    """Heteroskedastic penalty loadings psi_j = sqrt((1/n) sum_i z_ij^2 eps_i^2)."""
    return np.sqrt(np.mean(z_partial**2 * (eps**2)[:, None], axis=0))


def _post_lasso_residual(d: np.ndarray, z_partial: np.ndarray, support: np.ndarray) -> np.ndarray:
    if support.size == 0:
        return d
    coef, *_ = np.linalg.lstsq(z_partial[:, support], d, rcond=None)
    return d - z_partial[:, support] @ coef


def select_instruments(
    T: np.ndarray,
    pool: np.ndarray,
    controls: np.ndarray | None = None,
    pool_names: list[str] | None = None,
    lam: float | str = "plugin",
    c: float = 1.1,
    gamma: float | None = None,
    max_iter: int = 15,
    rng: np.random.Generator | None = None,
) -> InstrumentSearchResult:
    """Belloni plug-in Lasso first-stage selection over a candidate pool.

    ``lam`` is ``"plugin"`` (default, RNG-free), ``"cv"`` (LassoCV lambda,
    requires ``rng`` for fold shuffling), or an explicit positive float.
    See the module docstring for the pinned formulas and honesty policy.
    """
    t_all = np.asarray(T, dtype=float).ravel()
    z_all = np.asarray(pool, dtype=float)
    if z_all.ndim == 1:
        z_all = z_all[:, None]
    n_rows, p = z_all.shape
    if t_all.size != n_rows:
        raise ValueError(f"T has {t_all.size} rows but pool has {n_rows}")
    c_all = np.asarray(controls, dtype=float) if controls is not None else np.empty((n_rows, 0))
    if c_all.ndim == 1:
        c_all = c_all[:, None]
    if c_all.shape[0] != n_rows:
        raise ValueError(f"controls has {c_all.shape[0]} rows but pool has {n_rows}")
    if pool_names is None:
        pool_names = [f"z{j}" for j in range(1, p + 1)]
    if len(pool_names) != p:
        raise ValueError(f"pool_names has {len(pool_names)} entries but pool has {p} columns")
    if isinstance(lam, str):
        if lam not in ("plugin", "cv"):
            raise ValueError(f"lam must be 'plugin', 'cv', or a positive float, got {lam!r}")
    else:
        lam = float(lam)
        if not np.isfinite(lam) or lam <= 0:
            raise ValueError(f"explicit lam must be a positive finite float, got {lam}")
    if lam == "cv":
        if rng is None:
            raise ValueError("pass an explicit numpy Generator (lam='cv' shuffles CV folds)")
        if not isinstance(rng, np.random.Generator):
            raise TypeError(f"rng must be a numpy Generator, got {type(rng).__name__}")
    if c <= 0:
        raise ValueError(f"c must be > 0, got {c}")
    if gamma is not None and not 0 < gamma < 1:
        raise ValueError(f"gamma must be in (0, 1), got {gamma}")
    if max_iter < 1:
        raise ValueError(f"max_iter must be >= 1, got {max_iter}")

    finite = np.isfinite(t_all) & np.isfinite(z_all).all(axis=1) & np.isfinite(c_all).all(axis=1)
    n = int(finite.sum())
    extras: dict = {"n_dropped": int(n_rows - n)}
    nan = float("nan")
    if n < c_all.shape[1] + 3:
        extras["reason"] = "underdetermined: fewer than q + 3 finite rows"
        extras["dropped_zero_variance"] = []
        return _empty_result(p, nan, np.zeros(p), 0, extras)

    t_used = t_all[finite]
    cfull = np.c_[np.ones(n), c_all[finite]]
    d = _residualize(t_used, cfull)
    z_partial_all = _residualize(z_all[finite], cfull)

    # psi = inf semantics for zero-variance-after-partialling columns:
    # structural up-front exclusion, never a divide-by-zero. The threshold is
    # scale-normalized (audit item 24: no absolute floors) — a column is dead
    # when partialling wiped out all but an O(n * machine-eps) relative
    # remnant of its raw sum of squares (constants leave ~1e-32 relatively).
    col_ss = (z_partial_all**2).sum(axis=0)
    raw_ss = (z_all[finite] ** 2).sum(axis=0)
    dead = col_ss <= raw_ss * (n * np.finfo(float).eps) ** 2
    kept = np.flatnonzero(~dead)
    extras["dropped_zero_variance"] = [pool_names[j] for j in np.flatnonzero(dead)]
    if kept.size == 0:
        extras["reason"] = "no pool column with positive variance after partialling"
        return _empty_result(p, nan if isinstance(lam, str) else lam, np.zeros(p), 0, extras)
    z_partial = z_partial_all[:, kept]
    p_kept = kept.size

    d_ss = float(d @ d)
    if d_ss <= 0:
        extras["reason"] = "treatment has zero variance after partialling controls"
        return _empty_result(p, nan if isinstance(lam, str) else lam, np.zeros(p), 0, extras)

    # BCCH-style eps^0: residual of d on the 5 most correlated pool columns
    # (plain centered/partialled d when p < 5).
    if p_kept >= _INIT_CORR_COLUMNS:
        corr = np.abs(z_partial.T @ d) / np.sqrt(col_ss[kept] * d_ss)
        top = np.argsort(-corr)[:_INIT_CORR_COLUMNS]
        eps = _post_lasso_residual(d, z_partial, top)
    else:
        eps = d

    if gamma is None:
        gamma = 0.1 / np.log(max(n, p_kept))
    extras["gamma"] = float(gamma)

    if lam == "plugin":
        lam_val = float(2.0 * c * np.sqrt(n) * stats.norm.ppf(1.0 - gamma / (2.0 * p_kept)))
        extras["lam_source"] = "plugin"
    elif lam == "cv":
        # CV chooses lambda only; the iterated-loading fit below is then
        # identical to an explicit float lambda. Caveat (documented): a CV
        # lambda voids the plug-in sparsity guarantee.
        if n < 5:
            raise ValueError(f"lam='cv' needs n >= 5 finite rows for 5-fold CV, got {n}")
        psi0 = _loadings(z_partial, eps)
        folds = KFold(n_splits=5, shuffle=True, random_state=int(rng.integers(0, 2**32 - 1)))
        cv_fit = LassoCV(cv=folds, fit_intercept=False, max_iter=_SKLEARN_MAX_ITER)
        cv_fit.fit(z_partial / psi0, d)
        lam_val = float(2.0 * n * cv_fit.alpha_)
        extras["lam_source"] = "cv"
        extras["alpha_cv"] = float(cv_fit.alpha_)
    else:
        lam_val = lam
        extras["lam_source"] = "explicit"

    alpha = lam_val / (2.0 * n)
    n_iter = 0
    prev_support: np.ndarray | None = None
    support = np.empty(0, dtype=int)
    psi = np.ones(p_kept)
    b = np.zeros(p_kept)
    trace: list[list[int]] = []
    for it in range(1, max_iter + 1):
        psi_new = _loadings(z_partial, eps)
        if not np.all(psi_new > 0):
            # eps has hit exact zeros everywhere a column lives (perfect
            # post-Lasso fit): loadings degenerate, keep the last valid fit.
            extras["loading_degenerate"] = True
            break
        psi = psi_new
        fit = Lasso(alpha=alpha, fit_intercept=False, max_iter=_SKLEARN_MAX_ITER)
        fit.fit(z_partial / psi, d)
        b = fit.coef_
        support = np.flatnonzero(b != 0)
        trace.append(kept[support].tolist())
        n_iter = it
        if prev_support is not None and np.array_equal(support, prev_support):
            break
        prev_support = support
        eps = _post_lasso_residual(d, z_partial, support)
    extras["support_trace"] = trace

    pi_lasso = np.zeros(p)
    pi_lasso[kept] = b / psi
    loadings = np.zeros(p)
    loadings[kept] = psi
    if support.size == 0:
        return _empty_result(p, lam_val, loadings, n_iter, extras)

    coef, *_ = np.linalg.lstsq(z_partial[:, support], d, rcond=None)
    pi_post = np.zeros(p)
    orig_idx = kept[support]
    pi_post[orig_idx] = coef
    f_stat, partial_r2 = _first_stage_diagnostics(t_used, cfull, z_all[finite][:, orig_idx])
    return InstrumentSearchResult(
        selected=[pool_names[j] for j in orig_idx],
        pi_lasso=pi_lasso,
        pi_post=pi_post,
        lam=lam_val,
        loadings=loadings,
        first_stage_F=f_stat,
        partial_r2=partial_r2,
        weak=bool(f_stat < _WEAK_F_THRESHOLD) if np.isfinite(f_stat) else True,
        n_iter=n_iter,
        extras=extras,
    )
