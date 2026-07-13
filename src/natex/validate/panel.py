"""Panel validation battery for SuDDDS discoveries (audit items 1, 2, 18).

Three checks, all replacing or repairing the thesis validation stage:

* :func:`panel_randomization_test` — fitted-null Monte Carlo calibration of
  the max-LLR scan statistic. This is a **parametric bootstrap against the
  fitted null model, NOT an exact randomization test** (audit item 1): the
  observed data are not exchangeable with replicas drawn from a model fitted
  to them, so p-values use the +1-rank rule and are never claimed exact.
  Every replica refits its own background and reruns the full scan.
* :func:`composition_test` — chi-square independence of the record
  composition (by unit or covariate profile) pre vs post inside the discovery
  window. This is the audit-18 replacement for McCrary on calendar time,
  which is information-free when record times are design-determined.
* :func:`anticipation_test` — placebo jump estimates at pre-period cutoffs
  ``T0 - shift * step`` restricted to ``t < T0`` (never contaminated by the
  real jump), Holm-corrected across shifts.

Null replica kinds (audit item 18 — preserve unit/time dependence):

* ``"ar1_unit"`` (normal-model default) — unit random effects plus a
  stationary AR(1) over each unit's time-sorted records: smooth by
  construction (no jump under H0) while preserving within-unit serial
  dependence and between-unit level dispersion.
* ``"iid"`` — phase-2-style ``fitted + sqrt(sigma2) * N(0, 1)`` draws,
  offered for comparison only; documented as DEPENDENCE-BREAKING and
  anti-conservative on serially dependent panels.
* ``"bernoulli"`` — direct i.i.d. Bernoulli(p_hat) draws (audit item 2).
  Caveat: serial dependence of a binary panel treatment is NOT preserved by
  i.i.d. draws; the normal-model ``ar1_unit`` test on the same data (force
  ``model="normal"``, thesis-parity path) is the dependence-preserving
  cross-check.

Discovery never reads the outcome ``y``; failures are NaN, never 0.0; a
degenerate check reports ``passed = False``, never a silent pass.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from scipy import stats

from natex.data.spec import Dataset
from natex.did.background import DiDBackground, fit_did_background
from natex.did.panel import CategoricalPanel, build_panel
from natex.did.statistics import (
    WindowStats,
    double_beta_q,
    single_delta_stats,
    window_stats,
    working_residuals,
)
from natex.did.suddds import DiDDiscovery, SuDDDSResult, suddds_scan

_NULL_KINDS = ("auto", "ar1_unit", "iid", "bernoulli")
_PHI_MAX = 0.95


# ---------------------------------------------------------------------------
# panel randomization test (audit items 1, 2, 18)
# ---------------------------------------------------------------------------


@dataclass
class PanelRandomizationReport:
    p_value: float
    observed_max_llr: float
    null_max_llrs: np.ndarray
    q: int
    null_kind: str  # "ar1_unit" | "iid" | "bernoulli"


@dataclass
class _Ar1Params:
    """Dependence-preserving null-draw parameters fitted once on observed data."""

    fitted: np.ndarray  # (n,) background mean
    unit: np.ndarray  # (n,) unit codes
    n_units: int
    alpha_sd: float  # between-unit sd of unit-mean residuals
    phi: float  # pooled lag-1 autocorrelation, clipped to [0, 0.95]
    e_sd: float  # stationary sd of the AR(1) component
    innov_sd: float  # innovation sd: e_sd^2 * (1 - phi^2)
    order: np.ndarray  # (n,) argsort by (unit, t)
    starts: np.ndarray  # per-unit run starts into `order`
    ends: np.ndarray  # per-unit run ends into `order`


def _fit_ar1_unit(panel: CategoricalPanel, background: DiDBackground) -> _Ar1Params:
    """Estimate (sigma2_alpha, phi, innovation variance) from background residuals.

    Unit-mean residuals give the between-unit variance; within-unit demeaned
    residuals, time-sorted per unit, give the pooled lag-1 autocorrelation
    ``phi = sum(e_t e_{t-1}) / sum(e_{t-1}^2)`` (clipped to [0, 0.95]) and the
    pooled residual variance; the innovation variance matches the latter
    through the stationary AR(1) identity ``var_e = innov / (1 - phi^2)``.
    """
    assert background.r is not None
    r = background.r
    unit = panel.unit
    n_units = len(panel.unit_values)
    counts = np.bincount(unit, minlength=n_units).astype(float)
    means = np.bincount(unit, weights=r, minlength=n_units) / np.maximum(counts, 1.0)
    alpha_sd = float(np.std(means))
    e = r - means[unit]
    order = np.lexsort((panel.t, unit))
    e_sorted = e[order]
    u_sorted = unit[order]
    same = u_sorted[1:] == u_sorted[:-1]
    prev = e_sorted[:-1][same]
    curr = e_sorted[1:][same]
    den = float(prev @ prev)
    phi = float(prev @ curr) / den if den > 0.0 else 0.0
    phi = float(min(max(phi, 0.0), _PHI_MAX))
    var_e = float(np.mean(e**2))
    starts = np.flatnonzero(np.r_[True, ~same])
    ends = np.r_[starts[1:], u_sorted.size]
    return _Ar1Params(
        fitted=background.fitted,
        unit=unit,
        n_units=n_units,
        alpha_sd=alpha_sd,
        phi=phi,
        e_sd=float(np.sqrt(var_e)),
        innov_sd=float(np.sqrt(var_e * (1.0 - phi**2))),
        order=order,
        starts=starts,
        ends=ends,
    )


def _draw_ar1_unit(params: _Ar1Params, rng: np.random.Generator) -> np.ndarray:
    """theta* = fitted + alpha_u + stationary AR(1) over time-sorted records."""
    n = params.fitted.size
    alpha = rng.normal(0.0, params.alpha_sd, size=params.n_units)
    z = rng.standard_normal(n)
    e_sorted = np.empty(n)
    for s, e in zip(params.starts, params.ends, strict=True):
        e_sorted[s] = params.e_sd * z[s]
        for k in range(s + 1, e):
            e_sorted[k] = params.phi * e_sorted[k - 1] + params.innov_sd * z[k]
    r_star = np.empty(n)
    r_star[params.order] = e_sorted
    return params.fitted + alpha[params.unit] + r_star


def panel_randomization_test(
    dataset: Dataset,
    scan_result: SuDDDSResult,
    Q: int = 99,
    rng: np.random.Generator | None = None,
    scan_kwargs: dict | None = None,
    null: str = "auto",
) -> PanelRandomizationReport:
    """Fitted-null Monte Carlo calibration of the SuDDDS max-LLR statistic.

    Parametric bootstrap, NOT exact (audit item 1): replicas are drawn from a
    background model fitted to the observed data, each replica REFITS ITS OWN
    background and reruns :func:`natex.did.suddds.suddds_scan` with the SAME
    resolved configuration the observed scan recorded on ``scan_result`` —
    windows/method/restarts plus bins/degree/dims/min_side/n_rho/
    exhaustive_max_values (issue #13; ``scan_kwargs`` overrides explicitly) —
    and the shared ``rng``; the p-value is the +1-rank
    ``(1 + #{null >= observed}) / (Q + 1)``.

    ``null="auto"`` selects ``"bernoulli"`` for a Bernoulli-model result
    (direct Bernoulli(p_hat) draws, audit item 2 — i.i.d., so a binary
    treatment's serial dependence is NOT preserved; rerun with
    ``model="normal"`` for the dependence-preserving ``ar1_unit``
    cross-check) and ``"ar1_unit"`` otherwise (audit item 18). Replicas that
    return no discovery score 0.0 — the supremum over an empty candidate set,
    a documented convention shared with the phase-2 test, not a failure code.
    """
    if rng is None:
        raise ValueError("rng is required: pass one numpy Generator through every stochastic call")
    if not isinstance(rng, np.random.Generator):
        raise TypeError(f"rng must be a numpy Generator, got {type(rng).__name__}")
    if Q < 1:
        raise ValueError(f"Q must be >= 1, got {Q}")
    if null not in _NULL_KINDS:
        raise ValueError(f"null must be one of {_NULL_KINDS}, got {null!r}")
    if not scan_result.discoveries:
        raise ValueError("scan_result has no discoveries: nothing to calibrate")
    kind = scan_result.model
    null_kind = ("bernoulli" if kind == "bernoulli" else "ar1_unit") if null == "auto" else null
    if null_kind == "bernoulli" and kind != "bernoulli":
        raise ValueError("null='bernoulli' requires a Bernoulli-model scan result (audit item 2)")
    if null_kind in ("ar1_unit", "iid") and kind != "normal":
        raise ValueError(
            f"null={null_kind!r} requires a normal-model scan result; rerun the scan with "
            "model='normal' (thesis-parity path) for the dependence-preserving cross-check"
        )

    kwargs = dict(scan_kwargs or {})
    kwargs.setdefault("windows", scan_result.windows)
    kwargs.setdefault("method", scan_result.method)
    kwargs.setdefault("restarts", scan_result.restarts)
    kwargs.setdefault("model", kind)
    # Issue #13: every default comes from the RESOLVED config recorded on the
    # scan result, never a hardcoded fallback — replicas searching a smaller
    # space than the observed max-LLR understate the null maximum and give
    # anti-conservative p-values. ``scan_kwargs`` stays an explicit override.
    kwargs.setdefault("degree", scan_result.degree)
    kwargs.setdefault("min_side", scan_result.min_side)
    kwargs.setdefault("n_rho", scan_result.n_rho)
    kwargs.setdefault("exhaustive_max_values", scan_result.exhaustive_max_values)
    bins = kwargs.pop("bins", scan_result.bins)
    dims = kwargs.pop("dims", scan_result.dims)
    degree = kwargs["degree"]

    panel = build_panel(dataset, dims=dims, bins=bins)
    background = fit_did_background(panel, model=kind, degree=degree)
    if null_kind == "ar1_unit" and len(panel.unit_values) < 2:
        raise ValueError(
            "null='ar1_unit' requires >= 2 units for the unit-level draws; "
            "use null='iid' (documented as dependence-breaking) instead"
        )
    params = _fit_ar1_unit(panel, background) if null_kind == "ar1_unit" else None

    def draw() -> np.ndarray:
        if null_kind == "bernoulli":
            return rng.binomial(1, background.fitted).astype(float)
        if null_kind == "iid":
            assert background.sigma2 is not None
            return background.fitted + np.sqrt(background.sigma2) * rng.standard_normal(panel.n)
        assert params is not None
        return _draw_ar1_unit(params, rng)

    observed = float(scan_result.discoveries[0].llr)
    null_max = np.empty(Q)
    for q_i in range(Q):
        theta_star = draw()
        if null_kind == "bernoulli" and np.unique(theta_star).size < 2:
            # Issue #14: a one-class Bernoulli(p_hat) draw (likely when p_hat
            # is small) admits no background refit and no scoreable split —
            # its max-LLR is the supremum over an empty candidate set, 0.0 by
            # the documented convention above (the exact limit of the clipped
            # LLR). Keeping the draw preserves i.i.d. sampling from the
            # fitted null; redrawing would bias the null distribution.
            null_max[q_i] = 0.0
            continue
        # audit item 1: every replica refits its own background inside the
        # scan (background=None) — only the coded panel structure is reused.
        panel_star = replace(panel, theta=theta_star)
        res_star = suddds_scan(dataset, rng=rng, panel=panel_star, **kwargs)
        null_max[q_i] = res_star.discoveries[0].llr if res_star.discoveries else 0.0

    p = (1.0 + float(np.sum(null_max >= observed))) / (Q + 1.0)
    return PanelRandomizationReport(
        p_value=p,
        observed_max_llr=observed,
        null_max_llrs=null_max,
        q=Q,
        null_kind=null_kind,
    )


# ---------------------------------------------------------------------------
# composition test (audit 18 replacement for McCrary-on-time)
# ---------------------------------------------------------------------------


@dataclass
class CompositionReport:
    p_value: float
    statistic: float
    table: np.ndarray  # (profiles-or-units x 2) pre/post in-window counts
    passed: bool


def composition_test(
    panel: CategoricalPanel,
    discovery: DiDDiscovery,
    alpha: float = 0.05,
    by: str = "unit",
) -> CompositionReport:
    """Chi-square independence of record composition pre vs post in the window.

    McCrary on calendar time is information-free when record times are
    design-determined (audit item 18); what CAN break a panel discovery is a
    compositional shift — units or covariate profiles entering/leaving the
    panel at the cutoff. Counts records per ``by`` row on each side of
    ``discovery.t0`` inside ``[t0 - W, t0 + W)``, INSIDE the discovery's own
    subset mask (issue #16: out-of-mask records can neither fail an internally
    stable discovery nor dilute a masked-subgroup shift), and tests row/side
    independence with :func:`scipy.stats.chi2_contingency`. Rows with
    all-zero counts are dropped; fewer than 2 usable rows or an empty side
    yield ``p_value = NaN, passed = False`` — never a silent pass.
    """
    if by not in ("unit", "profile"):
        raise ValueError(f"by must be 'unit' or 'profile', got {by!r}")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must lie in (0, 1), got {alpha}")
    t = panel.t
    t0, w = float(discovery.t0), float(discovery.window)
    dmask = np.asarray(discovery.mask, dtype=bool)
    pre = (t >= t0 - w) & (t < t0) & dmask
    post = (t >= t0) & (t < t0 + w) & dmask
    rows = panel.unit if by == "unit" else panel.profile_id
    k = int(rows.max()) + 1 if rows.size else 0
    table = np.column_stack(
        [
            np.bincount(rows[pre], minlength=k),
            np.bincount(rows[post], minlength=k),
        ]
    ).astype(np.int64)
    table = table[table.sum(axis=1) > 0]
    if table.shape[0] < 2 or np.any(table.sum(axis=0) == 0):
        return CompositionReport(
            p_value=float("nan"), statistic=float("nan"), table=table, passed=False
        )
    chi2, p, _, _ = stats.chi2_contingency(table)
    return CompositionReport(
        p_value=float(p), statistic=float(chi2), table=table, passed=bool(p > alpha)
    )


# ---------------------------------------------------------------------------
# anticipation test (audit 18: pre-period placebo jumps)
# ---------------------------------------------------------------------------


@dataclass
class AnticipationReport:
    shifts: tuple[int, ...]
    estimates: np.ndarray  # placebo jump estimates at T0 - shift*step (pre-data only)
    p_values: np.ndarray  # two-sided z from the model's analytic variance
    p_holm: np.ndarray
    passed: bool


def _holm(p: np.ndarray) -> np.ndarray:
    """Holm step-down adjusted p-values; NaN entries are excluded and preserved."""
    out = np.full(p.shape, np.nan)
    idx = np.flatnonzero(~np.isnan(p))
    m = idx.size
    if m == 0:
        return out
    order = np.argsort(p[idx], kind="stable")
    adj = np.empty(m)
    running = 0.0
    for rank, j in enumerate(order):
        running = max(running, float((m - rank) * p[idx[j]]))
        adj[j] = min(running, 1.0)
    out[idx] = adj
    return out


def anticipation_test(
    panel: CategoricalPanel,
    background: DiDBackground,
    discovery: DiDDiscovery,
    shifts: tuple[int, ...] = (1, 2, 3),
    alpha: float = 0.05,
) -> AnticipationReport:
    """Pre-period placebo jumps at ``T0 - shift * step``, Holm across shifts.

    ``step`` is the median diff of unique panel times. Each placebo estimate
    uses the discovery's own window width and subset mask but is RESTRICTED
    to records with ``t < T0`` — never contaminated by the real jump. The
    estimator matches the discovery: ``single_delta`` uses the profiled
    ``Delta_hat = C/B`` with ``Var = 1/B_tilde``; otherwise the double-beta
    contrast ``q1 - q2`` with ``Var = 1/B1 + 1/B0`` (for a Bernoulli-model
    discovery the contrast runs on working residuals — a documented normal
    approximation for this diagnostic only). Two-sided normal p-values; a
    shift with insufficient two-sided support gets NaN (excluded from Holm,
    visible in the report). ``passed`` requires at least one usable shift AND
    every usable Holm p above ``alpha`` — all-degenerate is a fail, never a
    silent pass.
    """
    shifts = tuple(int(s) for s in shifts)
    if len(shifts) == 0 or any(s < 1 for s in shifts):
        raise ValueError(f"shifts must be a nonempty tuple of ints >= 1, got {shifts}")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must lie in (0, 1), got {alpha}")
    u = np.unique(panel.t)
    if u.size < 2:
        raise ValueError("anticipation_test requires >= 2 distinct time points")
    step = float(np.median(np.diff(u)))

    if background.kind == "normal":
        assert background.r is not None and background.sigma2 is not None
        r, sigma2 = background.r, background.sigma2
    else:
        # Documented normal approximation for this diagnostic only.
        r, sigma2 = working_residuals(panel.theta, background.fitted)

    sel = np.asarray(discovery.mask, dtype=bool) & (panel.t < discovery.t0)
    n_profiles = int(np.prod(panel.dim_sizes)) if panel.m else 1
    estimates = np.full(len(shifts), np.nan)
    p_values = np.full(len(shifts), np.nan)
    for i, shift in enumerate(shifts):
        t0p = float(discovery.t0 - shift * step)
        ws = window_stats(panel.t, r, sigma2, t0p, float(discovery.window))
        g1 = ws.g1 & sel
        g0 = ws.g0 & sel
        in_w = g1 | g0
        if not g1.any() or not g0.any():
            continue  # insufficient two-sided support: NaN, excluded from Holm
        wsel = WindowStats(
            in_window=in_w,
            g1=g1,
            c=np.where(in_w, ws.c, 0.0),
            b=np.where(in_w, ws.b, 0.0),
        )
        if discovery.method == "single_delta":
            c_prof, b_prof = single_delta_stats(wsel, panel.profile_id, n_profiles=n_profiles)
            b_sum = float(b_prof.sum())
            if b_sum <= 0.0:
                continue  # no identifying within-profile variation
            est, var = float(c_prof.sum()) / b_sum, 1.0 / b_sum
        else:
            q1, q2 = double_beta_q(wsel, np.ones((panel.n, 1), dtype=bool))
            b1 = float((wsel.b * g1).sum())
            b0 = float((wsel.b * g0).sum())
            if not (np.isfinite(q1[0]) and np.isfinite(q2[0])) or b1 <= 0.0 or b0 <= 0.0:
                continue
            est, var = float(q1[0] - q2[0]), 1.0 / b1 + 1.0 / b0
        estimates[i] = est
        p_values[i] = 2.0 * float(stats.norm.sf(abs(est) / np.sqrt(var)))

    p_holm = _holm(p_values)
    usable = ~np.isnan(p_holm)
    passed = bool(usable.any() and np.all(p_holm[usable] > alpha))
    return AnticipationReport(
        shifts=shifts, estimates=estimates, p_values=p_values, p_holm=p_holm, passed=passed
    )
