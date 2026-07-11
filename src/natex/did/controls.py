"""Control identification for discovered DD subsets (thesis section 6.3.2, repaired).

Three control constructions for a discovered treated subset ``s_tau`` at
cutoff ``t0``. Estimation uses the FULL pre/post split at ``discovery.t0``
(pre: ``t < t0``; post: ``t >= t0``), not the scan window.

* :func:`dd_control` — standard DD control (Eq 6.18) with the audit's typo
  repairs: every mean divides by the COUNT OF RECORDS ACTUALLY SUMMED (the
  printed ``1/|D \\ s_tau|`` and ``1/(T0 |.|)`` denominators assume a
  balanced panel), so unbalanced panels are handled; a time with zero
  control records gets a NaN counterfactual — never 0.
* :func:`synthetic_control` — UNIT-level simplex weights (audit typo: the
  printed Eq 6.19 indexes records; weights live on units) minimizing the
  pre-period trajectory mismatch, solved deterministically with SLSQP from
  the uniform start. DOCUMENTED DEVIATION from Abadie et al.: this is an
  outcome-only pre-fit — the covariate V-weight nesting of the original
  synthetic-control method is NOT implemented.
* :func:`gess_control` — Greedy Expansion Subset Search (Algorithm 9) with
  the audit item-14 repair: line 6 is **argmin** over candidate MSEs (the
  printed argmax is a typo) and the incumbent MSE initializes to +inf via
  the empty control of ``s_sup = s_tau``, so the first expansion producing
  a nonempty control always wins.

``pre_mse`` is Eq 6.17 reported as the MEAN over the DEFINED pre-period
``s_tau`` records (the printed sum assumes every record defined; the mean
keeps GESS candidates with different undefined-time counts comparable). It
is ``+inf`` when no pre-period record is defined — NEVER 0 on failure, and
``y0_hat`` is NaN wherever undefined. Records with non-finite ``y`` are
excluded from every mean and residual (they behave as absent records).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from natex.did.mdss import SubsetState
from natex.did.panel import CategoricalPanel
from natex.did.suddds import DiDDiscovery
from natex.estimate.simplex import (
    MISSING_W_TOL,
    fit_simplex_weights,
    weighted_counterfactual,
)

# Simplex fitting and missing-donor renormalization moved verbatim to
# natex.estimate.simplex (phase 5 task 6) so IV/SC donor selection shares
# the same deterministic fitter; both names re-exported for compatibility.
_MISSING_W_TOL = MISSING_W_TOL
_weighted_counterfactual_by_time = weighted_counterfactual


@dataclass
class ControlResult:
    """Counterfactual construction for the ``s_tau`` records of one discovery."""

    method: str  # "dd" | "synthetic" | "gess"
    y0_hat: np.ndarray  # (n_tau,) counterfactual per s_tau record, NaN where undefined
    pre_mse: float  # Eq 6.17 over defined pre records; +inf if none; NEVER 0 on failure
    control_mask: np.ndarray | None  # (n,) records forming the control set (dd, gess)
    weights: np.ndarray | None  # per-UNIT simplex weights (synthetic; Eq 6.19 repaired)
    alpha: float | None  # fixed-effect offset (dd, gess); NaN when undefined
    extras: dict = field(default_factory=dict)


def _require_outcome(panel: CategoricalPanel) -> np.ndarray:
    if panel.y is None:
        raise ValueError(
            "control identification requires an outcome (panel.y is None); "
            "discovery upstream never needed y, estimation does"
        )
    return np.asarray(panel.y, dtype=float)


def _profile_from_discovery(panel: CategoricalPanel, discovery: DiDDiscovery) -> list[np.ndarray]:
    """Per-dimension value masks of s_tau's covariate profile.

    Dimensions absent from ``discovery.subset_values`` are unconstrained
    (all-True); constrained dimensions include exactly the decoded values
    recorded at discovery time.
    """
    included: list[np.ndarray] = []
    for j, name in enumerate(panel.dim_names):
        vals = discovery.subset_values.get(name)
        if vals is None:
            included.append(np.ones(panel.dim_sizes[j], dtype=bool))
        else:
            included.append(np.isin(np.asarray(panel.dim_values[j]), np.asarray(vals)))
    return included


def _counterfactual(
    panel: CategoricalPanel,
    y: np.ndarray,
    tau_mask: np.ndarray,
    ctrl_mask: np.ndarray,
    t0: float,
) -> tuple[np.ndarray, float, float, int]:
    """Eqs 6.18/6.20 counterfactual with count-corrected denominators.

    Returns ``(y0_hat, alpha, pre_mse, n_undefined_times)`` where ``y0_hat``
    aligns with the ``s_tau`` records in panel order. All denominators are
    counts of records actually summed (unbalanced-panel tolerant); a time
    with zero control records yields NaN — never 0.
    """
    finite = np.isfinite(y)
    _, code = np.unique(panel.t, return_inverse=True)
    n_t = int(code.max()) + 1 if code.size else 0
    ctrl = ctrl_mask & finite

    c_cnt = np.bincount(code[ctrl], minlength=n_t).astype(float)
    c_sum = np.bincount(code[ctrl], weights=y[ctrl], minlength=n_t)
    c_mean = np.where(c_cnt > 0, c_sum / np.where(c_cnt > 0, c_cnt, 1.0), np.nan)

    pre = panel.t < t0
    tau_pre = tau_mask & pre & finite
    ctrl_pre = ctrl & pre
    if tau_pre.any() and ctrl_pre.any():
        alpha = float(y[tau_pre].mean() - y[ctrl_pre].mean())
    else:
        alpha = float("nan")

    tau_idx = np.flatnonzero(tau_mask)
    y0_hat = alpha + c_mean[code[tau_idx]]

    tau_t_present = np.bincount(code[tau_mask], minlength=n_t) > 0
    n_undefined = int(np.count_nonzero(tau_t_present & (c_cnt == 0)))

    ok = pre[tau_idx] & finite[tau_idx] & np.isfinite(y0_hat)
    if ok.any():
        resid = y[tau_idx][ok] - y0_hat[ok]
        pre_mse = float(np.mean(resid**2))
    else:
        pre_mse = float("inf")
    return y0_hat, alpha, pre_mse, n_undefined


def dd_control(panel: CategoricalPanel, discovery: DiDDiscovery) -> ControlResult:
    """Standard DD control (Eq 6.18, count-corrected): control = D \\ s_tau.

    ``y0_hat[i] = alpha + mean(y of control records at t_i)`` with
    ``alpha = mean(s_tau pre) - mean(control pre)``, every mean over the
    records actually present. Times with no control records are NaN and
    counted in ``extras["n_undefined_times"]``.
    """
    y = _require_outcome(panel)
    tau_mask = np.asarray(discovery.mask, dtype=bool)
    ctrl_mask = ~tau_mask
    y0_hat, alpha, pre_mse, n_undefined = _counterfactual(
        panel, y, tau_mask, ctrl_mask, discovery.t0
    )
    return ControlResult(
        method="dd",
        y0_hat=y0_hat,
        pre_mse=pre_mse,
        control_mask=ctrl_mask,
        weights=None,
        alpha=alpha,
        extras={
            "n_undefined_times": n_undefined,
            "n_control": int(ctrl_mask.sum()),
            "n_tau": int(tau_mask.sum()),
        },
    )


def synthetic_control(panel: CategoricalPanel, discovery: DiDDiscovery) -> ControlResult:
    """Synthetic control: unit-level simplex weights fit on pre-period trajectories.

    Control units are the units with NO ``s_tau`` records (partially treated
    units are excluded entirely), restricted to the DONOR POOL of units with
    finite-y records at every pre-period time where the treated subset has
    records (Abadie's balanced-donor requirement; without it one sparse unit
    voids every common time and the fit fails on any thin panel — the
    ``extras["n_donors_dropped"]`` count reports the exclusions). Weights
    ``w >= 0, sum(w) = 1`` minimize
    ``||ybar_tau(t) - sum_u w_u ybar_u(t)||^2`` over those pre-period times,
    solved with SLSQP from the uniform start (deterministic).

    DOCUMENTED DEVIATION: outcome-only pre-fit; Abadie et al.'s covariate
    V-weights are not implemented. ``y0_hat = sum_u w_u ybar_u(t)`` with no
    alpha offset (the weights absorb levels); missing donor cells are
    renormalized away while their weight mass is <= 0.1, beyond which the
    time is NaN — never 0.
    """
    y = _require_outcome(panel)
    tau_mask = np.asarray(discovery.mask, dtype=bool)
    finite = np.isfinite(y)
    _, code = np.unique(panel.t, return_inverse=True)
    n_t = int(code.max()) + 1 if code.size else 0
    n_units = int(panel.unit_values.shape[0])
    n_tau = int(tau_mask.sum())

    is_ctrl_unit = np.ones(n_units, dtype=bool)
    is_ctrl_unit[np.unique(panel.unit[tau_mask])] = False
    ctrl_units = np.flatnonzero(is_ctrl_unit)

    def _fail(reason: str) -> ControlResult:
        return ControlResult(
            method="synthetic",
            y0_hat=np.full(n_tau, np.nan),
            pre_mse=float("inf"),
            control_mask=None,
            weights=None,
            alpha=None,
            extras={"failure": reason, "control_units": ctrl_units, "n_tau": n_tau},
        )

    if ctrl_units.size == 0:
        return _fail("no control units (every unit has s_tau records)")

    # Unit-by-time outcome means over finite-y records.
    key = panel.unit * n_t + code
    cell_cnt = np.bincount(key[finite], minlength=n_units * n_t).reshape(n_units, n_t)
    cell_sum = np.bincount(
        key[finite], weights=y[finite], minlength=n_units * n_t
    ).reshape(n_units, n_t)
    unit_mean = np.where(cell_cnt > 0, cell_sum / np.maximum(cell_cnt, 1), np.nan)

    # Treated trajectory: per-time mean over s_tau records.
    tf = tau_mask & finite
    tau_cnt = np.bincount(code[tf], minlength=n_t)
    tau_sum = np.bincount(code[tf], weights=y[tf], minlength=n_t)
    y_tau = np.where(tau_cnt > 0, tau_sum / np.maximum(tau_cnt, 1), np.nan)

    times = np.unique(panel.t)
    common = (times < discovery.t0) & (tau_cnt > 0)
    if not common.any():
        return _fail("no pre-period times with s_tau records")
    # Donor pool: candidate units observed at EVERY common pre time. Dropping
    # incomplete units (instead of shrinking the time set) keeps thin panels
    # fittable — previously any one sparse unit voided all common times.
    n_candidates = ctrl_units.size
    complete = np.all(cell_cnt[ctrl_units][:, common] > 0, axis=1)
    ctrl_units = ctrl_units[complete]
    n_dropped = int(n_candidates - ctrl_units.size)
    if ctrl_units.size == 0:
        return _fail("no control unit has records at every s_tau pre-period time")

    y_fit = y_tau[common]  # (n_common,)
    y_ctrl = unit_mean[ctrl_units][:, common].T  # (n_common, n_c)

    # SLSQP from the uniform start on the scale-normalized SSE (the phase-3
    # scale-invariance fix lives in estimate.simplex; regression:
    # test_synthetic_control_scale_invariant_optimization).
    fit = fit_simplex_weights(y_fit, y_ctrl)
    w = fit.weights

    # Counterfactual per time: renormalized over present donors; NaN only when
    # the missing donor weight mass exceeds _MISSING_W_TOL.
    y0_by_time = _weighted_counterfactual_by_time(unit_mean[ctrl_units], w)  # (n_t,)

    tau_idx = np.flatnonzero(tau_mask)
    y0_hat = y0_by_time[code[tau_idx]]

    pre = panel.t < discovery.t0
    ok = pre[tau_idx] & finite[tau_idx] & np.isfinite(y0_hat)
    if ok.any():
        resid = y[tau_idx][ok] - y0_hat[ok]
        pre_mse = float(np.mean(resid**2))
    else:
        pre_mse = float("inf")

    tau_t_present = np.bincount(code[tau_mask], minlength=n_t) > 0
    n_undefined = int(np.count_nonzero(tau_t_present & ~np.isfinite(y0_by_time)))

    return ControlResult(
        method="synthetic",
        y0_hat=y0_hat,
        pre_mse=pre_mse,
        control_mask=None,
        weights=w,
        alpha=None,
        extras={
            "control_units": ctrl_units,
            "control_unit_values": np.asarray(panel.unit_values)[ctrl_units],
            "n_donors_dropped": n_dropped,
            "n_common_pre_times": int(common.sum()),
            "n_undefined_times": n_undefined,
            "converged": fit.converged,
            "fit_sse": fit.sse,  # de-normalized back to y units
            "note": "outcome-only pre-fit; Abadie covariate V-weights not implemented",
        },
    )


def gess_control(
    panel: CategoricalPanel, discovery: DiDDiscovery, full_dimension: bool = False
) -> ControlResult:
    """Greedy Expansion Subset Search (Algorithm 9, audit item 14 repaired).

    Starting from ``s_sup = s_tau`` (empty control, MSE = +inf), each step
    evaluates every monotone profile expansion — one covariate VALUE added
    along a constrained dimension (default), or one whole DIMENSION relaxed
    to all-True (``full_dimension=True``, Eq 6.21) — scores each candidate
    by the Eq 6.20/6.17 control MSE of ``s_c = s_sup' \\ s_tau`` (count-
    corrected denominators, alpha offset as in :func:`dd_control`), and
    takes the **argmin** (the printed argmax is a typo). It stops when no
    candidate strictly lowers the MSE; termination is bounded by the total
    value count because accepted masks grow monotonically.

    ``extras["mse_trace"]`` holds the incumbent MSE at initialization and
    after each accepted expansion (monotone nonincreasing);
    ``extras["expansions"]`` lists the accepted ``{"dim", "value"}`` steps
    (``value=None`` for a whole-dimension relaxation); ``extras["profile"]``
    is the final decoded ``s_sup`` profile (all-True dims omitted).
    """
    y = _require_outcome(panel)
    tau_mask = np.asarray(discovery.mask, dtype=bool)
    included = _profile_from_discovery(panel, discovery)

    def evaluate(inc: list[np.ndarray]) -> tuple[float, tuple]:
        ctrl = panel.subset_mask(inc) & ~tau_mask
        y0_hat, alpha, pre_mse, n_undefined = _counterfactual(
            panel, y, tau_mask, ctrl, discovery.t0
        )
        return pre_mse, (y0_hat, alpha, ctrl, n_undefined)

    best_mse, best_eval = evaluate(included)  # s_sup = s_tau -> empty control -> +inf
    mse_trace = [best_mse]
    expansions: list[dict] = []

    while True:
        candidates: list[tuple[int, int | None, list[np.ndarray]]] = []
        for j in range(panel.m):
            if included[j].all():
                continue  # unconstrained dim: nothing to add
            if full_dimension:
                cand = [m.copy() for m in included]
                cand[j][:] = True
                candidates.append((j, None, cand))
            else:
                for k in np.flatnonzero(~included[j]):
                    cand = [m.copy() for m in included]
                    cand[j][k] = True
                    candidates.append((j, int(k), cand))
        if not candidates:
            break

        evals = [evaluate(cand) for _, _, cand in candidates]
        # Audit item 14: argmin (the thesis prints argmax on line 6 — the
        # regression-tested bug); +inf failures are never selected over a
        # finite MSE, and NaN never appears (failures are +inf by policy).
        i_best = int(np.argmin([mse for mse, _ in evals]))
        if not evals[i_best][0] < best_mse:  # no strict improvement -> stop
            break
        j, k, cand = candidates[i_best]
        included = cand
        best_mse, best_eval = evals[i_best]
        mse_trace.append(best_mse)
        expansions.append(
            {
                "dim": panel.dim_names[j],
                "value": None if k is None else np.asarray(panel.dim_values[j])[k].item(),
            }
        )

    y0_hat, alpha, ctrl_mask, n_undefined = best_eval
    return ControlResult(
        method="gess",
        y0_hat=y0_hat,
        pre_mse=best_mse,
        control_mask=ctrl_mask,
        weights=None,
        alpha=alpha,
        extras={
            "mse_trace": mse_trace,
            "expansions": expansions,
            "profile": SubsetState(included).values(panel),
            "n_undefined_times": n_undefined,
            "n_control": int(ctrl_mask.sum()),
            "n_tau": int(tau_mask.sum()),
            "full_dimension": full_dimension,
        },
    )
