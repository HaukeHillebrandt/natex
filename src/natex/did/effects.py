"""Effect estimation for discovered DD subsets (thesis section 6.3.3, repaired).

Given a discovery ``(s_tau, T0)`` and a control construction from
:mod:`natex.did.controls`, :func:`did_effect` estimates the treated-subset
effect as the mean post-period gap ``y - y_hat(0)`` with the audit item-19
repair: for a CONTINUOUS treatment the raw DD contrast estimates
``zeta * tau`` (the theta-jump times the outcome effect), so it is
normalized by the dose ``delta_hat`` — the SAME DD contrast, with the SAME
fitted control structure, applied to ``theta`` instead of ``y``.

:func:`tau_randomization_test` is the audit item-5 repair of the thesis's
one-sided 95th-percentile placebo rule: two-sided STUDENTIZED statistic,
+1-rank p-value, and placebo subsets matched in shape (same number of
covariate profiles as ``s_tau``, same T0, drawn from profiles with no
``s_tau`` records). :func:`placebo_dimension_tests` runs the same test with
each free dimension's composition share as the outcome (thesis section
6.3.1(3)), Holm-corrected across dimensions.

Failure policy throughout: NaN, never 0.0; no result is fabricated when a
counterfactual, dose, or placebo pool is undefined.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from itertools import combinations
from math import comb
from typing import Protocol, runtime_checkable

import numpy as np

from natex.did.controls import (
    ControlResult,
    _counterfactual,
    _weighted_counterfactual_by_time,
    dd_control,
    gess_control,
    synthetic_control,
)
from natex.did.panel import CategoricalPanel
from natex.did.suddds import DiDDiscovery

_CONTROLS = {"dd": dd_control, "synthetic": synthetic_control, "gess": gess_control}
_DOSE_TOL = 1e-10  # |delta_hat| below this: normalization undefined -> NaN
_ENUM_MAX = 200  # Q="auto": enumerate when the pool yields <= this many placebos
_SAMPLE_Q = 199  # Q="auto" sampling fallback
_MIN_USABLE = 5  # fewer usable placebos -> p = NaN (never a fake 1.0)


@dataclass
class DiDEffect:
    """Effect estimate for one discovery under one control construction."""

    tau: float  # NaN on failure, never 0.0
    se: float  # CR1 cluster(=post-period)-robust SE of the record-weighted tau
    method: str  # control method name ("dd" | "synthetic" | "gess")
    pre_mse: float  # control fit quality (Eq 6.17, from the ControlResult)
    n_treated_post: int  # s_tau post records USED (finite y and counterfactual)
    dose: float | None  # theta DD contrast used for normalization; None when not applied
    extras: dict = field(default_factory=dict)


def _theta_is_binary(theta: np.ndarray) -> bool:
    vals = np.unique(theta[~np.isnan(theta)])
    return vals.size <= 2 and set(vals.tolist()) <= {0.0, 1.0}


def _resolve_control(
    panel: CategoricalPanel, discovery: DiDDiscovery, control: str | ControlResult
) -> ControlResult:
    if isinstance(control, ControlResult):
        return control
    if control not in _CONTROLS:
        raise ValueError(f"control must be one of {tuple(_CONTROLS)} or a ControlResult, "
                         f"got {control!r}")
    return _CONTROLS[control](panel, discovery)


def _apply_control_to(
    panel: CategoricalPanel,
    discovery: DiDDiscovery,
    control: ControlResult,
    v: np.ndarray,
) -> np.ndarray:
    """Counterfactual ``v0_hat`` for an arbitrary per-record variable ``v``.

    Reuses the FITTED control structure — the control record mask (dd, gess)
    or the unit-level simplex weights (synthetic) — and never refits, so the
    contrast applied to ``v`` is exactly the contrast that produced
    ``control.y0_hat`` (audit 19: "same contrast, same control set").
    Undefined cells are NaN, never 0.
    """
    tau_mask = np.asarray(discovery.mask, dtype=bool)
    n_tau = int(tau_mask.sum())
    v = np.asarray(v, dtype=float)

    if control.method == "synthetic":
        if control.weights is None:
            return np.full(n_tau, np.nan)
        w = np.asarray(control.weights, dtype=float)
        ctrl_units = np.asarray(control.extras["control_units"], dtype=np.int64)
        finite = np.isfinite(v)
        _, code = np.unique(panel.t, return_inverse=True)
        n_t = int(code.max()) + 1 if code.size else 0
        n_units = int(panel.unit_values.shape[0])
        key = panel.unit * n_t + code
        cell_cnt = np.bincount(key[finite], minlength=n_units * n_t).reshape(n_units, n_t)
        cell_sum = np.bincount(
            key[finite], weights=v[finite], minlength=n_units * n_t
        ).reshape(n_units, n_t)
        unit_mean = np.where(cell_cnt > 0, cell_sum / np.maximum(cell_cnt, 1), np.nan)
        v0_by_time = _weighted_counterfactual_by_time(unit_mean[ctrl_units], w)
        return v0_by_time[code[np.flatnonzero(tau_mask)]]

    if control.control_mask is None:
        return np.full(n_tau, np.nan)
    v0_hat, _alpha, _mse, _n_undef = _counterfactual(
        panel, v, tau_mask, np.asarray(control.control_mask, dtype=bool), discovery.t0
    )
    return v0_hat


def _mean_gap(
    panel: CategoricalPanel,
    discovery: DiDDiscovery,
    v: np.ndarray,
    v0_hat: np.ndarray,
) -> tuple[float, float, int, int, int]:
    """Mean post-period gap of ``v - v0_hat`` over the ``s_tau`` records.

    Returns ``(gap_mean, se, n_used, n_skipped, h)``: the record-level mean
    over usable post records, its time-cluster-robust standard error, the used
    and skipped (NaN cell) post-record counts, and the number of usable post
    periods ``h``. The SE is the CR1 cluster(=period) variance of the SAME
    record-weighted estimator (issue #17): with per-period usable counts
    ``n_g``, period means ``gbar_g`` and ``N = n_used``,
    ``se = sqrt(h/(h-1) * sum_g ((n_g/N) * (gbar_g - tau))^2)`` — reducing
    exactly to ``std(period_means, ddof=1)/sqrt(h)`` when cells are balanced —
    and NaN when fewer than two usable post periods.
    """
    tau_idx = np.flatnonzero(np.asarray(discovery.mask, dtype=bool))
    t_tau = panel.t[tau_idx]
    post = t_tau >= discovery.t0
    usable = post & np.isfinite(v[tau_idx]) & np.isfinite(v0_hat)
    n_used = int(usable.sum())
    n_skipped = int(post.sum()) - n_used
    if n_used == 0:
        return float("nan"), float("nan"), 0, n_skipped, 0
    gaps = v[tau_idx][usable] - v0_hat[usable]
    tau = float(gaps.mean())
    times, code = np.unique(t_tau[usable], return_inverse=True)
    h = int(times.size)
    if h >= 2:
        n_g = np.bincount(code).astype(float)
        gbar = np.bincount(code, weights=gaps) / n_g
        se = float(np.sqrt(h / (h - 1) * np.sum(((n_g / n_used) * (gbar - tau)) ** 2)))
    else:
        se = float("nan")
    return tau, se, n_used, n_skipped, h


def _dose_contrast(
    panel: CategoricalPanel, discovery: DiDDiscovery, control: ControlResult
) -> float:
    """theta DD contrast under the fitted control structure (audit 19).

    ``delta_hat = mean_post(theta - theta0_hat) - mean_pre(theta - theta0_hat)``
    over the usable ``s_tau`` records. The pre-mean term is ~0 for dd/gess
    (the alpha offset already balances the pre period) and guards the
    synthetic path, whose weights were fit on ``y`` levels, not ``theta``.
    NaN when either side has no usable record.
    """
    theta0 = _apply_control_to(panel, discovery, control, panel.theta)
    tau_idx = np.flatnonzero(np.asarray(discovery.mask, dtype=bool))
    t_tau = panel.t[tau_idx]
    gaps = panel.theta[tau_idx] - theta0
    ok = np.isfinite(gaps)
    post_ok = ok & (t_tau >= discovery.t0)
    pre_ok = ok & (t_tau < discovery.t0)
    if not post_ok.any() or not pre_ok.any():
        return float("nan")
    return float(gaps[post_ok].mean() - gaps[pre_ok].mean())


def did_effect(
    panel: CategoricalPanel,
    discovery: DiDDiscovery,
    control: str | ControlResult = "dd",
    dose_normalize: str | bool = "auto",
) -> DiDEffect:
    """Estimate the effect of one discovery: mean post-period gap, dose-repaired.

    ``tau_hat`` is the mean over ``s_tau`` post-period records of
    ``y - y_hat(0)``, skipping NaN counterfactual cells (count reported in
    ``extras["n_skipped_nan"]``); all cells NaN gives ``tau = NaN``, never 0.

    Dose normalization (audit item 19): with ``dose_normalize=True`` — or
    ``"auto"`` when ``theta`` is NOT binary in {0, 1} — the raw contrast
    (which estimates ``zeta * tau`` for a continuous treatment) is divided by
    ``delta_hat``, the same DD contrast applied to ``theta`` under the same
    fitted control structure; ``se`` is scaled by ``1/|delta_hat|`` alongside
    (delta-method with ``delta_hat`` treated as fixed), so the studentized
    ``tau/se`` is invariant to normalization. ``|delta_hat| < 1e-10`` gives
    ``tau = se = NaN`` with the near-zero dose still reported.

    Raises ``ValueError`` when ``panel.y`` is None: estimation requires the
    outcome even though discovery upstream never read it.
    """
    if panel.y is None:
        raise ValueError(
            "effect estimation requires an outcome (panel.y is None); "
            "discovery upstream never needed y, estimation does"
        )
    if not (dose_normalize == "auto" or isinstance(dose_normalize, bool)):
        raise ValueError(f"dose_normalize must be 'auto', True or False, got {dose_normalize!r}")
    ctrl = _resolve_control(panel, discovery, control)
    y = np.asarray(panel.y, dtype=float)
    tau_rf, se_rf, n_used, n_skipped, h = _mean_gap(panel, discovery, y, ctrl.y0_hat)

    apply_dose = (
        not _theta_is_binary(panel.theta) if dose_normalize == "auto" else bool(dose_normalize)
    )
    extras = {
        "tau_rf": tau_rf,
        "se_rf": se_rf,
        "n_skipped_nan": n_skipped,
        "n_post_periods": h,
    }
    if not apply_dose:
        return DiDEffect(
            tau=tau_rf,
            se=se_rf,
            method=ctrl.method,
            pre_mse=ctrl.pre_mse,
            n_treated_post=n_used,
            dose=None,
            extras=extras,
        )

    delta = _dose_contrast(panel, discovery, ctrl)
    if not np.isfinite(delta) or abs(delta) < _DOSE_TOL:
        tau, se = float("nan"), float("nan")
    else:
        tau, se = tau_rf / delta, se_rf / abs(delta)
    return DiDEffect(
        tau=tau,
        se=se,
        method=ctrl.method,
        pre_mse=ctrl.pre_mse,
        n_treated_post=n_used,
        dose=float(delta) if np.isfinite(delta) else float("nan"),
        extras=extras,
    )


# ---------------------------------------------------------------------------
# per-period gaps (reporting; phase report-paper task 2)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PeriodGaps:
    """Per-period treated-minus-control mean gaps (descriptive, for reporting).

    ``gap[i]`` is the mean of ``y - y0_hat`` over the usable ``s_tau`` records
    of period ``times[i]``; ``n[i]`` counts them. Feeds the pretrend figure:
    pre-``t0`` gaps near zero support the control fit, post-``t0`` gaps show
    the (raw, un-dose-normalized) effect path.
    """

    times: np.ndarray  # sorted unique usable s_tau periods (pre AND post)
    gap: np.ndarray  # per-period mean of y - y0_hat over usable s_tau records
    n: np.ndarray  # usable record count per period (int)
    t0: float
    control: str  # "dd" | "synthetic" | "gess"


def period_gaps(
    panel: CategoricalPanel,
    discovery: DiDDiscovery,
    control: str | ControlResult = "dd",
) -> PeriodGaps:
    """Per-period mean treated-minus-control gap for the pretrend figure.

    Descriptive ONLY — no new inference. The counterfactual reuses the FITTED
    control contrast via :func:`_resolve_control` + :func:`_apply_control_to`
    (audit 19: same contrast, same control set as :func:`did_effect`), so the
    ``n``-weighted average of the post-period gaps equals :func:`did_effect`'s
    reduced-form ``tau``. Gaps are RAW ``y`` contrasts, never dose-normalized.
    Periods with zero usable (finite ``y`` and counterfactual) records are
    OMITTED, never zero-filled.

    Raises ``ValueError`` when ``panel.y`` is None — reporting never
    fabricates outcomes.
    """
    if panel.y is None:
        raise ValueError(
            "period_gaps requires an outcome (panel.y is None); "
            "reporting never fabricates outcomes"
        )
    ctrl = _resolve_control(panel, discovery, control)
    y = np.asarray(panel.y, dtype=float)
    y0_hat = _apply_control_to(panel, discovery, ctrl, y)
    tau_idx = np.flatnonzero(np.asarray(discovery.mask, dtype=bool))
    usable = np.isfinite(y[tau_idx]) & np.isfinite(y0_hat)
    gaps = y[tau_idx][usable] - y0_hat[usable]
    times, code = np.unique(panel.t[tau_idx][usable], return_inverse=True)
    counts = np.bincount(code, minlength=times.size)  # >= 1 per unique time
    gap = np.bincount(code, weights=gaps, minlength=times.size) / np.maximum(counts, 1)
    return PeriodGaps(
        times=times,
        gap=gap,
        n=counts.astype(np.int64),
        t0=float(discovery.t0),
        control=ctrl.method,
    )


# ---------------------------------------------------------------------------
# pluggable estimator backends (spec non-goal boundary: interface only)
# ---------------------------------------------------------------------------


@runtime_checkable
class DiDEstimatorBackend(Protocol):
    """Pluggable effect-estimator interface (audit section 3 / spec non-goals).

    Staggered-adoption group-time ATT (Callaway-Sant'Anna) is a FUTURE
    backend; phase 3 deliberately ships only this protocol plus the default
    ``"mean_gap"`` backend, keeping the estimator swappable without touching
    discovery or control identification.
    """

    name: str

    def estimate(
        self, panel: CategoricalPanel, discovery: DiDDiscovery, control: ControlResult
    ) -> DiDEffect: ...


@dataclass
class MeanGapBackend:
    """Default backend: :func:`did_effect` with auto dose normalization."""

    name: str = "mean_gap"
    dose_normalize: str | bool = "auto"

    def estimate(
        self, panel: CategoricalPanel, discovery: DiDDiscovery, control: ControlResult
    ) -> DiDEffect:
        return did_effect(panel, discovery, control=control, dose_normalize=self.dose_normalize)


ESTIMATOR_BACKENDS: dict[str, DiDEstimatorBackend] = {"mean_gap": MeanGapBackend()}


# ---------------------------------------------------------------------------
# tau randomization test (audit item 5)
# ---------------------------------------------------------------------------


@dataclass
class TauRandomizationReport:
    """Two-sided studentized placebo test for one discovery's effect."""

    p_value: float
    observed: float  # studentized tau_hat / se
    null_stats: np.ndarray  # usable studentized placebo statistics
    q: int  # number of USABLE placebos (the +1-rank denominator is q + 1)
    mode: str  # "enumerate" | "sample"
    extras: dict = field(default_factory=dict)


def _profiles_to_values(
    panel: CategoricalPanel,
    profiles: tuple[int, ...],
    constrained_dims: set[str] | None = None,
) -> dict[str, list]:
    """Decoded per-dimension values covering the given full-profile ids.

    ``constrained_dims`` (audit 5 matched shapes) restricts the output to the
    dimensions the OBSERVED discovery constrains, so a placebo discovery has
    the same free/constrained dimension pattern as ``s_tau`` — a gess placebo
    then expands in the same space the observed gess did. Full-profile
    seeding (``None`` = all dims) starved every prop99 gess placebo: no state
    is within one value change of another on all 7 binned covariates. For
    multi-profile placebos the conjunction-of-unions cover can exceed the
    exact profile set; the placebo MASK stays exact — the cover only seeds
    gess expansion.
    """
    codes = np.unravel_index(np.asarray(profiles, dtype=np.int64), panel.dim_sizes)
    return {
        panel.dim_names[j]: np.unique(np.asarray(panel.dim_values[j])[codes[j]]).tolist()
        for j in range(panel.m)
        if constrained_dims is None or panel.dim_names[j] in constrained_dims
    }


def _without_records(panel: CategoricalPanel, drop: np.ndarray) -> CategoricalPanel:
    """Panel with ``drop`` records removed (dims/units unchanged)."""
    keep = ~np.asarray(drop, dtype=bool)
    return replace(
        panel,
        codes=panel.codes[keep],
        t=panel.t[keep],
        theta=panel.theta[keep],
        y=None if panel.y is None else np.asarray(panel.y)[keep],
        unit=panel.unit[keep],
    )


def _studentized(eff: DiDEffect) -> float:
    """tau/se; NaN when undefined — except exact zero movement, which is 0.

    ``tau == 0 and se == 0`` means every usable post-period gap is exactly 0:
    provably NO movement, the least extreme statistic possible, so it maps to
    0.0 (this arises structurally, e.g. composition shares of time-invariant
    covariate dimensions — the prop99 panel). Any other non-finite tau/se, or
    ``se == 0`` with ``tau != 0``, stays NaN: never a fabricated value.
    """
    if eff.tau == 0.0 and eff.se == 0.0:
        return 0.0
    if not np.isfinite(eff.tau) or not np.isfinite(eff.se) or eff.se == 0.0:
        return float("nan")
    return float(eff.tau / eff.se)


def tau_randomization_test(
    panel: CategoricalPanel,
    discovery: DiDDiscovery,
    control: str = "dd",
    Q: int | str = "auto",
    rng: np.random.Generator | None = None,
) -> TauRandomizationReport:
    """Two-sided studentized placebo test for tau_hat (audit item 5 in full).

    Placebo subsets are MATCHED IN SHAPE: each draws the same number of full
    covariate profiles as ``s_tau`` from the pool of profiles containing no
    ``s_tau`` records, keeps the discovery's T0, and constrains the SAME
    dimensions the observed discovery constrains (its ``subset_values`` keys)
    so a gess placebo expands in the same space the observed gess did — the
    placebo record mask itself stays the exact full-profile set. The statistic is the
    studentized ``|tau_p / se_p|`` against ``|tau_hat / se|``, so a planted
    NEGATIVE effect rejects (the thesis's one-sided 95th-percentile rule
    cannot). The p-value uses the +1-rank rule:
    ``(1 + #{|null| >= |observed|}) / (1 + q)`` over the ``q`` usable
    placebos. Effects are estimated in reduced form (``dose_normalize=False``)
    — the studentized statistic is invariant to dose normalization because
    :func:`did_effect` scales ``se`` by ``1/|delta_hat|`` alongside ``tau``.
    Placebo effects are computed with the ``s_tau`` records REMOVED from the
    panel (Abadie's placebo-in-space practice), so a real effect never
    contaminates placebo controls.

    ``Q="auto"`` enumerates every profile combination when the pool yields at
    most 200 placebos (deterministic; single-profile ``s_tau`` — the Prop 99
    case — is exactly placebo-in-space) and otherwise samples ``Q=199``
    (``rng`` required); an integer ``Q`` forces sampling. Placebo draws with
    NaN tau or se are dropped and counted in ``extras["n_failed"]``; fewer
    than 5 usable placebos gives ``p = NaN``, never a fake 1.0.

    Assumptions, stated (audit 5): the scan statistic is a function of
    ``(x, t, theta)`` only, so CONDITIONAL on the discovered ``(s_tau, T0)``
    this test uses ``y`` information not used in selection; placebo profiles
    are treated as exchangeable with ``s_tau`` under H0 — an assumption, not
    a theorem. The thesis's "independence of the two tests" claim is replaced
    by this precise conditional statement.
    """
    if rng is not None and not isinstance(rng, np.random.Generator):
        raise TypeError(f"rng must be a numpy Generator, got {type(rng).__name__}")
    mask = np.asarray(discovery.mask, dtype=bool)
    eff_obs = did_effect(panel, discovery, control=control, dose_normalize=False)
    observed = _studentized(eff_obs)

    pid = panel.profile_id
    tau_profiles = np.unique(pid[mask])
    k = int(tau_profiles.size)
    pool = np.setdiff1d(np.unique(pid), tau_profiles)

    n_enum = comb(int(pool.size), k) if pool.size >= k else 0
    if Q == "auto":
        mode = "enumerate" if n_enum <= _ENUM_MAX else "sample"
        n_draws = n_enum if mode == "enumerate" else _SAMPLE_Q
    elif isinstance(Q, int) and not isinstance(Q, bool):
        if Q < 1:
            raise ValueError(f"Q must be >= 1, got {Q}")
        mode, n_draws = "sample", Q
    else:
        raise ValueError(f"Q must be 'auto' or a positive int, got {Q!r}")
    if mode == "sample" and rng is None:
        raise ValueError("rng is required in sampling mode (pass one numpy Generator)")

    # Placebo panel: s_tau records removed so a real effect never leaks into
    # placebo controls; profile ids are unchanged (same dims, same coding).
    placebo_panel = _without_records(panel, mask)
    pid_p = placebo_panel.profile_id

    if mode == "enumerate":
        draws: list[tuple[int, ...]] = [tuple(c) for c in combinations(pool.tolist(), k)]
    else:
        draws = [tuple(sorted(rng.choice(pool, size=k, replace=False).tolist()))
                 for _ in range(n_draws)] if pool.size >= k else []

    stats: list[float] = []
    n_failed = 0
    for profiles in draws:
        p_mask = np.isin(pid_p, np.asarray(profiles, dtype=np.int64))
        if not p_mask.any():
            n_failed += 1
            continue
        p_disc = DiDDiscovery(
            subset_values=_profiles_to_values(
                panel, profiles, constrained_dims=set(discovery.subset_values)
            ),
            mask=p_mask,
            t0=discovery.t0,
            window=discovery.window,
            llr=float("nan"),
            model=discovery.model,
            method=discovery.method,
        )
        stat = _studentized(
            did_effect(placebo_panel, p_disc, control=control, dose_normalize=False)
        )
        if np.isnan(stat):
            n_failed += 1
        else:
            stats.append(stat)

    null_stats = np.asarray(stats, dtype=float)
    q = int(null_stats.size)
    if q < _MIN_USABLE or np.isnan(observed):
        p_value = float("nan")
    else:
        p_value = float((1 + int(np.sum(np.abs(null_stats) >= abs(observed)))) / (1 + q))
    return TauRandomizationReport(
        p_value=p_value,
        observed=observed,
        null_stats=null_stats,
        q=q,
        mode=mode,
        extras={
            "n_failed": n_failed,
            "n_pool_profiles": int(pool.size),
            "k_profiles": k,
            "n_draws": len(draws),
            "tau_hat": eff_obs.tau,
            "se_hat": eff_obs.se,
        },
    )


# ---------------------------------------------------------------------------
# per-dimension placebo tests (thesis section 6.3.1(3))
# ---------------------------------------------------------------------------


@dataclass
class PlaceboDimensionReport:
    """Composition placebos across the dimensions NOT defining ``s_tau``."""

    p_values: dict[str, float]
    p_holm: dict[str, float]
    passed: bool
    note: str | None = None
    extras: dict = field(default_factory=dict)


def _holm(p_values: dict[str, float]) -> dict[str, float]:
    """Holm step-down adjusted p-values; NaN entries excluded and preserved."""
    out: dict[str, float] = {name: float("nan") for name in p_values}
    usable = [(name, p) for name, p in p_values.items() if not np.isnan(p)]
    m = len(usable)
    running = 0.0
    for rank, (name, p) in enumerate(sorted(usable, key=lambda item: item[1])):
        running = max(running, (m - rank) * p)
        out[name] = min(running, 1.0)
    return out


def placebo_dimension_tests(
    panel: CategoricalPanel,
    discovery: DiDDiscovery,
    control: str = "dd",
    rng: np.random.Generator | None = None,
    Q: int | str = "auto",
    alpha: float = 0.05,
) -> PlaceboDimensionReport:
    """Composition placebo per free dimension, Holm across dimensions.

    For each dimension NOT defining ``s_tau`` (absent from
    ``discovery.subset_values``), the outcome is replaced by that dimension's
    per-record one-hot share — the indicator of its MODAL value within
    ``s_tau`` (ties break to the smallest code, deterministic; the tested
    value is reported in ``extras["modal_values"]``) — and the same
    two-sided studentized :func:`tau_randomization_test` runs at the
    discovery's T0: a rejection means the subset's composition on that
    dimension jumps at the cutoff, an anticipation/composition red flag.

    The tested dimension is REMOVED from the profile definition for its test:
    the one-hot share is constant within any full covariate profile, so
    full-profile placebos would be degenerate on the tested dimension while
    ``s_tau`` (free on it) is not — matched shapes (audit 5) here mean
    placebo cells that stay free on the tested dimension exactly as ``s_tau``
    does. Calibration evidence: with full-profile placebos the null p was
    anti-conservative (5/16 seeds below 0.06); with the reduced profiles it
    is uniform on the same seeds.

    ``passed`` is True iff every usable Holm-adjusted p exceeds ``alpha`` AND
    at least one dimension was usable — all-degenerate is a fail, never a
    silent pass — except vacuously True (with a note) when every dimension
    defines ``s_tau``.
    """
    free_dims = [
        (j, name)
        for j, name in enumerate(panel.dim_names)
        if name not in discovery.subset_values
    ]
    if not free_dims:
        return PlaceboDimensionReport(
            p_values={},
            p_holm={},
            passed=True,
            note="every dimension defines s_tau; vacuously passed",
        )

    mask = np.asarray(discovery.mask, dtype=bool)
    p_values: dict[str, float] = {}
    modal_values: dict[str, object] = {}
    reports: dict[str, TauRandomizationReport] = {}
    for j, name in free_dims:
        modal_code = int(np.bincount(panel.codes[mask, j], minlength=panel.dim_sizes[j]).argmax())
        modal_values[name] = np.asarray(panel.dim_values[j])[modal_code].item()
        share = (panel.codes[:, j] == modal_code).astype(float)
        keep = [jj for jj in range(panel.m) if jj != j]
        panel_j = CategoricalPanel(
            codes=panel.codes[:, keep],
            dim_names=[panel.dim_names[jj] for jj in keep],
            dim_values=[panel.dim_values[jj] for jj in keep],
            t=panel.t,
            theta=panel.theta,
            y=share,
            unit=panel.unit,
            unit_values=panel.unit_values,
            t_origin=panel.t_origin,
        )
        rep = tau_randomization_test(panel_j, discovery, control=control, Q=Q, rng=rng)
        p_values[name] = rep.p_value
        reports[name] = rep

    p_holm = _holm(p_values)
    usable = [p for p in p_holm.values() if not np.isnan(p)]
    passed = bool(usable) and all(p > alpha for p in usable)
    note = None if usable else "no dimension produced a usable placebo p-value"
    return PlaceboDimensionReport(
        p_values=p_values,
        p_holm=p_holm,
        passed=passed,
        note=note,
        extras={"modal_values": modal_values, "reports": reports},
    )
