"""SuDDDS driver: heterogeneous RDiT search (thesis Algorithms 6-7, repaired).

Algorithm 6 alternates two conditional optimizations — T0 given the subset
(Algorithm 7, exhaustive over unique in-subset times) and the subset given T0
(Algorithm 8, :func:`natex.did.mdss.mdss_optimize`) — restarted ``restarts``
times per window width W. Audit repairs (docs/math_audit_final.md):

* **Item 11** — a single global incumbent ``(s*, T0*, W*, llr*)`` is kept
  across ALL windows and restarts; every restart's converged local optimum is
  recorded, deduped on ``(mask bytes, t0)`` and ranked by LLR, so
  ``discoveries[0]`` is the global incumbent, never the last window's best.
* **Item 12** — Algorithm 7 requires a minimum two-sided support: a cutoff
  candidate qualifies only with ``>= min_side`` in-subset records on EACH
  side inside the window; :func:`optimize_t0` returns ``None`` when no
  candidate qualifies (the caller keeps its incumbent — never a silent
  empty-side score).
* **Item 19** — the observation model matches the treatment's type:
  ``model="auto"`` selects Bernoulli iff theta is binary in {0, 1};
  ``model="normal"`` remains forceable (thesis parity). The Bernoulli model
  orders MDSS priorities via working residuals (documented heuristic) while
  every LLR evaluation stays exact-Bernoulli.

The W grid is an unreported thesis hyperparameter (design spec section 10):
:func:`default_windows` derives a documented data-driven grid.

Discovery never reads the outcome ``y`` — this module touches only
``panel.codes``, ``panel.t``, ``panel.theta``, ``panel.unit`` and
``panel.profile_id``. NaN evaluator columns are never treated as 0.0 — a
failed candidate is simply never selected.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from natex.data.spec import Dataset
from natex.did.background import DiDBackground, fit_did_background
from natex.did.mdss import Evaluator, SingleDeltaPriority, SubsetState, mdss_optimize
from natex.did.panel import CategoricalPanel, build_panel
from natex.did.statistics import (
    WindowStats,
    bernoulli_window_llr_masks,
    double_beta_llr_masks,
    double_beta_q,
    single_delta_llr,
    window_stats,
    working_residuals,
)

MakeEvaluator = Callable[[float], tuple]
"""T0 -> (Evaluator, WindowStats, ...): a model-bound evaluator at cutoff T0.

Only the first two elements are read by :func:`optimize_t0`; the internal
factory appends the MDSS priority stats as a third element.
"""

_MODELS = ("auto", "normal", "bernoulli")
_METHODS = ("greedy", "wcc", "single_delta")
_MAX_ALTERNATIONS = 20
_TOL = 1e-12
_MAX_INIT_REDRAWS = 100


@dataclass
class DiDDiscovery:
    """One converged RDiT local optimum ``(s_tau, T0, W)`` with its LLR."""

    subset_values: dict[str, list]  # dim -> included decoded values (unconstrained dims omitted)
    mask: np.ndarray  # (n,) record membership of s_tau
    t0: float
    window: float
    llr: float
    model: str  # "normal" | "bernoulli"
    method: str  # "greedy" | "wcc" | "single_delta"
    p_value: float | None = None
    extras: dict = field(default_factory=dict)  # e.g. delta_hat, q1, q2, restart index


@dataclass
class SuDDDSResult:
    """Deduped, LLR-ranked discoveries; ``discoveries[0]`` is the global incumbent."""

    discoveries: list[DiDDiscovery]
    model: str
    method: str
    windows: tuple[float, ...]
    restarts: int

    def top(self, m: int) -> list[DiDDiscovery]:
        return self.discoveries[:m]


def default_windows(t: np.ndarray) -> tuple[float, ...]:
    """Data-driven W grid (the thesis never reports one — spec section 10 risk).

    With ``span = t.max() - t.min()`` and ``step`` = median diff of unique
    times, the grid is ``(span/8, span/4, span/2)`` snapped UP to multiples of
    ``step``, deduped, each ``>= 2 * step`` (a window narrower than two time
    steps cannot hold records on both sides of a cutoff).
    """
    t = np.asarray(t, dtype=float)
    u = np.unique(t)
    if u.size < 2:
        raise ValueError("default_windows requires >= 2 distinct time points")
    span = float(u[-1] - u[0])
    step = float(np.median(np.diff(u)))
    grid: list[float] = []
    for frac in (0.125, 0.25, 0.5):
        w = max(span * frac, 2.0 * step)
        w = float(np.ceil(w / step - 1e-9) * step)  # snap up to a multiple of step
        grid.append(w)
    return tuple(dict.fromkeys(grid))


def optimize_t0(
    make_evaluator: MakeEvaluator,
    t: np.ndarray,
    mask: np.ndarray,
    W: float,
    min_side: int = 3,
) -> tuple[float, float] | None:
    """Algorithm 7 with the audit-12 repair: exhaustive T0 over in-subset times.

    Candidates are the unique times of the current subset. A candidate
    qualifies only with ``>= min_side`` in-subset records on EACH side inside
    the window (``[T0-W, T0)`` pre / ``[T0, T0+W)`` post). Returns the
    ``(T0, llr)`` argmax over qualifying candidates — ties keep the earliest
    T0 — or ``None`` when no candidate qualifies (the caller keeps its
    incumbent; an empty side is never scored). NaN evaluations are treated as
    failures and never selected.
    """
    if min_side < 1:
        raise ValueError(f"min_side must be >= 1, got {min_side}")
    t = np.asarray(t, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    best: tuple[float, float] | None = None
    for T0 in np.unique(t[mask]):
        out = make_evaluator(float(T0))
        evaluator, ws = out[0], out[1]
        if int(np.count_nonzero(mask & ws.g0)) < min_side:
            continue
        if int(np.count_nonzero(mask & ws.g1)) < min_side:
            continue
        llr = float(np.asarray(evaluator(mask[:, None]), dtype=float)[0])
        if np.isnan(llr):
            continue  # failed evaluation: never selected, never 0.0
        if best is None or llr > best[1]:
            best = (float(T0), llr)
    return best


def _evaluator_factory(
    panel: CategoricalPanel, background: DiDBackground, model: str, method: str, W: float
) -> MakeEvaluator:
    """T0 -> (evaluator, window stats, mdss priority stats) for the bound model."""
    if model == "bernoulli":
        # Working residuals order priorities ONLY; evaluation stays exact-Bernoulli.
        r, sigma2 = working_residuals(panel.theta, background.fitted)
    else:
        assert background.r is not None and background.sigma2 is not None
        r, sigma2 = background.r, background.sigma2
    n_profiles = int(np.prod(panel.dim_sizes)) if panel.m else 1

    def make(T0: float) -> tuple[Evaluator, WindowStats, object]:
        ws = window_stats(panel.t, r, sigma2, T0, W)
        if method == "single_delta":
            prio = SingleDeltaPriority.from_window_stats(
                ws, panel.profile_id, n_profiles=n_profiles
            )

            def ev_sd(M: np.ndarray) -> np.ndarray:
                Mt = np.asarray(M, dtype=bool).T.astype(float)
                return np.asarray(single_delta_llr(Mt @ prio.c_rec, Mt @ prio.b_rec), dtype=float)

            return ev_sd, ws, prio
        if model == "bernoulli":

            def ev_bern(M: np.ndarray) -> np.ndarray:
                return bernoulli_window_llr_masks(panel.theta, background.eta, ws, M)

            return ev_bern, ws, ws

        def ev_db(M: np.ndarray) -> np.ndarray:
            return double_beta_llr_masks(ws, M)

        return ev_db, ws, ws

    return make


def _random_init(panel: CategoricalPanel, rng: np.random.Generator) -> SubsetState:
    """Per-dimension i.i.d. Bernoulli(1/2) value masks, redrawn until nonempty."""
    for _ in range(_MAX_INIT_REDRAWS):
        included = [rng.random(k) < 0.5 for k in panel.dim_sizes]
        if panel.subset_mask(included).any():
            return SubsetState(included=included)
    # Vanishingly unlikely with any usable panel; fall back to s = D (deterministic).
    return SubsetState(included=[np.ones(k, dtype=bool) for k in panel.dim_sizes])


def _alternate(
    panel: CategoricalPanel,
    factory: MakeEvaluator,
    method: str,
    state: SubsetState,
    W: float,
    rng: np.random.Generator,
    min_side: int,
    n_rho: int,
    exhaustive_max_values: int,
) -> tuple[SubsetState, float, float] | None:
    """One Algorithm 6 restart: alternate Alg 7 / Alg 8 until the LLR stalls.

    Returns the best consistent ``(state, t0, llr)`` seen in this restart, or
    ``None`` when no cutoff ever qualified (audit 12: the restart contributes
    nothing rather than scoring an empty side).
    """
    best: tuple[SubsetState, float, float] | None = None
    prev_llr = -np.inf
    for _ in range(_MAX_ALTERNATIONS):
        res = optimize_t0(factory, panel.t, state.mask(panel), W, min_side=min_side)
        if res is None:
            break  # keep the incumbent `best` — never an empty-side score
        t0, _llr_t0 = res
        evaluator, _ws, prio_stats = factory(t0)
        state, llr = mdss_optimize(
            panel,
            evaluator,
            method,
            prio_stats,
            rng,
            init=state,
            n_rho=n_rho,
            exhaustive_max_values=exhaustive_max_values,
        )
        if best is None or llr > best[2]:
            best = (state, t0, llr)
        if llr <= prev_llr + _TOL:
            break
        prev_llr = llr
    return best


def _discovery_extras(
    factory: MakeEvaluator, method: str, model: str, mask: np.ndarray, t0: float
) -> dict:
    """Model-specific diagnostics for one discovery (NaN on failure, never 0.0)."""
    _ev, ws, prio = factory(t0)
    extras: dict = {}
    if method == "single_delta":
        assert isinstance(prio, SingleDeltaPriority)
        m = mask.astype(float)
        b = float(m @ prio.b_rec)
        extras["delta_hat"] = float(m @ prio.c_rec) / b if b > 0.0 else float("nan")
    elif model == "normal":
        q1, q2 = double_beta_q(ws, mask[:, None])
        extras["q1"] = float(q1[0])
        extras["q2"] = float(q2[0])
    return extras


def suddds_scan(
    dataset: Dataset,
    windows: tuple[float, ...] | None = None,
    restarts: int = 8,
    model: str = "auto",
    method: str = "single_delta",
    bins: int = 4,
    degree: int = 1,
    dims: list[str] | None = None,
    rng: np.random.Generator | None = None,
    min_side: int = 3,
    n_rho: int = 10,
    exhaustive_max_values: int = 12,
    panel: CategoricalPanel | None = None,
    background: DiDBackground | None = None,
) -> SuDDDSResult:
    """Repaired Algorithm 6: heterogeneous RDiT search over (s, T0, W).

    Outer loop over ``windows`` (``None`` -> :func:`default_windows`); per
    window, ``restarts`` restarts — restart 0 initializes ``s = D`` (thesis
    practice), later restarts draw each dimension's value mask i.i.d.
    Bernoulli(1/2) from ``rng``, redrawing until the record mask is nonempty.
    Each restart alternates :func:`optimize_t0` (Alg 7) and
    :func:`natex.did.mdss.mdss_optimize` (Alg 8) until the LLR stops
    improving (> 1e-12) or 20 alternations. A global incumbent is kept across
    ALL windows and restarts (audit 11); every converged local optimum is
    recorded, deduped on ``(mask bytes, t0)`` keeping the max LLR, and ranked
    by LLR descending — ``discoveries[0]`` is the global incumbent.

    ``method="single_delta"`` requires the normal model (the profile GLR is a
    Gaussian statistic); ``model="auto"`` follows audit 19 (Bernoulli iff
    binary theta, ``"normal"`` forceable for thesis parity). ``panel`` /
    ``background`` accept precomputed inputs (validation replicas refit their
    own). The scan never touches the outcome ``y``.
    """
    if rng is None:
        raise ValueError("rng is required: pass one numpy Generator through every stochastic call")
    if not isinstance(rng, np.random.Generator):
        raise TypeError(f"rng must be a numpy Generator, got {type(rng).__name__}")
    if model not in _MODELS:
        raise ValueError(f"model must be one of {_MODELS}, got {model!r}")
    if method not in _METHODS:
        raise ValueError(f"method must be one of {_METHODS}, got {method!r}")
    if restarts < 1:
        raise ValueError(f"restarts must be >= 1, got {restarts}")

    if panel is None:
        panel = build_panel(dataset, dims=dims, bins=bins)
    theta_vals = np.unique(panel.theta[~np.isnan(panel.theta)])
    theta_binary = theta_vals.size <= 2 and set(theta_vals.tolist()) <= {0.0, 1.0}
    if model == "bernoulli" and not theta_binary:
        raise ValueError("model='bernoulli' requires binary theta in {0, 1} (audit item 19)")
    if background is None:
        background = fit_did_background(panel, model=model, degree=degree)
    if background.fitted.shape[0] != panel.n:
        raise ValueError(
            f"background was fitted on {background.fitted.shape[0]} records, panel has {panel.n}"
        )
    resolved_model = background.kind
    if model != "auto" and model != resolved_model:
        raise ValueError(f"model={model!r} conflicts with background kind {resolved_model!r}")
    if method == "single_delta" and resolved_model != "normal":
        raise ValueError(
            "method='single_delta' requires model='normal' (the profile GLR is a Gaussian "
            "statistic); force model='normal' for the thesis-parity path"
        )

    if windows is None:
        windows = default_windows(panel.t)
    windows = tuple(float(W) for W in windows)
    if len(windows) == 0:
        raise ValueError("windows must be nonempty")
    for W in windows:
        if not np.isfinite(W) or W <= 0.0:
            raise ValueError(f"every window must be a finite positive width, got {W}")

    # Audit item 11: one global incumbent across ALL windows and restarts, and
    # every restart's converged local optimum recorded.
    incumbent_llr = -np.inf
    records: list[tuple[float, SubsetState, float, float, dict]] = []
    for W in windows:
        factory = _evaluator_factory(panel, background, resolved_model, method, W)
        for restart in range(restarts):
            if restart == 0:
                state = SubsetState(included=[np.ones(k, dtype=bool) for k in panel.dim_sizes])
            else:
                state = _random_init(panel, rng)
            local = _alternate(
                panel, factory, method, state, W, rng, min_side, n_rho, exhaustive_max_values
            )
            if local is None:
                continue  # no qualifying cutoff in this restart (audit 12)
            state, t0, llr = local
            extras = {"restart": restart}
            extras.update(
                _discovery_extras(factory, method, resolved_model, state.mask(panel), t0)
            )
            records.append((llr, state, t0, W, extras))
            incumbent_llr = max(incumbent_llr, llr)

    # Dedup identical (mask bytes, t0) keeping the max LLR, then rank.
    Rec = tuple[float, np.ndarray, SubsetState, float, float, dict]
    best_by_key: dict[tuple[bytes, float], Rec] = {}
    for llr, state, t0, W, extras in records:
        mask = state.mask(panel)
        key = (mask.tobytes(), t0)
        if key not in best_by_key or llr > best_by_key[key][0]:
            best_by_key[key] = (llr, mask, state, t0, W, extras)
    ranked = sorted(
        best_by_key.values(), key=lambda rec: (-rec[0], rec[3], rec[1].tobytes())
    )
    discoveries = [
        DiDDiscovery(
            subset_values=state.values(panel),
            mask=mask,
            t0=t0,
            window=W,
            llr=llr,
            model=resolved_model,
            method=method,
            extras=extras,
        )
        for llr, mask, state, t0, W, extras in ranked
    ]
    assert not discoveries or discoveries[0].llr == incumbent_llr  # audit 11 invariant
    return SuDDDSResult(
        discoveries=discoveries,
        model=resolved_model,
        method=method,
        windows=windows,
        restarts=restarts,
    )
