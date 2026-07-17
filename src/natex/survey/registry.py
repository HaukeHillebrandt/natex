"""Method-family registry for the one-command survey.

Seven families in a FIXED order everywhere: rdd, did, kink, iv, sc, bunching, dee.
Each family carries a plain-language description (reused verbatim in the report), an
honest-inference caveat line, and REQUIREMENTS expressed as declarative predicates over
the intake profile / :class:`~natex.data.spec.DatasetSpec` / declared inputs ONLY —
never over dataset content. No predicate in this module accepts a DataFrame.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass, field
from typing import Callable

from natex.data.spec import DatasetSpec
from natex.intake.profiler import IntakeProfile


@dataclass(frozen=True)
class DeclaredInputs:
    """User/analyst-declared survey inputs; the ONLY non-profile evidence predicates may read."""

    time: str | None = None
    unit: str | None = None
    cutoffs: dict[str, float] = field(default_factory=dict)  # col -> cutoff value
    instruments: list[str] = field(default_factory=list)
    thresholds: dict[str, float] = field(default_factory=dict)  # col -> threshold value
    treated_unit: str | None = None  # sc hint (guidance/config only)
    t0: float | None = None  # sc hint


@dataclass(frozen=True)
class Requirement:
    key: str  # e.g. "needs_numeric_forcing"
    description: str  # human sentence, used verbatim in reasons
    user_suppliable: bool  # unmet+True -> needs_input; unmet+False -> inapplicable
    check: Callable[[IntakeProfile, DatasetSpec | None, DeclaredInputs], bool]


@dataclass(frozen=True)
class MethodFamily:
    name: str  # "rdd"|"did"|"kink"|"iv"|"sc"|"bunching"|"dee"
    title: str  # e.g. "Regression discontinuity (LoRD3 scan)"
    description: str  # ONE plain-language paragraph, reused verbatim in the report
    caveat: str  # honest-inference caveat line (phrasing pulled from the method cards)
    requirements: tuple[Requirement, ...]


# ---------------------------------------------------------------------------
# Named predicates. Every predicate reads ONLY IntakeProfile / DatasetSpec /
# DeclaredInputs attributes; the recording-stub test pins the exact profile
# attribute set they may touch.
# ---------------------------------------------------------------------------


def _has_treatment(p: IntakeProfile, s: DatasetSpec | None, d: DeclaredInputs) -> bool:
    return bool(p.treatment_candidates) or s is not None


def _has_numeric_forcing(p: IntakeProfile, s: DatasetSpec | None, d: DeclaredInputs) -> bool:
    return bool(p.forcing_candidates) or bool(s is not None and s.forcing)


def _has_panel(p: IntakeProfile, s: DatasetSpec | None, d: DeclaredInputs) -> bool:
    if p.panel_candidates:
        return True
    if d.unit and d.time:
        return True
    return bool(s is not None and s.time and s.unit)


def _has_time(p: IntakeProfile, s: DatasetSpec | None, d: DeclaredInputs) -> bool:
    return any(c.is_time_like for c in p.columns) or d.time is not None


def _has_outcome(p: IntakeProfile, s: DatasetSpec | None, d: DeclaredInputs) -> bool:
    if any(c.is_numeric and not c.is_binary and not c.is_time_like for c in p.columns):
        return True
    return bool(s is not None and s.outcome is not None)


def _has_declared_cutoff(p: IntakeProfile, s: DatasetSpec | None, d: DeclaredInputs) -> bool:
    return bool(d.cutoffs)


def _has_declared_instruments(p: IntakeProfile, s: DatasetSpec | None, d: DeclaredInputs) -> bool:
    return bool(d.instruments)


def _has_declared_threshold(p: IntakeProfile, s: DatasetSpec | None, d: DeclaredInputs) -> bool:
    return bool(d.thresholds)


def _min_rows(n: int) -> Callable[[IntakeProfile, DatasetSpec | None, DeclaredInputs], bool]:
    def check(p: IntakeProfile, s: DatasetSpec | None, d: DeclaredInputs) -> bool:
        return bool(p.n_rows >= n)

    return check


def _gp_extra_installed(p: IntakeProfile, s: DatasetSpec | None, d: DeclaredInputs) -> bool:
    # Environment predicate, still content-blind: checks installed packages only.
    return (
        importlib.util.find_spec("torch") is not None
        and importlib.util.find_spec("gpytorch") is not None
    )


# ---------------------------------------------------------------------------
# Requirement builders.
# ---------------------------------------------------------------------------


def _req_min_rows(n: int) -> Requirement:
    return Requirement(
        key="min_rows",
        description=f"at least {n} rows (conservative floor for this family)",
        user_suppliable=False,
        check=_min_rows(n),
    )


_REQ_BINARY_TREATMENT = Requirement(
    key="needs_binary_treatment",
    description="a binary treatment column (profiled treatment candidate or declared spec)",
    user_suppliable=False,
    check=_has_treatment,
)

_REQ_ANY_TREATMENT = Requirement(
    key="needs_binary_or_continuous_treatment",
    description="a treatment column (profiled treatment candidate or declared spec)",
    user_suppliable=False,
    check=_has_treatment,
)

_REQ_NUMERIC_FORCING = Requirement(
    key="needs_numeric_forcing",
    description="a numeric forcing (running) variable",
    user_suppliable=False,
    check=_has_numeric_forcing,
)

_REQ_PANEL = Requirement(
    key="needs_panel",
    description="a panel structure: unit and time columns",
    user_suppliable=True,
    check=_has_panel,
)

_REQ_OUTCOME = Requirement(
    key="needs_outcome",
    description="a continuous numeric outcome column",
    user_suppliable=False,
    check=_has_outcome,
)

_REQ_DECLARED_CUTOFF = Requirement(
    key="needs_declared_cutoff",
    description="no pre-declared cutoff (kink is candidate evaluation, not discovery)",
    user_suppliable=True,
    check=_has_declared_cutoff,
)

_REQ_DECLARED_INSTRUMENTS = Requirement(
    key="needs_candidate_instruments",
    description="declared candidate instrument columns (--instrument COL, repeatable)",
    user_suppliable=True,
    check=_has_declared_instruments,
)

_REQ_DECLARED_THRESHOLD = Requirement(
    key="needs_declared_threshold",
    description="a declared bunching threshold (--threshold COL=VALUE)",
    user_suppliable=True,
    check=_has_declared_threshold,
)

_REQ_GP_EXTRA = Requirement(
    key="needs_gp_extra",
    description='the gp extra is not installed (pip install "natex-discovery[gp]")',
    user_suppliable=False,
    check=_gp_extra_installed,
)


FAMILY_ORDER: tuple[str, ...] = ("rdd", "did", "kink", "iv", "sc", "bunching", "dee")

_FAMILY_LIST: tuple[MethodFamily, ...] = (
    MethodFamily(
        name="rdd",
        title="Regression discontinuity (LoRD3 scan)",
        description=(
            "Regression discontinuity designs arise when a rule assigns treatment to units on "
            "one side of a cutoff in a numeric running variable, so units just above and just "
            "below the cutoff are comparable except for treatment. In a survey run natex applies "
            "the LoRD3 scan to search local neighborhoods for discontinuity structure, validates "
            "the strongest candidates with placebo and density checks, and estimates the local "
            "treatment effect at the boundary. A credible verdict means the scan found a "
            "discontinuity that survives the validation battery, supporting a local causal "
            "comparison at the cutoff."
        ),
        caveat=(
            "Scan p-values are fitted-null Monte Carlo p-values, not exact; the density "
            "diagnostic is valid only on the frozen scan geometry, so oblique projections can "
            "still hide manipulation."
        ),
        requirements=(_req_min_rows(100), _REQ_BINARY_TREATMENT, _REQ_NUMERIC_FORCING),
    ),
    MethodFamily(
        name="did",
        title="Difference-in-differences (SuDDDS scan)",
        description=(
            "Difference-in-differences compares how outcomes change over time between units that "
            "adopt a policy and units that do not, removing stable unit differences and shared "
            "time shocks. In a survey run natex applies the SuDDDS scan over subsets and adoption "
            "windows of the panel, checks composition and anticipation, and reports "
            "difference-in-differences, synthetic-control-style, and GESS effect estimates for "
            "the strongest discovery. A credible verdict means a treated subset shows a "
            "post-adoption break that passes those checks."
        ),
        caveat=(
            "Scan p-values are fitted-null Monte Carlo p-values, not exact; validity rests on "
            "the composition and anticipation checks, and when the placebo battery fails the "
            "estimate is descriptive only."
        ),
        requirements=(_req_min_rows(60), _REQ_BINARY_TREATMENT, _REQ_PANEL),
    ),
    MethodFamily(
        name="kink",
        title="Regression kink (declared cutoffs)",
        description=(
            "A regression kink design uses a policy rule whose slope, not its level, changes at "
            "a known threshold of a running variable, so the outcome should change slope at the "
            "same point if the policy matters. In a survey run natex fits local polynomials on "
            "each side of every user-declared cutoff and reports the slope change "
            "(right-minus-left), scaled by the policy kink when one is given. A credible verdict "
            "means a clear, robust slope change at a pre-declared cutoff."
        ),
        caveat=(
            "Conventional local-polynomial Wald inference may retain smoothing bias; a kink in a "
            "calendar-time running variable is a before/after slope contrast and needs the "
            "calendar-time caveats, not a density test."
        ),
        requirements=(_req_min_rows(60), _REQ_NUMERIC_FORCING, _REQ_DECLARED_CUTOFF),
    ),
    MethodFamily(
        name="iv",
        title="Instrumental variables (honest Lasso selection)",
        description=(
            "Instrumental variables recover a causal effect when treatment is confounded, by "
            "using variables that move the treatment but affect the outcome only through it. In "
            "a survey run natex screens the declared candidate instruments with a Lasso-based "
            "first-stage search on a discovery half of the rows and estimates two-stage least "
            "squares on the held-out estimation half. A credible verdict means instruments were "
            "selected, the first stage stays strong on the estimation half, and the resulting "
            "estimate is stable."
        ),
        caveat=(
            "Instrument exclusion is untestable from data — the honest discovery/estimation "
            "split is the guarantee; a weak first stage on either half invalidates the Wald "
            "interval."
        ),
        requirements=(_req_min_rows(80), _REQ_ANY_TREATMENT, _REQ_DECLARED_INSTRUMENTS),
    ),
    MethodFamily(
        name="sc",
        title="Synthetic control (donor selection + in-space placebos)",
        description=(
            "Synthetic control builds a weighted combination of untreated units that tracks a "
            "treated unit's outcome before an intervention, then reads the effect as the "
            "post-intervention gap between the unit and its synthetic counterpart. In a survey "
            "run natex selects donor units from the panel, fits simplex weights on pre-period "
            "fit only, and runs in-space placebo tests over the donor pool. A credible verdict "
            "means good pre-period fit and a post-period gap that is extreme relative to the "
            "placebo distribution."
        ),
        caveat=(
            "Placebo inference is an in-space +1-rank RMSPE-ratio test with granularity "
            "1/(n_used+1); with few usable donors the smallest attainable p-value is large."
        ),
        requirements=(_req_min_rows(40), _REQ_PANEL, _REQ_OUTCOME),
    ),
    MethodFamily(
        name="bunching",
        title="Bunching at declared thresholds",
        description=(
            "Bunching designs look for excess mass in the distribution of a variable at a "
            "declared policy threshold — people or firms piling up just at a notch or kink in "
            "their incentives. In a survey run natex tests for a density break at each "
            "user-declared threshold with a binned Poisson model of counts near the threshold; "
            "it never searches for thresholds on its own. A credible verdict means the observed "
            "pile-up at a declared threshold is unlikely under a smooth density."
        ),
        caveat=(
            "Tests run at declared thresholds only — not searched, so no selection correction "
            "is needed; the density model is a binned-Poisson approximation."
        ),
        requirements=(_req_min_rows(60), _REQ_DECLARED_THRESHOLD),
    ),
    MethodFamily(
        name="dee",
        title="Discovered-experiment ensemble debiasing (DEE)",
        description=(
            "Discovered-experiment ensembles debias observational effect estimates by finding "
            "many small pockets of the data where treatment is as-good-as-random, estimating the "
            "bias of an observational learner inside those pockets, and subtracting a smoothed "
            "version of that bias elsewhere. In a survey run natex fits the ensemble on the "
            "declared treatment, forcing, and outcome columns and reports raw versus debiased "
            "effect estimates. A credible verdict means the discovered experiments cover the "
            "covariate space well enough for the debiasing to be trustworthy."
        ),
        caveat=(
            "Debiasing quality depends on how well the discovered experiments cover the "
            "covariate space; the output is a corrected estimate, not a hypothesis test."
        ),
        requirements=(
            _req_min_rows(200),
            _REQ_BINARY_TREATMENT,
            _REQ_NUMERIC_FORCING,
            _REQ_OUTCOME,
            _REQ_GP_EXTRA,
        ),
    ),
)

FAMILIES: dict[str, MethodFamily] = {f.name: f for f in _FAMILY_LIST}

assert tuple(FAMILIES) == FAMILY_ORDER  # insertion order == FAMILY_ORDER, by construction
