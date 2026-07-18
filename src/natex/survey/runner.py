"""One-command survey runner: every method family against one dataset.

Flow (plan task 5):

1. Load df (csv path or DataFrame), ``declared = DeclaredInputs(...)`` from the
   kwargs.
2. ``fam_rngs = dict(zip(FAMILY_ORDER, rng.spawn(7)))`` FIRST (stream
   stability), then ``intake = study(csv_or_df, context=context,
   guidance=guidance, rng=rng, out=out_dir/"intake")`` (study-style
   understanding; NullBackend heuristics when guidance=None; its
   guidance_log.jsonl is the survey's log).
3. ``plans, declared = resolve_applicability(intake.profile, None, declared,
   wrapped_guidance, context=context)`` where wrapped_guidance appends to the
   SAME ``out_dir/"intake"/guidance_log.jsonl`` via ``LoggedBackend``
   (GuidanceLog appends); ``guidance_log_path`` = that file when guidance is
   not None else None.
4. For each family in FAMILY_ORDER: not run => FamilyResult(status = "skipped"
   if analyst said no or heuristic inapplicable, "needs_input" if heuristic
   needs_input and no override, reason from the plan). Run => call
   ``_run_<family>(...)`` inside the documented isolation boundary
   (``except Exception``; BaseException still propagates).
5. Write per-family ``out_dir/families/<name>.json`` (jsonable full detail),
   figures under ``out_dir/figures/`` (task 7), ``survey.json``, then reports
   (task 8).

The survey layer adds NO new inference code: rdd and did REUSE
``natex.discover`` end to end; kink, iv, sc, bunching and dee REUSE
``regression_kink``/``sensitivity_grid``, ``discover_instruments``,
``unit_time_matrix``/``select_donors``/``sc_placebo_test``,
``binned_poisson_jump`` and ``lord3_scan``/``dee_debias`` respectively.
dee carries one runtime gate ON TOP of applicability: it only runs after a
CREDIBLE rdd family result (there must be a validated discovery to debias).
"""

from __future__ import annotations

import dataclasses
import json
import traceback
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from natex.data.spec import Dataset, DatasetSpec
from natex.dee.debias import dee_debias

# _effective_budget is discover's own budget resolution (defaults <- plan
# hints <- explicit dict); dee reuses it so its scan k matches the rdd
# family's effective budget exactly.
from natex.discover import DiscoverReport, _effective_budget, discover
from natex.intake.analyst import IntakeReport, study
from natex.iv.donors import sc_placebo_test, select_donors, unit_time_matrix
from natex.iv.pipeline import discover_instruments
from natex.jsonutil import jsonable
from natex.kink import regression_kink, sensitivity_grid
from natex.llm import GuidanceBackend, GuidanceLog, LoggedBackend
from natex.rdd.lord3 import lord3_scan
from natex.report.survey_html import render_survey_html, render_survey_md
from natex.survey.applicability import FamilyPlan, resolve_applicability
from natex.survey.figures import missing_matplotlib_reason, render_family_figures
from natex.survey.registry import FAMILIES, FAMILY_ORDER, DeclaredInputs
from natex.validate.density import binned_poisson_jump

# Holm step-down with NaN entries excluded and preserved — the SAME
# convention as placebo_tests/did.effects/validate.panel (plan task 6: no
# new math, plain step-down on the vector of per-cutoff/threshold p-values).
from natex.validate.placebo import _holm

ALPHA = 0.05  # verdict gate for scan/kink/bunching p-values
# sc placebo gate: the in-space +1-rank test has granularity 1/(n_used+1), so
# 0.05 is often unattainable with few donors — documented coarser gate.
SC_ALPHA = 0.10


@dataclass
class FamilyResult:
    family: str
    status: str  # credible|null|skipped|needs_input|failed
    reason: str  # one sentence, always set
    applicability: dict = field(default_factory=dict)  # FamilyPlan serialized
    key_numbers: dict = field(default_factory=dict)  # flat name->number (NaN -> null)
    diagnostics: dict = field(default_factory=dict)  # extras incl. attached caveats
    figures: dict[str, str] = field(default_factory=dict)  # name -> out_dir-relative posix path
    no_figure_reason: str | None = None  # set whenever figures are absent or incomplete
    details_path: str | None = None  # families/<name>.json relative to out_dir
    error: str | None = None  # verbatim str(exc) when status == "failed"


@dataclass
class SurveyResult:
    out_dir: Path
    families: dict[str, FamilyResult]  # ALWAYS all seven, FAMILY_ORDER order
    coverage: dict  # {"ran", "not_run", "rdd": <discover.searched or None>, "did": <...>}
    dataset: dict  # source/n_rows/n_cols/columns_truncated/time_column/time_range
    natex_version: str
    seed: int | None
    created: str  # UTC isoformat — the ONLY nondeterministic survey.json field
    context: str | None
    guidance_log_path: str | None
    report_html: str | None = None  # set by task 8
    report_md: str | None = None

    def to_json(self) -> str:
        # out_dir is deliberately NOT serialized: every recorded path is
        # out_dir-relative, and ``load`` restores out_dir from the file's own
        # location — keeping ``created`` the only nondeterministic field.
        payload = {
            "families": {n: dataclasses.asdict(f) for n, f in self.families.items()},
            "coverage": self.coverage,
            "dataset": self.dataset,
            "natex_version": self.natex_version,
            "seed": self.seed,
            "created": self.created,
            "context": self.context,
            "guidance_log_path": self.guidance_log_path,
            "report_html": self.report_html,
            "report_md": self.report_md,
        }
        return json.dumps(jsonable(payload), indent=1)

    def save(self) -> Path:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        path = self.out_dir / "survey.json"
        path.write_text(self.to_json(), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path) -> SurveyResult:
        path = Path(path)
        d = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            out_dir=path.parent,
            families={n: FamilyResult(**f) for n, f in d["families"].items()},
            coverage=d["coverage"],
            dataset=d["dataset"],
            natex_version=d["natex_version"],
            seed=d.get("seed"),
            created=d["created"],
            context=d.get("context"),
            guidance_log_path=d.get("guidance_log_path"),
            report_html=d.get("report_html"),
            report_md=d.get("report_md"),
        )


def _plan_dict(plan: FamilyPlan) -> dict:
    """FamilyPlan serialized: heuristic + analyst decision + recorded override."""
    return {
        "run": plan.run,
        "reason": plan.reason,
        "heuristic": {
            "status": plan.heuristic.status,
            "reason": plan.heuristic.reason,
            "unmet": list(plan.heuristic.unmet),
        },
        "config_hints": plan.config_hints.model_dump(),
        "override": plan.override,
        "guidance_error": plan.guidance_error,
    }


def _finite(v) -> bool:
    return v is not None and bool(np.isfinite(v))


def _natex_version() -> str:
    import natex  # lazy: natex/__init__ imports this module

    return natex.__version__


# ---------------------------------------------------------------------------
# Dataset construction for the rdd/did families (REUSING the intake report).
# ---------------------------------------------------------------------------


def _first_candidate_index(intake: IntakeReport, design: str) -> int | None:
    ranked = intake.search_plan.ranked()
    return next((i for i, c in enumerate(ranked) if c.design == design), None)


def _outcome_role_columns(intake: IntakeReport) -> set[str]:
    """Every known outcome-roled column (understanding guesses + prep roles)."""
    roles = {col for col, role in intake.prep_plan.column_roles.items() if role == "outcome"}
    return roles | {g.column for g in intake.understanding.outcomes}


def _rdd_dataset(df: pd.DataFrame, intake: IntakeReport) -> Dataset:
    """First ranked rdd candidate via ``intake.prepare``; else a constructed spec
    mirroring ``Dataset.from_csv`` defaults (covariates = all minus outcome-role
    columns; forcing = profiled forcing candidates minus {treatment, outcome})."""
    idx = _first_candidate_index(intake, "rdd")
    if idx is not None:
        return intake.prepare(df, candidate=idx)
    df2, _ = intake.prep_plan.apply(df)
    profile = intake.profile
    if not profile.treatment_candidates:
        raise ValueError("no treatment candidate available for a constructed rdd spec")
    treatment = profile.treatment_candidates[0]
    outcome = next(
        (g.column for g in intake.understanding.outcomes
         if g.column != treatment and g.column in df2.columns),
        None,
    )
    reserved = {treatment, outcome} | _outcome_role_columns(intake)
    covariates = [c for c in df2.columns if c not in reserved]
    forcing = [c for c in profile.forcing_candidates if c in covariates]
    spec = DatasetSpec(
        treatment=treatment, outcome=outcome, forcing=forcing, covariates=covariates
    )
    return Dataset(df2, spec)


def _did_dataset(df: pd.DataFrame, intake: IntakeReport, declared: DeclaredInputs) -> Dataset:
    """First ranked did candidate via ``intake.prepare``; else a constructed spec
    with unit/time from the declared inputs or the first profiled panel
    candidate (the unit column may be non-numeric: it stays out of forcing)."""
    idx = _first_candidate_index(intake, "did")
    if idx is not None:
        return intake.prepare(df, candidate=idx)
    df2, _ = intake.prep_plan.apply(df)
    profile = intake.profile
    panel = profile.panel_candidates[0] if profile.panel_candidates else (None, None)
    unit = declared.unit or panel[0]
    time = declared.time or panel[1]
    if unit is None or time is None:
        raise ValueError("no panel structure (unit, time) declared or profiled for did")
    if not profile.treatment_candidates:
        raise ValueError("no treatment candidate available for a constructed did spec")
    treatment = profile.treatment_candidates[0]
    outcome = next(
        (g.column for g in intake.understanding.outcomes
         if g.column not in (treatment, unit, time) and g.column in df2.columns),
        None,
    )
    reserved = {treatment, outcome, unit, time} | _outcome_role_columns(intake)
    covariates = [c for c in df2.columns if c not in reserved]
    spec = DatasetSpec(
        treatment=treatment, outcome=outcome, forcing=[], covariates=covariates,
        time=time, unit=unit,
    )
    return Dataset(df2, spec)


# ---------------------------------------------------------------------------
# Per-family runners. Each returns a FamilyResult; exceptions are handled by
# the survey()'s documented isolation boundary. ``artifacts`` is the family's
# live-object figure payload (plan task 7): the runner stashes the arrays and
# objects it already has in hand so the figure glue never re-reads the CSV.
# ---------------------------------------------------------------------------


def _scanless_result(family: str, rep: DiscoverReport, diagnostics: dict) -> FamilyResult:
    """No scanned config: failed with the first config error verbatim, unless
    every enumerated config was invalid — then null with a coverage reason."""
    searched = rep.searched
    first_error = next(
        (r.error for r in rep.configs if r.status == "failed" and r.error), None
    )
    if first_error is not None:
        return FamilyResult(
            family=family, status="failed",
            reason=f"no {family} configuration scanned — first config error recorded",
            diagnostics=diagnostics, error=first_error,
        )
    reason = (
        f"no {family} configuration could be scanned "
        f"({searched['n_invalid']} invalid of {searched['n_total']} enumerated)"
    )
    return FamilyResult(
        family=family, status="null", reason=reason, diagnostics=diagnostics,
        no_figure_reason=f"no figure: no {family} configuration scanned",
    )


def _run_rdd(
    df: pd.DataFrame,
    intake: IntakeReport,
    declared: DeclaredInputs,
    plan: FamilyPlan,
    budget: dict | None,
    fam_rng: np.random.Generator,
    fam_dir: Path,
    artifacts: dict,
) -> FamilyResult:
    ds = _rdd_dataset(df, intake)
    rep = discover(
        ds, design="rdd", search_plan=intake.search_plan, rng=fam_rng,
        budget=budget, out=fam_dir,
    )
    diagnostics = {"caveats": [FAMILIES["rdd"].caveat], "searched": rep.searched}
    best = rep.best()
    if best is None:
        return _scanless_result("rdd", rep, diagnostics)
    s = best.summary
    # Figure payload (task 7): the glue re-scans ds at the same effective
    # k/degree — presentational only, the randomization test is NOT re-run.
    eff_budget = _effective_budget(intake.search_plan, budget)
    artifacts.update(
        ds=ds, k=int(eff_budget["k"]), degree=int(eff_budget["degree"]),
        rng=fam_rng, summary=s,
    )
    p, density_p = best.p_value, s.get("density_p")
    if not _finite(p):
        status, reason = "null", "scan p-value unavailable — no credible discovery"
    elif p > ALPHA:
        status, reason = "null", f"scan p={p:.2f} above {ALPHA}"
    elif not s.get("placebo_passed"):
        # audit 3 phrasing: a failed placebo battery demotes the discovery.
        status, reason = "null", "descriptive only — placebo battery failed"
    elif not _finite(density_p):
        status, reason = "null", "density diagnostic unavailable — manipulation check inconclusive"
    elif density_p <= ALPHA:
        status, reason = "null", f"density test rejects (p={density_p:.3f}) — manipulation risk"
    else:
        status = "credible"
        reason = (
            f"scan p={p:.3f} at or below {ALPHA}, placebo battery passed, "
            f"density p={density_p:.2f}"
        )
    # placebo_holm in the scan summary is a per-covariate dict; key_numbers is
    # FLAT, so report the battery's decisive number — the min Holm-adjusted p
    # over usable entries (plain min over the reported vector, no new math) —
    # and keep the full dict in diagnostics. None when nothing is usable.
    holm = s.get("placebo_holm") or {}
    usable = [v for v in holm.values() if _finite(v)]
    key_numbers = {
        "llr": best.llr,
        "p_value": p,
        "density_p": density_p,
        "placebo_holm": min(usable) if usable else None,
        "n_configs_scanned": rep.searched["n_scanned"],
    }
    diagnostics["placebo_holm_by_covariate"] = holm
    eff = (s.get("effects") or {}).get("2sls")
    if eff:
        key_numbers.update(
            tau_2sls=eff["tau"], se_2sls=eff["se"],
            ci_low_2sls=eff["ci"][0], ci_high_2sls=eff["ci"][1],
        )
    diagnostics["best_candidate"] = best.candidate.model_dump()
    return FamilyResult(
        family="rdd", status=status, reason=reason,
        key_numbers=key_numbers, diagnostics=diagnostics,
    )


def _run_did(
    df: pd.DataFrame,
    intake: IntakeReport,
    declared: DeclaredInputs,
    plan: FamilyPlan,
    budget: dict | None,
    fam_rng: np.random.Generator,
    fam_dir: Path,
    artifacts: dict,
) -> FamilyResult:
    ds = _did_dataset(df, intake, declared)
    rep = discover(
        ds, design="did", search_plan=intake.search_plan, rng=fam_rng,
        budget=budget, out=fam_dir,
    )
    diagnostics = {"caveats": [FAMILIES["did"].caveat], "searched": rep.searched}
    best = rep.best()
    if best is None:
        return _scanless_result("did", rep, diagnostics)
    s = best.summary
    # Figure payload (task 7): the glue rebuilds the panel and re-runs the
    # scan at the same effective budget to recover the pretrend discovery.
    artifacts.update(
        ds=ds, budget=_effective_budget(intake.search_plan, budget),
        rng=fam_rng, summary=s,
    )
    p = best.p_value
    if not _finite(p):
        status, reason = "null", "scan p-value unavailable — no credible discovery"
    elif p > ALPHA:
        status, reason = "null", f"scan p={p:.2f} above {ALPHA}"
    elif not s.get("composition_passed"):
        status, reason = (
            "null", "composition check failed — the treated subset's makeup shifts at adoption"
        )
    elif not s.get("anticipation_passed"):
        status, reason = (
            "null", "anticipation check failed — pre-adoption trend break in the treated subset"
        )
    else:
        status = "credible"
        reason = f"scan p={p:.3f} at or below {ALPHA}, composition and anticipation checks passed"
    key_numbers = {
        "llr": best.llr,
        "p_value": p,
        "t0": s.get("t0"),
        "window": s.get("window"),
        "n_configs_scanned": rep.searched["n_scanned"],
    }
    for name in ("dd", "synthetic", "gess"):
        eff = (s.get("effects") or {}).get(name)
        if eff:
            key_numbers[f"tau_{name}"] = eff.get("tau")
            key_numbers[f"se_{name}"] = eff.get("se")
            key_numbers[f"p_{name}"] = eff.get("p")
            key_numbers[f"dose_{name}"] = eff.get("dose")  # audit 19: dose shown
    diagnostics["best_candidate"] = best.candidate.model_dump()
    diagnostics["subset_values"] = s.get("subset_values")
    return FamilyResult(
        family="did", status=status, reason=reason,
        key_numbers=key_numbers, diagnostics=diagnostics,
    )


# ---------------------------------------------------------------------------
# kink / iv / sc / bunching / dee runners (plan task 6). Shared helpers first.
# ---------------------------------------------------------------------------

_NO_OUTCOME_REASON = "no numeric outcome column identified"

# audit 18: McCrary/density on calendar time is information-free when record
# times are design-determined; composition/anticipation carry the burden.
_AUDIT18_CAVEAT = (
    "a density test on the calendar-time column {col!r} is information-free when "
    "record times are design-determined (audit 18); composition and anticipation "
    "checks are the informative diagnostics"
)

_CALENDAR_KINK_CAVEAT = (
    "the running variable {col!r} is calendar time: this kink is a before/after "
    "slope contrast, so composition and anticipation caveats apply — not a "
    "density test (audit 18)"
)


def _numeric_columns(intake: IntakeReport) -> set[str]:
    return {c.name for c in intake.profile.columns if c.is_numeric}


def _time_like_columns(intake: IntakeReport) -> set[str]:
    return {c.name for c in intake.profile.columns if c.is_time_like}


def _first_outcome_guess(
    intake: IntakeReport, df: pd.DataFrame, exclude: set[str]
) -> str | None:
    """First Understanding outcome guess that exists, is numeric, and is not excluded."""
    numeric = _numeric_columns(intake)
    return next(
        (g.column for g in intake.understanding.outcomes
         if g.column not in exclude and g.column in df.columns and g.column in numeric),
        None,
    )


def _two_sided_p(tau: float, se: float) -> float:
    """Normal two-sided p for tau/se; NaN — never 0.0 — when undefined."""
    if not (np.isfinite(tau) and np.isfinite(se)) or se <= 0.0:
        return float("nan")
    return float(2.0 * stats.norm.sf(abs(tau / se)))


def _nanmean(a) -> float:
    """Mean over finite entries; NaN (never 0.0, never a warning) when none."""
    arr = np.asarray(a, dtype=float).ravel()
    finite = arr[np.isfinite(arr)]
    return float(finite.mean()) if finite.size else float("nan")


def _min_holm(p_values: dict[str, float]) -> tuple[dict[str, float], float]:
    """Holm-adjust the vector; return (adjusted dict, min finite adjusted p or NaN)."""
    p_holm = _holm(p_values)
    usable = [v for v in p_holm.values() if np.isfinite(v)]
    return p_holm, (min(usable) if usable else float("nan"))


def _run_kink(
    df: pd.DataFrame,
    intake: IntakeReport,
    declared: DeclaredInputs,
    plan: FamilyPlan,
    budget: dict | None,
    fam_rng: np.random.Generator,
    fam_dir: Path,
    artifacts: dict,
) -> FamilyResult:
    numeric = _numeric_columns(intake)
    time_like = _time_like_columns(intake)
    caveats = [FAMILIES["kink"].caveat]
    skipped: dict[str, str] = {}  # per-cutoff recorded reasons (plan task 6)
    used_cutoffs: dict[str, float] = {}
    per_cutoff: dict[str, dict] = {}
    sensitivity: dict[str, list[dict]] = {}
    sensitivity_errors: dict[str, str] = {}
    p_values: dict[str, float] = {}
    saw_no_outcome = False

    for col, cut in declared.cutoffs.items():
        if col not in df.columns:
            skipped[col] = f"cutoff column {col!r} not in the dataset"
            continue
        if col not in numeric:
            skipped[col] = f"cutoff column {col!r} is not numeric"
            continue
        outcome = _first_outcome_guess(intake, df, exclude={col})
        if outcome is None:
            saw_no_outcome = True
            skipped[col] = _NO_OUTCOME_REASON
            continue
        r = df[col].to_numpy(dtype=float)
        y = df[outcome].to_numpy(dtype=float)
        c = float(cut)
        # Documented default with NO optimality claim: the median absolute
        # distance to the cutoff puts about half the sample in-window.
        bw = float(np.nanquantile(np.abs(r - c), 0.5))
        if not (np.isfinite(bw) and bw > 0.0):
            skipped[col] = (
                "default bandwidth (median |running - cutoff|) is not finite and positive"
            )
            continue
        try:
            est = regression_kink(y, r, policy_kink=1.0, cutoff=c, bandwidth=bw)
        except (ValueError, np.linalg.LinAlgError) as exc:
            skipped[col] = f"kink fit failed: {exc}"
            continue
        used_cutoffs[col] = c
        # With policy_kink=1, tau IS the reduced-form outcome slope kink
        # (right-minus-left, kink-card convention).
        p = _two_sided_p(est.tau, est.se)
        p_values[col] = p
        per_cutoff[col] = {
            "cutoff": c, "outcome": outcome, "tau": est.tau, "se": est.se,
            "ci": list(est.ci), "bandwidth": bw, "n_used": est.n_used, "p_value": p,
        }
        # Figure payload (task 7): kink_fit_plot per usable cutoff.
        artifacts.setdefault("cutoffs", []).append({
            "column": col, "running": r, "outcome_values": y, "cutoff": c,
            "bandwidth": bw, "estimate": est,
        })
        bws = [0.5 * bw, bw, 2.0 * bw]
        try:
            grid = sensitivity_grid(y, r, bandwidths=bws, policy_kink=1.0, cutoff=c)
            sensitivity[col] = [
                {"bandwidth": h, "tau": g.tau, "se": g.se}
                for h, g in zip(bws, grid, strict=True)
            ]
        except (ValueError, np.linalg.LinAlgError) as exc:
            sensitivity_errors[col] = str(exc)
        if col in time_like:
            caveats.append(_CALENDAR_KINK_CAVEAT.format(col=col))

    diagnostics = {
        "caveats": caveats, "cutoffs": used_cutoffs, "per_cutoff": per_cutoff,
        "sensitivity": sensitivity,
    }
    if skipped:
        diagnostics["skipped_cutoffs"] = skipped
    if sensitivity_errors:
        diagnostics["sensitivity_errors"] = sensitivity_errors

    if not per_cutoff:
        reason = _NO_OUTCOME_REASON if saw_no_outcome else (
            "no usable declared cutoff ("
            + "; ".join(f"{c}: {r}" for c, r in skipped.items()) + ")"
        )
        return FamilyResult(
            family="kink", status="needs_input", reason=reason,
            diagnostics=diagnostics,
        )

    p_holm, min_holm = _min_holm(p_values)
    diagnostics["p_holm"] = p_holm
    m = len(p_values)
    if not np.isfinite(min_holm):
        status = "null"
        reason = "kink fits degenerate — no finite kink p-value at any declared cutoff"
    elif min_holm <= ALPHA:
        status = "credible"
        reason = (
            f"min Holm-adjusted kink p={min_holm:.3g} at or below {ALPHA} "
            f"across {m} declared cutoff(s)"
        )
    else:
        status = "null"
        reason = (
            f"min Holm-adjusted kink p={min_holm:.2f} above {ALPHA} "
            f"across {m} declared cutoff(s)"
        )
    first = per_cutoff[next(iter(per_cutoff))]
    key_numbers = {
        "tau": first["tau"], "se": first["se"],
        "ci_low": first["ci"][0], "ci_high": first["ci"][1],
        "bandwidth": first["bandwidth"], "n_used": first["n_used"],
        "p_value": first["p_value"], "min_holm_p": min_holm,
    }
    return FamilyResult(
        family="kink", status=status, reason=reason,
        key_numbers=key_numbers, diagnostics=diagnostics,
    )


def _run_iv(
    df: pd.DataFrame,
    intake: IntakeReport,
    declared: DeclaredInputs,
    plan: FamilyPlan,
    budget: dict | None,
    fam_rng: np.random.Generator,
    fam_dir: Path,
    artifacts: dict,
) -> FamilyResult:
    numeric = _numeric_columns(intake)
    treatment = next(
        (g.column for g in intake.understanding.treatments if g.column in df.columns),
        None,
    )
    if treatment is None:
        if not intake.profile.treatment_candidates:
            raise ValueError("no treatment candidate available for the iv family")
        treatment = intake.profile.treatment_candidates[0]

    dropped: dict[str, str] = {}
    pool: list[str] = []
    for col in declared.instruments:
        if col not in df.columns:
            dropped[col] = "not in the dataset"
        elif col not in numeric:
            dropped[col] = "not numeric"
        elif col == treatment:
            dropped[col] = "is the treatment column"
        else:
            pool.append(col)
    if not pool:
        reason = (
            "no usable declared instrument column ("
            + "; ".join(f"{c}: {r}" for c, r in dropped.items()) + ")"
        )
        return FamilyResult(
            family="iv", status="needs_input", reason=reason,
            diagnostics={"caveats": [FAMILIES["iv"].caveat],
                         "dropped_instruments": dropped},
        )
    outcome = _first_outcome_guess(intake, df, exclude={treatment, *pool})

    res = discover_instruments(
        df, treatment, pool, outcome=outcome, honest=True, rng=fam_rng
    )
    search, est = res.search, res.estimate
    diagnostics = {
        "caveats": [FAMILIES["iv"].caveat],
        "honest_split": (
            f"instruments selected on a {res.n_discovery}-row discovery half and "
            f"estimated on a held-out {res.n_estimation}-row half"
        ),
        "treatment": treatment, "outcome": outcome, "pool": pool,
        "selected": list(search.selected),
        "first_stage_F_discovery": search.first_stage_F,
        "partial_r2_discovery": search.partial_r2,
    }
    if dropped:
        diagnostics["dropped_instruments"] = dropped
    key_numbers: dict = {
        "n_selected": len(search.selected),
        "first_stage_F": est.first_stage_F if est is not None else search.first_stage_F,
        "partial_r2": est.partial_r2 if est is not None else search.partial_r2,
        "n_discovery": res.n_discovery, "n_estimation": res.n_estimation,
    }
    if est is not None:
        key_numbers.update(
            tau=est.tau, se=est.se, ci_low=est.ci[0], ci_high=est.ci[1],
            j_p=est.j_p, ar_kind=est.ar_kind,
        )
        if est.ar_ci is not None:
            key_numbers.update(ar_ci_low=est.ar_ci[0], ar_ci_high=est.ar_ci[1])
        # Figure payload (task 7): forest of the 2SLS row (+ finite AR interval).
        artifacts["estimate"] = est

    if not search.selected:
        status, reason = "null", "no instrument selected from the declared pool"
    elif search.weak or (est is not None and est.weak_instrument):
        # audit 10: first-stage relevance is measured, never assumed — a weak
        # first stage on EITHER half demotes the family.
        weak_f = search.first_stage_F if search.weak else est.first_stage_F
        half = "discovery" if search.weak else "estimation"
        status = "null"
        if _finite(weak_f):
            reason = (
                f"weak first stage (F={weak_f:.1f} on the {half} half) — "
                "instrument relevance not established (audit 10)"
            )
        else:
            reason = (
                f"first-stage F unavailable on the {half} half — "
                "instrument relevance not established (audit 10)"
            )
    else:
        f_used = key_numbers["first_stage_F"]
        tail = (
            " (selection only — no outcome column identified)"
            if est is None else " on the held-out estimation half"
        )
        status = "credible"
        reason = (
            f"{len(search.selected)} instrument(s) selected with "
            f"first-stage F={f_used:.1f}{tail}"
        )
    return FamilyResult(
        family="iv", status=status, reason=reason,
        key_numbers=key_numbers, diagnostics=diagnostics,
        no_figure_reason=(
            "no figure: selection-only run (no outcome column)"
            if outcome is None else None
        ),
    )


def _run_sc(
    df: pd.DataFrame,
    intake: IntakeReport,
    declared: DeclaredInputs,
    plan: FamilyPlan,
    budget: dict | None,
    fam_rng: np.random.Generator,
    fam_dir: Path,
    artifacts: dict,
) -> FamilyResult:
    profile = intake.profile
    panel = profile.panel_candidates[0] if profile.panel_candidates else (None, None)
    unit = declared.unit or panel[0]
    time = declared.time or panel[1]
    if unit is None or time is None:
        raise ValueError("no panel structure (unit, time) declared or profiled for sc")
    outcome = _first_outcome_guess(
        intake, df, exclude={unit, time, *profile.treatment_candidates}
    )
    diagnostics: dict = {
        "caveats": [FAMILIES["sc"].caveat], "unit": unit, "time": time,
        "outcome": outcome,
    }
    if outcome is None:
        return FamilyResult(
            family="sc", status="needs_input", reason=_NO_OUTCOME_REASON,
            diagnostics=diagnostics,
        )
    Y, units, times = unit_time_matrix(df, unit, time, outcome)

    treated = declared.treated_unit
    t0 = declared.t0
    if treated is None or t0 is None:
        tcol = next((c for c in profile.treatment_candidates if c in df.columns), None)
        if tcol is None:
            return FamilyResult(
                family="sc", status="needs_input",
                reason=(
                    "could not identify a treated unit (no binary treatment column "
                    "profiled); provide treated_unit/t0 via guidance"
                ),
                diagnostics=diagnostics,
            )
        t_on = df[tcol].to_numpy(dtype=float) == 1.0
        if treated is None:
            ever = pd.unique(df.loc[t_on, unit])  # per-unit ever-treated set
            if len(ever) != 1:
                return FamilyResult(
                    family="sc", status="needs_input",
                    reason=(
                        f"could not identify a single treated unit from {tcol!r} "
                        f"(found {len(ever)}); provide treated_unit/t0 via guidance"
                    ),
                    diagnostics=diagnostics,
                )
            treated = ever[0]
        if t0 is None:
            t0 = float(df.loc[t_on, time].min())  # min(time where T == 1)
    t0 = float(t0)
    diagnostics["treated_unit"] = treated
    diagnostics["t0"] = t0

    sel = select_donors(Y, units, times, treated, t0)
    if "failure" in sel.extras:
        failure = str(sel.extras["failure"])
        return FamilyResult(
            family="sc", status="failed", reason=failure, error=failure,
            diagnostics=diagnostics,
        )
    # Figure payload (task 7): per-period treated-minus-synthetic gap.
    artifacts.update(times=sel.times, gaps=sel.effect_by_time, t0=t0)
    rep = sc_placebo_test(Y, units, times, treated, t0)
    diagnostics["n_skipped_placebos"] = rep.n_skipped
    key_numbers = {
        "att_post": sel.att_post, "pre_rmspe": sel.pre_rmspe,
        "post_rmspe": sel.post_rmspe, "ratio_treated": rep.ratio_treated,
        "p_value": rep.p_value, "n_donors": len(sel.donors),
        "n_placebos": len(rep.ratios),
    }
    p = rep.p_value  # audit 5: the +1-rank RMSPE-ratio p, verbatim
    if not _finite(p):
        status, reason = "null", "too few usable placebos (<5) for the ratio test"
    elif p <= SC_ALPHA:
        status = "credible"
        reason = f"in-space placebo RMSPE-ratio p={p:.3f} at or below {SC_ALPHA}"
    else:
        status = "null"
        reason = f"in-space placebo RMSPE-ratio p={p:.2f} above {SC_ALPHA}"
    return FamilyResult(
        family="sc", status=status, reason=reason,
        key_numbers=key_numbers, diagnostics=diagnostics,
    )


def _run_bunching(
    df: pd.DataFrame,
    intake: IntakeReport,
    declared: DeclaredInputs,
    plan: FamilyPlan,
    budget: dict | None,
    fam_rng: np.random.Generator,
    fam_dir: Path,
    artifacts: dict,
) -> FamilyResult:
    numeric = _numeric_columns(intake)
    time_like = _time_like_columns(intake)
    caveats = [FAMILIES["bunching"].caveat]
    skipped: dict[str, str] = {}
    per_threshold: dict[str, dict] = {}
    p_values: dict[str, float] = {}

    for col, thr in declared.thresholds.items():
        if col not in df.columns:
            skipped[col] = f"threshold column {col!r} not in the dataset"
            continue
        if col not in numeric:
            skipped[col] = f"threshold column {col!r} is not numeric"
            continue
        t = float(thr)
        s = df[col].to_numpy(dtype=float) - t
        rep = binned_poisson_jump(s)  # drops non-finite s itself
        p_values[col] = rep.p_value
        per_threshold[col] = {
            "threshold": t, "p_value": rep.p_value, "theta": rep.theta,
            "se": rep.se,
            "n_finite": int(np.isfinite(s).sum()),
        }
        # Figure payload (task 7): bunching_hist per usable threshold.
        artifacts.setdefault("thresholds", []).append({
            "column": col, "values": df[col].to_numpy(dtype=float),
            "threshold": t, "p_value": rep.p_value,
        })
        if col in time_like:
            caveats.append(_AUDIT18_CAVEAT.format(col=col))

    diagnostics: dict = {"caveats": caveats, "per_threshold": per_threshold}
    if skipped:
        diagnostics["skipped_thresholds"] = skipped
    if not per_threshold:
        reason = (
            "no usable declared threshold ("
            + "; ".join(f"{c}: {r}" for c, r in skipped.items()) + ")"
        )
        return FamilyResult(
            family="bunching", status="needs_input", reason=reason,
            diagnostics=diagnostics,
        )

    p_holm, min_holm = _min_holm(p_values)
    diagnostics["p_holm"] = p_holm
    if not np.isfinite(min_holm):
        status = "null"
        reason = "density fits degenerate at every declared threshold"
    elif min_holm <= ALPHA:
        status = "credible"
        reason = (
            "density discontinuity at a declared threshold — bunching/manipulation "
            f"signal (min Holm-adjusted p={min_holm:.3g})"
        )
    else:
        status = "null"
        reason = (
            "no density discontinuity at the declared threshold(s) "
            f"(min Holm-adjusted p={min_holm:.2f})"
        )
    first = per_threshold[next(iter(per_threshold))]
    key_numbers = {
        "p_value": first["p_value"], "theta": first["theta"],
        "n_finite": first["n_finite"], "min_holm_p": min_holm,
    }
    return FamilyResult(
        family="bunching", status=status, reason=reason,
        key_numbers=key_numbers, diagnostics=diagnostics,
    )


# dee query lattice: fewer points per forcing dim than the CLI debias
# default (15) — the survey favors breadth over surface resolution.
_DEE_GRID = 8


def _run_dee(
    df: pd.DataFrame,
    intake: IntakeReport,
    declared: DeclaredInputs,
    plan: FamilyPlan,
    budget: dict | None,
    fam_rng: np.random.Generator,
    fam_dir: Path,
    artifacts: dict,
) -> FamilyResult:
    # The survey() loop already gated on a CREDIBLE rdd family result; this
    # runner rebuilds the same rdd Dataset and debiases over its discoveries.
    ds = _rdd_dataset(df, intake)
    eff = _effective_budget(intake.search_plan, budget)
    scan = lord3_scan(ds, k=int(eff["k"]), degree=1, rng=fam_rng)
    # query lattice: _DEE_GRID points per forcing dim over observed ranges
    axes = [
        np.linspace(ds.Z[:, j].min(), ds.Z[:, j].max(), _DEE_GRID)
        for j in range(ds.Z.shape[1])
    ]
    mesh = np.meshgrid(*axes, indexing="ij")
    query = np.column_stack([m.ravel() for m in mesh])
    res = dee_debias(
        ds, query, scan, m_prime=min(10, len(scan.discoveries)), rng=fam_rng
    )
    n_experiments = len(res.vknn.experiments)
    n_used = int(np.asarray(res.used, dtype=bool).sum())
    # Figure payload (task 7): forest of per-experiment local-2SLS taus.
    artifacts.update(
        tau=[e.tau for e in res.effects], se=[e.se for e in res.effects]
    )
    key_numbers = {
        "w_debias": res.weights.w_debias,
        "n_experiments": n_experiments,
        "n_experiments_used": n_used,
        "mean_cate_raw": _nanmean(res.cate_raw),
        "mean_cate_debiased": _nanmean(res.cate_debiased),
        "mean_cate_direct": _nanmean(res.cate_direct),
    }
    diagnostics = {
        "caveats": [FAMILIES["dee"].caveat],
        "weights_strategy": res.weights.strategy,
        "scan_k": int(eff["k"]),
        "n_query": int(query.shape[0]),
        "n_experiments_used_bias": res.diagnostics.get("n_experiments_used_bias"),
        "dropped_experiments": res.diagnostics.get("dropped"),
    }
    if "reason" in res.diagnostics:
        status, reason = "null", str(res.diagnostics["reason"])
    else:
        # Documented status-semantics stretch: dee is a surface fit, not a
        # hypothesis test — "credible" here means the fit completed with a
        # usable experiment ensemble (the method card explains).
        status = "credible"
        reason = f"debiased CATE surface fitted over {n_used} experiments"
    return FamilyResult(
        family="dee", status=status, reason=reason,
        key_numbers=key_numbers, diagnostics=diagnostics,
    )


# ---------------------------------------------------------------------------
# survey()
# ---------------------------------------------------------------------------


def _dataset_meta(df: pd.DataFrame, source: str, declared: DeclaredInputs, intake) -> dict:
    time_col = declared.time or next(
        (c.name for c in intake.profile.columns if c.is_time_like), None
    )
    time_range = None
    if time_col is not None and time_col in df.columns:
        col = df[time_col].dropna()
        if len(col):
            time_range = [col.min(), col.max()]
    return {
        "source": source,
        "n_rows": int(len(df)),
        "n_cols": int(df.shape[1]),
        "columns_truncated": [str(c) for c in df.columns[:20]],
        "time_column": time_col,
        "time_range": time_range,
    }


def survey(
    csv_or_df: str | Path | pd.DataFrame,
    *,
    context: str | None = None,
    guidance: GuidanceBackend | None = None,
    rng: np.random.Generator | None = None,
    out_dir: str | Path,
    budget: dict | None = None,  # passed through discover() — its keys, its validation
    time: str | None = None,
    unit: str | None = None,
    cutoffs: dict[str, float] | None = None,
    instruments: list[str] | None = None,
    thresholds: dict[str, float] | None = None,
    seed: int | None = None,  # metadata only; rng governs randomness
) -> SurveyResult:
    """Run every applicable method family against one dataset; see module docstring."""
    if rng is None:
        raise ValueError("pass an explicit numpy Generator (reproducibility contract)")
    out_dir = Path(out_dir)
    if isinstance(csv_or_df, (str, Path)):
        source = str(csv_or_df)
        df = pd.read_csv(csv_or_df)
    else:
        source = "<dataframe>"
        df = csv_or_df
    declared = DeclaredInputs(
        time=time, unit=unit,
        cutoffs=dict(cutoffs or {}), instruments=list(instruments or []),
        thresholds=dict(thresholds or {}),
    )

    # ONE upfront spawn in registry order, BEFORE any other rng use: a skipped
    # family never shifts another family's stream.
    fam_rngs = dict(zip(FAMILY_ORDER, rng.spawn(7), strict=True))

    intake = study(csv_or_df, context=context, guidance=guidance, rng=rng,
                   out=out_dir / "intake")

    wrapped: GuidanceBackend | None = None
    guidance_log_path: str | None = None
    if guidance is not None:
        # Appends to the SAME jsonl study() just wrote (GuidanceLog appends).
        log = GuidanceLog(out_dir / "intake" / "guidance_log.jsonl")
        wrapped = LoggedBackend(guidance, log)
        guidance_log_path = "intake/guidance_log.jsonl"  # out_dir-relative
    plans, declared = resolve_applicability(
        intake.profile, None, declared, wrapped, context=context
    )

    families: dict[str, FamilyResult] = {}
    ran: list[str] = []
    not_run: dict[str, str] = {}
    family_artifacts: dict[str, dict] = {}  # live-object figure payloads (task 7)
    for name in FAMILY_ORDER:
        plan = plans[name]
        if not plan.run:
            status = (
                "needs_input"
                if plan.heuristic.status == "needs_input" and plan.override is None
                else "skipped"
            )
            families[name] = FamilyResult(
                family=name, status=status, reason=plan.reason,
                applicability=_plan_dict(plan),
                no_figure_reason=f"no figure: family did not run ({plan.reason})",
            )
            not_run[name] = plan.reason
            continue
        if name == "dee" and families["rdd"].status != "credible":
            # Runtime gate ON TOP of applicability (plan task 6): dee debiases
            # a VALIDATED rdd discovery; without one there is nothing to debias.
            reason = "no validated rdd discovery to debias"
            families[name] = FamilyResult(
                family=name, status="skipped", reason=reason,
                applicability=_plan_dict(plan),
                no_figure_reason=f"no figure: family did not run ({reason})",
            )
            not_run[name] = reason
            continue
        fam_dir = out_dir / "families" / name
        runner_fn = globals()[f"_run_{name}"]  # module attribute: monkeypatchable
        try:
            result = runner_fn(df, intake, declared, plan, budget, fam_rngs[name],
                               fam_dir, family_artifacts.setdefault(name, {}))
        except Exception as exc:  # noqa: BLE001 — DOCUMENTED isolation boundary (plan task 5):
            # a family failure must never abort the survey; BaseException
            # (KeyboardInterrupt/SystemExit) still propagates.
            result = FamilyResult(
                family=name, status="failed", reason="family raised",
                error=str(exc),
                diagnostics={"traceback": traceback.format_exc()},
            )
        result.applicability = _plan_dict(plan)
        # NaN -> None NOW (not just at save): FamilyResult equality across a
        # save/load round trip must hold, and NaN != NaN would break it.
        result.key_numbers = jsonable(result.key_numbers)
        result.diagnostics = jsonable(result.diagnostics)
        families[name] = result
        ran.append(name)

    # ------------------------------------------------------------------
    # Figures (plan task 7) — presentation only, never a verdict change.
    # matplotlib importability is probed ONCE per survey.
    mpl_reason = missing_matplotlib_reason()
    for name in ran:
        fam = families[name]
        if fam.status in ("failed", "needs_input"):
            fam.figures = {}
            fam.no_figure_reason = f"no figure: family did not run ({fam.reason})"
            continue
        if mpl_reason is not None:
            fam.figures = {}
            fam.no_figure_reason = mpl_reason
            continue
        artifacts = family_artifacts.get(name)
        if not artifacts:
            # keep a runner-recorded reason (scanless rdd/did, selection-only iv)
            if fam.no_figure_reason is None:
                fam.no_figure_reason = f"no figure: nothing to draw for the {name} family"
            continue
        fam.figures, fam.no_figure_reason = render_family_figures(name, artifacts, out_dir)

    fam_root = out_dir / "families"
    for name in ran:
        fam_root.mkdir(parents=True, exist_ok=True)
        families[name].details_path = f"families/{name}.json"
        detail = jsonable(dataclasses.asdict(families[name]))
        (fam_root / f"{name}.json").write_text(json.dumps(detail, indent=1), encoding="utf-8")

    coverage = {
        "ran": ran,
        "not_run": not_run,
        "rdd": families["rdd"].diagnostics.get("searched"),
        "did": families["did"].diagnostics.get("searched"),
    }
    result = SurveyResult(
        out_dir=out_dir,
        families=families,
        coverage=coverage,
        dataset=jsonable(_dataset_meta(df, source, declared, intake)),
        natex_version=_natex_version(),
        seed=seed,
        created=datetime.now(UTC).isoformat(),
        context=context,
        guidance_log_path=guidance_log_path,
    )
    result.save()

    # Reports (plan task 8): report.md ALWAYS (pure Python); report.html only
    # with the report extra — a missing jinja2 degrades to the md report plus
    # a recorded install message, never an exception. Both renderers consume
    # the JSON-NATIVE dict, the same path a re-render from survey.json takes.
    payload = json.loads(result.to_json())
    result.report_md = render_survey_md(payload, out_dir).name  # out_dir-relative
    try:
        result.report_html = render_survey_html(payload, out_dir).name
    except ImportError as exc:  # message names natex-discovery[report]
        result.report_html = None
        coverage.setdefault("notes", []).append(str(exc))
    result.save()  # re-save with the report paths recorded
    return result
