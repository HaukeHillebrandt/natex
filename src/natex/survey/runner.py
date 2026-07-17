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
``natex.discover`` end to end; the remaining family runners arrive in plan
task 6 (until then they raise NotImplementedError, which the isolation
boundary records as an honest ``failed`` — never a silent absence).
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

from natex.data.spec import Dataset, DatasetSpec
from natex.discover import DiscoverReport, discover
from natex.intake.analyst import IntakeReport, study
from natex.jsonutil import jsonable
from natex.llm import GuidanceBackend, GuidanceLog, LoggedBackend
from natex.survey.applicability import FamilyPlan, resolve_applicability
from natex.survey.registry import FAMILIES, FAMILY_ORDER, DeclaredInputs

ALPHA = 0.05  # verdict gate for scan/kink/bunching p-values
# sc placebo gate: the in-space +1-rank test has granularity 1/(n_used+1), so
# 0.05 is often unattainable with few donors — documented coarser gate.
SC_ALPHA = 0.10

_NO_FIGURE_YET = "no figure: figure generation arrives with the report stage (plan task 7)"


@dataclass
class FamilyResult:
    family: str
    status: str  # credible|null|skipped|needs_input|failed
    reason: str  # one sentence, always set
    applicability: dict = field(default_factory=dict)  # FamilyPlan serialized
    key_numbers: dict = field(default_factory=dict)  # flat name->number (NaN -> null)
    diagnostics: dict = field(default_factory=dict)  # extras incl. attached caveats
    figures: dict[str, str] = field(default_factory=dict)  # name -> out_dir-relative posix path
    no_figure_reason: str | None = None  # set whenever figures == {}
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
# the survey()'s documented isolation boundary.
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
            no_figure_reason=f"no figure: no {family} configuration scanned",
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
        no_figure_reason=_NO_FIGURE_YET,
    )


def _run_did(
    df: pd.DataFrame,
    intake: IntakeReport,
    declared: DeclaredInputs,
    plan: FamilyPlan,
    budget: dict | None,
    fam_rng: np.random.Generator,
    fam_dir: Path,
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
        no_figure_reason=_NO_FIGURE_YET,
    )


def _not_implemented(name: str):
    def runner(df, intake, declared, plan, budget, fam_rng, fam_dir) -> FamilyResult:
        raise NotImplementedError(f"{name} family runner arrives in plan task 6")

    return runner


# Plan task 6 replaces these with real runners; until then a family that
# should run reports an honest "failed" through the isolation boundary.
_run_kink = _not_implemented("kink")
_run_iv = _not_implemented("iv")
_run_sc = _not_implemented("sc")
_run_bunching = _not_implemented("bunching")
_run_dee = _not_implemented("dee")


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
                no_figure_reason="no figure: family did not run",
            )
            not_run[name] = plan.reason
            continue
        fam_dir = out_dir / "families" / name
        runner_fn = globals()[f"_run_{name}"]  # module attribute: monkeypatchable
        try:
            result = runner_fn(df, intake, declared, plan, budget, fam_rngs[name], fam_dir)
        except Exception as exc:  # noqa: BLE001 — DOCUMENTED isolation boundary (plan task 5):
            # a family failure must never abort the survey; BaseException
            # (KeyboardInterrupt/SystemExit) still propagates.
            result = FamilyResult(
                family=name, status="failed", reason="family raised",
                error=str(exc),
                diagnostics={"traceback": traceback.format_exc()},
                no_figure_reason="no figure: family failed",
            )
        result.applicability = _plan_dict(plan)
        # NaN -> None NOW (not just at save): FamilyResult equality across a
        # save/load round trip must hold, and NaN != NaN would break it.
        result.key_numbers = jsonable(result.key_numbers)
        result.diagnostics = jsonable(result.diagnostics)
        families[name] = result
        ran.append(name)

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
    return result
