"""Per-family applicability: content-blind heuristics + advisory analyst pass.

:func:`heuristic_applicability` is a pure function of the intake profile, an
optional declared :class:`DatasetSpec`, and user-declared inputs: no rng, no
I/O, no DataFrame parameter — CONTENT-BLIND by construction. Always returns
all seven families in :data:`FAMILY_ORDER`.

:func:`resolve_applicability` layers the optional ``method_applicability``
guidance task on top: the analyst may override heuristics BOTH ways, every
override is recorded (``heuristic_said`` vs ``analyst_said`` + reason), and
analyst proposals feed config (cutoffs, instruments, thresholds, treated
unit) — never statistics.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass

from pydantic import BaseModel, Field

from natex.data.spec import DatasetSpec

# Uniform fallback policy, SAME exception tuple as intake.analyst: backend
# transport/timeout failures and JSON/schema violations fall back to
# heuristics — never a crash, never a bare except.
from natex.intake.analyst import _FALLBACK_EXCEPTIONS
from natex.intake.profiler import IntakeProfile
from natex.jsonutil import jsonable
from natex.llm import GuidanceBackend, GuidanceRequest
from natex.survey.registry import FAMILIES, DeclaredInputs


@dataclass
class FamilyVerdict:
    family: str
    status: str  # "applicable" | "inapplicable" | "needs_input"
    reason: str  # met: "all requirements met"; unmet: joined requirement descriptions
    unmet: list[str]  # unmet requirement keys, registry order


def heuristic_applicability(
    profile: IntakeProfile,
    spec: DatasetSpec | None,
    declared: DeclaredInputs,
) -> dict[str, FamilyVerdict]:
    """Evaluate every registry requirement; all requirements met -> ``applicable``.

    Otherwise: every unmet requirement ``user_suppliable`` -> ``needs_input``,
    else ``inapplicable``. Reasons join the unmet requirement descriptions
    verbatim in registry order (deterministic).
    """
    verdicts: dict[str, FamilyVerdict] = {}
    for name, family in FAMILIES.items():  # insertion order == FAMILY_ORDER
        unmet = [r for r in family.requirements if not r.check(profile, spec, declared)]
        if not unmet:
            verdicts[name] = FamilyVerdict(
                family=name, status="applicable", reason="all requirements met", unmet=[]
            )
            continue
        status = "needs_input" if all(r.user_suppliable for r in unmet) else "inapplicable"
        verdicts[name] = FamilyVerdict(
            family=name,
            status=status,
            reason="; ".join(r.description for r in unmet),
            unmet=[r.key for r in unmet],
        )
    return verdicts


# ---------------------------------------------------------------------------
# method_applicability response models. Schema-safe by construction: NO dict
# fields (llm/api._make_strict sets additionalProperties: false on every
# object node, which would destroy a dict[str, Model] field) and NO pydantic
# field constraints (_strict_schema strips them) — lists of keyed submodels.
# ---------------------------------------------------------------------------


class ConfigValueHint(BaseModel):
    """One column=value proposal; cutoffs and thresholds share this shape."""

    column: str
    value: float


class ConfigHints(BaseModel):
    """Analyst config proposals — feed configuration ONLY, never statistics."""

    cutoffs: list[ConfigValueHint] = Field(default_factory=list)
    instruments: list[str] = Field(default_factory=list)
    thresholds: list[ConfigValueHint] = Field(default_factory=list)
    treated_unit: str | None = None
    t0: float | None = None


class FamilyDecision(BaseModel):
    """One family's analyst decision: run or not, why, and optional hints."""

    family: str
    run: bool
    reason: str = ""
    config_hints: ConfigHints = Field(default_factory=ConfigHints)


class ApplicabilityResponse(BaseModel):
    """Full ``method_applicability`` reply: one decision per family."""

    families: list[FamilyDecision]


@dataclass
class FamilyPlan:
    """Resolved per-family decision: analyst (when available) over heuristic."""

    family: str
    run: bool
    reason: str
    heuristic: FamilyVerdict
    config_hints: ConfigHints
    override: dict | None  # {"heuristic_said", "analyst_said", "reason"} when run differs
    guidance_error: str | None  # backend/parse failure -> fell back to heuristics


def _heuristic_plan(verdict: FamilyVerdict, error: str | None = None) -> FamilyPlan:
    return FamilyPlan(
        family=verdict.family,
        run=verdict.status == "applicable",
        reason=verdict.reason,
        heuristic=verdict,
        config_hints=ConfigHints(),
        override=None,
        guidance_error=error,
    )


def _clean_hints(
    hints: ConfigHints, known: set[str], numeric: set[str]
) -> tuple[ConfigHints, list[str]]:
    """Drop hints naming unknown/non-numeric columns; return (kept, notes)."""
    notes: list[str] = []

    def _bad(kind: str, column: str) -> bool:
        if column not in known:
            notes.append(f"{kind} hint dropped: column {column!r} not in profile")
            return True
        if column not in numeric:
            notes.append(f"{kind} hint dropped: column {column!r} is not numeric")
            return True
        return False

    return (
        ConfigHints(
            cutoffs=[h for h in hints.cutoffs if not _bad("cutoff", h.column)],
            instruments=[c for c in hints.instruments if not _bad("instrument", c)],
            thresholds=[h for h in hints.thresholds if not _bad("threshold", h.column)],
            treated_unit=hints.treated_unit,
            t0=hints.t0,
        ),
        notes,
    )


def _merge_declared(declared: DeclaredInputs, plans: dict[str, FamilyPlan]) -> DeclaredInputs:
    """Fold cleaned hints into the declared inputs; explicit declarations WIN."""
    cutoffs = dict(declared.cutoffs)
    thresholds = dict(declared.thresholds)
    instruments = list(declared.instruments)
    treated_unit = declared.treated_unit
    t0 = declared.t0
    for plan in plans.values():  # FAMILY_ORDER
        hints = plan.config_hints
        for h in hints.cutoffs:
            cutoffs.setdefault(h.column, h.value)
        for h in hints.thresholds:
            thresholds.setdefault(h.column, h.value)
        for col in hints.instruments:
            if col not in instruments:
                instruments.append(col)
        if treated_unit is None:
            treated_unit = hints.treated_unit
        if t0 is None:
            t0 = hints.t0
    return dataclasses.replace(
        declared,
        cutoffs=cutoffs,
        thresholds=thresholds,
        instruments=instruments,
        treated_unit=treated_unit,
        t0=t0,
    )


def resolve_applicability(
    profile: IntakeProfile,
    spec: DatasetSpec | None,
    declared: DeclaredInputs,
    guidance: GuidanceBackend | None,
    *,
    context: str | None = None,
) -> tuple[dict[str, FamilyPlan], DeclaredInputs]:
    """Heuristics -> optional one-shot method_applicability request -> merged declared inputs.

    ``guidance=None``: no request fires; ``run = (status == "applicable")``;
    overrides all None. Backend given: ONE ``GuidanceRequest`` carrying the
    profile, context, declared inputs, per-family descriptors (with per-
    requirement ``met`` flags) and the heuristic verdicts. Parse failures /
    ``_FALLBACK_EXCEPTIONS`` fall back to heuristics per family with
    ``guidance_error`` recorded. Unknown family names in the reply are dropped
    (recorded); families the reply omits keep their heuristic decision.

    Hint hygiene: cutoff/threshold/instrument columns must exist in the
    profile and be numeric; ``treated_unit`` must be a string (model-
    enforced); invalid hints are DROPPED and recorded in the family's
    ``guidance_error`` — never a crash, and hints never overwrite an
    explicitly declared value (CLI/user declarations win). Returns the merged
    :class:`DeclaredInputs` used by the runner. Hints feed config ONLY — no
    statistic is touched here.
    """
    verdicts = heuristic_applicability(profile, spec, declared)
    if guidance is None:
        return {n: _heuristic_plan(v) for n, v in verdicts.items()}, declared

    request = GuidanceRequest(
        task="method_applicability",
        payload={
            "profile": json.loads(profile.to_json()),
            "context": context,
            "declared": jsonable(dataclasses.asdict(declared)),
            "families": [
                {
                    "name": f.name,
                    "title": f.title,
                    "description": f.description,
                    "requirements": [
                        {
                            "key": r.key,
                            "description": r.description,
                            "met": r.check(profile, spec, declared),
                        }
                        for r in f.requirements
                    ],
                }
                for f in FAMILIES.values()
            ],
            "heuristics": {
                n: {"status": v.status, "reason": v.reason, "unmet": v.unmet}
                for n, v in verdicts.items()
            },
        },
        schema_hint=ApplicabilityResponse.model_json_schema(),
    )
    try:
        reply = ApplicabilityResponse.model_validate(guidance.complete(request).content)
    except _FALLBACK_EXCEPTIONS as exc:
        error = f"method_applicability: {exc} -- fell back to heuristics"
        return {n: _heuristic_plan(v, error=error) for n, v in verdicts.items()}, declared

    decisions: dict[str, FamilyDecision] = {}
    unknown: list[str] = []
    for decision in reply.families:
        if decision.family in verdicts:
            decisions[decision.family] = decision  # last mention wins on duplicates
        else:
            unknown.append(decision.family)

    known_cols = {c.name for c in profile.columns}
    numeric_cols = {c.name for c in profile.columns if c.is_numeric}
    plans: dict[str, FamilyPlan] = {}
    for name, verdict in verdicts.items():  # FAMILY_ORDER
        decision = decisions.get(name)
        if decision is None:  # omitted family keeps its heuristic decision
            plans[name] = _heuristic_plan(verdict)
            continue
        hints, notes = _clean_hints(decision.config_hints, known_cols, numeric_cols)
        heuristic_run = verdict.status == "applicable"
        override = None
        if decision.run != heuristic_run:
            override = {
                "heuristic_said": heuristic_run,
                "analyst_said": decision.run,
                "reason": decision.reason,
            }
        plans[name] = FamilyPlan(
            family=name,
            run=decision.run,
            reason=decision.reason,
            heuristic=verdict,
            config_hints=hints,
            override=override,
            guidance_error="; ".join(notes) if notes else None,
        )

    if unknown:  # dropped, but recorded — never silently absent
        note = "unknown families in guidance reply dropped: " + ", ".join(unknown)
        for plan in plans.values():
            plan.guidance_error = (
                f"{plan.guidance_error}; {note}" if plan.guidance_error else note
            )

    return plans, _merge_declared(declared, plans)
