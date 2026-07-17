"""Heuristic per-family applicability from registry predicates alone.

Pure function of the intake profile, an optional declared :class:`DatasetSpec`,
and user-declared inputs: no rng, no I/O, no DataFrame parameter — CONTENT-BLIND
by construction. Always returns all seven families in :data:`FAMILY_ORDER`.
"""

from __future__ import annotations

from dataclasses import dataclass

from natex.data.spec import DatasetSpec
from natex.intake.profiler import IntakeProfile
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
