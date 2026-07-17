"""One-command systematic design survey: run a dataset against all method families."""

from natex.survey.applicability import (
    ApplicabilityResponse,
    ConfigHints,
    ConfigValueHint,
    FamilyDecision,
    FamilyPlan,
    FamilyVerdict,
    heuristic_applicability,
    resolve_applicability,
)
from natex.survey.registry import (
    FAMILIES,
    FAMILY_ORDER,
    DeclaredInputs,
    MethodFamily,
    Requirement,
)
from natex.survey.runner import FamilyResult, SurveyResult, survey

__all__ = [
    "FAMILIES",
    "FAMILY_ORDER",
    "ApplicabilityResponse",
    "ConfigHints",
    "ConfigValueHint",
    "DeclaredInputs",
    "FamilyDecision",
    "FamilyPlan",
    "FamilyResult",
    "FamilyVerdict",
    "MethodFamily",
    "Requirement",
    "SurveyResult",
    "heuristic_applicability",
    "resolve_applicability",
    "survey",
]
