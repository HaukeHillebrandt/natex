"""One-command systematic design survey: run a dataset against all method families."""

from natex.survey.applicability import FamilyVerdict, heuristic_applicability
from natex.survey.registry import (
    FAMILIES,
    FAMILY_ORDER,
    DeclaredInputs,
    MethodFamily,
    Requirement,
)

__all__ = [
    "FAMILIES",
    "FAMILY_ORDER",
    "DeclaredInputs",
    "FamilyVerdict",
    "MethodFamily",
    "Requirement",
    "heuristic_applicability",
]
