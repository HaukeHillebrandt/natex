"""Understanding / DesignCandidate / SearchPlan: the analyst pass's structured output.

``Understanding`` is what a guidance backend believes about the dataset (spec 6a);
``SearchPlan`` ranks :class:`DesignCandidate` configurations for ``natex.discover``
(spec 6b: ranked candidates are scanned FIRST, the exhaustive remainder is still
scanned within budget — a plan orders the search, it never truncates it).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator


class ColumnGuess(BaseModel):
    """One column proposed for a role, with the reason the guess fired."""

    column: str
    reason: str = ""


class DiDStructure(BaseModel):
    """A (unit, time) pair that plausibly indexes a panel."""

    unit: str
    time: str
    reason: str = ""


class Understanding(BaseModel):
    """A backend's structured belief about the dataset (guidance task ``understand``)."""

    shape: Literal["cross-section", "time-series", "panel", "aggregated-cells"]
    unit_of_observation: str = "row"
    treatments: list[ColumnGuess] = Field(default_factory=list)
    outcomes: list[ColumnGuess] = Field(default_factory=list)
    forcing: list[ColumnGuess] = Field(default_factory=list)
    did_structures: list[DiDStructure] = Field(default_factory=list)
    quirks: list[str] = Field(default_factory=list)
    notes: str = ""


class DesignCandidate(BaseModel):
    """One concrete design configuration for discovery to scan."""

    design: Literal["rdd", "did"]
    treatment: str
    outcome: str | None = None
    forcing: list[str] = Field(default_factory=list)  # rdd: must be nonempty
    unit: str | None = None  # did only
    time: str | None = None  # did: must be set
    rationale: str = ""
    priority: int = 0  # 0 = scan first

    @model_validator(mode="after")
    def _design_requirements(self) -> DesignCandidate:
        if self.design == "rdd" and not self.forcing:
            raise ValueError("rdd candidate requires a nonempty forcing list")
        if self.design == "did" and self.time is None:
            raise ValueError("did candidate requires time")
        return self

    def key(self) -> tuple:
        """Dedup key used by ``discover()``; forcing order-insensitive for rdd."""
        if self.design == "rdd":
            return ("rdd", self.treatment, tuple(sorted(self.forcing)))
        return ("did", self.treatment, self.unit, self.time)


class SearchPlan(BaseModel):
    """Ranked design candidates plus budget hints (guidance task ``search_plan``)."""

    candidates: list[DesignCandidate] = Field(default_factory=list)
    budget: dict = Field(default_factory=dict)  # hints: k, q, coarse, n_coarse, max_configs

    def ranked(self) -> list[DesignCandidate]:
        """Candidates by ascending priority; stable (list order breaks ties)."""
        return sorted(self.candidates, key=lambda c: c.priority)
