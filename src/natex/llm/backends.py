"""Guidance backends: request/response models, protocol, and MockBackend.

Spec 6c contract: guidance PROPOSES and VETOES but never fabricates
statistics. Every hook is optional — the pipeline produces identical
statistical output with or without a backend (a veto is only ever a flag in
the results). Every request+response is appended to the JSONL guidance log
(see ``natex.llm.log``) for reproducibility.
"""

from __future__ import annotations

import json
from typing import Literal, Protocol, runtime_checkable

from pydantic import BaseModel, Field

TASKS = (
    "understand",
    "prepare",
    "search_plan",
    "interpret_discovery",
    "audit_assumptions",
    "review_control_group",
)

Task = Literal[
    "understand",
    "prepare",
    "search_plan",
    "interpret_discovery",
    "audit_assumptions",
    "review_control_group",
]


class GuidanceRequest(BaseModel):
    """One guidance question posed to a backend."""

    task: Task
    payload: dict
    schema_hint: dict = Field(default_factory=dict)  # JSON schema the content should satisfy


class GuidanceResponse(BaseModel):
    """Structured answer from a backend."""

    content: dict
    raw_text: str = ""
    backend: str


@runtime_checkable
class GuidanceBackend(Protocol):
    """Anything that can answer a GuidanceRequest."""

    name: str

    def complete(self, request: GuidanceRequest) -> GuidanceResponse: ...


# One short instruction paragraph per task, shared by AgentBackend request
# files and both API backends' prompts.
TASK_INSTRUCTIONS: dict[str, str] = {
    "understand": (
        "You are given a column-level profile of a tabular dataset plus any user-supplied "
        "context. Describe what each column most likely measures and which columns could serve "
        "as treatment, outcome, forcing (running) variable, time, or unit identifiers. Answer "
        "as JSON matching the provided schema; do not invent columns."
    ),
    "prepare": (
        "Propose a declarative data-preparation plan (column roles, encodings, discretization, "
        "drops, row filters, optional subsample) for the profiled dataset. You may only name "
        "existing columns and the allowed operations in the schema; you never write code — the "
        "plan is executed by natex itself."
    ),
    "search_plan": (
        "Rank candidate natural-experiment designs (RDD forcing variables, DiD unit/time "
        "panels) for the prepared dataset, most promising first, with brief reasons and budget "
        "hints. Ranked candidates are scanned first; the exhaustive remainder is still scanned "
        "within budget, so omissions are never silently dropped."
    ),
    "interpret_discovery": (
        "You are shown a validated discovery (location, neighborhood, validation p-values) "
        "WITHOUT raw outcome values. Suggest a plausible institutional mechanism for the "
        "discontinuity and note anything that looks artifactual. Your notes are advisory only "
        "and never alter the statistics."
    ),
    "audit_assumptions": (
        "Given a discovery's validation summary (randomization, placebo, density tests), flag "
        "identification-assumption risks (manipulation, confounded thresholds, anticipation). "
        "Respond as JSON per the schema; you may recommend a veto, which is recorded as a flag "
        "only — no statistic is dropped or changed."
    ),
    "review_control_group": (
        "Review the proposed DiD control group (GESS expansion profile and composition "
        "summary) for plausibility as a counterfactual. You may veto it; the veto is recorded "
        "as a flag alongside the unchanged effect estimate, never in place of it."
    ),
}


class MockBackend:
    """Canned responses for tests. Pops ``responses`` in order; records every request."""

    name = "mock"

    def __init__(self, responses: list[dict | GuidanceResponse]):
        self._responses = list(responses)
        self._n_total = len(self._responses)
        self.requests: list[GuidanceRequest] = []

    def complete(self, request: GuidanceRequest) -> GuidanceResponse:
        self.requests.append(request)
        if not self._responses:
            raise RuntimeError(
                f"MockBackend exhausted after {self._n_total} responses (task={request.task!r})"
            )
        item = self._responses.pop(0)
        if isinstance(item, GuidanceResponse):
            return item
        return GuidanceResponse(content=item, raw_text=json.dumps(item), backend=self.name)
