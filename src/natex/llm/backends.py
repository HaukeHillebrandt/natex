"""Guidance backends: request/response models, protocol, and MockBackend.

Spec 6c contract: guidance PROPOSES and VETOES but never fabricates
statistics. Every hook is optional — the pipeline produces identical
statistical output with or without a backend (a veto is only ever a flag in
the results). Every request+response is appended to the JSONL guidance log
(see ``natex.llm.log``) for reproducibility.
"""

from __future__ import annotations

import json
import re
from typing import Literal, Protocol, runtime_checkable

from pydantic import BaseModel, Field

TASKS = (
    "understand",
    "prepare",
    "search_plan",
    "interpret_discovery",
    "audit_assumptions",
    "review_control_group",
    "method_applicability",
)

Task = Literal[
    "understand",
    "prepare",
    "search_plan",
    "interpret_discovery",
    "audit_assumptions",
    "review_control_group",
    "method_applicability",
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
    "method_applicability": (
        "You are shown a dataset profile, user context, declared inputs, per-family method "
        "descriptors and natex's heuristic applicability verdicts. For each family decide run "
        "true/false with a reason a non-statistician can follow, and optionally propose config "
        "hints (kink cutoffs, candidate instrument columns, bunching thresholds, a "
        "synthetic-control treated unit and t0) grounded in the context. You may override the "
        "heuristics in either direction; overrides are recorded, and your hints feed "
        "configuration only — never statistics."
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


# --- NullBackend heuristics: named module-level constants -------------------
_AGG_MAX_ROWS = 5000  # "aggregated-cells" only plausible below this row count
_QUIRK_MISSING = 0.2  # missing_frac above which a column is flagged as a quirk
_DROP_MISSING = 0.5  # missing_frac above which prepare() drops the column
_PREFIX_EXPLAINED = 0.95  # missingness this prefix-concentrated is structural (issue #6)
_BOUNDARY_TOL = 0.01  # boundary rows within this fraction of n_rows share one filter
_SUB_N = 20000  # subsample size when the profile is large
_SUB_MAX = 50000  # rows above which prepare() proposes a subsample
_COARSE_MIN = 10000  # rows above which the search-plan budget suggests coarse scan
_PLAN_K = 50  # search-plan budget hint: neighborhood size
_PLAN_Q = 99  # search-plan budget hint: randomization-test draws
_PLAN_N_COARSE = 2000  # search-plan budget hint: coarse-pass centers
# column names that suggest each row is an aggregated cell, not one observation
_COUNT_NAME = re.compile(r"^(n|count|n_obs|weight|wt|pop|population|cells?)$", re.IGNORECASE)


class NullBackend:
    """Deterministic profile-only heuristics — the no-LLM degradation path (spec 6a).

    Every answer is derived from the request payload alone: NO rng, NO data
    access, NO network. Identical payload => identical content and bitwise
    identical ``raw_text``. NullBackend NEVER vetoes. Per-task heuristics:

    ``understand`` (payload ``{"profile": <IntakeProfile dict>, "context": str|None}``):
      * ``shape``: ``"panel"`` if ``panel_candidates`` else ``"aggregated-cells"``
        if ``n_rows < _AGG_MAX_ROWS`` and some column name matches ``_COUNT_NAME``
        else ``"time-series"`` if some time-like column has ``n_unique == n_rows``
        else ``"cross-section"``.
      * ``unit_of_observation``: first panel candidate's unit column, else ``"row"``.
      * ``treatments`` = profile ``treatment_candidates`` (reason "binary 0/1 column").
      * ``outcomes`` = numeric, non-binary, non-time-like columns (reason
        "numeric, non-binary").
      * ``forcing`` = profile ``forcing_candidates`` minus time-like columns
        (reason "numeric with >= 20 distinct values").
      * ``did_structures`` from ``panel_candidates`` (reason "unit x time grid
        covers >= 95% of rows").
      * ``quirks``: "<col>: constant column" for ``n_unique <= 1``;
        "<col>: {pct}% missing" for ``missing_frac > _QUIRK_MISSING``.
      * ``notes`` = "NullBackend heuristics (no LLM)".

    ``prepare`` (payload adds ``"understanding"`` and ``"seed"``): PrepPlan dict with
    ``drop_cols`` = constant columns + columns with ``missing_frac > _DROP_MISSING``,
    EXCEPT clusters of >= 2 such columns whose missingness is structural (issue #6):
    >= ``_PREFIX_EXPLAINED`` of it lies in the all-missing row prefix before
    ``first_valid_index``, the boundary rows agree within ``_BOUNDARY_TOL * n_rows``,
    a fully observed monotone column can express the shared boundary (profile
    ``boundary_values``), and every member's post-boundary missing_frac falls
    below ``_QUIRK_MISSING`` — those columns are kept and ONE shared row filter
    ``monotone_col >= boundary value`` is emitted instead (still profile-only:
    all evidence is precomputed by the profiler).
    ``subsample = {"n": _SUB_N, "seed": payload["seed"]}`` iff ``n_rows > _SUB_MAX``;
    everything else empty (profile-only degradation, spec 6a).

    ``search_plan`` (payload ``{"profile": <post-prep profile>, "understanding", "context"}``):
    understanding guesses naming columns absent from the (post-prep) profile are
    discarded first (the understanding predates the prep plan's drops); then
    for each treatment guess ``t`` (profile order) one ``rdd`` candidate — outcome =
    first outcome guess != t (else None), forcing = all forcing guesses not in
    {t, outcome}, skipped if forcing empty; then for each (t x did_structure) one
    ``did`` candidate (unit/time from the structure, outcome as above). ``priority``
    is the running index; ``rationale`` states the rule that fired. ``budget`` =
    ``{"k": _PLAN_K, "q": _PLAN_Q, "coarse": n_rows > _COARSE_MIN, "n_coarse": _PLAN_N_COARSE}``.

    ``interpret_discovery``: a deterministic sentence naming the design, the
    dominant forcing column (max |influence|) and the location; ``matched_policies``
    empty, ``confounded_risk`` "unknown".

    ``audit_assumptions``: every assumption "unreviewed", ``veto`` False, one caveat
    naming the limitation.

    ``review_control_group``: ``face_valid`` None, ``veto`` False (NullBackend never
    vetoes), reason reports ``n_expansions``.

    ``method_applicability``: pure echo of ``payload["heuristics"]`` — one decision
    per family with ``run = (status == "applicable")``, the heuristic reason
    verbatim, and empty config hints (NullBackend proposes nothing).
    """

    name = "null"

    def complete(self, request: GuidanceRequest) -> GuidanceResponse:
        handlers = {
            "understand": self._understand,
            "prepare": self._prepare,
            "search_plan": self._search_plan,
            "interpret_discovery": self._interpret_discovery,
            "audit_assumptions": self._audit_assumptions,
            "review_control_group": self._review_control_group,
            "method_applicability": self._method_applicability,
        }
        handler = handlers.get(request.task)
        if handler is None:  # unreachable through GuidanceRequest, but never silent
            raise ValueError(f"NullBackend has no heuristic for task {request.task!r}")
        content = handler(request.payload)
        return GuidanceResponse(
            content=content, raw_text=json.dumps(content, sort_keys=True), backend=self.name
        )

    @staticmethod
    def _understand(payload: dict) -> dict:
        prof = payload["profile"]
        cols = prof.get("columns", [])
        n_rows = int(prof.get("n_rows", 0))
        panel = [tuple(p) for p in prof.get("panel_candidates", [])]
        time_like = {c["name"] for c in cols if c.get("is_time_like")}

        if panel:
            shape = "panel"
        elif n_rows < _AGG_MAX_ROWS and any(_COUNT_NAME.match(c["name"]) for c in cols):
            shape = "aggregated-cells"
        elif any(c.get("is_time_like") and c["n_unique"] == n_rows for c in cols):
            shape = "time-series"
        else:
            shape = "cross-section"

        quirks: list[str] = []
        for c in cols:
            if c["n_unique"] <= 1:
                quirks.append(f"{c['name']}: constant column")
            if c["missing_frac"] > _QUIRK_MISSING:
                quirks.append(f"{c['name']}: {round(100 * c['missing_frac'])}% missing")

        return {
            "shape": shape,
            "unit_of_observation": panel[0][0] if panel else "row",
            "treatments": [
                {"column": t, "reason": "binary 0/1 column"}
                for t in prof.get("treatment_candidates", [])
            ],
            "outcomes": [
                {"column": c["name"], "reason": "numeric, non-binary"}
                for c in cols
                if c.get("is_numeric") and not c.get("is_binary") and not c.get("is_time_like")
            ],
            "forcing": [
                {"column": f, "reason": "numeric with >= 20 distinct values"}
                for f in prof.get("forcing_candidates", [])
                if f not in time_like
            ],
            "did_structures": [
                {"unit": u, "time": t, "reason": "unit x time grid covers >= 95% of rows"}
                for (u, t) in panel
            ],
            "quirks": quirks,
            "notes": "NullBackend heuristics (no LLM)",
        }

    @staticmethod
    def _prefix_filters(prof: dict, high: list[dict]) -> tuple[list[dict], set[str]]:
        """Shared structural-missingness filters (issue #6), profile-only.

        Among the high-missingness columns ``high``, cluster those whose
        missingness is >= ``_PREFIX_EXPLAINED`` prefix-explained and whose
        boundary rows agree within ``_BOUNDARY_TOL * n_rows``. A cluster of
        >= 2 columns is rescued when a monotone column has a recorded value at
        the cluster boundary and every member's post-boundary missing_frac
        falls below ``_QUIRK_MISSING``: one ``>=`` filter per cluster.
        Returns ``(filters, rescued column names)``.
        """
        n_rows = int(prof.get("n_rows", 0))
        # profile dicts arrive JSON-round-tripped: boundary keys are strings
        boundary_values = {int(b): v for b, v in (prof.get("boundary_values") or {}).items()}
        if not n_rows or not boundary_values:
            return [], set()
        structural = sorted(
            (
                (int(c["first_valid_index"]), c)
                for c in high
                if c.get("first_valid_index")
                and c.get("prefix_missing_frac") is not None
                and c["prefix_missing_frac"] >= _PREFIX_EXPLAINED
            ),
            key=lambda item: (item[0], item[1]["name"]),
        )
        # monotone columns that can express a boundary; time-like ones first
        # (stable sort keeps profile order within each tier — deterministic)
        monotone = [
            c["name"]
            for c in sorted(
                (c for c in prof.get("columns", []) if c.get("is_monotone") and c["n_unique"] > 1),
                key=lambda c: not c.get("is_time_like"),
            )
        ]
        clusters: list[list[tuple[int, dict]]] = []
        for row, col in structural:
            if clusters and row - clusters[-1][0][0] <= _BOUNDARY_TOL * n_rows:
                clusters[-1].append((row, col))
            else:
                clusters.append([(row, col)])
        filters: list[dict] = []
        rescued: set[str] = set()
        for cluster in clusters:
            if len(cluster) < 2:
                continue  # a shared structural boundary needs >= 2 witnesses
            boundary = max(row for row, _ in cluster)
            values = boundary_values.get(boundary) or {}
            expressed_by = next((m for m in monotone if m in values), None)
            tail = n_rows - boundary
            if expressed_by is None or tail <= 0:
                continue
            residual = [
                c["missing_frac"] * n_rows * (1.0 - c["prefix_missing_frac"]) / tail
                for _, c in cluster
            ]
            if max(residual) >= _QUIRK_MISSING:
                continue
            filters.append({"col": expressed_by, "op": ">=", "value": values[expressed_by]})
            rescued |= {c["name"] for _, c in cluster}
        return filters, rescued

    @staticmethod
    def _prepare(payload: dict) -> dict:
        prof = payload["profile"]
        cols = prof.get("columns", [])
        drop = [c["name"] for c in cols if c["n_unique"] <= 1]
        high = [c for c in cols if c["missing_frac"] > _DROP_MISSING and c["name"] not in drop]
        # Issue #6: structurally prefix-missing column clusters are kept behind
        # one shared boundary filter instead of being dropped wholesale.
        filters, rescued = NullBackend._prefix_filters(prof, high)
        drop += [c["name"] for c in high if c["name"] not in rescued]
        subsample = None
        if int(prof.get("n_rows", 0)) > _SUB_MAX:
            subsample = {"n": _SUB_N, "seed": int(payload.get("seed", 0))}
        return {
            "version": 1,
            "column_roles": {},
            "encodings": {},
            "discretize": {},
            "drop_cols": drop,
            "subsample": subsample,
            "filters": filters,
        }

    @staticmethod
    def _search_plan(payload: dict) -> dict:
        prof = payload["profile"]
        und = payload.get("understanding") or {}
        # The profile is POST-prep while the understanding guesses are PRE-prep:
        # guesses naming dropped columns would only produce candidates that
        # study()'s column validation drops again (dogfood finding), so keep a
        # guess only when its column still exists in the profile.
        known = {c["name"] for c in prof.get("columns", [])}
        treatments = [g["column"] for g in und.get("treatments", []) if g["column"] in known]
        outcome_guesses = [g["column"] for g in und.get("outcomes", []) if g["column"] in known]
        forcing_guesses = [g["column"] for g in und.get("forcing", []) if g["column"] in known]
        structures = [
            s for s in und.get("did_structures", [])
            if s["unit"] in known and s["time"] in known
        ]

        def first_outcome(t: str) -> str | None:
            return next((o for o in outcome_guesses if o != t), None)

        candidates: list[dict] = []
        for t in treatments:
            outcome = first_outcome(t)
            forcing = [f for f in forcing_guesses if f != t and f != outcome]
            if not forcing:
                continue  # rdd needs a forcing variable; nothing left after t/outcome
            candidates.append(
                {
                    "design": "rdd",
                    "treatment": t,
                    "outcome": outcome,
                    "forcing": forcing,
                    "rationale": (
                        f"NullBackend: binary treatment '{t}' with numeric forcing "
                        f"candidates {forcing}"
                    ),
                    "priority": len(candidates),
                }
            )
        for t in treatments:
            outcome = first_outcome(t)
            for s in structures:
                candidates.append(
                    {
                        "design": "did",
                        "treatment": t,
                        "outcome": outcome,
                        "unit": s["unit"],
                        "time": s["time"],
                        "rationale": (
                            f"NullBackend: binary treatment '{t}' on panel "
                            f"({s['unit']} x {s['time']})"
                        ),
                        "priority": len(candidates),
                    }
                )
        return {
            "candidates": candidates,
            "budget": {
                "k": _PLAN_K,
                "q": _PLAN_Q,
                "coarse": int(prof.get("n_rows", 0)) > _COARSE_MIN,
                "n_coarse": _PLAN_N_COARSE,
            },
        }

    @staticmethod
    def _interpret_discovery(payload: dict) -> dict:
        summary = payload.get("summary") or {}
        candidate = payload.get("candidate") or {}
        design = summary.get("design") or candidate.get("design") or "unknown"
        influence = summary.get("forcing_influence") or {}
        if influence:
            # max |influence|; ties broken by column name (sorted keys) for determinism
            dominant = max(sorted(influence), key=lambda c: abs(float(influence[c])))
            dominant_txt = (
                f"dominant forcing column '{dominant}' "
                f"(|influence|={abs(float(influence[dominant])):.3g})"
            )
        else:
            dominant_txt = "no forcing influence reported"
        location_parts = [
            f"{key}={json.dumps(summary[key], sort_keys=True, default=str)}"
            for key in ("center_z", "t0", "window", "subset_values")
            if summary.get(key) is not None
        ]
        location_txt = ", ".join(location_parts) if location_parts else "unspecified"
        return {
            "summary": f"{design} discovery: {dominant_txt}; location {location_txt}",
            "matched_policies": [],
            "confounded_risk": "unknown",
            "note": "NullBackend: no domain knowledge applied",
        }

    @staticmethod
    def _audit_assumptions(payload: dict) -> dict:
        return {
            "excludability": "unreviewed",
            "monotonicity": "unreviewed",
            "sutva": "unreviewed",
            "veto": False,
            "caveats": ["NullBackend: assumption audit requires a human or LLM reviewer"],
        }

    @staticmethod
    def _method_applicability(payload: dict) -> dict:
        """Echo the heuristic verdicts: NullBackend never overrides, never hints."""
        return {
            "families": [
                {
                    "family": name,
                    "run": h["status"] == "applicable",
                    "reason": h["reason"],
                    "config_hints": {
                        "cutoffs": [],
                        "instruments": [],
                        "thresholds": [],
                        "treated_unit": None,
                        "t0": None,
                    },
                }
                for name, h in payload["heuristics"].items()
            ]
        }

    @staticmethod
    def _review_control_group(payload: dict) -> dict:
        expansions = payload.get("expansions") or []
        return {
            "face_valid": None,
            "veto": False,  # NullBackend NEVER vetoes
            "reason": (
                "NullBackend performs no substantive review; "
                f"n_expansions={len(expansions)}"
            ),
        }
