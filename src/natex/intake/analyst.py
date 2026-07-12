"""Stage-0 analyst pipeline: ``natex.study()`` + ``IntakeReport`` (spec 6a).

``study()`` profiles the raw data, asks a guidance backend to understand it,
propose a declarative :class:`~natex.intake.prep.PrepPlan` (validated against
the REAL dataframe and executed ONLY by natex code — the LLM never emits
code), and rank design candidates into a
:class:`~natex.intake.plans.SearchPlan`. Everything degrades gracefully: any
backend/parse failure falls back to :class:`~natex.llm.NullBackend`
heuristics, recorded by name in ``guidance_errors`` (or re-raised under
``strict=True``). Guidance is advisory only — it orders the search, it never
touches a statistic.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from pydantic import ValidationError

from natex.data.spec import Dataset, DatasetSpec
from natex.intake.plans import DesignCandidate, SearchPlan, Understanding
from natex.intake.prep import PrepPlan
from natex.intake.profiler import ColumnProfile, IntakeProfile, profile
from natex.jsonutil import jsonable
from natex.llm import (
    GuidanceBackend,
    GuidanceLog,
    GuidanceRequest,
    LoggedBackend,
    NullBackend,
)

# Exceptions that trigger the uniform fallback-to-NullBackend policy (never a
# bare except): backend transport/timeout failures, JSON/schema violations,
# and PrepPlan rejections against the real dataframe.
_FALLBACK_EXCEPTIONS = (ValidationError, ValueError, TimeoutError, RuntimeError)


@dataclass
class IntakeReport:
    """Serializable record of one ``study()`` run (RAW profile, pre-prep)."""

    profile: IntakeProfile
    understanding: Understanding
    prep_plan: PrepPlan
    search_plan: SearchPlan
    guidance_log_path: str | None
    context: str | None
    source: str  # csv path, or "<dataframe>"
    guidance_errors: list[str]  # fallbacks, dropped candidates, snooping warnings
    prep_log: list[str]  # PrepPlan.apply log of the study() run
    _df: pd.DataFrame | None = field(default=None, repr=False, compare=False)  # not serialized

    def to_json(self) -> str:
        payload = {
            "profile": dataclasses.asdict(self.profile),
            "understanding": self.understanding.model_dump(),
            "prep_plan": self.prep_plan.model_dump(),
            "search_plan": self.search_plan.model_dump(),
            "guidance_log_path": self.guidance_log_path,
            "context": self.context,
            "source": self.source,
            "guidance_errors": list(self.guidance_errors),
            "prep_log": list(self.prep_log),
        }
        return json.dumps(jsonable(payload), indent=1)

    def save(self, out: str | Path) -> Path:
        """Write ``out/intake_report.json`` (full report) and ``out/prep_plan.json``
        (the plan alone, editable by the user); return the report path."""
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
        report_path = out / "intake_report.json"
        report_path.write_text(self.to_json(), encoding="utf-8")
        plan_json = json.dumps(jsonable(self.prep_plan.model_dump()), indent=1)
        (out / "prep_plan.json").write_text(plan_json, encoding="utf-8")
        return report_path

    @classmethod
    def load(cls, path: str | Path) -> IntakeReport:
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        p = d["profile"]
        prof = IntakeProfile(
            n_rows=int(p["n_rows"]),
            columns=[ColumnProfile(**c) for c in p["columns"]],
            panel_candidates=[tuple(pair) for pair in p.get("panel_candidates", [])],
            forcing_candidates=list(p.get("forcing_candidates", [])),
            treatment_candidates=list(p.get("treatment_candidates", [])),
        )
        return cls(
            profile=prof,
            understanding=Understanding.model_validate(d["understanding"]),
            prep_plan=PrepPlan.model_validate(d["prep_plan"]),
            search_plan=SearchPlan.model_validate(d["search_plan"]),
            guidance_log_path=d.get("guidance_log_path"),
            context=d.get("context"),
            source=d["source"],
            guidance_errors=list(d.get("guidance_errors", [])),
            prep_log=list(d.get("prep_log", [])),
        )

    def prepare(self, df: pd.DataFrame | None = None, candidate: int = 0) -> Dataset:
        """Re-apply the prep plan and build a :class:`Dataset` for one candidate.

        Frame resolution: ``df`` arg, else the study() frame, else re-read
        ``source`` if it is an existing csv path. Spec mirrors
        ``Dataset.from_csv`` defaults: covariates = all prepared columns minus
        {treatment, outcome} and minus columns the prep plan roles as
        ``ignore``/``time``/``unit`` (unless the candidate itself uses them) —
        role-ignored columns must not leak into the scan's covariate set, the
        placebo battery, or listwise deletion (dogfood finding).
        """
        frame = df if df is not None else self._df
        if frame is None and self.source != "<dataframe>" and Path(self.source).exists():
            frame = pd.read_csv(self.source)
        if frame is None:
            raise ValueError("no dataframe available; pass df=")
        df2, _ = self.prep_plan.apply(frame)
        ranked = self.search_plan.ranked()
        if not 0 <= candidate < len(ranked):
            raise ValueError(
                f"candidate index {candidate} out of range for {len(ranked)} ranked candidates"
            )
        c = ranked[candidate]
        roles = self.prep_plan.column_roles
        used = {c.treatment, c.outcome, c.unit, c.time, *c.forcing}
        covariates = [
            col for col in df2.columns
            if col not in {c.treatment, c.outcome}
            and not (roles.get(col) in ("ignore", "time", "unit") and col not in used)
        ]
        if c.design == "rdd":
            spec = DatasetSpec(
                treatment=c.treatment, outcome=c.outcome,
                forcing=list(c.forcing), covariates=covariates,
            )
        else:  # did
            spec = DatasetSpec(
                treatment=c.treatment, outcome=c.outcome, forcing=[],
                covariates=covariates, time=c.time, unit=c.unit,
            )
        # Dataset's constructor enforces numeric forcing/time etc. — errors
        # propagate: they are real spec bugs, never silently absorbed.
        return Dataset(df2, spec)


def _candidate_columns(c: DesignCandidate) -> list[str]:
    cols = [c.treatment, *c.forcing]
    cols += [x for x in (c.outcome, c.unit, c.time) if x is not None]
    return cols


def study(
    csv_or_df: str | Path | pd.DataFrame,
    context: str | None = None,
    guidance: GuidanceBackend | None = None,
    rng: np.random.Generator | None = None,
    out: str | Path | None = None,
    strict: bool = False,
) -> IntakeReport:
    """Stage-0 analyst pass: profile -> understand -> prepare -> search_plan.

    The task order is FIXED — ``understand``, ``prepare``, ``search_plan`` —
    which is the MockBackend contract (canned responses are consumed in that
    order). Backend/parse failures fall back to NullBackend heuristics and are
    recorded in ``guidance_errors`` (``strict=True`` re-raises as ValueError
    naming the step). When ``out`` is given, every request+response (including
    fallbacks) is appended to ``out/guidance_log.jsonl`` and the report is
    saved to ``out``. The PrepPlan subsample seed is drawn from ``rng`` and
    stored IN the plan, so the serialized plan replays bitwise.
    """
    if rng is None:
        raise ValueError("pass an explicit numpy Generator (reproducibility contract)")
    if isinstance(csv_or_df, (str, Path)):
        source = str(csv_or_df)
        df = pd.read_csv(csv_or_df)
    else:
        source = "<dataframe>"
        df = csv_or_df

    backend: GuidanceBackend = guidance if guidance is not None else NullBackend()
    fallback: GuidanceBackend = NullBackend()
    guidance_log_path: str | None = None
    if out is not None:
        log = GuidanceLog(Path(out) / "guidance_log.jsonl")
        backend = LoggedBackend(backend, log)
        fallback = LoggedBackend(fallback, log)
        guidance_log_path = str(log.path)

    errors: list[str] = []

    def ask(request: GuidanceRequest, validate: Callable):
        """Uniform fallback policy: the failed attempt stays in the log (it was
        appended when the backend answered); the Null fallback is logged too."""
        try:
            return validate(backend.complete(request).content)
        except _FALLBACK_EXCEPTIONS as exc:
            if strict:
                raise ValueError(f"{request.task}: {exc}") from exc
            errors.append(f"{request.task}: {exc} -- fell back to NullBackend heuristics")
            return validate(fallback.complete(request).content)

    prof = profile(df)
    prof_dict = json.loads(prof.to_json())

    understanding: Understanding = ask(
        GuidanceRequest(
            task="understand",
            payload={"profile": prof_dict, "context": context},
            schema_hint=Understanding.model_json_schema(),
        ),
        Understanding.model_validate,
    )

    # One draw per run, whether or not the plan subsamples: rng usage is
    # identical on every path, so identical seed => identical report.
    subsample_seed = int(rng.integers(2**31 - 1))

    def validate_prep(content: dict) -> PrepPlan:
        plan = PrepPlan.model_validate(content)
        plan.validate_against(df)  # spec 6a: validate against the ACTUAL dataframe
        return plan

    prep_plan: PrepPlan = ask(
        GuidanceRequest(
            task="prepare",
            payload={
                "profile": prof_dict,
                "understanding": understanding.model_dump(),
                "seed": subsample_seed,
                "context": context,
            },
            schema_hint=PrepPlan.model_json_schema(),
        ),
        validate_prep,
    )
    if prep_plan.subsample is not None:
        prep_plan.subsample.seed = subsample_seed  # plan-carried seed: bitwise replay

    df2, prep_log = prep_plan.apply(df)
    prof2 = profile(df2)

    search_request = GuidanceRequest(
        task="search_plan",
        payload={
            "profile": json.loads(prof2.to_json()),
            "understanding": understanding.model_dump(),
            "context": context,
        },
        schema_hint=SearchPlan.model_json_schema(),
    )
    search_plan: SearchPlan = ask(search_request, SearchPlan.model_validate)

    known = set(df2.columns)

    def keep_known(plan: SearchPlan, origin: str = "") -> SearchPlan:
        """Drop candidates naming columns absent from the PREPARED frame."""
        surviving: list[DesignCandidate] = []
        for i, c in enumerate(plan.candidates):
            unknown = sorted({col for col in _candidate_columns(c) if col not in known})
            if unknown:
                errors.append(
                    f"search_plan{origin}: dropped candidate {i} ({c.design}, "
                    f"treatment='{c.treatment}'): unknown columns {unknown}"
                )
            else:
                surviving.append(c)
        return SearchPlan(candidates=surviving, budget=plan.budget)

    search_plan = keep_known(search_plan)
    if not search_plan.candidates:
        errors.append(
            "search_plan: no candidates survived column validation "
            "-- fell back to NullBackend heuristics"
        )
        # The fallback plan is column-validated too: without this, a fallback
        # that repeats the same invalid candidates (dogfood finding: with the
        # Null backend the fallback IS the backend that just failed) would put
        # unknown columns straight back into the final plan.
        search_plan = keep_known(
            SearchPlan.model_validate(fallback.complete(search_request).content),
            " (fallback)",
        )

    # Snooping guard (audit 1 lineage): prep steps that touch a surviving
    # candidate's outcome column are flagged — warning only, never a failure.
    outcome_cols = {c.outcome for c in search_plan.candidates if c.outcome is not None}
    for f in prep_plan.filters:
        if f.col in outcome_cols:
            errors.append(
                f"prep filter touches candidate outcome '{f.col}' -- possible outcome snooping"
            )
    for col in prep_plan.drop_cols:
        if col in outcome_cols:
            errors.append(
                f"prep drop touches candidate outcome '{col}' -- possible outcome snooping"
            )

    report = IntakeReport(
        profile=prof,
        understanding=understanding,
        prep_plan=prep_plan,
        search_plan=search_plan,
        guidance_log_path=guidance_log_path,
        context=context,
        source=source,
        guidance_errors=errors,
        prep_log=prep_log,
        _df=df,
    )
    if out is not None:
        report.save(out)
    return report
