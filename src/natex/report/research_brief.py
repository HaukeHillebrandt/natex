"""Deep-research handoff: ``research-brief.md`` (spec section 7, task 8).

The brief is a ready-to-paste query for the user's deep-research skill
(Gemini Interactions API) — natex itself performs NO research calls. It is
PURE deterministic text built from ``bundle.results`` alone: no network, no
rng, no LLM, no timestamps, so reruns are byte-identical. Every number goes
through :func:`natex.report.paper._fmt` (None/NaN -> em dash) — the strings
"nan" and "None" never reach the page.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING

from natex.report.paper import (
    _EM,
    _candidate_label,
    _data_context,
    _effects_rows,
    _fmt,
    _validation_block,
)

if TYPE_CHECKING:
    from natex.report.bundle import ResultsBundle

BRIEF_FILENAME = "research-brief.md"

_NO_EFFECTS_NO_OUTCOME = "no effect estimates (no outcome column was provided)"
_NO_EFFECTS = "no effect estimates were recorded for the scanned configurations"

_NOTE = (
    "*AI-generated research brief — verify every claim and citation before "
    "circulation. Built by natex from results.json only; natex performs no "
    "research calls itself.*"
)

_OPEN_QUESTIONS = (
    "Confounded cutoffs: does the discovered threshold coincide with other "
    "rules that also change treatment or outcomes at the same point?",
    "Competing policies: do other programs or policies switch at the same "
    "threshold (or at the same date), contaminating the contrast?",
    "External validity: the estimate is local to units near the threshold "
    "(or to the treated subset) — how far do the findings generalize?",
    "Selection into the forcing variable: can units manipulate or sort on "
    "the forcing variable (or anticipate the policy date)? The density test "
    "above is only a partial check.",
)

_GENERIC_QUESTIONS = (
    "What institutional or administrative rules govern how the treatment was "
    "assigned in this setting?",
    "What natural experiments have been documented in comparable datasets or "
    "settings?",
    "What known confounders should any quasi-experimental analysis of this "
    "kind of data address?",
)

_HOW_TO_USE = (
    "Paste this brief verbatim as the query to your deep-research skill (it "
    "runs on the Gemini Interactions API as a background process); natex "
    "does not call any research API itself. Ask the research agent for a "
    "source-linked literature review that answers the numbered questions "
    "above and flags any prior study of the same threshold, subset, or "
    "policy date. Verify every returned claim and citation by hand before "
    "it informs the paper draft — the draft carries an AI-generated banner "
    "for the same reason."
)


def _scanned_configs(results: dict) -> list[dict]:
    return [
        cfg for cfg in (results.get("configs") or [])
        if isinstance(cfg, dict) and cfg.get("status") == "scanned"
    ]


def _center_str(cand: dict, summ: dict) -> str:
    """Cutoff center in raw forcing coordinates (``center_z`` is ``ds.Z`` at
    the discovered center — the unstandardized forcing matrix)."""
    names = list(cand.get("forcing") or [])
    center = summ.get("center_z") or []
    if names and len(names) == len(center):
        pairs = zip(names, center, strict=True)
        return ", ".join(f"{n} = {_fmt(v)}" for n, v in pairs)
    return ", ".join(_fmt(v) for v in center) or _EM


def _top_influence(summ: dict) -> str:
    """The forcing variable with the largest |normal| component."""
    infl = summ.get("forcing_influence")
    if not isinstance(infl, dict):
        return _EM
    finite = {
        str(k): float(v) for k, v in infl.items()
        if isinstance(v, (int, float)) and math.isfinite(float(v))
    }
    if not finite:
        return _EM
    top = max(finite, key=finite.get)  # ties: first in insertion order
    return f"{top} (|normal component| = {_fmt(finite[top])})"


def _subset_str(summ: dict) -> str:
    sv = summ.get("subset_values")
    if not isinstance(sv, dict) or not sv:
        return "all units"
    return "; ".join(
        f"{dim} in {{{', '.join(_fmt(v) for v in (vals or []))}}}"
        for dim, vals in sv.items()
    )


def _context_lines(results: dict) -> list[str]:
    lines: list[str] = []
    data = _data_context(results.get("data"))
    if data is None:
        lines.append("- (no data description was recorded in this bundle)")
    else:
        lines += [
            f"- Rows: {data['n_rows']}; source: {data['source']}",
            f"- Treatment: {data['treatment']}; outcome: {data['outcome']}",
            f"- Forcing variables: {data['forcing']}",
            f"- Time column: {data['time']}; unit column: {data['unit']}; "
            f"covariate columns: {data['n_covariates']}",
        ]
    intake = results.get("intake")
    context = intake.get("context") if isinstance(intake, dict) else None
    if context:
        lines.append(f"- User-provided context: {context}")
    searched = results.get("searched")
    if isinstance(searched, dict):
        s = {k: int(searched.get(k) or 0)
             for k in ("n_total", "n_scanned", "n_skipped_budget",
                       "n_failed", "n_invalid")}
        lines.append(
            f"- Coverage: of {s['n_total']} enumerated configurations, "
            f"{s['n_scanned']} were scanned, {s['n_skipped_budget']} skipped "
            f"by budget, {s['n_failed']} failed, {s['n_invalid']} invalid."
        )
    else:
        lines.append("- Coverage: search coverage was not recorded for this bundle.")
    return lines


def _design_lines(cfg: dict) -> list[str]:
    cand = cfg.get("candidate") or {}
    summ = cfg.get("summary") or {}
    lines = [f"### {_candidate_label(cand)}"]
    if cand.get("design") == "rdd":
        lines += [
            f"- Cutoff center (raw forcing coordinates): {_center_str(cand, summ)}",
            f"- Top forcing influence: {_top_influence(summ)}",
        ]
    else:
        lines += [
            f"- Treated subset: {_subset_str(summ)}",
            f"- Onset T0 = {_fmt(summ.get('t0'))}; window = {_fmt(summ.get('window'))}",
        ]
    lines.append(
        f"- Monte Carlo p = {_fmt(cfg.get('p_value'))} "
        f"(max LLR = {_fmt(cfg.get('llr'))})"
    )
    return lines


def _effects_lines(results: dict, scanned: list[dict]) -> list[str]:
    lines: list[str] = []
    for cfg in scanned:
        cand = cfg.get("candidate") or {}
        rows = _effects_rows(cand.get("design") or "",
                             (cfg.get("summary") or {}).get("effects"))
        if not rows:
            continue
        lines.append(f"For {_candidate_label(cand)}:")
        for row in rows:
            flags = "" if row["flags"] == _EM else f" — {row['flags']}"
            lines.append(
                f"- {row['estimator']}: tau = {row['tau']}, se = {row['se']}, "
                f"95% CI {row['ci']}{flags}"
            )
        lines.append("")
    if lines:
        return lines[:-1]  # drop the trailing spacer
    data = results.get("data") or {}
    if not data.get("outcome"):
        return [_NO_EFFECTS_NO_OUTCOME]
    return [_NO_EFFECTS]


def _validation_lines(scanned: list[dict]) -> list[str]:
    if not scanned:
        return ["No configuration was scanned, so no validation was run."]
    lines: list[str] = []
    for cfg in scanned:
        v = _validation_block(cfg)
        label = _candidate_label(cfg.get("candidate") or {})
        if v["design"] == "rdd":
            lines.append(
                f"- {label}: Monte Carlo p = {v['p_value']}; covariate "
                f"placebo {v['placebo']} (Holm-adjusted p = "
                f"{v['placebo_holm']}); density test p = {v['density_p']}"
            )
        else:
            lines.append(
                f"- {label}: Monte Carlo p = {v['p_value']} (null: "
                f"{v['null_kind']}); composition check {v['composition']}; "
                f"anticipation check {v['anticipation']}"
            )
    return lines


def _literature_questions(results: dict, scanned: list[dict]) -> list[str]:
    data = results.get("data") or {}
    questions: list[str] = []
    for cfg in scanned:
        cand = cfg.get("candidate") or {}
        summ = cfg.get("summary") or {}
        treatment = cand.get("treatment") or "the treatment"
        outcome = cand.get("outcome") or data.get("outcome")
        if cand.get("design") == "rdd":
            forcing = ", ".join(cand.get("forcing") or []) or "the forcing variable"
            center = _center_str(cand, summ)
            questions += [
                f"What statutory or administrative rules set a threshold "
                f"near {center} on {forcing}?",
                f"What quasi-experimental studies exploit thresholds in "
                f"{forcing} in comparable settings?",
                f"What mechanisms could make {treatment} change "
                f"discontinuously at a threshold in {forcing}, and could "
                f"units sort across it?",
            ]
        else:
            time = cand.get("time") or "time"
            t0 = _fmt(summ.get("t0"))
            subset = _subset_str(summ)
            questions += [
                f"What other policies changed at {time} = {t0} that could "
                f"affect {subset}?",
                f"What documented intervention explains why {treatment} "
                f"switched on for {subset} at {time} = {t0}?",
                f"What quasi-experimental studies exploit staggered adoption "
                f"of {treatment} in comparable settings?",
            ]
        if outcome:
            questions.append(
                f"What published effect estimates exist for {treatment} "
                f"on {outcome}?"
            )
    seen: set[str] = set()
    unique = [q for q in questions if not (q in seen or seen.add(q))]
    for generic in _GENERIC_QUESTIONS:  # always hand off >= 3 questions
        if len(unique) >= 3:
            break
        if generic not in unique:
            unique.append(generic)
    return unique


def _brief_text(results: dict) -> str:
    scanned = _scanned_configs(results)
    data = results.get("data") or {}
    intake = results.get("intake") or {}
    source = data.get("source") or intake.get("source") or "dataset"

    lines: list[str] = [f"# Research brief: {source}", "", _NOTE, ""]

    lines += ["## Discovery context", "", *_context_lines(results), ""]

    lines += ["## Discovered designs", ""]
    if scanned:
        for cfg in scanned:
            lines += [*_design_lines(cfg), ""]
    else:
        lines += ["No configuration was scanned in this bundle.", ""]

    lines += ["## Effect estimates", "", *_effects_lines(results, scanned), ""]

    lines += ["## Validation status", "", *_validation_lines(scanned), ""]

    lines += ["## Open questions", "", *(f"- {q}" for q in _OPEN_QUESTIONS), ""]

    lines += ["## Literature questions for deep research", ""]
    lines += [f"{i}. {q}" for i, q in
              enumerate(_literature_questions(results, scanned), start=1)]
    lines.append("")

    lines += ["## How to use", "", _HOW_TO_USE, ""]
    return "\n".join(lines)


def research_brief(bundle: ResultsBundle, out: str | Path) -> Path:
    """Write ``research-brief.md``. ``out`` is a directory unless it ends
    with ``.md``, in which case the brief is written exactly there. Pure
    string generation from ``bundle.results`` — byte-identical on rerun."""
    out = Path(out)
    if str(out).endswith(".md"):
        path = out
        path.parent.mkdir(parents=True, exist_ok=True)
    else:
        out.mkdir(parents=True, exist_ok=True)
        path = out / BRIEF_FILENAME
    path.write_text(_brief_text(bundle.results), encoding="utf-8")
    return path
