"""Survey report renderers (phase survey task 8): report.md always + report.html.

:func:`render_survey_md` is pure Python (no jinja2) and ALWAYS works on a
core-only install; :func:`render_survey_html` imports jinja2 lazily and
raises ImportError naming ``natex-discovery[report]`` when the extra is
missing. BOTH consume the JSON-NATIVE dict (the content of ``survey.json``),
so ``natex survey`` and any re-render from a saved file share one path.

Every number passes through the paper renderer's :func:`~natex.report.paper._fmt`
(em dash for missing/non-finite — the strings "nan" and "None" never reach a
rendered page). The HTML report is SELF-CONTAINED: inline CSS, every figure
PNG read from ``out_dir`` and embedded as a base64 data URI, no external
requests; a missing PNG file degrades to the no-figure reason, never a
broken ``<img>``. Status badges use the Okabe–Ito palette, matching
:mod:`natex.report.figures`.

This module RENDERS numbers the survey already computed — no inference.
"""

from __future__ import annotations

import base64
from pathlib import Path

from natex.report.paper import _fmt
from natex.survey.registry import FAMILIES, FAMILY_ORDER

__all__ = ["SURVEY_BANNER", "render_survey_html", "render_survey_md"]

SURVEY_BANNER = "AI-generated — verify before citing"

_EM = "—"
_REPORT_EXTRA_MSG = (
    'render_survey_html requires the report extra: pip install "natex-discovery[report]"'
)

# status -> (Okabe–Ito color, icon); unknown statuses degrade to gray "?".
_BADGES = {
    "credible": ("#009E73", "✓"),
    "null": ("#999999", "○"),
    "skipped": ("#0072B2", "–"),
    "needs_input": ("#E69F00", "⚠"),
    "failed": ("#D55E00", "✗"),
}
_UNKNOWN_BADGE = ("#666666", "?")


def _value_text(v) -> str:
    """Compact human string for a nested JSON value; scalars via ``_fmt``
    (so None/non-finite become the em dash at every depth)."""
    if isinstance(v, dict):
        inner = ", ".join(f"{k}: {_value_text(x)}" for k, x in v.items())
        return "{" + inner + "}" if inner else _EM
    if isinstance(v, (list, tuple)):
        inner = ", ".join(_value_text(x) for x in v)
        return "[" + inner + "]" if inner else _EM
    return _fmt(v)


def _said(v) -> str:
    """Override verdicts are run/not-run booleans (applicability contract);
    tolerate strings for hand-built dicts."""
    if isinstance(v, bool):
        return "run" if v else "do not run"
    return str(v) if isinstance(v, str) and v else _fmt(v)


def _applicability_lines(app) -> tuple[str, str, str]:
    """(heuristic line, override line, guidance-error line); empties for absent."""
    if not isinstance(app, dict) or not app:
        return _EM, "", ""
    h = app.get("heuristic") or {}
    decision = "family ran" if app.get("run") else "family did not run"
    line = (
        f"heuristic: {h.get('status') or _EM} ({h.get('reason') or _EM}); {decision}"
    )
    override = app.get("override")
    over_line = ""
    if isinstance(override, dict):
        over_line = (
            f"heuristic said {_said(override.get('heuristic_said'))}; "
            f"analyst said {_said(override.get('analyst_said'))} — "
            f"{override.get('reason') or _EM}"
        )
    err = app.get("guidance_error")
    return line, over_line, (str(err) if err else "")


def _figure_entries(fam: dict, out_dir: Path | None) -> list[dict]:
    """Figure map -> render entries. Markdown mode (``out_dir=None``) records
    paths only; html mode embeds base64 or degrades to the no-figure reason."""
    entries: list[dict] = []
    for name, rel in (fam.get("figures") or {}).items():
        entry = {"name": str(name), "path": str(rel), "b64": "", "note": ""}
        if out_dir is not None:
            path = out_dir / str(rel)
            if path.is_file():
                entry["b64"] = base64.b64encode(path.read_bytes()).decode("ascii")
            else:  # missing file: reason instead, never a broken <img>
                entry["note"] = (
                    fam.get("no_figure_reason")
                    or f"no figure: file not found ({rel})"
                )
        entries.append(entry)
    return entries


def _family_context(name: str, fam: dict, out_dir: Path | None) -> dict:
    reg = FAMILIES.get(name)
    status = str(fam.get("status") or "unknown")
    color, icon = _BADGES.get(status, _UNKNOWN_BADGE)
    app_line, over_line, guidance_err = _applicability_lines(fam.get("applicability"))
    diagnostics = fam.get("diagnostics") or {}
    caveats = [str(c) for c in (diagnostics.get("caveats") or [])]
    if not caveats and reg is not None:
        caveats = [reg.caveat]  # skipped families still show the registry caveat
    return {
        "name": name,
        "title": reg.title if reg is not None else name,
        "description": reg.description if reg is not None else "",
        "status": status,
        "status_label": status.replace("_", " "),
        "icon": icon,
        "color": color,
        "reason": str(fam.get("reason") or _EM),
        "applicability_line": app_line,
        "override_line": over_line,
        "guidance_error": guidance_err,
        "key_numbers": [(str(k), _fmt(v)) for k, v in (fam.get("key_numbers") or {}).items()],
        "figures": _figure_entries(fam, out_dir),
        "no_figure_reason": str(fam.get("no_figure_reason") or ""),
        "diagnostics": [
            (str(k), _value_text(v))
            for k, v in diagnostics.items()
            if k not in ("caveats", "traceback")  # caveats shown apart; tracebacks stay in json
        ],
        "caveats": caveats,
        "error": str(fam.get("error") or ""),
    }


def _survey_context(result: dict, out_dir: Path | None) -> dict:
    """Shared pre-formatted context (pure dict function, no jinja2). Family
    order: FAMILY_ORDER first, then any unknown extras in input order."""
    dataset = result.get("dataset") or {}
    source = str(dataset.get("source") or _EM)
    tr = dataset.get("time_range")
    if isinstance(tr, (list, tuple)) and len(tr) == 2:
        time_range = f"{_fmt(tr[0])} – {_fmt(tr[1])}"
    else:
        time_range = _EM
    fams_in = result.get("families") or {}
    names = [n for n in FAMILY_ORDER if n in fams_in]
    names += [n for n in fams_in if n not in FAMILY_ORDER]
    return {
        "banner": SURVEY_BANNER,
        "title": f"Systematic design survey — {source}",
        "source": source,
        "shape": f"{_fmt(dataset.get('n_rows'))} rows × {_fmt(dataset.get('n_cols'))} columns",
        "time_column": str(dataset.get("time_column") or _EM),
        "time_range": time_range,
        "version": str(result.get("natex_version") or _EM),
        "seed": _fmt(result.get("seed")),
        "created": str(result.get("created") or _EM),
        "context": str(result.get("context") or ""),
        "families": [_family_context(n, fams_in[n] or {}, out_dir) for n in names],
        "notes": [str(x) for x in (result.get("coverage") or {}).get("notes") or []],
    }


def _cell(s: str) -> str:
    """Markdown table cell hygiene: no pipes or newlines inside a cell."""
    return str(s).replace("|", "\\|").replace("\n", " ")


def render_survey_md(result: dict, out_dir: str | Path) -> Path:
    """Write ``out_dir/report.md`` from the JSON-native survey dict.

    Pure Python — this is the fallback report that ALWAYS works, so it must
    not import jinja2 (or anything optional). Figures are referenced by
    their out_dir-relative paths, not embedded.
    """
    out_dir = Path(out_dir)
    ctx = _survey_context(result, out_dir=None)
    lines: list[str] = [
        f"# {ctx['title']}",
        "",
        f"> **{ctx['banner']}**",
        "",
        f"- dataset: {ctx['source']} ({ctx['shape']})",
        f"- time column: {ctx['time_column']} (range {ctx['time_range']})",
        f"- natex-discovery {ctx['version']}, seed {ctx['seed']}, created {ctx['created']}",
    ]
    if ctx["context"]:
        lines.append(f"- context: {ctx['context']}")
    for note in ctx["notes"]:
        lines.append(f"- note: {note}")
    lines += ["", "## Verdicts", "", "| family | status | reason |", "|---|---|---|"]
    for fam in ctx["families"]:
        lines.append(
            f"| {fam['name']} | {fam['icon']} {_cell(fam['status_label'])} "
            f"| {_cell(fam['reason'])} |"
        )
    for fam in ctx["families"]:
        lines += ["", f"## {fam['title']}", ""]
        if fam["description"]:
            lines += [fam["description"], ""]
        lines.append(f"**Verdict ({fam['status_label']}):** {fam['reason']}")
        lines += ["", f"**Applicability:** {fam['applicability_line']}"]
        if fam["override_line"]:
            lines += ["", f"**Override:** {fam['override_line']}"]
        if fam["guidance_error"]:
            lines += ["", f"Guidance note: {fam['guidance_error']}"]
        if fam["key_numbers"]:
            lines += ["", "| quantity | value |", "|---|---|"]
            for k, v in fam["key_numbers"]:
                lines.append(f"| {_cell(k)} | {_cell(v)} |")
        if fam["figures"]:
            lines += ["", "Figures (paths relative to this report's directory):", ""]
            for fig in fam["figures"]:
                lines.append(f"- {fig['name']}: {fig['path']}")
        if fam["no_figure_reason"]:
            lines += ["", fam["no_figure_reason"]]
        if fam["diagnostics"]:
            lines += ["", "Diagnostics:", ""]
            for k, v in fam["diagnostics"]:
                lines.append(f"- {k}: {v}")
        if fam["error"]:
            lines += ["", f"Error: {fam['error']}"]
        for caveat in fam["caveats"]:
            lines += ["", f"*Caveat: {caveat}*"]
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _env():
    """Jinja2 environment (lazy import; StrictUndefined so context/template
    drift fails loudly; autoescape ON — free-text reasons may carry <>)."""
    try:
        import jinja2
    except ImportError as exc:
        raise ImportError(_REPORT_EXTRA_MSG) from exc
    return jinja2.Environment(
        loader=jinja2.FileSystemLoader(Path(__file__).parent / "templates"),
        undefined=jinja2.StrictUndefined,
        autoescape=True,
        trim_blocks=True,
        lstrip_blocks=True,
    )


def render_survey_html(result: dict, out_dir: str | Path) -> Path:
    """Write the self-contained ``out_dir/report.html`` from the JSON-native
    survey dict; figures are read from ``out_dir`` and embedded as base64.
    Raises ImportError naming ``natex-discovery[report]`` without jinja2."""
    out_dir = Path(out_dir)
    env = _env()
    text = env.get_template("survey.html.j2").render(**_survey_context(result, out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "report.html"
    path.write_text(text, encoding="utf-8")
    return path
