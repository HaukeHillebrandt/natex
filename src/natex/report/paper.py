"""Paper renderer under the ``report`` extra (spec section 7).

Everything here RENDERS numbers already computed by discover/validate/
estimate — no new inference. jinja2 is imported lazily (:func:`_env`), so
importing this module works on a core-only install; calling
:func:`render_paper` without the extra raises ImportError naming
``natex-discovery[report]``.

The context builder :func:`_paper_context` is a pure dict function (fully
testable without jinja2). Every number passes through :func:`_fmt`: finite
values become fixed-significant-digit strings, missing values become an em
dash — the strings "nan" and "None" never reach a rendered page.
"""

from __future__ import annotations

import math
import re
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from natex.report.bundle import ResultsBundle

BANNER = "AI-generated draft — verify all claims before circulation"

_CARDS = {"rdd": "lord3.md", "did": "suddds.md"}  # design -> method card file
_EM = "—"
_FORMATS = ("md", "latex")
_REPORT_EXTRA_MSG = (
    'render_paper requires the report extra: pip install "natex-discovery[report]"'
)

_REFERENCES = (
    "Herlands, W., McFowland III, E., Wilson, A. G., & Neill, D. B. (2018). "
    "Automated Local Regression Discontinuity Design Discovery. In Proceedings "
    "of the 24th ACM SIGKDD International Conference on Knowledge Discovery & "
    "Data Mining (KDD '18), 1512-1520.",
    "Herlands, W. (2019). Change modeling for understanding our world and the "
    "counterfactual one(s). PhD thesis, Carnegie Mellon University.",
    "Jakubowski, B., Somanchi, S., McFowland III, E., & Neill, D. B. (2023). "
    "Exploiting Discovered Regression Discontinuities to Debias "
    "Conditioned-on-observable Estimators. Journal of Machine Learning Research.",
)

_CORRECTIONS_NOTE = (
    "Corrections vs the printed papers: natex deviates from the published "
    "formulas wherever the two-model math audit found errors — "
    "`docs/math_audit_final.md` is the authoritative document (it wins every "
    "conflict) and the README section \"Corrections vs the papers\" lists the "
    "headline repairs. In particular, every p-value reported here is a "
    "+1-rank Monte Carlo estimate calibrated against a fitted null (a "
    "parametric bootstrap), never an exact randomization p-value (audit "
    "item 1)."
)


@dataclass(frozen=True)
class PaperResult:
    markdown: Path | None
    tex: Path | None
    pdf: Path | None
    compiled: bool
    message: str


_TABLE_OMITTED = "(table omitted in LaTeX rendering --- see the markdown card)"
_TECTONIC_MISSING = (
    "tectonic not found — wrote paper.tex; install tectonic "
    "(https://tectonic-typesetting.github.io) to compile, or use --format md"
)

_CODE_RE = re.compile(r"`([^`]*)`")
_LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)\s]+)\)")
_BOLD_RE = re.compile(r"\*\*(.+?)\*\*")
_EMPH_RE = re.compile(r"(?<!\*)\*([^*\n]+)\*(?!\*)")
_HEADING_RE = re.compile(r"^(#{1,4})\s+(.*)$")
_ITEM_RE = re.compile(r"^[-*]\s+(.*)$")
_HEADINGS = {
    1: r"\section*",
    2: r"\subsection*",
    3: r"\subsubsection*",
    4: r"\paragraph",
}


def texesc(s) -> str:
    """Escape LaTeX specials in ORDER (backslash first): ``\\ { } & % $ # _ ~ ^``.

    Em/en dashes are additionally normalized to the LaTeX en-dash ligature
    ``--`` so :func:`_fmt`'s missing-value em dash renders as an en dash in
    tables (plan contract) and prose stays inside Latin Modern's glyph set.
    NOT idempotent (escaping twice escapes the introduced backslashes) —
    apply exactly once, at render time. No character is ever dropped.
    """
    s = str(s)
    s = s.replace("\\", "\x00")  # sentinel: keep introduced braces unescaped
    s = s.replace("{", r"\{").replace("}", r"\}")
    s = s.replace("\x00", r"\textbackslash{}")
    for ch, rep in (
        ("&", r"\&"),
        ("%", r"\%"),
        ("$", r"\$"),
        ("#", r"\#"),
        ("_", r"\_"),
        ("~", r"\textasciitilde{}"),
        ("^", r"\textasciicircum{}"),
    ):
        s = s.replace(ch, rep)
    return s.replace("—", "--").replace("–", "--")


def _tex_emph(text: str) -> str:
    parts = _EMPH_RE.split(text)
    return "".join(
        texesc(p) if i % 2 == 0 else r"\emph{" + texesc(p) + "}"
        for i, p in enumerate(parts)
    )


def _tex_bold(text: str) -> str:
    parts = _BOLD_RE.split(text)
    return "".join(
        _tex_emph(p) if i % 2 == 0 else r"\textbf{" + _tex_emph(p) + "}"
        for i, p in enumerate(parts)
    )


def _tex_prose(text: str) -> str:
    """Non-code fragment: [t](u) -> t\\footnote{\\texttt{u}}, then bold/emph."""
    pieces = _LINK_RE.split(text)  # [text, label, url, text, label, url, ...]
    out: list[str] = []
    i = 0
    while i < len(pieces):
        if i % 3 == 0:
            out.append(_tex_bold(pieces[i]))
            i += 1
        else:
            label, url = pieces[i], pieces[i + 1]
            out.append(_tex_bold(label) + r"\footnote{\texttt{" + texesc(url) + "}}")
            i += 2
    return "".join(out)


def _tex_inline(line: str) -> str:
    """One markdown line -> LaTeX: code spans first, prose rules elsewhere."""
    parts = _CODE_RE.split(line)
    return "".join(
        _tex_prose(p) if i % 2 == 0 else r"\texttt{" + texesc(p) + "}"
        for i, p in enumerate(parts)
    )


def _md_to_tex(md: str) -> str:
    """Minimal, bounded markdown -> LaTeX for method-card bodies ONLY.

    LOSSY BY DESIGN: #/##/###/#### become starred sections (\\paragraph for
    ####); ``code`` -> \\texttt; **b** -> \\textbf; *i* -> \\emph; ``-``/``*``
    bullets -> itemize (blank/indented lines inside a list stay inside it as
    item continuations); [t](u) -> t\\footnote{\\texttt{u}}; markdown tables
    are replaced by one omission marker pointing at the markdown card; every
    other line is texesc'd verbatim. Numbered lists, nesting, and block
    quotes degrade to escaped plain text.
    """
    out: list[str] = []
    in_list = False
    in_table = False
    for line in md.splitlines():
        stripped = line.strip()
        if stripped.startswith("|"):
            if not in_table:
                if in_list:
                    out.append(r"\end{itemize}")
                    in_list = False
                out.append(_TABLE_OMITTED)
                in_table = True
            continue
        in_table = False
        heading = _HEADING_RE.match(stripped)
        item = _ITEM_RE.match(stripped)
        if in_list and not item:
            # blank or indented lines continue the current item; anything
            # else (heading, new paragraph at column 0) closes the list
            if stripped and not line.startswith(" "):
                out.append(r"\end{itemize}")
                in_list = False
        if heading is not None:
            if in_list:
                out.append(r"\end{itemize}")
                in_list = False
            cmd = _HEADINGS[len(heading.group(1))]
            out.append(cmd + "{" + _tex_inline(heading.group(2)) + "}")
        elif item is not None:
            if not in_list:
                out.append(r"\begin{itemize}")
                in_list = True
            out.append(r"\item " + _tex_inline(item.group(1)))
        else:
            out.append(_tex_inline(line))
    if in_list:
        out.append(r"\end{itemize}")
    return "\n".join(out)


def _compile_tex(tex: Path, timeout: int = 300) -> tuple[Path | None, bool, str]:
    """Compile ``tex`` with tectonic when present; NEVER raises for a missing
    or failed compiler. Returns ``(pdf_or_None, compiled, message)``."""
    exe = shutil.which("tectonic")
    if exe is None:
        return None, False, _TECTONIC_MISSING
    try:
        proc = subprocess.run(
            [exe, tex.name], cwd=tex.parent,
            capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return None, False, f"tectonic timed out after {timeout}s compiling {tex.name}"
    if proc.returncode != 0:
        return None, False, (proc.stdout + proc.stderr)[-2000:]
    pdf = tex.with_suffix(".pdf")
    if not pdf.is_file():
        return None, False, "tectonic reported success but wrote no PDF"
    return pdf, True, "compiled"


def _fmt(x, nd: int = 3) -> str:
    """Render-safe value: finite numbers to ``nd`` significant digits, bools
    to yes/no, missing/non-finite to an em dash — never "nan"/"None"."""
    if x is None:
        return _EM
    if isinstance(x, bool):
        return "yes" if x else "no"
    if isinstance(x, int):
        return str(x)
    try:
        v = float(x)
    except (TypeError, ValueError):
        s = str(x)
        return s if s else _EM
    if not math.isfinite(v):
        return _EM
    return f"{v:.{nd}g}"


def _cards_dir(explicit: str | Path | None) -> Path | None:
    """Method-card directory: explicit arg, else the repo checkout relative to
    this file (src layout, works for editable installs), else cwd, else None
    (installed wheel without a checkout -> placeholder text)."""
    if explicit is not None:
        return Path(explicit)
    repo = Path(__file__).resolve().parents[3] / "docs" / "method_cards"
    if repo.is_dir():
        return repo
    cwd = Path.cwd() / "docs" / "method_cards"
    if cwd.is_dir():
        return cwd
    return None


def _candidate_label(cand: dict) -> str:
    """Human-readable one-line label for a candidate dict (coverage lists).

    Missing fields render as an em dash — a plain single-scan bundle (F-D1)
    does not record the treatment column name, and "None" must never reach
    the page.
    """
    treatment = cand.get("treatment") or _EM
    if cand.get("design") == "rdd":
        forcing = ", ".join(cand.get("forcing") or []) or _EM
        return f"rdd: {treatment} ~ {forcing}"
    label = f"did: {treatment} over {cand.get('time') or _EM}"
    unit = cand.get("unit")
    return label if unit is None else f"{label} by {unit}"


def _passfail(x) -> str:
    if x is None:
        return _EM
    return "passed" if x else "failed"


def _data_context(data: dict | None) -> dict | None:
    if not isinstance(data, dict):
        return None
    covariates = data.get("covariates")
    return {
        "n_rows": _fmt(data.get("n_rows")),
        "treatment": data.get("treatment") or _EM,
        "outcome": data.get("outcome") or _EM,
        "forcing": ", ".join(data.get("forcing") or []) or _EM,
        "time": data.get("time") or _EM,
        "unit": data.get("unit") or _EM,
        "n_covariates": _fmt(len(covariates)) if isinstance(covariates, list) else _EM,
        "source": data.get("source") or _EM,
    }


def _intake_context(intake: dict | None) -> dict | None:
    if not isinstance(intake, dict):
        return None
    u = intake.get("understanding")
    if not isinstance(u, dict):
        return None
    return {
        "shape": u.get("shape") or _EM,
        "unit_of_observation": u.get("unit_of_observation") or _EM,
        "quirks": [str(q) for q in u.get("quirks") or []],
        "notes": str(u.get("notes") or ""),
    }


def _method_cards(designs: list[str], cards_dir: str | Path | None) -> list[dict]:
    """One card per design: repo card body verbatim, or the placeholder."""
    cdir = _cards_dir(cards_dir)
    cards: list[dict] = []
    for design in designs:
        fname = _CARDS.get(design)
        body = None
        title = f"Method card ({design})"
        if fname is not None and cdir is not None:
            path = cdir / fname
            if path.is_file():
                body = path.read_text(encoding="utf-8").strip()
                for line in body.splitlines():
                    if line.startswith("#"):
                        title = line.lstrip("#").strip()
                        break
        if body is None:
            body = (
                f"(method card `{fname}` not available in this installation — "
                "see the natex repository's docs/method_cards/)"
            )
        cards.append({"design": design, "title": title, "body": body})
    return cards


def _effects_rows(design: str, effects) -> list[dict]:
    """Effects table rows: estimator | tau | SE | 95% CI | flags (all strings).

    rdd flags carry the weak-IV first-stage diagnostic (audit 3/10); did rows
    compute the CI as tau +/- 1.96 se and carry the placebo p in the flags
    (the SE footnote lives in the template)."""
    rows: list[dict] = []
    if not isinstance(effects, dict):
        return rows
    if design == "rdd":
        for name in ("2sls", "wald"):
            block = effects.get(name)
            if not isinstance(block, dict):
                continue
            ci = block.get("ci") or [None, None]
            if block.get("weak_instrument"):
                flags = f"weak IV (first-stage t = {_fmt(block.get('first_stage_t'))})"
            else:
                flags = _EM
            rows.append({
                "estimator": name,
                "tau": _fmt(block.get("tau")),
                "se": _fmt(block.get("se")),
                "ci": f"[{_fmt(ci[0])}, {_fmt(ci[1])}]",
                "flags": flags,
            })
        return rows
    for name in ("dd", "synthetic", "gess"):
        block = effects.get(name)
        if not isinstance(block, dict):
            continue
        tau, se = block.get("tau"), block.get("se")
        lo = hi = None
        if (isinstance(tau, (int, float)) and isinstance(se, (int, float))
                and math.isfinite(float(tau)) and math.isfinite(float(se))):
            lo, hi = tau - 1.96 * se, tau + 1.96 * se
        flags = [f"placebo p = {_fmt(block.get('p'))}"]
        if block.get("vetoed_by_guidance"):
            flags.append("guidance veto (advisory)")
        rows.append({
            "estimator": name,
            "tau": _fmt(tau),
            "se": _fmt(se),
            "ci": f"[{_fmt(lo)}, {_fmt(hi)}]",
            "flags": "; ".join(flags),
        })
    return rows


def _best_block(cfg: dict) -> dict:
    """Best-config detail: rdd cutoff center/influence/density, did subset/T0."""
    cand = cfg.get("candidate") or {}
    summ = cfg.get("summary") or {}
    design = cand.get("design") or _EM
    best = {
        "design": design,
        "label": _candidate_label(cand),
        "llr": _fmt(cfg.get("llr")),
        "p_value": _fmt(cfg.get("p_value")),
    }
    if design == "rdd":
        names = list(cand.get("forcing") or [])
        center = summ.get("center_z") or []
        if names and len(names) == len(center):
            pairs = zip(names, center, strict=True)
            best["center"] = ", ".join(f"{n} = {_fmt(v)}" for n, v in pairs)
        else:
            best["center"] = ", ".join(_fmt(v) for v in center) or _EM
        infl = summ.get("forcing_influence") or {}
        best["forcing_influence"] = (
            ", ".join(f"{k} = {_fmt(v)}" for k, v in infl.items()) or _EM
        )
        best["density_p"] = _fmt(summ.get("density_p"))
    else:
        sv = summ.get("subset_values") or {}
        best["subset"] = (
            "; ".join(
                f"{dim} in {{{', '.join(str(v) for v in vals)}}}"
                for dim, vals in sv.items()
            )
            or "all units"
        )
        best["t0"] = _fmt(summ.get("t0"))
        best["window"] = _fmt(summ.get("window"))
    return best


def _validation_block(cfg: dict) -> dict:
    """Robustness narrative inputs for the best config, all pre-formatted."""
    cand = cfg.get("candidate") or {}
    summ = cfg.get("summary") or {}
    design = cand.get("design") or _EM
    v = {"design": design, "p_value": _fmt(cfg.get("p_value"))}
    if design == "rdd":
        v["placebo"] = _passfail(summ.get("placebo_passed"))
        holm = summ.get("placebo_holm")
        if isinstance(holm, dict):
            finite = {k: p for k, p in holm.items()
                      if isinstance(p, (int, float)) and math.isfinite(float(p))}
            if finite:
                worst = min(finite, key=finite.get)
                v["placebo_holm"] = f"{_fmt(finite[worst])} (smallest; covariate {worst})"
            elif not holm:
                # Empty battery: never the dict repr "{}" (F-D3), and say WHY
                # a vacuous battery reports "passed".
                v["placebo_holm"] = f"{_EM} (no covariate was testable)"
            else:
                v["placebo_holm"] = _EM
        else:
            v["placebo_holm"] = _fmt(holm)
        v["density_p"] = _fmt(summ.get("density_p"))
    else:
        v["null_kind"] = str(summ.get("null_kind") or _EM)
        v["composition"] = _passfail(summ.get("composition_passed"))
        v["anticipation"] = _passfail(summ.get("anticipation_passed"))
    return v


def _paper_context(bundle: ResultsBundle, fmt: str,
                   cards_dir: str | Path | None = None) -> dict:
    """Pure template context builder — no jinja2, no inference, no file writes
    (method cards are only read). All numbers pre-formatted via :func:`_fmt`."""
    r = bundle.results
    configs = [c for c in (r.get("configs") or []) if isinstance(c, dict)]

    designs: list[str] = []
    for cfg in configs:
        design = (cfg.get("candidate") or {}).get("design")
        if cfg.get("status") == "scanned" and design and design not in designs:
            designs.append(design)

    discovery_rows = [
        {
            "design": (cfg.get("candidate") or {}).get("design") or _EM,
            "source": cfg.get("source") or _EM,
            "status": cfg.get("status") or _EM,
            "llr": _fmt(cfg.get("llr")),
            "p_value": _fmt(cfg.get("p_value")),
            "n_discoveries": int(cfg.get("n_discoveries") or 0),
        }
        for cfg in configs
    ]

    best_index = r.get("best_index")
    best_cfg = None
    if best_index is not None and 0 <= int(best_index) < len(configs):
        best_cfg = configs[int(best_index)]
    best = _best_block(best_cfg) if best_cfg else None
    validation = _validation_block(best_cfg) if best_cfg else None
    effects = (best_cfg.get("summary") or {}).get("effects") if best_cfg else None
    effects_rows = _effects_rows(best["design"], effects) if best else []

    searched = r.get("searched")
    coverage = None
    if isinstance(searched, dict):
        coverage = {
            key: int(searched.get(key) or 0)
            for key in ("n_total", "n_scanned", "n_skipped_budget",
                        "n_failed", "n_invalid")
        }
        coverage["not_searched"] = [
            f"{_candidate_label(cfg.get('candidate') or {})} ({cfg.get('status')})"
            for cfg in configs
            if cfg.get("status") != "scanned"
        ]

    ext = "pdf" if fmt == "latex" else "png"
    figures = []
    for entry in r.get("figures") or []:
        if not isinstance(entry, dict):
            continue
        rel = entry.get(ext) or entry.get("png")  # pdf preferred for latex
        if not rel:
            continue
        relpath = rel if Path(rel).is_absolute() else f"../{rel}"
        figures.append({"name": entry.get("name") or _EM, "relpath": relpath})

    data = _data_context(r.get("data"))
    version = r.get("natex_version") or _EM
    title = "Automated natural-experiment discovery report"
    if data is not None and data["treatment"] != _EM:
        title += f": {data['treatment']}"
        if data["outcome"] != _EM:
            title += f" -> {data['outcome']}"

    references = [*_REFERENCES,
                  f"natex-discovery {version} — automated natural-experiment "
                  "discovery software (LoRD3 lineage). "
                  "https://github.com/HaukeHillebrandt/natex"]

    return {
        "banner": BANNER,
        "title": title,
        "version": version,
        "seed": _fmt(r.get("seed")),
        "created": r.get("created") or datetime.now(timezone.utc).isoformat(),
        "data": data,
        "intake": _intake_context(r.get("intake")),
        "designs": designs,
        "method_cards": _method_cards(designs, cards_dir),
        "discovery_rows": discovery_rows,
        "best": best,
        "validation": validation,
        "effects_rows": effects_rows,
        "coverage": coverage,
        "figures": figures,
        "references": references,
        "corrections_note": _CORRECTIONS_NOTE,
    }


def _env():
    """Jinja2 environment (lazy import; StrictUndefined so context/template
    drift fails loudly instead of rendering blanks)."""
    try:
        import jinja2
    except ImportError as exc:
        raise ImportError(_REPORT_EXTRA_MSG) from exc
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(Path(__file__).parent / "templates"),
        undefined=jinja2.StrictUndefined,
        autoescape=False,
        trim_blocks=True,
        lstrip_blocks=True,
    )
    env.filters["fmt"] = _fmt
    env.filters["texesc"] = texesc
    env.filters["md_to_tex"] = _md_to_tex
    return env


def render_paper(bundle: ResultsBundle, format: str = "md",
                 out_dir: str | Path | None = None, *,
                 cards_dir: str | Path | None = None) -> PaperResult:
    """Render the bundle to a paper draft. ``format="md"`` always works with
    the report extra installed. ``format="latex"`` writes ``paper.tex``
    (markdown is NOT written on that branch) and compiles it when tectonic is
    on PATH — a missing/failed compiler degrades to a message, never an
    exception. ``out_dir`` defaults to ``bundle.paper_dir``."""
    if format not in _FORMATS:
        raise ValueError(f"format must be one of {_FORMATS}, got {format!r}")
    env = _env()
    ctx = _paper_context(bundle, format, cards_dir=cards_dir)
    out = Path(out_dir) if out_dir is not None else bundle.paper_dir
    out.mkdir(parents=True, exist_ok=True)
    if format == "latex":
        text = env.get_template("paper.tex.j2").render(**ctx)
        tex_path = out / "paper.tex"
        tex_path.write_text(text, encoding="utf-8")
        pdf, compiled, message = _compile_tex(tex_path)
        return PaperResult(
            markdown=None, tex=tex_path, pdf=pdf, compiled=compiled,
            message=message,
        )
    text = env.get_template("paper.md.j2").render(**ctx)
    path = out / "paper.md"
    path.write_text(text, encoding="utf-8")
    return PaperResult(
        markdown=path, tex=None, pdf=None, compiled=False,
        message=f"markdown paper rendered to {path}",
    )
