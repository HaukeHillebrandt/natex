"""Markdown paper renderer (phase report-paper task 4).

Context-builder tests run WITHOUT jinja2 (the builder is a pure dict
function); render tests ``importorskip("jinja2")`` per test so a core-only
install stays green. The figures test additionally skips without matplotlib.
"""

from __future__ import annotations

import json
import re
import shutil
import sys

import numpy as np
import pytest

from natex.report.bundle import ResultsBundle
from natex.report.paper import BANNER, _fmt, _paper_context, render_paper
from report_helpers import make_did_bundle, make_rdd_bundle, make_scan_payload_bundle

CONTEXT_KEYS = {
    "banner", "title", "version", "seed", "created", "data", "intake",
    "designs", "method_cards", "discovery_rows", "best", "validation",
    "effects_rows", "coverage", "figures", "references", "corrections_note",
}


@pytest.fixture(scope="module")
def rdd(tmp_path_factory):
    return make_rdd_bundle(tmp_path_factory.mktemp("rdd_paper"))


@pytest.fixture(scope="module")
def did(tmp_path_factory):
    return make_did_bundle(tmp_path_factory.mktemp("did_paper"))


# ---------------------------------------------------------------------------
# _fmt: numbers become strings, missing becomes an em dash — never nan/None
# ---------------------------------------------------------------------------


def test_fmt_helper():
    assert _fmt(None) == "—"
    assert _fmt(float("nan")) == "—"
    assert _fmt(float("inf")) == "—"
    assert _fmt(0.123456) == "0.123"
    assert _fmt(7) == "7"
    assert _fmt(True) == "yes" and _fmt(False) == "no"


# ---------------------------------------------------------------------------
# 1. context builder on the rdd bundle (NO jinja2 needed)
# ---------------------------------------------------------------------------


def test_context_rdd(rdd):
    bundle, _report, _ds = rdd
    ctx = _paper_context(bundle, "md")
    assert set(ctx) == CONTEXT_KEYS
    assert ctx["designs"] == ["rdd"]
    assert len(ctx["discovery_rows"]) == len(bundle.results["configs"])
    rows = [r for r in ctx["effects_rows"] if r["estimator"] == "2sls"]
    assert len(rows) == 1
    assert rows[0]["tau"] != "—" and np.isfinite(float(rows[0]["tau"]))
    assert "flags" in rows[0]  # weak-IV flag field (audit 3)
    searched = bundle.results["searched"]
    cov = ctx["coverage"]
    for key in ("n_total", "n_scanned", "n_skipped_budget", "n_failed", "n_invalid"):
        assert cov[key] == searched[key]
    assert isinstance(cov["not_searched"], list)
    dump = json.dumps(ctx)
    assert "nan" not in dump.lower()  # no raw NaN, no 'nan' in any string
    assert "None" not in dump  # null is fine; the string "None" never renders


# ---------------------------------------------------------------------------
# 2. missing pieces degrade, never crash
# ---------------------------------------------------------------------------


def test_context_no_outcome(tmp_path):
    bundle, _, ds = make_rdd_bundle(tmp_path, with_outcome=False)
    assert ds.y is None
    ctx = _paper_context(bundle, "md")
    assert ctx["effects_rows"] == []
    assert set(ctx) == CONTEXT_KEYS


def test_context_empty_cards_dir(rdd, tmp_path):
    bundle, _, _ = rdd
    ctx = _paper_context(bundle, "md", cards_dir=tmp_path)  # empty dir
    assert len(ctx["method_cards"]) == 1
    body = ctx["method_cards"][0]["body"]
    assert body.startswith("(method card")
    assert "not available in this installation" in body


# ---------------------------------------------------------------------------
# 3. render_paper(bundle, "md") — paths and PaperResult contract
# ---------------------------------------------------------------------------


def test_render_md_paths(rdd):
    pytest.importorskip("jinja2")
    bundle, _, _ = rdd
    res = render_paper(bundle, "md")
    assert res.markdown is not None and res.markdown.is_file()
    assert res.markdown.parent == bundle.paper_dir
    assert res.tex is None and res.pdf is None
    assert res.compiled is False
    assert "markdown" in res.message


# ---------------------------------------------------------------------------
# 4. rendered content: banner, lineage, corrections, coverage, em dashes
# ---------------------------------------------------------------------------


def test_render_md_content(tmp_path):
    pytest.importorskip("jinja2")
    bundle, _, _ = make_rdd_bundle(tmp_path)
    best = bundle.results["configs"][bundle.results["best_index"]]
    best["summary"]["effects"]["2sls"]["se"] = float("nan")  # -> null on save
    bundle.save()
    reloaded = ResultsBundle.load(tmp_path)
    res = render_paper(reloaded, "md")
    text = res.markdown.read_text(encoding="utf-8")
    # banner is the first body line after the title
    lines = [ln for ln in text.splitlines() if ln.strip()]
    assert lines[0].startswith("# ")
    assert lines[1] == f"> **{BANNER}**"
    assert text.count("Herlands") >= 2
    assert text.count("Jakubowski") >= 1
    assert "math_audit_final.md" in text
    assert "Monte Carlo" in text
    s = reloaded.results["searched"]
    sentence = (
        f"Of {s['n_total']} enumerated configurations, {s['n_scanned']} were "
        f"scanned, {s['n_skipped_budget']} skipped by budget, {s['n_failed']} "
        f"failed, {s['n_invalid']} invalid."
    )
    assert sentence in text
    assert "Method card — LoRD3" in text
    (row_2sls,) = [ln for ln in text.splitlines() if ln.startswith("| 2sls ")]
    assert "| — |" in row_2sls  # the NaN'd se renders as an em dash
    assert re.search(r"\bnan\b|\bNone\b", text) is None


# ---------------------------------------------------------------------------
# 5. figures embedded when the manifest is non-empty (plot extra)
# ---------------------------------------------------------------------------


def test_render_md_embeds_figures(tmp_path):
    pytest.importorskip("jinja2")
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    from natex.rdd.lord3 import lord3_scan
    from natex.report.figures import rdd_figures

    bundle, _, ds = make_rdd_bundle(tmp_path)
    scan = lord3_scan(ds, k=25, rng=np.random.default_rng(0))
    rdd_figures(bundle, ds, scan)
    res = render_paper(bundle, "md")
    text = res.markdown.read_text(encoding="utf-8")
    assert "../figures/discovery_scatter.png" in text


def test_render_md_plain_scan_bundle(tmp_path):
    """F-D1: `natex paper` on a plain single-scan results.json bundle renders
    the run — discovery row, validation numbers, effects — never the empty
    'No configuration was scanned' document README promises it accepts."""
    pytest.importorskip("jinja2")
    bundle, payload = make_scan_payload_bundle(tmp_path)
    res = render_paper(bundle, "md")
    text = res.markdown.read_text(encoding="utf-8")
    assert "No configuration was scanned" not in text
    assert _fmt(payload["scan"]["p_value"]) in text
    assert _fmt(payload["effects"]["2sls"]["tau"]) in text
    assert _fmt(payload["validation"]["density_p"]) in text
    assert "Method card — LoRD3" in text
    assert re.search(r"\bnan\b|\bNone\b", text) is None


# ---------------------------------------------------------------------------
# 6. did bundle renders; bad format raises ValueError naming the value
# ---------------------------------------------------------------------------


def test_render_did(did):
    pytest.importorskip("jinja2")
    bundle, _, _ = did
    res = render_paper(bundle, "md")
    text = res.markdown.read_text(encoding="utf-8")
    assert "SuDDDS" in text  # the did method card was pulled in
    assert "Treated subset" in text
    assert "T0" in text


def test_render_bad_format(did):
    bundle, _, _ = did
    with pytest.raises(ValueError, match="banana"):
        render_paper(bundle, "banana")


# ---------------------------------------------------------------------------
# 7. import guard: module imports core-clean; render names the report extra
# ---------------------------------------------------------------------------


def test_render_import_guard(rdd, monkeypatch):
    bundle, _, _ = rdd
    monkeypatch.setitem(sys.modules, "jinja2", None)
    with pytest.raises(ImportError, match=r"natex-discovery\[report\]"):
        render_paper(bundle, "md")


def test_module_import_without_jinja2(monkeypatch):
    monkeypatch.setitem(sys.modules, "jinja2", None)
    monkeypatch.delitem(sys.modules, "natex.report.paper", raising=False)
    import natex.report.paper  # noqa: F401  (must not touch jinja2)


def test_package_export():
    import natex.report

    assert natex.report.render_paper is render_paper


# ---------------------------------------------------------------------------
# 8. texesc: every LaTeX special escaped (backslash first), nothing dropped
# ---------------------------------------------------------------------------


def test_texesc_escapes_every_special():
    from natex.report.paper import texesc

    out = texesc("50% & _x_ #$ ~^ \\")
    assert "\\%" in out
    assert "\\&" in out
    assert "\\_x\\_" in out
    assert "\\#" in out
    assert "\\$" in out
    assert "\\textasciitilde{}" in out
    assert "\\textasciicircum{}" in out
    assert "\\textbackslash{}" in out
    assert "50" in out  # no character dropped
    assert texesc("{a}") == "\\{a\\}"
    assert texesc("—") == "--"  # em dash -> LaTeX en dash (missing table cells)


# ---------------------------------------------------------------------------
# 9. _md_to_tex: bounded method-card converter (lossy by design)
# ---------------------------------------------------------------------------


def test_md_to_tex_bounded_converter():
    from natex.report.paper import _md_to_tex

    md = "\n".join([
        "# Head",
        "",
        "Plain foo_bar with **bold** and *ital* and `code_x` and",
        "[audit](../math_audit_final.md).",
        "",
        "- first item",
        "- second `it_em`",
        "",
        "| a | b |",
        "|---|---|",
        "| 1 | 2 |",
        "",
        "tail",
    ])
    tex = _md_to_tex(md)
    assert "\\section*{Head}" in tex
    assert "\\textbf{bold}" in tex
    assert "\\emph{ital}" in tex
    assert "\\texttt{code\\_x}" in tex
    assert "\\begin{itemize}" in tex and "\\end{itemize}" in tex
    assert tex.count("\\item ") == 2
    assert "\\footnote{\\texttt{../math\\_audit\\_final.md}}" in tex
    assert "table omitted in LaTeX rendering" in tex
    assert "| a | b |" not in tex
    assert "foo\\_bar" in tex
    assert re.search(r"(?<!\\)_", tex) is None  # every underscore escaped
    assert "tail" in tex


# ---------------------------------------------------------------------------
# 10. render_paper(bundle, "latex"): tex written, data-derived strings escaped
# ---------------------------------------------------------------------------


def _hostile_rdd_bundle(tmp_path):
    """rdd bundle with a forcing column renamed to the LaTeX-hostile 'x_0%'."""
    bundle, _, _ = make_rdd_bundle(tmp_path)
    r = bundle.results
    r["data"]["forcing"] = ["x_0%" if f == "x0" else f for f in r["data"]["forcing"]]
    for cfg in r["configs"]:
        cand = cfg.get("candidate") or {}
        if cand.get("forcing"):
            cand["forcing"] = ["x_0%" if f == "x0" else f for f in cand["forcing"]]
        infl = (cfg.get("summary") or {}).get("forcing_influence")
        if isinstance(infl, dict) and "x0" in infl:
            infl["x_0%"] = infl.pop("x0")
    return bundle


def test_render_latex_content(tmp_path, monkeypatch):
    pytest.importorskip("jinja2")
    import natex.report.paper as paper_mod

    monkeypatch.setattr(paper_mod.shutil, "which", lambda _cmd: None)
    bundle = _hostile_rdd_bundle(tmp_path)
    res = render_paper(bundle, "latex")
    assert res.markdown is None  # markdown NOT written on the latex branch
    assert res.tex is not None and res.tex.is_file()
    assert res.tex.parent == bundle.paper_dir
    text = res.tex.read_text(encoding="utf-8")
    assert "\\documentclass{article}" in text
    for pkg in ("graphicx", "booktabs", "hyperref"):
        assert f"\\usepackage{{{pkg}}}" in text
    assert "\\fbox{" in text  # framed banner after \maketitle
    assert "AI-generated draft" in text
    for sec in ("Introduction", "Data", "Methods", "Results", "Robustness"):
        assert f"\\section{{{sec}}}" in text
    assert "\\begin{thebibliography}" in text
    assert text.count("Herlands") >= 2
    assert "x\\_0\\%" in text
    assert "x_0%" not in text  # the raw form never survives
    assert re.search(r"\bnan\b|\bNone\b", text) is None


# ---------------------------------------------------------------------------
# 11. tectonic missing: graceful message, tex still written, never raises
# ---------------------------------------------------------------------------


def test_render_latex_tectonic_missing(rdd, monkeypatch):
    pytest.importorskip("jinja2")
    import natex.report.paper as paper_mod

    monkeypatch.setattr(paper_mod.shutil, "which", lambda _cmd: None)
    bundle, _, _ = rdd
    res = render_paper(bundle, "latex")  # must not raise
    assert res.compiled is False
    assert res.pdf is None
    assert "tectonic" in res.message
    assert res.tex is not None and res.tex.is_file()


# ---------------------------------------------------------------------------
# 12. figures manifest -> \includegraphics with the pdf variant preferred
# ---------------------------------------------------------------------------


def test_render_latex_includes_figures(tmp_path, monkeypatch):
    pytest.importorskip("jinja2")
    import natex.report.paper as paper_mod

    monkeypatch.setattr(paper_mod.shutil, "which", lambda _cmd: None)
    bundle, _, _ = make_rdd_bundle(tmp_path)
    png = bundle.figures_dir / "discovery_scatter.png"
    pdf = bundle.figures_dir / "discovery_scatter.pdf"
    png.write_bytes(b"\x89PNG stub")
    pdf.write_bytes(b"%PDF stub")
    bundle.add_figure("discovery_scatter", png, pdf)
    res = render_paper(bundle, "latex")
    text = res.tex.read_text(encoding="utf-8")
    assert "\\includegraphics" in text
    assert "../figures/discovery_scatter.pdf" in text


# ---------------------------------------------------------------------------
# 13. real tectonic compile (skips in CI; runs locally when installed)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(shutil.which("tectonic") is None, reason="tectonic not installed")
def test_render_latex_real_compile(tmp_path):
    pytest.importorskip("jinja2")
    bundle, _, _ = make_rdd_bundle(tmp_path)
    res = render_paper(bundle, "latex")
    assert res.compiled is True, res.message
    assert res.pdf is not None and res.pdf.exists()
    assert res.message == "compiled"
