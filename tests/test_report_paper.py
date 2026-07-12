"""Markdown paper renderer (phase report-paper task 4).

Context-builder tests run WITHOUT jinja2 (the builder is a pure dict
function); render tests ``importorskip("jinja2")`` per test so a core-only
install stays green. The figures test additionally skips without matplotlib.
"""

from __future__ import annotations

import json
import re
import sys

import numpy as np
import pytest

from natex.report.bundle import ResultsBundle
from natex.report.paper import BANNER, _fmt, _paper_context, render_paper
from report_helpers import make_did_bundle, make_rdd_bundle

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
