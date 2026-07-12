"""CLI ``natex paper`` (phase report-paper task 6).

Contract under test: renders the AI-draft paper from a results bundle dir
(``ResultsBundle.save`` output or a bare ``discover --out`` dir with only
``discover_report.json``); bad format / unloadable bundle / missing jinja2
exit 2 with a clean message (no traceback); success exits 0 and prints the
artifact paths — INCLUDING when tectonic is absent (paper.tex written, skip
message echoed verbatim).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from natex.cli import app
from report_helpers import make_rdd_bundle

runner = CliRunner()


@pytest.fixture(scope="module")
def rdd(tmp_path_factory):
    return make_rdd_bundle(tmp_path_factory.mktemp("cli_paper_rdd"))


# ---------------------------------------------------------------------------
# 1. happy path: markdown paper from a saved bundle
# ---------------------------------------------------------------------------


def test_paper_md_happy_path(rdd):
    pytest.importorskip("jinja2")
    bundle, _report, _ds = rdd
    result = runner.invoke(app, ["paper", "--bundle", str(bundle.out_dir)])
    assert result.exit_code == 0, result.output
    md_path = bundle.paper_dir / "paper.md"
    assert str(md_path) in result.output
    assert md_path.is_file()
    assert "AI-generated" in result.output


# ---------------------------------------------------------------------------
# 2. bad format / unloadable bundle exit 2 with clean messages
# ---------------------------------------------------------------------------


def test_paper_bad_format(rdd):
    bundle, _report, _ds = rdd
    result = runner.invoke(
        app, ["paper", "--bundle", str(bundle.out_dir), "--format", "banana"]
    )
    assert result.exit_code == 2
    assert "banana" in result.output


def test_paper_missing_bundle(tmp_path):
    result = runner.invoke(app, ["paper", "--bundle", str(tmp_path / "nope")])
    assert result.exit_code == 2
    assert "results.json" in result.output


# ---------------------------------------------------------------------------
# 3. latex without tectonic: exit 0, tex written, skip message echoed
# ---------------------------------------------------------------------------


def test_paper_latex_tectonic_missing(rdd, tmp_path, monkeypatch):
    pytest.importorskip("jinja2")
    import natex.report.paper as paper_mod

    monkeypatch.setattr(paper_mod.shutil, "which", lambda _cmd: None)
    bundle, _report, _ds = rdd
    out = tmp_path / "tex_out"
    result = runner.invoke(
        app,
        ["paper", "--bundle", str(bundle.out_dir), "--format", "latex",
         "--out", str(out)],
    )
    assert result.exit_code == 0, result.output
    assert "tectonic" in result.output
    assert (out / "paper.tex").is_file()
    assert str(out / "paper.tex") in result.output


# ---------------------------------------------------------------------------
# 4. jinja2 missing: exit 2 with the [report] extra install message
# ---------------------------------------------------------------------------


def test_paper_jinja2_missing(rdd, monkeypatch):
    monkeypatch.setitem(sys.modules, "jinja2", None)
    bundle, _report, _ds = rdd
    result = runner.invoke(app, ["paper", "--bundle", str(bundle.out_dir)])
    assert result.exit_code == 2
    assert "natex-discovery[report]" in result.output


# ---------------------------------------------------------------------------
# 5. bare discover --out dir (only discover_report.json) works end to end
# ---------------------------------------------------------------------------


def test_paper_from_discover_only_dir(rdd, tmp_path):
    pytest.importorskip("jinja2")
    _bundle, report, _ds = rdd
    d = tmp_path / "discover_out"
    d.mkdir()
    report.save(d)
    assert (d / "discover_report.json").is_file()
    assert not (d / "results.json").exists()
    result = runner.invoke(app, ["paper", "--bundle", str(d)])
    assert result.exit_code == 0, result.output
    md_path = Path(d) / "paper" / "paper.md"
    assert md_path.is_file()
    assert str(md_path) in result.output
