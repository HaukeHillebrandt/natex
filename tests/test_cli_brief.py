"""CLI ``natex brief`` (phase skills-docs task 1).

Contract under test: `natex brief --bundle DIR` wraps the existing
`natex.report.research_brief()` API — writes ``research-brief.md`` next to
the bundle by default, `--out` accepts a dir or an exact ``.md`` path,
reruns are byte-identical (the lit-review skill relies on that), and an
unloadable bundle exits 2 with a clean message (no traceback), mirroring
`natex paper`.
"""

from __future__ import annotations

from typer.testing import CliRunner

from natex.cli import app
from report_helpers import make_rdd_bundle

runner = CliRunner()


def test_brief_writes_default_path(tmp_path):
    make_rdd_bundle(tmp_path)
    result = runner.invoke(app, ["brief", "--bundle", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert "research-brief.md" in result.output
    brief_path = tmp_path / "research-brief.md"
    assert brief_path.exists()
    assert "## Literature questions for deep research" in brief_path.read_text()


def test_brief_out_md_writes_exactly_there(tmp_path):
    make_rdd_bundle(tmp_path)
    target = tmp_path / "sub" / "my-brief.md"
    result = runner.invoke(
        app, ["brief", "--bundle", str(tmp_path), "--out", str(target)]
    )
    assert result.exit_code == 0, result.output
    assert target.exists()


def test_brief_reruns_byte_identical(tmp_path):
    make_rdd_bundle(tmp_path)
    brief_path = tmp_path / "research-brief.md"

    result = runner.invoke(app, ["brief", "--bundle", str(tmp_path)])
    assert result.exit_code == 0, result.output
    first = brief_path.read_bytes()

    result = runner.invoke(app, ["brief", "--bundle", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert brief_path.read_bytes() == first


def test_brief_missing_bundle_exits_2(tmp_path):
    result = runner.invoke(app, ["brief", "--bundle", str(tmp_path / "nope")])
    assert result.exit_code == 2
    assert "Traceback" not in result.output
