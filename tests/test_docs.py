"""Documentation contracts for the reporting phase (plan task 9).

README must carry the "From discovery to paper" section (three-command flow,
Python API, deep-research handoff, manual Google Docs route, extras install
lines, the human-in-the-loop banner rule) and the phase status file must
exist with the run-of-record commands. Pure text assertions — no rendering.
"""

from pathlib import Path

import re

from natex.report.paper import BANNER

ROOT = Path(__file__).resolve().parents[1]
README = (ROOT / "README.md").read_text(encoding="utf-8")

SECTION_HEADER = "## From discovery to paper"


def _section() -> str:
    start = README.index(SECTION_HEADER)
    nxt = README.index("\n## ", start + len(SECTION_HEADER))
    return README[start:nxt]


def _flat() -> str:
    """Section text with whitespace normalized, so hard-wrapped prose still
    matches multi-word substrings."""
    return " ".join(_section().split())


def test_section_sits_between_quickstart_and_backtests():
    i_quick = README.index("## Quickstart")
    i_paper = README.index(SECTION_HEADER)
    i_back = README.index("## Backtests on real data")
    assert i_quick < i_paper < i_back


def test_three_command_flow_documented():
    sec = _section()
    assert "natex study" in sec
    assert "natex discover" in sec and "--plan" in sec
    assert "natex paper --bundle" in sec
    assert "--format" in sec and "tectonic" in sec


def test_python_api_flow_documented():
    sec = _section()
    for name in (
        "ResultsBundle.from_discover",
        "rdd_figures",
        "did_figures",
        "render_paper",
        "research_brief",
    ):
        assert name in sec, f"missing Python API name: {name}"


def test_deep_research_handoff_documented():
    flat = _flat()
    assert "research-brief.md" in flat
    assert "deep-research" in flat.lower() or "deep research" in flat.lower()
    # natex hands off the query; it never performs research calls itself
    assert "no research calls" in flat


def test_google_docs_route_is_manual_and_says_so():
    flat = _flat()
    assert "Google Docs" in flat and "Google Drive" in flat
    assert "integrate the Google Docs API" in flat  # "...does NOT integrate..."
    low = flat.lower()
    assert "does **not** integrate" in low or "does not integrate" in low


def test_extras_install_lines_present():
    sec = _section()
    for extra in ("[report]", "[plot]", "[paperbanana]"):
        assert f"natex-discovery{extra}" in sec, f"missing install line for {extra}"


def test_banner_rule_quotes_the_real_banner():
    # README must quote the banner verbatim so the rule and the code agree.
    assert BANNER in _flat()


def test_roadmap_reporting_row_done():
    assert re.search(r"\|\s*7\s*\|\s*\*\*Done\*\*", README) is not None
    assert "docs/status/phase-report-paper.md" in README


def test_phase_status_file_exists_with_run_of_record():
    status = (ROOT / "docs" / "status" / "phase-report-paper.md").read_text(
        encoding="utf-8"
    )
    assert "uv run ruff check src tests" in status
    assert "uv run pytest -q" in status
    # known limitations the plan requires the status doc to record
    assert "_md_to_tex" in status
    assert "paperbanana" in status
