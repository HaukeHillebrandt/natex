"""Deep-research handoff brief (phase report-paper task 8).

Pure text generation — no optional deps, no network, no rng: every test here
runs on a core-only install. Determinism is asserted byte-for-byte.
"""

from __future__ import annotations

import re

import pytest

from natex.report.bundle import ResultsBundle
from natex.report.paper import _fmt
from natex.report.research_brief import research_brief
from report_helpers import make_did_bundle, make_rdd_bundle

HEADINGS = (
    "## Discovery context",
    "## Discovered designs",
    "## Effect estimates",
    "## Validation status",
    "## Open questions",
    "## Literature questions for deep research",
    "## How to use",
)


@pytest.fixture(scope="module")
def rdd(tmp_path_factory):
    return make_rdd_bundle(tmp_path_factory.mktemp("rdd_brief"))


@pytest.fixture(scope="module")
def did(tmp_path_factory):
    return make_did_bundle(tmp_path_factory.mktemp("did_brief"))


def _lit_section(text: str) -> str:
    tail = text.split("## Literature questions for deep research", 1)[1]
    return tail.split("\n## ", 1)[0]


# ---------------------------------------------------------------------------
# 1. rdd bundle: path, all eight headings, numbered questions, cutoff center
# ---------------------------------------------------------------------------


def test_rdd_brief_headings_and_questions(rdd, tmp_path):
    bundle, _report, _ds = rdd
    path = research_brief(bundle, tmp_path)
    assert path == tmp_path / "research-brief.md"
    text = path.read_text(encoding="utf-8")
    assert text.splitlines()[0].startswith("# Research brief: ")
    for heading in HEADINGS:  # + the title line above = all eight headings
        assert heading in text
    numbered = [ln for ln in _lit_section(text).splitlines() if re.match(r"^\d+\.", ln)]
    assert len(numbered) >= 3
    assert all(ln.rstrip().endswith("?") for ln in numbered)
    assert "= {}" not in text  # empty holm dict renders as an em dash
    # the cutoff center coordinates (raw forcing space, formatted) appear
    best = bundle.results["configs"][bundle.results["best_index"]]
    for value in best["summary"]["center_z"]:
        assert _fmt(value) in text


def test_rdd_weak_iv_flag(tmp_path):
    bundle, _, _ = make_rdd_bundle(tmp_path)
    eff = bundle.results["configs"][bundle.results["best_index"]]["summary"]["effects"]
    eff["2sls"]["weak_instrument"] = True
    eff["2sls"]["first_stage_t"] = 1.2
    text = research_brief(bundle, tmp_path).read_text(encoding="utf-8")
    assert "weak IV (first-stage t = 1.2)" in text


# ---------------------------------------------------------------------------
# 2. did bundle: T0 and a treated-subset value string appear
# ---------------------------------------------------------------------------


def test_did_brief_t0_and_subset(did, tmp_path):
    bundle, _, _ = did
    text = research_brief(bundle, tmp_path).read_text(encoding="utf-8")
    summ = bundle.results["configs"][bundle.results["best_index"]]["summary"]
    assert f"T0 = {_fmt(summ['t0'])}" in text
    assert summ["subset_values"], "did fixture must carry a treated subset"
    dim, vals = next(iter(summ["subset_values"].items()))
    assert dim in text
    assert _fmt(vals[0]) in text


# ---------------------------------------------------------------------------
# 3. determinism: byte-identical reruns; "nan"/"None" never rendered
# ---------------------------------------------------------------------------


def test_byte_identical_reruns(rdd, tmp_path):
    bundle, _, _ = rdd
    first = research_brief(bundle, tmp_path / "a").read_bytes()
    second = research_brief(bundle, tmp_path / "b").read_bytes()
    assert first == second
    rerun = research_brief(bundle, tmp_path / "a")  # overwrite in place
    assert rerun.read_bytes() == first
    assert re.search(rb"\bnan\b|\bNone\b", first) is None


def test_nan_renders_as_em_dash(tmp_path):
    bundle, _, _ = make_rdd_bundle(tmp_path)
    best = bundle.results["configs"][bundle.results["best_index"]]
    best["summary"]["effects"]["2sls"]["se"] = float("nan")  # -> null on save
    bundle.save()
    reloaded = ResultsBundle.load(tmp_path)
    text = research_brief(reloaded, tmp_path).read_text(encoding="utf-8")
    assert re.search(r"\bnan\b|\bNone\b", text) is None
    assert "—" in text


# ---------------------------------------------------------------------------
# 4. out endswith .md -> exact path; no-outcome bundle degrades, never crashes
# ---------------------------------------------------------------------------


def test_out_md_written_exactly_there(rdd, tmp_path):
    bundle, _, _ = rdd
    target = tmp_path / "handoff" / "my-brief.md"
    path = research_brief(bundle, target)
    assert path == target
    assert target.is_file()


def test_no_outcome_bundle(tmp_path):
    bundle, _, ds = make_rdd_bundle(tmp_path, with_outcome=False)
    assert ds.y is None
    text = research_brief(bundle, tmp_path).read_text(encoding="utf-8")
    assert "no effect estimates (no outcome column was provided)" in text
    for heading in HEADINGS:
        assert heading in text
    assert re.search(r"\bnan\b|\bNone\b", text) is None


def test_package_export():
    import natex.report

    assert natex.report.research_brief is research_brief
