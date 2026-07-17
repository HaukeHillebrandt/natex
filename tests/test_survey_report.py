"""Survey report renderers (phase survey task 8): report.md always + report.html.

Contract under test: ``survey()`` ALWAYS writes ``report.md`` (pure Python,
no jinja2) and records it on the result; ``report.html`` needs the report
extra — with jinja2 monkeypatched away the survey still completes,
``report_html`` is None and ``coverage["notes"]`` carries the
natex-discovery[report] install message. The html is SELF-CONTAINED (inline
CSS, base64-embedded figures, no external resources), renders one
``<section id=...>`` per family with status badges, records analyst
overrides ("heuristic said ...; analyst said ..."), and never renders the
strings "nan"/"None" (em-dash convention). Both renderers consume the
JSON-native survey.json dict, so a re-render from a saved file shares one
path with the live run.

Stochastic notes: reuses the task-5 synthetic rdd recipe at q=9/k=25
(verdict pinned only to {credible, null}, figures exist either way —
test_survey_figures precedent); everything else is deterministic
doctored-dict rendering.
"""

import json
import re
import sys

import numpy as np
import pandas as pd
import pytest

from natex.data.synthetic import make_synthetic
from natex.report.survey_html import SURVEY_BANNER, render_survey_html, render_survey_md
from natex.survey import survey
from natex.survey.registry import FAMILIES, FAMILY_ORDER

_BUDGET = {"q": 9, "k": 25}  # small explicit test budget (plan task 5 convention)
_KINK_SKIP = "no pre-declared cutoff (kink is candidate evaluation, not discovery)"
_STATUSES = ("credible", "null", "skipped", "needs_input", "failed")


def _trap_words(text: str) -> list[str]:
    """Napkin trap words: the word nan (any case) and the exact word None."""
    return re.findall(r"(?i)\bnan\b", text) + re.findall(r"\bNone\b", text)


def _synthetic_csv(root):
    """make_synthetic(n=300) binary-treatment CSV (tests/test_survey_runner.py recipe)."""
    ds, _ = make_synthetic(
        n=300, px=3, pz=2, zeta=6.0, kind="binary", rng=np.random.default_rng(0)
    )
    df = ds.df.copy()
    df.insert(df.columns.get_loc("T"), "holiday",
              np.random.default_rng(1).integers(0, 2, len(df)))
    path = root / "synthetic.csv"
    df.to_csv(path, index=False)
    return path


def _plain_cross_section(seed=0, n=200):
    """Pure rng normals x0..x3: no binary column, no panel, nothing to run."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame(rng.normal(size=(n, 4)), columns=[f"x{i}" for i in range(4)])


def _fam(name, status):
    """Doctored JSON-native FamilyResult dict with deliberate None values."""
    return {
        "family": name,
        "status": status,
        "reason": f"doctored {status.replace('_', ' ')} outcome",
        "applicability": {
            "run": status in ("credible", "null", "failed"),
            "reason": "all requirements met",
            "heuristic": {
                "status": "applicable", "reason": "all requirements met", "unmet": [],
            },
            "config_hints": {
                "cutoffs": [], "instruments": [], "thresholds": [],
                "treated_unit": None, "t0": None,
            },
            "override": None,
            "guidance_error": None,
        },
        "key_numbers": {"tau": 1.25, "p_value": None},
        "diagnostics": {"caveats": [FAMILIES[name].caveat], "n_used": 42, "missing": None},
        "figures": {},
        "no_figure_reason": "no figure: doctored result",
        "details_path": None,
        "error": None,
    }


def _doctored_result():
    """All seven families with one family per status across the 5-value set,
    a recorded kink override, and a failed family carrying an error."""
    statuses = dict(rdd="credible", did="null", kink="needs_input", iv="failed",
                    sc="skipped", bunching="null", dee="skipped")
    families = {n: _fam(n, statuses[n]) for n in FAMILY_ORDER}
    families["kink"]["applicability"]["override"] = {
        "heuristic_said": False,
        "analyst_said": True,
        "reason": "the context names a plausible cutoff",
    }
    families["iv"]["error"] = "doctored failure message"
    return {
        "families": families,
        "coverage": {"ran": ["rdd", "did", "iv", "bunching"], "not_run": {},
                     "rdd": None, "did": None},
        "dataset": {"source": "doctored.csv", "n_rows": 100, "n_cols": 5,
                    "columns_truncated": ["a", "b"], "time_column": None,
                    "time_range": None},
        "natex_version": "0.0.0-test",
        "seed": None,
        "created": "2026-07-17T00:00:00+00:00",
        "context": None,
        "guidance_log_path": None,
        "report_html": None,
        "report_md": None,
    }


def test_report_md_always(monkeypatch, tmp_path):
    """No jinja2: report.md still written and recorded; report_html None with
    the [report]-extra install message in coverage notes."""
    monkeypatch.setitem(sys.modules, "jinja2", None)
    out = tmp_path / "out"
    res = survey(_plain_cross_section(), rng=np.random.default_rng(1), out_dir=out)

    assert res.report_md == "report.md"
    md = (out / "report.md").read_text(encoding="utf-8")
    assert SURVEY_BANNER in md
    for family in FAMILIES.values():
        assert family.title in md
    assert _KINK_SKIP in md
    assert _trap_words(md) == []

    assert res.report_html is None
    assert any("natex-discovery[report]" in n for n in res.coverage["notes"])

    # the re-saved survey.json carries the report paths
    d = json.loads((out / "survey.json").read_text(encoding="utf-8"))
    assert d["report_md"] == "report.md"
    assert d["report_html"] is None


def test_report_html_contract(tmp_path):
    """rdd synthetic run: self-contained html with all seven sections, at
    least one embedded base64 PNG, and the banner."""
    pytest.importorskip("jinja2")
    pytest.importorskip("matplotlib")
    csv = _synthetic_csv(tmp_path)
    out = tmp_path / "out"
    res = survey(str(csv), rng=np.random.default_rng(0), out_dir=out, budget=_BUDGET)

    assert res.report_md == "report.md"
    assert res.report_html == "report.html"
    html = (out / "report.html").read_text(encoding="utf-8")
    assert SURVEY_BANNER in html
    for name in FAMILY_ORDER:
        assert f'<section id="{name}">' in html
    assert html.count("data:image/png;base64,") >= 1


def test_badges_and_trap_words(tmp_path):
    """Doctored one-family-per-status render: all five badge classes used in
    the body; no "nan"/"None" anywhere in the rendered text."""
    pytest.importorskip("jinja2")
    html = render_survey_html(_doctored_result(), tmp_path).read_text(encoding="utf-8")
    for status in _STATUSES:
        assert re.search(rf'class="badge badge-{status}"', html), status
    assert _trap_words(html) == []
    # the kink override is rendered in html too
    assert "heuristic said" in html and "analyst said" in html


def test_html_self_contained(tmp_path):
    """No external requests: no http(s) img src, no <link>, no <script>."""
    pytest.importorskip("jinja2")
    html = render_survey_html(_doctored_result(), tmp_path).read_text(encoding="utf-8")
    assert '<img src="http' not in html
    assert "<link" not in html
    assert "<script" not in html


def test_override_rendered(tmp_path):
    """Kink override: both the heuristic and the analyst verdicts appear with
    the analyst's reason (md path — must work without jinja2)."""
    md = render_survey_md(_doctored_result(), tmp_path).read_text(encoding="utf-8")
    assert "heuristic said do not run" in md
    assert "analyst said run" in md
    assert "the context names a plausible cutoff" in md
    assert _trap_words(md) == []


def test_md_html_render_from_loaded_json(tmp_path):
    """Both renderers run off json.loads(survey.json) — JSON-native contract;
    the re-rendered report.md is byte-identical to the live run's."""
    pytest.importorskip("jinja2")
    out = tmp_path / "out"
    survey(_plain_cross_section(), rng=np.random.default_rng(1), out_dir=out)
    d = json.loads((out / "survey.json").read_text(encoding="utf-8"))

    re_dir = tmp_path / "rerender"
    md = render_survey_md(d, re_dir)
    html = render_survey_html(d, re_dir)
    assert md.read_text(encoding="utf-8") == (out / "report.md").read_text(encoding="utf-8")
    assert SURVEY_BANNER in html.read_text(encoding="utf-8")
