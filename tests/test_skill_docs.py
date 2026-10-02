"""Docs and agent skills carry the inference rules the pipeline enforces:
the six-value verdict vocabulary (inconclusive is not null), the
placebo-calibrated gate for calendar-time kinks, the size-not-power reading
of placebo rejections, and a writing contract that keeps the slop out of
generated papers."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8")


def test_survey_method_card_documents_inconclusive_and_the_calibrated_kink_gate():
    text = _read("docs/method_cards/survey.md")
    low = text.lower()
    assert "`inconclusive`" in text
    assert "placebo-calibrated" in low
    assert "19" in text  # the placebo-position floor
    assert "role space" in low or "role_space" in low
    assert "effects_by_outcome" in text


def test_kink_method_card_reads_placebo_rejections_as_size_not_power():
    low = _read("docs/method_cards/kink.md").lower()
    assert "not a power" in low or "not evidence of power" in low
    assert "min_attainable_p" in low
    assert "19" in low


def test_survey_skill_and_readme_list_inconclusive():
    assert "inconclusive" in _read("skills/natex-survey/SKILL.md")
    assert "inconclusive" in _read("README.md")
    assert "inconclusive" in _read("AGENTS.md")


def test_write_paper_skill_carries_a_style_contract():
    text = _read("skills/natex-write-paper/SKILL.md")
    assert "Style contract" in text  # its own numbered section
    low = text.lower()
    for phrase in ("em dash", "placebo-calibrated", "once", "title"):
        assert phrase in low, phrase


def test_discover_skill_explains_coverage_and_per_outcome_effects():
    text = _read("skills/discover-natural-experiments/SKILL.md")
    assert "role_space" in text and "effects_by_outcome" in text
    assert "excluded" in text
