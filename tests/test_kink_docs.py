"""Documentation contract for the known-cutoff kink-design surface."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_kink_method_card_covers_estimands_and_sign_convention():
    path = ROOT / "docs" / "method_cards" / "kink.md"
    assert path.is_file()
    text = path.read_text(encoding="utf-8")
    low = " ".join(text.lower().split())
    for phrase in (
        "right-minus-left",
        "sharp rkd",
        "fuzzy rkd",
        "sharp dik",
        "fuzzy dik",
        "fieller",
        "hc1",
        "cr1",
    ):
        assert phrase in low


def test_kink_method_card_states_paper_and_identification_boundaries():
    text = (ROOT / "docs" / "method_cards" / "kink.md").read_text(encoding="utf-8")
    low = " ".join(text.lower().split())
    assert "dp18313" in low
    assert "known cutoff" in low or "known-cutoff" in low
    assert "no automatic" in low and "bandwidth" in low
    assert "composition" in low and "same-sign" in low
    assert "parallel" in low and "time-stable" in low
    assert "smoothing bias" in low


def test_readme_teaches_kink_cli_and_python_api():
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    for marker in (
        "natex kink",
        "regression_kink",
        "difference_in_kinks",
        "--policy-kink",
        "--policy-kink-change",
        "docs/method_cards/kink.md",
    ):
        assert marker in text


def test_agent_card_lists_kink_command_and_method_card():
    text = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
    assert "`kink`" in text
    assert "docs/method_cards/kink.md" in text


def test_discover_skill_points_agents_at_known_cutoff_kink_designs():
    path = ROOT / "skills" / "discover-natural-experiments" / "SKILL.md"
    text = path.read_text(encoding="utf-8")
    assert "natex kink" in text
    assert "docs/method_cards/kink.md" in text
    low = " ".join(text.lower().split())
    assert "known" in low and "cutoff" in low


def test_kink_status_records_green_gate_and_deliberate_boundaries():
    path = ROOT / "docs" / "status" / "phase-kinks.md"
    assert path.is_file()
    text = path.read_text(encoding="utf-8")
    assert "uv run pytest -q" in text
    assert "uv run ruff check src tests" in text
    assert "known-cutoff" in text
    assert "fuzzy DiK" in text
