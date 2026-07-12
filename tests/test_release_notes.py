"""Release-notes and phase-status contracts for v0.1.0 (phase skills-docs, task 8).

`docs/release_notes/v0.1.0.md` is passed verbatim to `gh release create
--notes-file`, so it renders OUTSIDE the repo: every repo link must be an
absolute `https://github.com/HaukeHillebrandt/natex/blob/v0.1.0/` URL.
Pure text assertions; the backtest table is checked against the live
`REGISTRY` keys, never a hardcoded list.
"""

from __future__ import annotations

from pathlib import Path

from natex.data.registry import REGISTRY

ROOT = Path(__file__).resolve().parents[1]
NOTES_PATH = ROOT / "docs" / "release_notes" / "v0.1.0.md"
BLOB = "https://github.com/HaukeHillebrandt/natex/blob/v0.1.0/"

SKILLS = (
    "discover-natural-experiments",
    "natex-write-paper",
    "natex-lit-review",
)


def _notes() -> str:
    return NOTES_PATH.read_text(encoding="utf-8")


def test_notes_file_exists():
    assert NOTES_PATH.is_file()


def test_summary_paragraph_before_first_heading():
    notes = _notes()
    head = notes[: notes.index("\n## ")]
    # drop the title line(s); what remains must be a real summary paragraph
    body = " ".join(
        line for line in head.splitlines() if line and not line.startswith("#")
    )
    assert len(" ".join(body.split())) > 100, "no summary paragraph before first ##"


def test_methods_table_covers_methods_with_correction_column():
    notes = _notes()
    header = next(
        line
        for line in notes.splitlines()
        if line.lstrip().startswith("|") and "Method" in line
    )
    assert "Correction" in header, "methods table lacks a Correction column"
    for method in ("LoRD3", "SuDDDS", "DEE", "IV"):
        assert method in notes, f"method {method} missing from release notes"
    assert ("SC" in notes) or ("synthetic control" in notes.lower())


def test_backtest_table_covers_every_registry_key():
    notes = _notes()
    for key in REGISTRY:
        assert key in notes, f"registry key {key} missing from backtest table"


def test_guidance_summary_names_backends_and_advisory_guarantee():
    lowered = _notes().lower()
    for word in ("null", "agent", "anthropic", "gemini", "advisory"):
        assert word in lowered, f"guidance summary missing: {word}"


def test_links_are_absolute_blob_urls():
    notes = _notes()
    targets = [f"skills/{name}/SKILL.md" for name in SKILLS]
    targets += ["docs/method_cards", "docs/math_audit_final.md"]
    for target in targets:
        assert BLOB + target in notes, f"missing absolute link: {BLOB}{target}"


def test_install_lines():
    assert "uv add git+https://github.com/HaukeHillebrandt/natex" in _notes()


def test_pypi_pending_note():
    notes = _notes()
    assert "not yet on PyPI" in notes
    assert "natex-discovery" in notes
    assert "pending" in notes


def test_phase_status_doc_has_run_of_record():
    status = (ROOT / "docs" / "status" / "phase-skills-docs.md").read_text(
        encoding="utf-8"
    )
    assert "uv run ruff check src tests" in status
    assert "uv run pytest -q" in status
