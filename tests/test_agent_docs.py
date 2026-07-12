"""Text contracts for the root agent docs (phase skills-docs task 5: AGENTS.md).

Task 6 extends this module with the CLAUDE.md rows. Shared parsing helpers
live in ``tests/doc_helpers.py`` (plain module, not a conftest) and are also
used by ``tests/test_skills.py``.
"""

from __future__ import annotations

import json
import re

from doc_helpers import ROOT, flat, json_blocks, registered_commands, commands_taught

from natex.llm.backends import TASKS

AGENTS_PATH = ROOT / "AGENTS.md"


def agents_text() -> str:
    assert AGENTS_PATH.exists(), "missing AGENTS.md at repo root"
    return AGENTS_PATH.read_text(encoding="utf-8")


def test_agents_md_exists_and_opens_with_what_natex_is():
    text = agents_text()
    paragraphs = [p for p in re.split(r"\n\s*\n", text) if p.strip()]
    prose = [p for p in paragraphs if not p.lstrip().startswith("#")]
    assert prose, "AGENTS.md has no prose paragraph"
    opening = flat(prose[0]).lower()
    assert "natural experiment" in opening
    assert "discovery" in opening


def test_cli_table_covers_every_command():
    text = agents_text()
    cmds = registered_commands()
    expected = {
        "datasets",
        "fetch-data",
        "study",
        "discover",
        "debias",
        "instruments",
        "donors",
        "paper",
        "brief",
    }
    assert expected <= cmds, f"help parser lost commands: {sorted(expected - cmds)}"
    for name in cmds:
        assert f"`{name}`" in text, f"AGENTS.md CLI table missing `{name}`"
    # Reverse drift guard: everything AGENTS.md teaches must be a real command.
    unknown = commands_taught(text) - cmds
    assert not unknown, f"AGENTS.md teaches unregistered natex commands: {sorted(unknown)}"


def test_protocol_spec_with_parsing_example():
    text = agents_text()
    low = flat(text)
    for marker in ("guidance/requests", "guidance/responses", "schema_hint", "respond_to"):
        assert marker in low, f"missing protocol marker {marker!r}"
    assert "{seq:04d}_{task}.json" in low or "0000_understand.json" in low
    for task in TASKS:
        assert task in low, f"missing guidance task name {task!r}"
    blocks = json_blocks(text)
    assert len(blocks) >= 2, "need at least a worked request and a worked response example"
    parsed = [json.loads(b) for b in blocks]
    request_keys = {"task", "payload", "schema_hint", "instructions", "respond_to"}
    assert any(
        request_keys <= set(p) for p in parsed if isinstance(p, dict)
    ), "no example request JSON carries all five protocol keys"


def test_pointers_exist_on_disk():
    text = agents_text()
    assert "docs/method_cards" in text
    assert "docs/math_audit_final.md" in text
    assert (ROOT / "docs" / "method_cards").is_dir()
    assert (ROOT / "docs" / "math_audit_final.md").is_file()
    referenced = set(re.findall(r"docs/method_cards/([A-Za-z0-9_.-]+\.md)", text))
    assert referenced, "AGENTS.md names no individual method cards"
    for name in sorted(referenced):
        assert (ROOT / "docs" / "method_cards" / name).is_file(), (
            f"AGENTS.md references docs/method_cards/{name} which does not exist"
        )


def test_testing_conventions():
    low = flat(agents_text())
    for phrase in ("uv run pytest -q", "-m backtest", "NATEX_DATA", "never commit"):
        assert phrase in low, f"missing testing-convention phrase {phrase!r}"
