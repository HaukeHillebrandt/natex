"""Shared helpers for the docs/skills text-contract tests.

Plain helper module (NOT a conftest): imported by ``tests/test_skills.py``
and ``tests/test_agent_docs.py``. Dependency-free beyond typer's CliRunner —
frontmatter and fenced blocks are parsed with tiny regex splitters, never
pyyaml/markdown.
"""

from __future__ import annotations

import re
from pathlib import Path

from typer.testing import CliRunner

from natex.cli import app

ROOT = Path(__file__).resolve().parents[1]


def flat(text: str) -> str:
    """Whitespace-flattened text so hard-wrapped prose matches substrings."""
    return " ".join(text.split())


def fenced_blocks(text: str) -> list[str]:
    """Contents of every triple-backtick fenced block, any language tag."""
    return re.findall(r"```[^\n]*\n(.*?)```", text, re.S)


def json_blocks(text: str) -> list[str]:
    """Contents of every ```json fenced block."""
    return re.findall(r"```json\n(.*?)```", text, re.S)


def registered_commands() -> set[str]:
    """Command names parsed from ``natex --help`` (rich-box or plain layout)."""
    output = CliRunner().invoke(app, ["--help"]).output
    tail = output[output.index("Commands") :]
    cmds = set()
    for line in tail.splitlines():
        m = re.match(r"^[│|]?\s{1,3}([a-z][a-z0-9-]+)", line)
        if m:
            cmds.add(m.group(1))
    return cmds


def commands_taught(text: str) -> set[str]:
    """Every ``natex <command>`` invocation inside fenced blocks of ``text``."""
    taught: set[str] = set()
    for block in fenced_blocks(text):
        taught |= set(re.findall(r"\bnatex[ \t]+([a-z][a-z0-9-]+)", block))
    return taught
