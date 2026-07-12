"""Text contracts for the agent skills in ``skills/`` (phase skills-docs tasks 2-4).

Dependency-free: frontmatter is parsed with a tiny splitter (single-line
``key: value`` pairs only — the authoring house rule keeps ``description``
on ONE line), never pyyaml. ``SKILL_DIRS`` starts with the discover skill;
tasks 3 and 4 append their directory name as their failing-test-first step.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from typer.testing import CliRunner

from natex.cli import app
from natex.llm.backends import TASKS

ROOT = Path(__file__).resolve().parents[1]
SKILLS = ROOT / "skills"

# Grows per task: task 3 appends "natex-write-paper", task 4 "natex-lit-review".
SKILL_DIRS = [
    "discover-natural-experiments",
]

KEBAB = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")

DISCOVER = "discover-natural-experiments"


def frontmatter(path: Path) -> tuple[dict[str, str], str]:
    """Parse leading '---' YAML block: single-line 'key: value' pairs only.

    Returns (meta, body). Raises AssertionError on malformed frontmatter.
    """
    text = path.read_text(encoding="utf-8")
    assert text.startswith("---\n"), f"{path}: missing frontmatter"
    end = text.index("\n---", 4)
    meta = {}
    for line in text[4:end].splitlines():
        key, _, value = line.partition(":")
        meta[key.strip()] = value.strip()
    return meta, text[end + 4 :]


def skill_path(d: str) -> Path:
    return SKILLS / d / "SKILL.md"


def body_of(d: str) -> str:
    _, body = frontmatter(skill_path(d))
    return body


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


def test_help_parser_sanity():
    """Guard the --help parser itself: the known commands must all be found."""
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


# --- shared contracts, every skill directory --------------------------------


@pytest.mark.parametrize("d", SKILL_DIRS)
def test_skill_file_exists(d):
    assert skill_path(d).exists(), f"missing skills/{d}/SKILL.md"


@pytest.mark.parametrize("d", SKILL_DIRS)
def test_frontmatter_name_matches_dir(d):
    meta, _ = frontmatter(skill_path(d))
    assert meta.get("name") == d
    assert KEBAB.match(meta["name"]), f"skill name not kebab-case: {meta['name']!r}"


@pytest.mark.parametrize("d", SKILL_DIRS)
def test_description_is_one_paragraph_with_triggers(d):
    meta, _ = frontmatter(skill_path(d))
    desc = meta.get("description", "")
    assert desc, f"skills/{d}: empty description"
    assert "\n" not in desc
    assert len(desc) > 100, f"skills/{d}: description too short to carry triggers"


@pytest.mark.parametrize("d", SKILL_DIRS)
def test_safety_warnings_present(d):
    low = flat(body_of(d)).lower()
    for phrase in ("never fabricate", "validation battery", "ai-generated", "verify"):
        assert phrase in low, f"skills/{d}: missing safety phrase {phrase!r}"


@pytest.mark.parametrize("d", SKILL_DIRS)
def test_only_real_cli_commands_are_taught(d):
    cmds = registered_commands()
    taught: set[str] = set()
    for block in fenced_blocks(body_of(d)):
        taught |= set(re.findall(r"\bnatex[ \t]+([a-z][a-z0-9-]+)", block))
    unknown = taught - cmds
    assert not unknown, f"skills/{d} teaches unregistered natex commands: {sorted(unknown)}"


# --- discover-natural-experiments specifics ---------------------------------


def test_discover_skill_triggers():
    meta, _ = frontmatter(skill_path(DISCOVER))
    desc = meta["description"]
    assert "find natural experiments in this dataset" in desc
    assert "discover RDDs" in desc


def test_discover_skill_install_paths():
    body = body_of(DISCOVER)
    assert "uv add git+https://github.com/HaukeHillebrandt/natex" in body
    assert "uv sync" in body


def test_discover_skill_serves_protocol():
    text = flat(body_of(DISCOVER))
    markers = [
        "--backend agent",
        "guidance/requests",
        "guidance/responses",
        "schema_hint",
        "respond_to",
        "intake_report.json",
        "natex discover --plan",
        "results.json",
    ]
    pos = -1
    for marker in markers:
        nxt = text.find(marker)
        assert nxt != -1, f"missing protocol marker {marker!r}"
        assert nxt > pos, f"protocol marker {marker!r} appears out of workflow order"
        pos = nxt
    for task in TASKS:
        assert task in text, f"missing guidance task name {task!r}"


def test_discover_skill_example_json_parses():
    blocks = json_blocks(body_of(DISCOVER))
    assert len(blocks) >= 2, "need at least a worked request and a worked response example"
    parsed = [json.loads(b) for b in blocks]
    request_keys = {"task", "payload", "schema_hint", "instructions", "respond_to"}
    assert any(
        request_keys <= set(p) for p in parsed if isinstance(p, dict)
    ), "no example request JSON carries all five protocol keys"


def test_discover_skill_states_honest_inference():
    low = flat(body_of(DISCOVER)).lower()
    for phrase in (
        "fitted-null monte carlo",
        "not exact",
        "+1-rank",
        "weak_instrument",
        "honest split",
        "advisory",
        "discovery",
        "estimation",
    ):
        assert phrase in low, f"missing honest-inference phrase {phrase!r}"
