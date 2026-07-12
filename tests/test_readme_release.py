"""README release contracts (phase skills-docs, task 7).

The README must link the three agent skills with one-line descriptions and an
install line, carry a Project status section (phases 1-8 Done, pinned real
test counts, a |Dataset|Design|Result| backtest table over every REGISTRY
key), and paste REAL captured CLI output for `natex datasets` and a seeded
`natex discover` demo. Pure text assertions — no rendering, no CLI runs.
"""

from pathlib import Path

import re

from natex.data.registry import REGISTRY

ROOT = Path(__file__).resolve().parents[1]
README = (ROOT / "README.md").read_text(encoding="utf-8")

# Pinned collected-test counts, from a fresh `uv run pytest --collect-only -q`
# (and `-m backtest`) at authoring time. Drift is a conscious two-line edit:
# update these constants AND the same numbers in README's Project status text.
N_NONBACKTEST = 798
N_BACKTEST = 32

SKILLS = (
    "discover-natural-experiments",
    "natex-write-paper",
    "natex-lit-review",
)


def _section(header: str) -> str:
    start = README.index(header)
    nxt = README.find("\n## ", start + len(header))
    return README[start:] if nxt == -1 else README[start:nxt]


def _fences(text: str) -> list[str]:
    return re.findall(r"```[^\n]*\n(.*?)```", text, flags=re.S)


def test_agent_skills_section():
    sec = _section("## Agent skills")
    for name in SKILLS:
        link = f"skills/{name}/SKILL.md"
        assert link in sec, f"missing link: {link}"
        assert (ROOT / link).is_file(), f"linked file does not exist: {link}"
        # one-line description: the bullet's link is followed by an em-dash or
        # colon and non-empty text
        m = re.search(rf"\[{re.escape(name)}\]\([^)]+\)\s*(—|:)\s*\S", sec)
        assert m is not None, f"missing one-line description for {name}"


def test_skills_install_line():
    sec = _section("## Agent skills")
    assert ".claude/skills" in sec
    assert ("ln -s" in sec) or ("cp -" in sec), "no symlink/copy install line"


def test_project_status_table():
    sec = _section("## Project status")
    for phase in range(1, 9):
        assert re.search(
            rf"^\|\s*{phase}\s*\|.*\bDone\b", sec, flags=re.M
        ), f"phase {phase} row not marked Done"
    assert str(N_NONBACKTEST) in sec, "pinned non-backtest count missing"
    assert str(N_BACKTEST) in sec, "pinned backtest count missing"
    assert re.search(r"\|\s*Dataset\s*\|\s*Design\s*\|\s*Result\s*\|", sec)
    for key in REGISTRY:
        assert key in sec, f"registry key {key} missing from backtest table"


def test_real_output_pasted():
    fences = _fences(README)
    # `natex datasets`: command plus genuine output lines in a fenced block
    datasets = [
        f
        for f in fences
        if "uv run natex datasets" in f
        and "rows=" in f
        and ("found" in f or "missing" in f)
    ]
    assert datasets, "no fenced block pairing `uv run natex datasets` with rows=/found output"
    # discover demo: generating snippet is copy-paste runnable ...
    assert any("make_synthetic" in f and "natex discover" in f for f in fences)
    # ... and its genuine output block carries the results:/LLR markers
    discover_out = [
        f for f in fences if "results:" in f and ("llr" in f.lower() or "p=" in f)
    ]
    assert discover_out, "no fenced discover-output block with results: and llr/p="


def test_quickstart_coherence():
    assert "guidance/requests" in README
    assert "out/agent" not in README, "stale guidance dir name out/agent"
    paper = _section("## From discovery to paper")
    assert "natex brief" in paper
    assert re.search(r"\|\s*8\s*\|\s*\*\*Done\*\*", README), "roadmap row 8 not Done"
    assert "Next:" not in README, '"Next:" still introduces the roadmap table'
