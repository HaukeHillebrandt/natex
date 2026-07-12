"""CLI agent-backend default request/response dir (phase skills-docs task 1).

Contract under test: `natex study --backend agent` and the plan branch of
`natex discover --backend agent` build the AgentBackend with workdir
``OUT/guidance`` by default (the dir the agent skills document), `--workdir`
overrides it, and both help texts name ``OUT/guidance`` (never ``OUT/agent``).
The AgentBackend is monkeypatched to a fake that records its workdir and
answers via the deterministic NullBackend heuristics — no polling, no files.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from natex.cli import app
from natex.llm.backends import NullBackend

runner = CliRunner()


@pytest.fixture
def recorded_workdirs(monkeypatch):
    """Monkeypatch natex.cli.AgentBackend to a recorder; return the record."""
    recorded: list[Path] = []

    class FakeAgentBackend:
        name = "agent"

        def __init__(self, workdir):
            self.workdir = Path(workdir)
            recorded.append(self.workdir)

        def complete(self, request):
            return NullBackend().complete(request)

    monkeypatch.setattr("natex.cli.AgentBackend", FakeAgentBackend)
    return recorded


def _tiny_csv(root) -> Path:
    """20-row csv with binary T and numeric y, x — deterministic, no rng."""
    df = pd.DataFrame(
        {
            "T": [0, 1] * 10,
            "y": [float(i) for i in range(20)],
            "x": [0.5 * i for i in range(20)],
        }
    )
    path = root / "tiny.csv"
    df.to_csv(path, index=False)
    return path


def test_study_agent_default_workdir_is_out_guidance(tmp_path, recorded_workdirs):
    csv = _tiny_csv(tmp_path)
    out = tmp_path / "out"
    result = runner.invoke(
        app, ["study", str(csv), "--backend", "agent", "--out", str(out)]
    )
    assert result.exit_code == 0, result.output
    assert recorded_workdirs == [out / "guidance"]


def test_study_agent_workdir_flag_overrides(tmp_path, recorded_workdirs):
    csv = _tiny_csv(tmp_path)
    out = tmp_path / "out"
    custom = tmp_path / "custom"
    result = runner.invoke(
        app,
        ["study", str(csv), "--backend", "agent", "--out", str(out),
         "--workdir", str(custom)],
    )
    assert result.exit_code == 0, result.output
    assert recorded_workdirs == [custom]
    assert not (out / "guidance").exists()


def test_discover_plan_agent_default_workdir(tmp_path, recorded_workdirs):
    csv = _tiny_csv(tmp_path)
    out = tmp_path / "out"
    result = runner.invoke(
        app, ["study", str(csv), "--backend", "null", "--out", str(out)]
    )
    assert result.exit_code == 0, result.output
    assert recorded_workdirs == []  # null backend never builds an AgentBackend

    out2 = tmp_path / "out2"
    runner.invoke(
        app,
        ["discover", str(csv), "--plan", str(out / "intake_report.json"),
         "--backend", "agent", "--out", str(out2)],
    )
    assert recorded_workdirs == [out2 / "guidance"]


def test_help_text_names_guidance_dir():
    for command in ("study", "discover"):
        result = runner.invoke(app, [command, "--help"])
        assert result.exit_code == 0, result.output
        assert "OUT/guidance" in result.output
        assert "OUT/agent" not in result.output
