"""Release version contracts (phase skills-docs, task 8).

`pyproject.toml`'s `project.version` and `natex.__version__` are declared in
two places; both must carry the same plain semver release string (no dev
suffix). Parsed with stdlib `tomllib` — no new dependency.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import natex

ROOT = Path(__file__).resolve().parents[1]


def test_version_synced():
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert pyproject["project"]["version"] == natex.__version__


def test_version_is_release():
    # plain X.Y.Z — no .dev/.rc suffix; stays valid for future releases
    assert re.fullmatch(r"\d+\.\d+\.\d+", natex.__version__)
