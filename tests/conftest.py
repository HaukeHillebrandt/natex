"""Shared test configuration: keep CLI output free of ANSI escapes.

Typer force-enables rich terminal rendering when GITHUB_ACTIONS (or
FORCE_COLOR / PY_COLORS) is set, so on CI the CliRunner output contains
ANSI escapes that split option names into separately styled fragments
(e.g. ``--outcome`` renders as two ``-`` segments), breaking substring
assertions that pass locally.  Typer reads these env vars once at import
time, so neutralise them here — conftest runs before any test module
imports typer.
"""

import os

os.environ["_TYPER_FORCE_DISABLE_TERMINAL"] = "1"
os.environ["NO_COLOR"] = "1"
os.environ.pop("FORCE_COLOR", None)
os.environ.pop("PY_COLORS", None)
