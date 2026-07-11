"""Append-only JSONL guidance log (spec 6c reproducibility).

One JSON line per request+response, for EVERY backend (including Null/Mock),
so a results bundle always carries the full guidance transcript.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from natex.llm.backends import GuidanceBackend, GuidanceRequest, GuidanceResponse


class GuidanceLog:
    """Append-only JSONL log of guidance requests and responses."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            with self.path.open("r", encoding="utf-8") as fh:
                self.n_entries = sum(1 for line in fh if line.strip())
        else:
            self.n_entries = 0

    def append(self, request: GuidanceRequest, response: GuidanceResponse) -> None:
        entry = {
            "seq": self.n_entries,
            "ts": datetime.now(UTC).isoformat(),
            "task": request.task,
            "backend": response.backend,
            "request": request.model_dump(),
            "response": response.model_dump(),
        }
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry) + "\n")
        self.n_entries += 1


class LoggedBackend:
    """Decorator: delegates to ``inner``, appends every request+response to ``log``."""

    def __init__(self, inner: GuidanceBackend, log: GuidanceLog):
        self._inner = inner
        self._log = log
        self.name = inner.name  # the log records the REAL backend

    def complete(self, request: GuidanceRequest) -> GuidanceResponse:
        response = self._inner.complete(request)
        self._log.append(request, response)
        return response
