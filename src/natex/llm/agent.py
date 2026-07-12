"""File-based subscription-mode guidance backend (design spec 6c).

A calling coding agent subscribes to ``workdir/requests/`` and answers by
writing JSON files into ``workdir/responses/``; natex never calls an API
(zero cost, fully offline). Guidance stays advisory — see
:mod:`natex.llm.backends` for the contract.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from pathlib import Path

from natex.llm.backends import TASK_INSTRUCTIONS, GuidanceRequest, GuidanceResponse


class AgentBackend:
    """File-based request/response guidance for a calling coding agent (spec 6c, zero API cost).

    ``complete()`` writes ``workdir/requests/{seq:04d}_{task}.json``, prints one
    instruction line via ``echo``, then polls
    ``workdir/responses/{seq:04d}_{task}.json`` until it parses as JSON or
    ``timeout`` elapses.
    """

    name = "agent"

    def __init__(
        self,
        workdir: str | Path,
        poll_interval: float = 0.5,
        timeout: float = 600.0,
        echo: Callable[[str], None] = print,
    ):
        self.workdir = Path(workdir)
        self.requests_dir = self.workdir / "requests"
        self.responses_dir = self.workdir / "responses"
        self.requests_dir.mkdir(parents=True, exist_ok=True)
        self.responses_dir.mkdir(parents=True, exist_ok=True)
        self.poll_interval = float(poll_interval)
        self.timeout = float(timeout)
        self.echo = echo
        # Restart-safe, monotone: continue after any requests already on disk.
        self._seq = len(list(self.requests_dir.iterdir()))

    def complete(self, request: GuidanceRequest) -> GuidanceResponse:
        seq = self._seq
        self._seq += 1
        filename = f"{seq:04d}_{request.task}.json"
        request_path = self.requests_dir / filename
        response_path = self.responses_dir / filename

        body = request.model_dump() | {
            "instructions": TASK_INSTRUCTIONS[request.task],
            "respond_to": str(response_path),
        }
        request_path.write_text(json.dumps(body, indent=1))
        self.echo(
            f"natex guidance request ({request.task}): "
            f"answer by writing JSON matching schema_hint to {response_path}"
        )

        deadline = time.monotonic() + self.timeout
        while True:
            if response_path.exists():
                text = response_path.read_text()
                if text:
                    try:
                        parsed = json.loads(text)
                    except json.JSONDecodeError:
                        pass  # partial write — keep polling
                    else:
                        if not isinstance(parsed, dict):
                            raise ValueError(
                                f"guidance response at {response_path} must be a JSON "
                                f"object, got {type(parsed).__name__}"
                            )
                        if isinstance(parsed.get("content"), dict):
                            content = parsed["content"]
                        else:
                            content = parsed
                        return GuidanceResponse(
                            content=content, raw_text=text, backend=self.name
                        )
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"no guidance response at {response_path} after {self.timeout:.0f}s; "
                    f"write JSON matching the schema_hint in {request_path} to that path "
                    f"and re-run, or use --backend null"
                )
            time.sleep(self.poll_interval)
