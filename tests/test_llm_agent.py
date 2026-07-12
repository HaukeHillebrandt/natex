"""AgentBackend file-based subscription protocol (plan task 4).

Polling-test policy: poll_interval <= 0.05 s and timeout <= 5 s in every test
so nothing blocks the suite; the 600 s production default is asserted on the
constructor signature only, never waited on.
"""

from __future__ import annotations

import inspect
import json
import threading
import time

import pytest

from natex.llm import AgentBackend, GuidanceRequest


def _answer_later(path, obj, delay=0.1):
    """Write ``obj`` as JSON to ``path`` after ``delay`` seconds, in a thread."""

    def _write():
        time.sleep(delay)
        path.write_text(json.dumps(obj))

    t = threading.Thread(target=_write)
    t.start()
    return t


def test_constructor_defaults():
    sig = inspect.signature(AgentBackend.__init__)
    assert sig.parameters["poll_interval"].default == 0.5
    assert sig.parameters["timeout"].default == 600.0
    assert sig.parameters["echo"].default is print
    assert AgentBackend.name == "agent"


def test_round_trip_via_thread(tmp_path):
    echoed: list[str] = []
    backend = AgentBackend(tmp_path, poll_interval=0.02, timeout=5.0, echo=echoed.append)
    request = GuidanceRequest(
        task="understand",
        payload={"columns": ["a", "b"]},
        schema_hint={"type": "object"},
    )
    response_path = tmp_path / "responses" / "0000_understand.json"
    t = _answer_later(response_path, {"shape": "panel"})
    response = backend.complete(request)
    t.join()

    assert response.content == {"shape": "panel"}
    assert response.backend == "agent"
    assert response.raw_text == json.dumps({"shape": "panel"})

    request_path = tmp_path / "requests" / "0000_understand.json"
    assert request_path.exists()
    body = json.loads(request_path.read_text())
    assert body["task"] == "understand"
    assert body["payload"] == {"columns": ["a", "b"]}
    assert body["schema_hint"] == {"type": "object"}
    assert body["instructions"]
    assert body["respond_to"] == str(response_path)

    assert len(echoed) == 1
    assert str(response_path) in echoed[0]


def test_envelope_shape(tmp_path):
    backend = AgentBackend(tmp_path, poll_interval=0.02, timeout=5.0, echo=lambda s: None)
    request = GuidanceRequest(task="understand", payload={})
    response_path = tmp_path / "responses" / "0000_understand.json"
    t = _answer_later(response_path, {"content": {"a": 1}, "note": "x"})
    response = backend.complete(request)
    t.join()
    assert response.content == {"a": 1}


def test_partial_write_tolerated(tmp_path):
    backend = AgentBackend(tmp_path, poll_interval=0.02, timeout=5.0, echo=lambda s: None)
    request = GuidanceRequest(task="understand", payload={})
    response_path = tmp_path / "responses" / "0000_understand.json"
    response_path.parent.mkdir(parents=True, exist_ok=True)
    response_path.write_text("{invalid")
    t = _answer_later(response_path, {"ok": True})
    response = backend.complete(request)
    t.join()
    assert response.content == {"ok": True}


def test_timeout_names_response_path_and_null_escape(tmp_path):
    backend = AgentBackend(tmp_path, poll_interval=0.02, timeout=0.15, echo=lambda s: None)
    request = GuidanceRequest(task="understand", payload={})
    response_path = tmp_path / "responses" / "0000_understand.json"
    start = time.monotonic()
    with pytest.raises(TimeoutError) as excinfo:
        backend.complete(request)
    elapsed = time.monotonic() - start
    assert elapsed < 1.0
    assert str(response_path) in str(excinfo.value)
    assert "--backend null" in str(excinfo.value)


def test_sequencing_and_restart_safe_seq(tmp_path):
    echoed: list[str] = []
    backend = AgentBackend(tmp_path, poll_interval=0.02, timeout=5.0, echo=echoed.append)

    t1 = _answer_later(tmp_path / "responses" / "0000_understand.json", {"shape": "panel"})
    backend.complete(GuidanceRequest(task="understand", payload={}))
    t1.join()

    t2 = _answer_later(tmp_path / "responses" / "0001_prepare.json", {"drops": []})
    backend.complete(GuidanceRequest(task="prepare", payload={}))
    t2.join()

    assert (tmp_path / "requests" / "0000_understand.json").exists()
    assert (tmp_path / "requests" / "0001_prepare.json").exists()

    fresh = AgentBackend(tmp_path, poll_interval=0.02, timeout=5.0, echo=echoed.append)
    t3 = _answer_later(tmp_path / "responses" / "0002_search_plan.json", {"candidates": []})
    fresh.complete(GuidanceRequest(task="search_plan", payload={}))
    t3.join()
    assert (tmp_path / "requests" / "0002_search_plan.json").exists()


def test_non_dict_response_raises_value_error(tmp_path):
    backend = AgentBackend(tmp_path, poll_interval=0.02, timeout=5.0, echo=lambda s: None)
    request = GuidanceRequest(task="understand", payload={})
    response_path = tmp_path / "responses" / "0000_understand.json"
    t = _answer_later(response_path, [1, 2, 3])
    with pytest.raises(ValueError):
        backend.complete(request)
    t.join()
