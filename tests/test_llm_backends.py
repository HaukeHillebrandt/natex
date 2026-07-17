"""Tests for natex.llm core: models, protocol, MockBackend, guidance log.

Deterministic (no RNG, no network): no statistical calibration needed.
"""

import json

import pytest
from pydantic import ValidationError

from natex.llm import (
    TASKS,
    GuidanceBackend,
    GuidanceLog,
    GuidanceRequest,
    GuidanceResponse,
    LoggedBackend,
    MockBackend,
)
from natex.llm.backends import TASK_INSTRUCTIONS


class TestGuidanceModels:
    def test_bogus_task_rejected(self):
        with pytest.raises(ValidationError):
            GuidanceRequest(task="bogus", payload={})

    def test_all_seven_tasks_validate(self):
        assert len(TASKS) == 7
        for task in TASKS:
            req = GuidanceRequest(task=task, payload={"a": 1})
            assert req.task == task
            assert req.schema_hint == {}

    def test_schema_hint_kept(self):
        req = GuidanceRequest(
            task="understand", payload={}, schema_hint={"type": "object"}
        )
        assert req.schema_hint == {"type": "object"}


class TestMockBackend:
    def test_responses_in_order_and_dict_wrapping(self):
        b = MockBackend([{"k": 1}, GuidanceResponse(content={"k": 2}, backend="other")])
        r1 = b.complete(GuidanceRequest(task="understand", payload={}))
        r2 = b.complete(GuidanceRequest(task="prepare", payload={}))
        assert r1.content == {"k": 1}
        assert r1.backend == "mock"
        assert json.loads(r1.raw_text) == r1.content
        assert r2.content == {"k": 2}
        assert r2.backend == "other"

    def test_requests_recorded_in_call_order(self):
        b = MockBackend([{}, {}, {}])
        for task in ("search_plan", "understand", "audit_assumptions"):
            b.complete(GuidanceRequest(task=task, payload={}))
        assert [r.task for r in b.requests] == [
            "search_plan",
            "understand",
            "audit_assumptions",
        ]

    def test_exhaustion_raises_runtime_error_naming_task(self):
        b = MockBackend([{"only": True}])
        b.complete(GuidanceRequest(task="understand", payload={}))
        with pytest.raises(RuntimeError, match="review_control_group"):
            b.complete(GuidanceRequest(task="review_control_group", payload={}))

    def test_protocol_runtime_checkable(self):
        assert isinstance(MockBackend([]), GuidanceBackend)


class TestGuidanceLog:
    def test_appends_jsonl_lines(self, tmp_path):
        path = tmp_path / "sub" / "guidance.jsonl"
        log = GuidanceLog(path)
        req = GuidanceRequest(task="understand", payload={"x": 1})
        resp = GuidanceResponse(content={"y": 2}, backend="mock")
        log.append(req, resp)
        log.append(req, resp)
        lines = path.read_text().splitlines()
        assert len(lines) == 2
        for i, line in enumerate(lines):
            entry = json.loads(line)
            assert set(entry) == {"seq", "ts", "task", "backend", "request", "response"}
            assert entry["seq"] == i
            assert entry["task"] == "understand"
            assert entry["backend"] == "mock"
            assert entry["request"]["payload"] == {"x": 1}
            assert entry["response"]["content"] == {"y": 2}

    def test_reopen_appends_not_truncates(self, tmp_path):
        path = tmp_path / "guidance.jsonl"
        req = GuidanceRequest(task="prepare", payload={})
        resp = GuidanceResponse(content={}, backend="mock")
        GuidanceLog(path).append(req, resp)
        log2 = GuidanceLog(path)
        log2.append(req, resp)
        lines = path.read_text().splitlines()
        assert len(lines) == 2
        assert json.loads(lines[1])["seq"] == 1
        assert log2.n_entries == 2


class TestLoggedBackend:
    def test_delegates_and_logs(self, tmp_path):
        log = GuidanceLog(tmp_path / "g.jsonl")
        inner = MockBackend([{"a": 1}, {"b": 2}])
        wrapped = LoggedBackend(inner, log)
        assert wrapped.name == "mock"
        assert isinstance(wrapped, GuidanceBackend)
        r1 = wrapped.complete(GuidanceRequest(task="understand", payload={}))
        assert r1.content == {"a": 1}
        assert r1.backend == "mock"
        assert log.n_entries == 1
        wrapped.complete(GuidanceRequest(task="prepare", payload={}))
        lines = (tmp_path / "g.jsonl").read_text().splitlines()
        assert len(lines) == 2
        assert json.loads(lines[1])["task"] == "prepare"


class TestTaskInstructions:
    def test_exactly_seven_nonempty(self):
        assert set(TASK_INSTRUCTIONS) == set(TASKS)
        for v in TASK_INSTRUCTIONS.values():
            assert isinstance(v, str) and v.strip()
