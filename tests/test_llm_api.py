"""Tests for natex.llm.api: Anthropic + Gemini backends behind the [llm] extra.

NO network, NO API keys: fake clients are injected via ``_client``; the real-SDK
smoke tests skip gracefully when the extra is not installed, so CI stays green.
"""

import copy
import importlib.util
import json
import sys
from types import SimpleNamespace

import pytest

from natex.intake.prep import PrepPlan
from natex.llm import AnthropicBackend, GeminiBackend, GuidanceRequest
from natex.llm.api import _prompt, _strict_schema
from natex.llm.backends import TASK_INSTRUCTIONS

STRIPPED_KEYS = (
    "minimum",
    "maximum",
    "exclusiveMinimum",
    "exclusiveMaximum",
    "multipleOf",
    "minLength",
    "maxLength",
    "minItems",
    "maxItems",
    "pattern",
)


def _iter_dict_nodes(node):
    if isinstance(node, dict):
        yield node
        for value in node.values():
            yield from _iter_dict_nodes(value)
    elif isinstance(node, list):
        for value in node:
            yield from _iter_dict_nodes(value)


def _has_module(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


class TestStrictSchema:
    def test_prepplan_schema_all_object_nodes_strict(self):
        strict = _strict_schema(PrepPlan.model_json_schema())
        object_nodes = [n for n in _iter_dict_nodes(strict) if n.get("type") == "object"]
        assert object_nodes  # top level + $defs models at minimum
        for node in object_nodes:
            assert node["additionalProperties"] is False
            assert node["required"] == list(node.get("properties", {}))

    def test_no_stripped_constraint_key_remains(self):
        # Regression guard: Subsample.n keeps gt OUT of the schema via a
        # field_validator (task 2); nothing here may reintroduce constraints.
        strict = _strict_schema(PrepPlan.model_json_schema())
        for node in _iter_dict_nodes(strict):
            for key in STRIPPED_KEYS:
                assert key not in node

    def test_strips_constraint_keys_recursively(self):
        schema = {
            "type": "object",
            "properties": {
                "n": {"type": "integer", "exclusiveMinimum": 0, "maximum": 10},
                "s": {"type": "string", "minLength": 1, "pattern": "^a"},
                "arr": {
                    "type": "array",
                    "minItems": 1,
                    "items": {"type": "number", "multipleOf": 2},
                },
                "u": {"anyOf": [{"type": "integer", "minimum": 3}, {"type": "null"}]},
            },
            "$defs": {"Inner": {"type": "object", "properties": {"x": {"maxLength": 5}}}},
        }
        strict = _strict_schema(schema)
        for node in _iter_dict_nodes(strict):
            for key in STRIPPED_KEYS:
                assert key not in node
        assert strict["required"] == ["n", "s", "arr", "u"]
        assert strict["$defs"]["Inner"]["additionalProperties"] is False
        assert strict["$defs"]["Inner"]["required"] == ["x"]

    def test_input_dict_unmutated(self):
        schema = PrepPlan.model_json_schema()
        before = copy.deepcopy(schema)
        _strict_schema(schema)
        assert schema == before


class TestPrompt:
    def test_prompt_contains_instructions_payload_and_json_directive(self):
        req = GuidanceRequest(task="audit_assumptions", payload={"b": 2, "a": 1})
        text = _prompt(req)
        assert text.startswith(TASK_INSTRUCTIONS["audit_assumptions"])
        assert json.dumps({"b": 2, "a": 1}, indent=1, sort_keys=True, default=str) in text
        assert text.endswith("Respond with a single JSON object.")


class TestImportGuards:
    def test_anthropic_missing_names_extra(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "anthropic", None)
        with pytest.raises(ImportError, match="natex-discovery\\[llm\\]"):
            AnthropicBackend()

    def test_gemini_missing_names_extra(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "google", None)
        monkeypatch.setitem(sys.modules, "google.genai", None)
        with pytest.raises(ImportError, match="natex-discovery\\[llm\\]"):
            GeminiBackend()


class FakeAnthropicClient:
    """Records messages.create kwargs; returns a canned text content block."""

    def __init__(self, text: str = '{"veto": false}'):
        self.calls: list[dict] = []
        self._text = text
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(content=[SimpleNamespace(type="text", text=self._text)])


class FakeGeminiClient:
    """Records models.generate_content kwargs; returns a canned .text response."""

    def __init__(self, text: str = '{"veto": false}'):
        self.calls: list[dict] = []
        self._text = text
        self.models = SimpleNamespace(generate_content=self._generate)

    def _generate(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(text=self._text)


class SchemaRejectingGeminiClient(FakeGeminiClient):
    """Simulates an older google-genai SDK without response_json_schema."""

    def _generate(self, **kwargs):
        self.calls.append(kwargs)
        if "response_json_schema" in (kwargs.get("config") or {}):
            raise TypeError("generate_content() got an unexpected keyword 'response_json_schema'")
        return SimpleNamespace(text=self._text)


SCHEMA_HINT = {"type": "object", "properties": {"veto": {"type": "boolean"}}}


class TestAnthropicBackend:
    def test_complete_with_schema_hint(self):
        fake = FakeAnthropicClient()
        backend = AnthropicBackend(_client=fake)
        req = GuidanceRequest(task="audit_assumptions", payload={"k": 1}, schema_hint=SCHEMA_HINT)
        resp = backend.complete(req)
        assert resp.content == {"veto": False}
        assert resp.backend == "anthropic"
        assert resp.raw_text == '{"veto": false}'
        (kwargs,) = fake.calls
        assert kwargs["model"] == "claude-sonnet-5"
        assert kwargs["max_tokens"] == 4096
        assert kwargs["messages"] == [{"role": "user", "content": _prompt(req)}]
        fmt = kwargs["output_config"]["format"]
        assert fmt["type"] == "json_schema"
        assert fmt["schema"]["additionalProperties"] is False
        assert fmt["schema"]["required"] == ["veto"]

    def test_no_output_config_without_schema_hint(self):
        fake = FakeAnthropicClient()
        backend = AnthropicBackend(_client=fake)
        backend.complete(GuidanceRequest(task="understand", payload={}))
        (kwargs,) = fake.calls
        assert "output_config" not in kwargs

    def test_non_json_text_raises_value_error_with_snippet(self):
        text = "x" * 500
        backend = AnthropicBackend(_client=FakeAnthropicClient(text=text))
        with pytest.raises(ValueError, match="non-JSON") as excinfo:
            backend.complete(GuidanceRequest(task="understand", payload={}))
        assert "x" * 200 in str(excinfo.value)
        assert "x" * 201 not in str(excinfo.value)

    def test_json_non_object_raises_value_error(self):
        backend = AnthropicBackend(_client=FakeAnthropicClient(text="[1, 2]"))
        with pytest.raises(ValueError, match="not a JSON object"):
            backend.complete(GuidanceRequest(task="understand", payload={}))

    def test_no_text_block_raises_value_error(self):
        fake = FakeAnthropicClient()
        fake.messages = SimpleNamespace(
            create=lambda **kw: SimpleNamespace(content=[SimpleNamespace(type="thinking")])
        )
        backend = AnthropicBackend(_client=fake)
        with pytest.raises(ValueError, match="no text block"):
            backend.complete(GuidanceRequest(task="understand", payload={}))


class TestGeminiBackend:
    def test_complete_with_schema_hint(self):
        fake = FakeGeminiClient()
        backend = GeminiBackend(_client=fake)
        req = GuidanceRequest(task="review_control_group", payload={"e": []}, schema_hint=SCHEMA_HINT)
        resp = backend.complete(req)
        assert resp.content == {"veto": False}
        assert resp.backend == "gemini"
        assert resp.raw_text == '{"veto": false}'
        (kwargs,) = fake.calls
        assert kwargs["model"] == "gemini-3.1-pro"
        assert kwargs["contents"] == _prompt(req)
        config = kwargs["config"]
        assert config["response_mime_type"] == "application/json"
        assert config["response_json_schema"]["additionalProperties"] is False
        assert config["response_json_schema"]["required"] == ["veto"]

    def test_no_schema_key_without_schema_hint(self):
        fake = FakeGeminiClient()
        backend = GeminiBackend(_client=fake)
        backend.complete(GuidanceRequest(task="understand", payload={}))
        (kwargs,) = fake.calls
        assert kwargs["config"] == {"response_mime_type": "application/json"}

    def test_type_error_falls_back_once_to_schema_free_config(self):
        fake = SchemaRejectingGeminiClient()
        backend = GeminiBackend(_client=fake)
        req = GuidanceRequest(task="understand", payload={}, schema_hint=SCHEMA_HINT)
        resp = backend.complete(req)
        assert resp.content == {"veto": False}
        assert len(fake.calls) == 2
        assert "response_json_schema" in fake.calls[0]["config"]
        assert fake.calls[1]["config"] == {"response_mime_type": "application/json"}

    def test_type_error_without_schema_propagates(self):
        fake = FakeGeminiClient()

        def _boom(**kwargs):
            raise TypeError("unrelated TypeError")

        fake.models = SimpleNamespace(generate_content=_boom)
        backend = GeminiBackend(_client=fake)
        with pytest.raises(TypeError, match="unrelated"):
            backend.complete(GuidanceRequest(task="understand", payload={}))

    def test_non_json_text_raises_value_error_with_snippet(self):
        text = "y" * 500
        backend = GeminiBackend(_client=FakeGeminiClient(text=text))
        with pytest.raises(ValueError, match="non-JSON") as excinfo:
            backend.complete(GuidanceRequest(task="understand", payload={}))
        assert "y" * 200 in str(excinfo.value)
        assert "y" * 201 not in str(excinfo.value)

    def test_json_non_object_raises_value_error(self):
        backend = GeminiBackend(_client=FakeGeminiClient(text='"just a string"'))
        with pytest.raises(ValueError, match="not a JSON object"):
            backend.complete(GuidanceRequest(task="understand", payload={}))


@pytest.mark.skipif(not _has_module("anthropic"), reason="anthropic SDK not installed")
class TestAnthropicSmoke:
    def test_construct_with_real_sdk_no_network(self):
        backend = AnthropicBackend(api_key="test-key")
        assert backend.name == "anthropic"
        assert backend.model == "claude-sonnet-5"


@pytest.mark.skipif(not _has_module("google.genai"), reason="google-genai SDK not installed")
class TestGeminiSmoke:
    def test_construct_with_real_sdk_no_network(self):
        backend = GeminiBackend(api_key="test-key")
        assert backend.name == "gemini"
        assert backend.model == "gemini-3.1-pro"
