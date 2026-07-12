"""API-mode guidance backends: Anthropic (Claude) and Gemini, behind the ``[llm]`` extra.

Both backends send one user message built from :data:`TASK_INSTRUCTIONS` and the
request payload, ask for structured JSON output (Anthropic: ``output_config`` with a
``json_schema`` format; Gemini: ``response_mime_type`` + ``response_json_schema``),
and parse the reply into a :class:`~natex.llm.backends.GuidanceResponse`.

The SDKs are imported lazily inside the constructors so this module always imports;
a missing SDK raises ``ImportError`` naming ``pip install 'natex-discovery[llm]'``.

Non-determinism note: API backends are inherently non-deterministic; reproducibility
comes from the guidance log — a run is replayable by feeding the logged responses
back through :class:`~natex.llm.backends.MockBackend`.
"""

from __future__ import annotations

import copy
import json

from natex.llm.backends import TASK_INSTRUCTIONS, GuidanceRequest, GuidanceResponse

_INSTALL_MSG = "requires the {pkg!r} package: pip install 'natex-discovery[llm]'"

# JSON-schema constraint keys unsupported by strict structured-output modes.
_STRIP_KEYS = (
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


def _make_strict(node: object) -> None:
    """In-place walk: strict object nodes, stripped constraints, known recursion keys."""
    if not isinstance(node, dict):
        return
    for key in _STRIP_KEYS:
        node.pop(key, None)
    if node.get("type") == "object":
        node["additionalProperties"] = False
        node["required"] = list(node.get("properties", {}))
    for mapping_key in ("properties", "$defs"):
        sub = node.get(mapping_key)
        if isinstance(sub, dict):
            for child in sub.values():
                _make_strict(child)
    items = node.get("items")
    if isinstance(items, list):
        for child in items:
            _make_strict(child)
    else:
        _make_strict(items)
    for union_key in ("anyOf", "allOf"):
        sub = node.get(union_key)
        if isinstance(sub, list):
            for child in sub:
                _make_strict(child)


def _strict_schema(schema: dict) -> dict:
    """Deep-copied JSON schema made structured-output-safe.

    Every object node gets ``additionalProperties: false`` and
    ``required = list(properties)``; unsupported constraint keys
    (:data:`_STRIP_KEYS`) are removed; recursion covers
    ``properties``/``items``/``anyOf``/``allOf``/``$defs``. Pure function with
    no SDK dependency; the input is never mutated.
    """
    out = copy.deepcopy(schema)
    _make_strict(out)
    return out


def _prompt(request: GuidanceRequest) -> str:
    """Task instructions + sorted-JSON payload + a single-JSON-object directive."""
    return (
        TASK_INSTRUCTIONS[request.task]
        + "\n\nInput payload (JSON):\n"
        + json.dumps(request.payload, indent=1, sort_keys=True, default=str)
        + "\n\nRespond with a single JSON object."
    )


def _parse_json_object(text: str, backend: str) -> dict:
    """``json.loads`` the reply; require a dict; errors carry a <=200-char snippet."""
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{backend} backend returned non-JSON text: {text[:200]!r}") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"{backend} backend reply is not a JSON object: {text[:200]!r}")
    return parsed


class AnthropicBackend:
    """Claude via the Anthropic Messages API with structured (json_schema) output."""

    name = "anthropic"

    def __init__(
        self,
        model: str = "claude-sonnet-5",
        api_key: str | None = None,
        max_tokens: int = 4096,
        _client=None,
    ):
        self.model = model
        self.max_tokens = max_tokens
        if _client is not None:  # test injection: no SDK, no network
            self._client = _client
            return
        try:
            import anthropic
        except ImportError:
            raise ImportError(
                f"AnthropicBackend {_INSTALL_MSG.format(pkg='anthropic')}"
            ) from None
        self._client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()

    def complete(self, request: GuidanceRequest) -> GuidanceResponse:
        kwargs: dict = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": [{"role": "user", "content": _prompt(request)}],
        }
        if request.schema_hint:
            kwargs["output_config"] = {
                "format": {
                    "type": "json_schema",
                    "schema": _strict_schema(request.schema_hint),
                }
            }
        response = self._client.messages.create(**kwargs)
        text = next(
            (block.text for block in response.content if getattr(block, "type", None) == "text"),
            None,
        )
        if text is None:
            raise ValueError("anthropic backend response contained no text block")
        content = _parse_json_object(text, self.name)
        return GuidanceResponse(content=content, raw_text=text, backend=self.name)


class GeminiBackend:
    """Gemini via google-genai with JSON output (``response_json_schema``).

    On SDKs without ``response_json_schema`` support, ``generate_content`` raises
    ``TypeError``; the call falls back ONCE to the schema-free JSON config
    (``response_mime_type`` only) so the request still succeeds — the schema then
    constrains nothing and :func:`_parse_json_object` remains the only validation.
    """

    name = "gemini"

    def __init__(
        self,
        model: str = "gemini-3.1-pro",
        api_key: str | None = None,
        _client=None,
    ):
        self.model = model
        if _client is not None:  # test injection: no SDK, no network
            self._client = _client
            return
        try:
            from google import genai
        except ImportError:
            raise ImportError(
                f"GeminiBackend {_INSTALL_MSG.format(pkg='google-genai')}"
            ) from None
        self._client = genai.Client(api_key=api_key) if api_key else genai.Client()

    def complete(self, request: GuidanceRequest) -> GuidanceResponse:
        prompt = _prompt(request)
        config: dict = {"response_mime_type": "application/json"}
        if request.schema_hint:
            config["response_json_schema"] = _strict_schema(request.schema_hint)
        try:
            resp = self._client.models.generate_content(
                model=self.model, contents=prompt, config=config
            )
        except TypeError:
            if "response_json_schema" not in config:
                raise
            resp = self._client.models.generate_content(
                model=self.model,
                contents=prompt,
                config={"response_mime_type": "application/json"},
            )
        text = resp.text
        content = _parse_json_object(text, self.name)
        return GuidanceResponse(content=content, raw_text=text, backend=self.name)
