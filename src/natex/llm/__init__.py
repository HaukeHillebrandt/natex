"""LLM-as-analyst guidance layer (design spec section 6c).

Backends answer :class:`GuidanceRequest`\\ s with structured
:class:`GuidanceResponse`\\ s; guidance is always advisory and always logged.
"""

from natex.llm.agent import AgentBackend
from natex.llm.api import AnthropicBackend, GeminiBackend
from natex.llm.backends import (
    TASKS,
    GuidanceBackend,
    GuidanceRequest,
    GuidanceResponse,
    MockBackend,
    NullBackend,
)
from natex.llm.log import GuidanceLog, LoggedBackend

__all__ = [
    "TASKS",
    "AgentBackend",
    "AnthropicBackend",
    "GeminiBackend",
    "GuidanceBackend",
    "GuidanceLog",
    "GuidanceRequest",
    "GuidanceResponse",
    "LoggedBackend",
    "MockBackend",
    "NullBackend",
]
