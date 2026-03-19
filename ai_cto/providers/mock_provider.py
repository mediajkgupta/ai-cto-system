"""
mock_provider.py — Mock LLM provider for tests / no-API-key runs.

When agents use get_provider().complete(...) in mock mode, they receive
a plausible-looking response string. The actual mock *node functions*
(mock_planning_node, mock_coding_node, etc.) in mock_llm.py bypass the
agents entirely and are injected at the graph level, so MockProvider.complete()
is only reached in edge cases (e.g. a new agent added without a mock node).

Returns a minimal valid JSON payload matching the format agents expect so
the pipeline degrades gracefully rather than crashing.
"""

import logging
from ai_cto.providers.base import LLMProvider

logger = logging.getLogger(__name__)

_FALLBACK_RESPONSE = """{
  "files": {
    "main.py": "# Mock fallback\\nprint('mock')"
  },
  "entry_point": "main.py",
  "explanation": "Mock provider fallback response."
}"""


class MockProvider(LLMProvider):
    """Lightweight mock — returns a static JSON response, no network call."""

    def complete(self, system: str, user: str, max_tokens: int = 2048) -> str:
        logger.info(
            "[MockProvider] Returning fallback response (no API call). "
            "system[:40]=%r", system[:40]
        )
        return _FALLBACK_RESPONSE
